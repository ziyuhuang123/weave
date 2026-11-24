"""Weave turnkey autotune: compile-time calibration + simple runtime auto-selection.

Two parts, one module:

Part A -- calibration (run once per GPU model / EP size / MoE shape)
    python -m deep_gemm.mega.autotune --num-experts 512 --num-topk 10 \
        --hidden 8192 --intermediate-hidden 2048 --num-processes 4
or, inside an already-initialised process group,
    deep_gemm.mega.autotune_moe(group, num_experts, num_topk, hidden, intermediate_hidden)

  1. Hardware fingerprint from the live CUDA runtime (compute capability, device
     name, SM count, shared memory per SM, L2 size, EP = group size).  Nothing is
     keyed on a card name: every SM-count dependent value is a fraction of the
     queried SM count (aligned to the cluster size 2 on SM10x).
  2. Per S bucket, a small architecture-family candidate space pruned by the
     expected rows per expert M_exp = S * topk / E, every candidate compiled
     (optionally with the 0-spill ptxas check) and accuracy-guarded against the
     same-precision reference output before it is timed; failures are skipped
     and logged, never fatal.
  3. Staged search per bucket (GEMM tuple -> one co-knob at a time -> combine ->
     second GEMM tuple), each step a paired palindromic ABBA comparison against
     the current incumbent, slowest rank per window.
  4. Break-even probes for the four runtime rules (dedup, comm-owned scatter,
     skewed-routing spatial tier, ready-group size).
  The result is one JSON profile per (hardware, EP, model shape, precision):
    ~/.deep_gemm/weave_autotune/{hw_id}__E{E}_K{K}_H{H}_I{I}_S{shared}_{mma}.json
  (override the directory with WEAVE_AUTOTUNE_DIR or the file with WEAVE_AUTOTUNE_FILE).

Part B -- runtime selection (``weave_mega_moe`` calls ``select_launch_env``)
  * Any explicit public/legacy WEAVE_* tuning variable in the environment ->
    manual mode: nothing is injected, the user's settings are used verbatim.
  * WEAVE_AUTOTUNE=0 -> compiled defaults, nothing is injected.
  * Otherwise the profile is loaded once (in-memory cache), S_global = tokens *
    EP is mapped to the nearest calibrated bucket (log2 distance), the bucket's
    measured winner env is used, and -- when routing statistics are available
    and the bucket has calibrated rule variants -- the four threshold rules
    adapt it.  Only variants that were compiled and accuracy-verified during
    calibration can be produced; anything else falls back to the bucket base.
  * No profile -> architecture-family cold-start defaults.
  The chosen env is applied only for the duration of the backend call.

The pure-Python half of this module (fingerprints, candidates, profile schema,
bucket lookup, rules) imports no torch so it can be unit-tested anywhere.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "weave_autotune/v1"
TOOL_VERSION = "1.0"
DEFAULT_S_BUCKETS: Tuple[int, ...] = (16, 128, 512, 1024, 4096, 8192, 16384)

# Every WEAVE_* name the selector may inject.  Anything else in a profile is
# rejected at load time (a profile can never smuggle an internal switch).
INJECTABLE_KNOBS: Tuple[str, ...] = (
    "WEAVE_DISPATCH_ACTIVE_SMS", "WEAVE_SCATTER_ACTIVE_SMS", "WEAVE_DISPATCH_WARPS",
    "WEAVE_SCATTER_WARPS", "WEAVE_CU_WAVE_SPLIT", "WEAVE_V_HEAD", "WEAVE_READY_GROUP_SIZE",
    "WEAVE_SCATTER_OWNER", "WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG",
    "WEAVE_REG_PULL_BYTES", "WEAVE_DEDUP_MODE", "WEAVE_SCATTER_STAGING",
)

# Presence of any of these means "the user is tuning by hand": manual mode.
# WEAVE_PRECISION_MODE only selects the backend and is not a tuning knob.
_MANUAL_PREFIXES: Tuple[str, ...] = ("WEAVE_",)
# Eager timing: a GPU-side spin (in SM clock cycles, ~1 ms at 2 GHz) queued before the timed
# launch, so the host launch path of every arm is hidden and the events time the kernel only
# (host-side selection overhead is reported separately by the e2e benchmark).
EAGER_HEAD_START_CYCLES = 2_000_000
# Register-spill policy of the compile guard, per kernel family.
#   reject  a candidate whose ptxas build reports any spill store/load is never measured
#           (Blackwell: the MegaMoE kernels build with 0 B spill, see the compile proof)
#   record  spill bytes are recorded in the verdict file but do not reject: on sm_90a the
#           baseline kernels themselves spill (the official DeepGEMM SM90 L1 kernel included),
#           so the measured latency + the accuracy guard decide
SPILL_POLICY = {"blackwell": "reject", "hopper": "record"}
_NON_TUNING_ENV: frozenset = frozenset({
    "WEAVE_PRECISION_MODE", "WEAVE_KNOB_PRECISION_MODE",
    "WEAVE_AUTOTUNE", "WEAVE_AUTOTUNE_DIR", "WEAVE_AUTOTUNE_FILE",
    "WEAVE_AUTOTUNE_ROUTING", "WEAVE_AUTOTUNE_VERBOSE", "WEAVE_REPO_ROOT",
})

SCATTER_EPI = {"WEAVE_SCATTER_OWNER": "epilogue_only", "WEAVE_SCATTER_STAGING": "reg_2stage"}
SCATTER_COMM = {"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem"}


# =============================================================================
# Hardware fingerprint
# =============================================================================

def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "gpu"


@dataclass(frozen=True)
class HardwareFingerprint:
    cc_major: int
    cc_minor: int
    device_name: str
    num_sms: int
    smem_per_sm: int
    l2_bytes: int
    ep: int
    gcn_arch: str = ""

    @property
    def hw_id(self) -> str:
        if self.gcn_arch:
            return f"{slug(self.gcn_arch)}_{slug(self.device_name)}_{self.num_sms}cu_ep{self.ep}"
        return f"sm{self.cc_major}{self.cc_minor}_{slug(self.device_name)}_{self.num_sms}sm_ep{self.ep}"

    @property
    def family(self) -> Optional[str]:
        return arch_family(self.cc_major, self.gcn_arch)

    @property
    def sm_align(self) -> int:
        # SM10x launches 2-CTA clusters; CDNA3 partitions CUs evenly across 4/8 XCDs.
        if self.family == "cdna3":
            return 4
        return 2 if self.cc_major == 10 else 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.gcn_arch:
            d.pop("gcn_arch", None)
        d["hw_id"] = self.hw_id
        d["family"] = self.family
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HardwareFingerprint":
        return HardwareFingerprint(int(d["cc_major"]), int(d["cc_minor"]), str(d["device_name"]),
                                   int(d["num_sms"]), int(d.get("smem_per_sm", 0)),
                                   int(d.get("l2_bytes", 0)), int(d["ep"]),
                                   str(d.get("gcn_arch", "")))


def arch_family(cc_major: int, gcn_arch: str = "") -> Optional[str]:
    """MegaMoE kernel family of a compute capability / GCN arch (None = unsupported)."""
    if gcn_arch and str(gcn_arch).lower().startswith("gfx94"):
        return "cdna3"       # gfx942 (MI308X 80-CU / MI300X 304-CU / MI325X 304-CU)
    if cc_major == 10:
        return "blackwell"   # sm_100a (B200) and sm_103a (B300) share the SM10x kernels
    if cc_major == 9:
        return "hopper"      # sm_90a (H20 / H100 / H800 / H200)
    return None


def query_fingerprint(ep: int, device: Optional[int] = None) -> HardwareFingerprint:
    import torch
    dev = torch.cuda.current_device() if device is None else device
    p = torch.cuda.get_device_properties(dev)
    smem = int(getattr(p, "shared_memory_per_multiprocessor", 0) or
               getattr(p, "max_shared_memory_per_multiprocessor", 0) or 0)
    l2 = int(getattr(p, "L2_cache_size", 0) or getattr(p, "l2_cache_size", 0) or 0)
    gcn_raw = str(getattr(p, "gcnArchName", "") or "")
    gcn = gcn_raw.split(":")[0] if gcn_raw else ""
    return HardwareFingerprint(int(p.major), int(p.minor), str(p.name), int(p.multi_processor_count),
                               smem, l2, int(ep), gcn)


def model_key(num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int,
              num_shared_experts: int, mma_type: str) -> str:
    return (f"E{int(num_experts)}_K{int(num_topk)}_H{int(hidden)}_I{int(intermediate_hidden)}"
            f"_S{int(num_shared_experts)}_{mma_type}")


def default_autotune_dir() -> Path:
    d = os.environ.get("WEAVE_AUTOTUNE_DIR")
    if d:
        return Path(d).expanduser()
    return Path.home() / ".deep_gemm" / "weave_autotune"


def profile_path(hw: HardwareFingerprint, num_experts: int, num_topk: int, hidden: int,
                 intermediate_hidden: int, num_shared_experts: int, mma_type: str) -> Path:
    f = os.environ.get("WEAVE_AUTOTUNE_FILE")
    if f:
        return Path(f).expanduser()
    return default_autotune_dir() / (
        f"{hw.hw_id}__{model_key(num_experts, num_topk, hidden, intermediate_hidden, num_shared_experts, mma_type)}.json")


def default_mma_type(family: Optional[str]) -> str:
    return "fp8xfp8" if family in ("hopper", "cdna3") else "fp8xfp4"


# =============================================================================
# Candidate space (architecture family generic, SM count proportional)
# =============================================================================

def align_down(x: int, a: int) -> int:
    return max(a, (int(x) // a) * a)


def sm_fraction(frac: float, hw: HardwareFingerprint) -> int:
    """round(frac * num_sms), aligned down to the cluster size, clamped to [align, num_sms]."""
    v = align_down(int(round(frac * hw.num_sms)), hw.sm_align)
    return int(min(max(v, hw.sm_align), align_down(hw.num_sms, hw.sm_align)))


def m_expected(s_global: int, num_topk: int, num_experts: int) -> float:
    return float(s_global) * float(num_topk) / float(max(1, num_experts))


def _pair(g1: str, g2: Optional[str] = None) -> Dict[str, str]:
    return {"WEAVE_GEMM1_CONFIG": g1, "WEAVE_GEMM2_CONFIG": g1 if g2 is None else g2}


def gemm_candidates(family: str, m_exp: float) -> List[Dict[str, str]]:
    """3-5 plausible whole-GEMM tuples around M_exp (never BM=240 at tiny M)."""
    if family == "blackwell":
        if m_exp <= 12:
            tuples = [_pair("16:8:256"), _pair("32:16:128"), _pair("64:16:128")]
        elif m_exp <= 48:
            tuples = [_pair("32:16:128"), _pair("64:16:128"), _pair("80:8:128:0:2"),
                      _pair("96:16", "96:8")]
        elif m_exp <= 128:
            tuples = [_pair("64:16:128"), _pair("80:8", "80:8"), _pair("96:16", "96:8"),
                      _pair("128:32:128:0:1")]
        else:
            tuples = [_pair("96:16", "96:8"), _pair("128:32:128:0:1"), _pair("192:32", "192:32"),
                      _pair("240:24", "240:8")]
        return tuples
    if family == "hopper":
        if m_exp <= 32:
            pairs = [(16, 16), (16, 32), (32, 64), (64, 64)]
        elif m_exp <= 96:
            pairs = [(32, 64), (64, 64), (128, 128)]
        else:
            pairs = [(64, 64), (128, 128)]
        return [_pair(str(a), str(b)) for a, b in pairs]
    if family == "cdna3":
        if m_exp <= 16:
            pairs = [(16, 16), (32, 32)]
        elif m_exp <= 64:
            pairs = [(32, 32), (64, 64)]
        else:
            pairs = [(64, 64), (128, 128), (0, 0)]
        return [_pair(str(a), str(b)) for a, b in pairs]
    raise ValueError(f"unsupported architecture family {family!r}")


def coknob_variants(family: str, hw: HardwareFingerprint, s_global: int,
                    base_env: Dict[str, str]) -> List[Tuple[str, str, Dict[str, str]]]:
    """Single co-knob variants on top of a base env: (group, name, delta)."""
    out: List[Tuple[str, str, Dict[str, str]]] = []
    base_comm = base_env.get("WEAVE_SCATTER_OWNER") == "comm_only"
    if family == "blackwell":
        out.append(("pull", "reg_pull_1024", {"WEAVE_REG_PULL_BYTES": "1024"}))
        out.append(("pull", "reg_pull_2048", {"WEAVE_REG_PULL_BYTES": "2048"}))
        out.append(("scatter", "scatter_comm_smem" if not base_comm else "scatter_epilogue",
                    dict(SCATTER_EPI if base_comm else SCATTER_COMM)))
        out.append(("sms", "dispatch_sms_0.5", {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.5, hw))}))
        out.append(("sms", "dispatch_sms_0.9", {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.9, hw))}))
        if s_global <= 2048:
            out.append(("head", "v_head_1", {"WEAVE_V_HEAD": "1"}))
        return out
    if family == "hopper":
        out.append(("scatter", "scatter_epilogue" if base_comm else "scatter_comm_smem",
                    dict(SCATTER_EPI if base_comm else SCATTER_COMM)))
        out.append(("warps", "dispatch_warps_8", {"WEAVE_DISPATCH_WARPS": "8"}))
        out.append(("warps", "dispatch_warps_4", {"WEAVE_DISPATCH_WARPS": "4"}))
        out.append(("warps", "scatter_warps_8", {"WEAVE_SCATTER_WARPS": "8"}))
        d85, d67 = sm_fraction(0.85, hw), sm_fraction(0.67, hw)
        out.append(("sms", "dispatch_sms_0.85", {"WEAVE_DISPATCH_ACTIVE_SMS": str(d85)}))
        out.append(("sms", "dispatch_sms_0.67", {"WEAVE_DISPATCH_ACTIVE_SMS": str(d67)}))
        if base_comm:
            out.append(("sms", "dispatch_scatter_sms_0.67",
                        {"WEAVE_DISPATCH_ACTIVE_SMS": str(d67), "WEAVE_SCATTER_ACTIVE_SMS": str(d67)}))
        out.append(("head", "v_head_10n", {"WEAVE_V_HEAD": str(10 * hw.num_sms)}))
        return out
    if family == "cdna3":
        out.append(("waves", "cu_wave_4_4", {"WEAVE_CU_WAVE_SPLIT": "4-4"}))
        out.append(("waves", "cu_wave_0_8", {"WEAVE_CU_WAVE_SPLIT": "0-8"}))
        out.append(("waves", "cu_wave_8_0", {"WEAVE_CU_WAVE_SPLIT": "8-0"}))
        out.append(("warps", "scatter_warps_8", {"WEAVE_SCATTER_WARPS": "8", "WEAVE_V_HEAD": "0"}))
        out.append(("scatter", "scatter_epilogue", {"WEAVE_SCATTER_OWNER": "epilogue_only"}))
        out.append(("sms", "dispatch_cus_0.6", {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.6, hw))}))
        out.append(("pull", "reg_pull_8192", {"WEAVE_REG_PULL_BYTES": "8192"}))
        return out
    raise ValueError(f"unsupported architecture family {family!r}")


def cold_start_env(family: Optional[str], hw: Optional[HardwareFingerprint] = None,
                   s_global: int = 0) -> Dict[str, str]:
    """Defaults used when no calibrated profile exists (no card names, no SM constants).

    Blackwell: the compiled defaults already carry the host shape-adaptive GEMM
    tuple and all-SM dispatch, so nothing is injected.  Hopper: comm-owned
    scatter through the SMEM ring is the configuration every measured SM90
    winner shares, so it is the family default.  CDNA3: 0-8 pure compute waves
    for small decode S <= 128 and 4-4 comm/compute split with 8192B wave64
    register pull for S > 128.
    """
    if family == "hopper":
        return dict(SCATTER_COMM)
    if family == "cdna3":
        return {
            "WEAVE_CU_WAVE_SPLIT": "0-8" if (0 < s_global <= 128) else "4-4",
            "WEAVE_SCATTER_OWNER": "comm_only",
            "WEAVE_SCATTER_STAGING": "reg_2stage",
            "WEAVE_REG_PULL_BYTES": "8192",
            "WEAVE_DEDUP_MODE": "1",
        }
    return {}


def comm_heavy_candidates(family: str, hw: HardwareFingerprint) -> List[Dict[str, str]]:
    if family == "blackwell":
        return [{"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.9, hw))},
                {"WEAVE_REG_PULL_BYTES": "2048"}]
    if family == "cdna3":
        return [{"WEAVE_CU_WAVE_SPLIT": "4-4", "WEAVE_SCATTER_WARPS": "8", "WEAVE_V_HEAD": "0"}]
    return [{"WEAVE_DISPATCH_WARPS": "8"}]


def comp_heavy_candidates(family: str, hw: HardwareFingerprint) -> List[Dict[str, str]]:
    if family == "blackwell":
        return [{"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.5, hw))}]
    if family == "cdna3":
        return [{"WEAVE_CU_WAVE_SPLIT": "0-8"},
                {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.6, hw))}]
    return [{"WEAVE_DISPATCH_WARPS": "4"},
            {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.67, hw))}]


def env_key(env: Dict[str, Any]) -> str:
    return ";".join(f"{k}={env[k]}" for k in sorted(env))


def normalize_env(env: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for k, v in env.items():
        if k not in INJECTABLE_KNOBS:
            raise ValueError(f"{k} is not an injectable public Weave knob")
        out[k] = str(v)
    return out


def static_legal(env: Dict[str, str], family: str) -> Optional[str]:
    """Python mirror of the host DG_HOST_ASSERTs that are cheap to check up front.

    Returns a reason string when the env is certainly rejected by the host.
    Anything subtler is caught by the compile step itself.
    """
    owner = env.get("WEAVE_SCATTER_OWNER")
    staging = env.get("WEAVE_SCATTER_STAGING")
    if staging in ("smem", "hbm_dedup", "1", "2") and owner in ("epilogue_only", "0"):
        return "staged scatter requires comm_only owner"
    if int(env.get("WEAVE_SCATTER_ACTIVE_SMS", "0") or 0) > 0 and owner in ("epilogue_only", "0"):
        return "scatter SM throttle requires comm_only owner"
    rp = int(env.get("WEAVE_REG_PULL_BYTES", "0") or 0)
    legal_rp = (0, 1024, 2048, 4096, 8192) if family == "cdna3" else (0, 1024, 2048, 4096)
    if rp not in legal_rp:
        return "illegal WEAVE_REG_PULL_BYTES"
    for k in ("WEAVE_DISPATCH_WARPS", "WEAVE_SCATTER_WARPS"):
        if int(env.get(k, "0") or 0) not in (0, 4, 8):
            return f"illegal {k}"
    if int(env.get("WEAVE_READY_GROUP_SIZE", "0") or 0) not in (0, 1, 2, 4, 8, 16):
        return "illegal WEAVE_READY_GROUP_SIZE"
    if family in ("hopper", "cdna3"):
        g1 = env.get("WEAVE_GEMM1_CONFIG", "0")
        g2 = env.get("WEAVE_GEMM2_CONFIG", "0")
        if ":" in g1 or ":" in g2 or "," in g1 or "," in g2:
            return "SM90/CDNA3 accepts bare BM GEMM configs only"
    if family == "cdna3":
        ws = env.get("WEAVE_CU_WAVE_SPLIT")
        if ws is not None and ws not in ("0", "1", "2", "3", "auto", "0-8", "4-4", "8-0"):
            return "CDNA3 WEAVE_CU_WAVE_SPLIT must be auto, 0-8, 4-4, or 8-0"
    return None


def gemm2_block_m(env: Dict[str, str], default: int = 128) -> int:
    g = env.get("WEAVE_GEMM2_CONFIG") or env.get("WEAVE_GEMM1_CONFIG") or "0"
    try:
        bm = int(re.split(r"[:,]", g)[0])
    except ValueError:
        bm = 0
    return bm if bm > 0 else default


# =============================================================================
# Routing statistics and the four runtime rules
# =============================================================================

@dataclass
class RoutingStats:
    """Receiver-side routing statistics of the bottleneck rank.

    m_rem   remote routed rows received (one row per (token, expert) slot)
    d_in    unique (src_rank, token) remote pulls (= rows after pull dedup)
    d_out   FC2 output rows scattered back to remote ranks (per-slot rows)
    k_tiles GEMM2 M tiles = sum_e ceil(m_e / BM2) over the rank's local experts
    m_max   hottest local expert row count
    mean_m  mean rows per local expert
    """
    m_rem: int
    d_in: int
    d_out: int
    k_tiles: int
    m_max: int
    mean_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RoutingStats":
        return RoutingStats(int(d["m_rem"]), int(d["d_in"]), int(d["d_out"]), int(d["k_tiles"]),
                            int(d["m_max"]), float(d["mean_m"]))


def routing_stats_from_topk(topk_global, num_experts: int, world: int, bm2: int) -> RoutingStats:
    """Exact per-receiver statistics from the global (S, K) top-k table (CPU or GPU tensor).

    Per receiver rank j: tokens of rank r != j hitting j contribute their slot
    count to m_rem[j] and 1 to d_in[j].  The returned record takes the maximum
    over receivers of every field (the slowest rank bounds the layer).
    """
    import torch
    ti = topk_global.to(torch.int64)
    S = ti.shape[0]
    tokens = S // world
    epr = num_experts // world
    valid = ti >= 0
    dst = torch.where(valid, ti // epr, torch.full_like(ti, -1))
    src = (torch.arange(S, device=ti.device) // max(1, tokens)).clamp(max=world - 1)
    hist = torch.bincount(ti[valid], minlength=num_experts)
    per_rank = []
    for j in range(world):
        hit_slots = (dst == j).sum(dim=1)
        remote = src != j
        m_rem = int(hit_slots[remote].sum())
        d_in = int((hit_slots[remote] > 0).sum())
        h = hist[j * epr:(j + 1) * epr]
        k_tiles = int(((h + bm2 - 1) // bm2).sum())
        m_max = int(h.max()) if h.numel() else 0
        mean_m = float(h.sum()) / float(max(1, epr))
        per_rank.append(RoutingStats(m_rem, d_in, m_rem, k_tiles, m_max, mean_m))
    return merge_rank_stats(per_rank)


def merge_rank_stats(per_rank: Sequence["RoutingStats"]) -> "RoutingStats":
    """Bottleneck view: max of every volume field; (m_max, mean_m) as the pair of the most skewed rank."""
    hot = max(per_rank, key=lambda r: r.m_max / max(r.mean_m, 1e-9))
    return RoutingStats(max(r.m_rem for r in per_rank), max(r.d_in for r in per_rank),
                        max(r.d_out for r in per_rank), max(r.k_tiles for r in per_rank),
                        hot.m_max, hot.mean_m)


def skew_ratio(stats: RoutingStats) -> float:
    return float(stats.m_max) / max(stats.mean_m, 1e-9)


RULE_ORDER = ("dedup", "comm_scatter", "spatial_tier", "ready_group")


def rule_flips(bucket: Dict[str, Any], thresholds: Dict[str, Any],
               stats: Optional[RoutingStats]) -> Dict[str, Dict[str, str]]:
    """The env deltas the four calibrated threshold rules request for `stats`."""
    flips: Dict[str, Dict[str, str]] = {}
    if stats is None:
        return flips
    enabled = bucket.get("rules_enabled", {})
    # 1. dedup: enough rows saved by pulling each remote token once
    thr = thresholds.get("dedup_min_saved_rows")
    if thr is not None and enabled.get("dedup", True):
        flips["dedup"] = {"WEAVE_DEDUP_MODE": "1" if stats.m_rem - stats.d_in >= int(thr) else "0"}
    # 2. comm-owned scatter: enough remote FC2 rows and enough GEMM2 tiles to hide it
    rows, tiles = thresholds.get("comm_scatter_min_rows"), thresholds.get("comm_scatter_min_tiles")
    if rows is not None and tiles is not None and enabled.get("comm_scatter", True):
        use = stats.d_out >= int(rows) and stats.k_tiles >= int(tiles)
        flips["comm_scatter"] = dict(SCATTER_COMM if use else SCATTER_EPI)
    # 3. spatial tier relative to the bucket's uniform-routing expectation
    exp = bucket.get("uniform_stats")
    if exp and enabled.get("spatial_tier", True):
        exp_tiles, exp_pull = float(exp["k_tiles"]), float(exp["d_in"])
        if stats.k_tiles < 0.6 * exp_tiles and stats.d_in > exp_pull and thresholds.get("skew_comm_heavy"):
            flips["spatial_tier"] = dict(thresholds["skew_comm_heavy"])
        elif stats.k_tiles > 1.5 * exp_tiles and thresholds.get("skew_comp_heavy"):
            flips["spatial_tier"] = dict(thresholds["skew_comp_heavy"])
    # 4. ready-group publication under hot-expert skew
    rgs = int(thresholds.get("skew_rgs") or 0)
    if rgs > 0 and enabled.get("ready_group", True):
        flips["ready_group"] = {"WEAVE_READY_GROUP_SIZE": str(rgs if stats.m_max > 3.0 * stats.mean_m else 0)}
    return flips


def compose_env(base: Dict[str, str], flips: Dict[str, Dict[str, str]],
                names: Iterable[str]) -> Dict[str, str]:
    env = dict(base)
    for n in names:
        env.update(flips.get(n, {}))
    # canonical form: drop explicit zeros that equal the compiled default
    for k in ("WEAVE_READY_GROUP_SIZE",):
        if env.get(k) == "0":
            env.pop(k)
    return env


def apply_rules(bucket: Dict[str, Any], thresholds: Dict[str, Any],
                stats: Optional[RoutingStats]) -> Tuple[Dict[str, str], List[str]]:
    """Bucket base env adapted by the rules, restricted to verified variants.

    Returns (env, fired_rule_names).  If the full flip set is not a verified
    variant, lower-priority flips are dropped (ready_group, then spatial_tier,
    comm_scatter, dedup) until a verified env is found; the base env is always
    verified.
    """
    base = dict(bucket["env"])
    flips = rule_flips(bucket, thresholds, stats)
    if not flips:
        return base, []
    verified = {env_key(v) for v in bucket.get("verified_variants", [])}
    verified.add(env_key(base))
    active = [n for n in RULE_ORDER if n in flips]
    # Every subset, largest first, then by rule priority (dedup > comm_scatter > spatial > ready_group).
    import itertools
    subsets = []
    for r in range(len(active), 0, -1):
        subsets += list(itertools.combinations(active, r))
    for keep in subsets:
        env = compose_env(base, flips, keep)
        if env_key(env) in verified:
            fired = [n for n in keep if env_key(compose_env(base, flips, [n])) != env_key(base)]
            return env, fired
    return base, []


# =============================================================================
# Profile (JSON) schema
# =============================================================================

class ProfileError(ValueError):
    pass


def validate_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ProfileError(f"not a {SCHEMA} profile")
    for key in ("hw", "model", "buckets", "thresholds"):
        if key not in doc:
            raise ProfileError(f"profile misses {key!r}")
    HardwareFingerprint.from_dict(doc["hw"])
    buckets = doc["buckets"]
    if not isinstance(buckets, list) or not buckets:
        raise ProfileError("profile has no buckets")
    last = 0
    for b in buckets:
        s = int(b["S"])
        if s <= last:
            raise ProfileError("buckets must be strictly increasing in S")
        last = s
        normalize_env(b["env"])
        for v in b.get("verified_variants", []):
            normalize_env(v)
    th = doc["thresholds"]
    for key in ("dedup_min_saved_rows", "comm_scatter_min_rows", "comm_scatter_min_tiles", "skew_rgs"):
        if key not in th:
            raise ProfileError(f"thresholds miss {key!r}")
    for key in ("skew_comm_heavy", "skew_comp_heavy"):
        normalize_env(th.get(key) or {})
    return doc


def load_profile(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return validate_profile(json.load(f))


def save_profile(doc: Dict[str, Any], path: os.PathLike) -> Path:
    validate_profile(doc)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, p)
    return p


def select_bucket(doc: Dict[str, Any], s_global: int) -> Dict[str, Any]:
    """Nearest calibrated bucket in log2(S) (ties go to the larger bucket)."""
    s = max(1, int(s_global))
    best, best_d = None, None
    for b in doc["buckets"]:
        d = abs(math.log2(s) - math.log2(max(1, int(b["S"]))))
        if best is None or d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and int(b["S"]) > int(best["S"])):
            best, best_d = b, d
    return best


# =============================================================================
# Runtime selector (Part B)
# =============================================================================

_MANUAL_PREFIXES_B: Tuple[bytes, ...] = tuple(x.encode() for x in _MANUAL_PREFIXES)


def manual_env_keys(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """WEAVE_* tuning variables set (non-empty) in the environment.

    Hot path (called per launch): scans the raw key table of os.environ without
    decoding every entry, and decodes only the WEAVE_* candidates.
    """
    if environ is None:
        data = getattr(os.environ, "_data", None)
        if isinstance(data, dict):
            out = []
            for k in data:
                if isinstance(k, bytes):
                    if not k.startswith(_MANUAL_PREFIXES_B):
                        continue
                    k = k.decode("utf-8", "surrogateescape")
                elif not k.startswith(_MANUAL_PREFIXES):
                    continue
                if k not in _NON_TUNING_ENV and os.environ.get(k, "") != "":
                    out.append(k)
            return sorted(out)
        environ = os.environ
    return sorted(k for k, v in environ.items()
                  if k.startswith(_MANUAL_PREFIXES) and k not in _NON_TUNING_ENV and v != "")


@dataclass
class Selection:
    env: Dict[str, str]
    source: str                 # manual | disabled | profile | cold_start | unsupported
    bucket_s: Optional[int] = None
    rules_fired: List[str] = field(default_factory=list)
    profile_path: Optional[str] = None


class RuntimeSelector:
    """Process-wide profile cache + per-call env selection."""

    def __init__(self):
        self._lock = threading.Lock()
        self._profiles: Dict[Tuple, Optional[Dict[str, Any]]] = {}
        self._fingerprints: Dict[Tuple[int, int], HardwareFingerprint] = {}
        self._stats_cache: Dict[Tuple, Tuple[int, RoutingStats]] = {}
        self._sticky: Dict[Tuple, Tuple[Dict[str, str], List[str]]] = {}
        self._fast: Dict[Tuple, Selection] = {}
        self.last: Optional[Selection] = None

    def clear(self):
        with self._lock:
            self._profiles.clear()
            self._fingerprints.clear()
            self._stats_cache.clear()
            self._sticky.clear()
            self._fast.clear()

    # -- profile lookup ------------------------------------------------------
    def fingerprint(self, ep: int, device: int) -> HardwareFingerprint:
        key = (device, ep)
        fp = self._fingerprints.get(key)
        if fp is None:
            fp = query_fingerprint(ep, device)
            self._fingerprints[key] = fp
        return fp

    def profile_for(self, hw: HardwareFingerprint, mk: Tuple) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        key = (hw, mk, os.environ.get("WEAVE_AUTOTUNE_FILE"), os.environ.get("WEAVE_AUTOTUNE_DIR"))
        if key in self._profiles:
            doc = self._profiles[key]
            return doc, (doc or {}).get("_path")
        path = profile_path(hw, *mk)
        doc = None
        if path.is_file():
            try:
                doc = load_profile(path)
                got = HardwareFingerprint.from_dict(doc["hw"])
                m = doc["model"]
                same_model = (int(m["E"]), int(m["K"]), int(m["H"]), int(m["I"]), int(m.get("shared", 0)),
                              str(m["mma_type"])) == tuple(mk)
                if got.hw_id != hw.hw_id or not same_model:
                    _warn_once(f"Weave autotune profile {path} was calibrated for {got.hw_id} / {m}, "
                               f"not {hw.hw_id} / {mk}; ignoring it")
                    doc = None
                else:
                    doc["_path"] = str(path)
            except Exception as exc:  # a broken profile never breaks a launch
                _warn_once(f"Weave autotune profile {path} unusable ({exc}); using cold-start defaults")
                doc = None
        with self._lock:
            self._profiles[key] = doc
        return doc, (str(path) if doc else None)

    # -- per call -------------------------------------------------------------
    def select(self, *, ep: int, device: int, num_experts: int, num_topk: int, hidden: int,
               intermediate_hidden: int, num_shared_experts: int, mma_type: str, s_global: int,
               stats_fn=None, buffer_key=None, capturing: bool = False) -> Selection:
        if os.environ.get("WEAVE_AUTOTUNE", "1").strip().lower() in ("0", "off", "false", "no"):
            sel = Selection({}, "disabled")
            self.last = sel
            return sel
        manual = manual_env_keys()
        if manual:
            sel = Selection({}, "manual")
            self.last = sel
            return sel
        mk = (int(num_experts), int(num_topk), int(hidden), int(intermediate_hidden),
              int(num_shared_experts), str(mma_type))
        fast_key = (device, ep, mk, int(s_global), buffer_key)
        hw = self.fingerprint(ep, device)
        doc, path = self.profile_for(hw, mk)
        if doc is None:
            fam = hw.family
            sel = Selection(cold_start_env(fam, hw, s_global), "cold_start" if fam else "unsupported")
            self.last = sel
            return sel
        bucket = select_bucket(doc, s_global)
        th = doc["thresholds"]
        rules_possible = bool(bucket.get("verified_variants"))
        if not rules_possible:
            sel = self._fast.get(fast_key)
            if sel is None:
                sel = Selection(dict(bucket["env"]), "profile", int(bucket["S"]), [], path)
                self._fast[fast_key] = sel
            self.last = sel
            return sel
        sticky_key = (buffer_key, int(s_global))
        if capturing or stats_fn is None or \
                os.environ.get("WEAVE_AUTOTUNE_ROUTING", "auto").strip().lower() in ("0", "off", "static"):
            env, fired = self._sticky.get(sticky_key, (dict(bucket["env"]), []))
            sel = Selection(env, "profile", int(bucket["S"]), list(fired), path)
            self.last = sel
            return sel
        stats = stats_fn(gemm2_block_m(bucket["env"]))
        env, fired = apply_rules(bucket, th, stats)
        with self._lock:
            self._sticky[sticky_key] = (env, fired)
        sel = Selection(env, "profile", int(bucket["S"]), fired, path)
        self.last = sel
        return sel


_SELECTOR = RuntimeSelector()
_WARNED: set = set()


def _warn_once(msg: str):
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    import warnings
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def get_runtime_selector() -> RuntimeSelector:
    return _SELECTOR


_ENV_LOCK = threading.RLock()


@contextlib.contextmanager
def scoped_env(env: Dict[str, str]):
    """Apply `env` for the duration of a launch and restore the previous values."""
    if not env:
        yield
        return
    with _ENV_LOCK:
        saved = {k: os.environ.get(k) for k in env}
        try:
            for k, v in env.items():
                os.environ[k] = v
            yield
        finally:
            for k, old in saved.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old


def _gpu_routing_stats(sym_buffer, num_tokens: int, bm2: int) -> RoutingStats:
    """Rank-consistent receiver-side routing statistics (one SUM all-reduce + one sync).

    Every rank contributes, per destination rank j, its routed slots and its
    unique tokens hitting j (own rank excluded) plus its expert histogram; the
    summed vector is identical on every rank, so every rank takes the same
    decision (the selected kernels are collective across the EP group).
    """
    import torch
    import torch.distributed as dist
    group = sym_buffer.group
    world = group.size()
    rank = group.rank()
    E = int(sym_buffer.num_experts)
    epr = E // world
    ti = sym_buffer.topk_idx[:num_tokens].to(torch.int64)
    valid = ti >= 0
    dst = torch.where(valid, ti // epr, torch.full_like(ti, world))
    slots = torch.zeros((num_tokens, world + 1), dtype=torch.int64, device=ti.device)
    slots.scatter_add_(1, dst, torch.ones_like(dst))
    slots = slots[:, :world]
    routes = slots.sum(0)
    uniq = (slots > 0).sum(0)
    routes[rank] = 0
    uniq[rank] = 0
    hist = torch.bincount(ti[valid], minlength=E)
    vec = torch.cat([routes, uniq, hist]).to(torch.int64)
    if world > 1:
        dist.all_reduce(vec, group=group)
    v = vec.tolist()
    routes_g, uniq_g, hist_g = v[:world], v[world:2 * world], v[2 * world:]
    per_rank = []
    for j in range(world):
        h = hist_g[j * epr:(j + 1) * epr]
        k_tiles = sum((x + bm2 - 1) // bm2 for x in h)
        m_max = max(h) if h else 0
        mean_m = sum(h) / float(max(1, epr))
        per_rank.append(RoutingStats(routes_g[j], uniq_g[j], routes_g[j], k_tiles, m_max, mean_m))
    return merge_rank_stats(per_rank)


def select_launch_env(sym_buffer, y, mma_type: str, routing_stats=None,
                      num_global_tokens: Optional[int] = None) -> Selection:
    """Runtime auto-selection for one weave_mega_moe launch (see module docstring)."""
    import torch
    group = sym_buffer.group
    ep = group.size()
    device = y.device.index if y.device.index is not None else torch.cuda.current_device()
    tokens = int(y.size(0))
    s_global = int(num_global_tokens) if num_global_tokens else tokens * ep
    capturing = bool(torch.cuda.is_current_stream_capturing())

    stats_fn = None
    if routing_stats is not None:
        rs = routing_stats if isinstance(routing_stats, RoutingStats) else RoutingStats.from_dict(routing_stats)
        stats_fn = lambda _bm2: rs  # noqa: E731
    elif not capturing:
        key = (id(sym_buffer), tokens)

        def stats_fn(bm2, _key=key):
            ver = int(getattr(sym_buffer.topk_idx, "_version", 0))
            hit = _SELECTOR._stats_cache.get(_key + (bm2,))
            if hit is not None and hit[0] == ver:
                return hit[1]
            st = _gpu_routing_stats(sym_buffer, tokens, bm2)
            _SELECTOR._stats_cache[_key + (bm2,)] = (ver, st)
            return st

    return _SELECTOR.select(
        ep=ep, device=device, num_experts=int(sym_buffer.num_experts), num_topk=int(sym_buffer.num_topk),
        hidden=int(sym_buffer.hidden), intermediate_hidden=int(sym_buffer.intermediate_hidden),
        num_shared_experts=int(getattr(sym_buffer, "num_shared_experts", 0) or 0), mma_type=mma_type,
        s_global=s_global, stats_fn=stats_fn, buffer_key=id(sym_buffer), capturing=capturing)


# =============================================================================
# Part A: calibration (torch, distributed)
# =============================================================================

def _e2m1_table(device):
    import torch
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    return torch.cat([mags, -mags])


def _dequant_fp4(packed, sf, gran_k: int = 32):
    """packed int8 (n, k/2) + float sf (n, k/gran) -> fp32 (n, k)."""
    import torch
    tab = _e2m1_table(packed.device)
    p = packed.view(torch.uint8).to(torch.int64)
    lo, hi = p & 0x0F, (p >> 4) & 0x0F
    n, k2 = packed.shape
    codes = torch.stack([lo, hi], dim=-1).view(n, 2 * k2)
    vals = tab[codes]
    return (vals.view(n, -1, gran_k) * sf.view(n, -1, 1)).view(n, 2 * k2)


def _quantize_grouped_fp8_block_128_128(w):
    """Block (128, 128) FP8 quantization along (N, K) (same recipe as the SM90 tests)."""
    import torch
    g, n, k = w.shape
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)
    sf = amax / 448.0
    w_fp8 = (w_view / sf.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


def make_routing(kind: str, s_global: int, num_experts: int, num_topk: int, world: int,
                 seed: int = 20261123, skew: float = 2.0):
    """See `_make_routing_cpu`; always built on the CPU (init_dist sets the default device to CUDA)."""
    import torch
    with torch.device("cpu"):
        return _make_routing_cpu(kind, s_global, num_experts, num_topk, world, seed, skew)


def _make_routing_cpu(kind: str, s_global: int, num_experts: int, num_topk: int, world: int,
                      seed: int = 20261123, skew: float = 2.0):
    """Deterministic global (S, K) top-k table, identical on every rank (CPU generator).

    uniform       top-k of i.i.d. scores
    skew          scores + linspace(skew, 0, E) bias over a random expert permutation
    zipf:a        Zipf(a) expert popularity (logit -a*log(rank+1)) + Gumbel noise
    spread        each token's experts spread round-robin over ranks (minimal multiplicity)
    concentrated  each token's experts drawn from 2 ranks (high multiplicity)
    """
    import torch
    g = torch.Generator().manual_seed(int(seed) * 1000003 + int(s_global) * 7 + num_experts)
    E, K = num_experts, num_topk
    epr = E // world
    if kind == "uniform" or kind == "skew" or kind.startswith("zipf"):
        scores = torch.randn((s_global, E), generator=g)
        if kind == "skew":
            bias = torch.linspace(float(skew), 0.0, E)[torch.randperm(E, generator=g)]
            scores = scores + bias.unsqueeze(0)
        elif kind.startswith("zipf"):
            a = float(kind.split(":", 1)[1]) if ":" in kind else 1.2
            rank_of = torch.randperm(E, generator=g).float()
            u = torch.rand((s_global, E), generator=g).clamp(1e-9, 1 - 1e-9)
            scores = -a * torch.log(rank_of + 1.0).unsqueeze(0) - torch.log(-torch.log(u))
        w, idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
        w = torch.softmax(w, dim=-1)
    elif kind in ("spread", "concentrated"):
        idx = torch.empty((s_global, K), dtype=torch.int64)
        if kind == "spread":
            rank_order = torch.stack([torch.randperm(world, generator=g) for _ in range(min(s_global, 1))])
            off = torch.randint(0, world, (s_global,), generator=g)
            for k in range(K):
                r = (off + k) % world
                slot = k // world
                local = torch.randint(0, epr, (s_global,), generator=g)
                idx[:, k] = r * epr + (local + slot * 7919) % epr
            del rank_order
        else:
            r0 = torch.randint(0, world, (s_global,), generator=g)
            r1 = (r0 + 1 + torch.randint(0, world - 1, (s_global,), generator=g)) % world if world > 1 else r0
            for k in range(K):
                r = torch.where(torch.tensor(k % 2 == 0), r0, r1)
                local = torch.randint(0, epr, (s_global,), generator=g)
                idx[:, k] = r * epr + local
        # de-duplicate experts within a token (re-draw collisions locally, then a
        # per-row fallback that draws K distinct experts from the same rank set)
        for _ in range(8):
            srt, _ = torch.sort(idx, dim=1)
            dup = (srt[:, 1:] == srt[:, :-1]).any(dim=1)
            if not bool(dup.any()):
                break
            rows = dup.nonzero().flatten()
            for k in range(1, K):
                clash = (idx[rows, k:k + 1] == idx[rows, :k]).any(dim=1)
                rr = rows[clash]
                idx[rr, k] = (idx[rr, k] // epr) * epr + (idx[rr, k] % epr + 1 + k) % epr
        srt, _ = torch.sort(idx, dim=1)
        for r in (srt[:, 1:] == srt[:, :-1]).any(dim=1).nonzero().flatten().tolist():
            ranks = sorted(set((idx[r] // epr).tolist()))
            pool = torch.cat([torch.arange(q * epr, (q + 1) * epr) for q in ranks])
            if pool.numel() < K:
                pool = torch.arange(E)
            idx[r] = pool[torch.randperm(pool.numel(), generator=g)[:K]]
        w = torch.softmax(torch.randn((s_global, K), generator=g), dim=-1)
    else:
        raise ValueError(f"unknown routing kind {kind!r}")
    return idx.to(torch.int64).contiguous(), w.to(torch.float32).contiguous()


class MoEWorkload:
    """One rank's MoE layer (weights, symmetric buffer, routing) for calibration/benchmarks."""

    def __init__(self, group, hw: HardwareFingerprint, num_experts: int, num_topk: int, hidden: int,
                 intermediate_hidden: int, mma_type: str, max_tokens_per_rank: int, seed: int = 20261123,
                 activation_clamp: float = 10.0):
        import torch
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4
        self.torch = torch
        self.dg = deep_gemm
        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        self.hw = hw
        self.family = hw.family
        self.E, self.K, self.H, self.I = num_experts, num_topk, hidden, intermediate_hidden
        self.epr = num_experts // self.world
        self.mma = mma_type
        self.clamp = activation_clamp
        self.seed = seed
        if self.family == "blackwell" and mma_type != "fp8xfp4":
            raise NotImplementedError(f"autotune measurement supports fp8xfp4 on SM10x (got {mma_type})")
        if self.family == "hopper" and mma_type != "fp8xfp8":
            raise NotImplementedError(f"autotune measurement supports fp8xfp8 on SM90 (got {mma_type})")
        gen_dev = torch.Generator(device="cuda")
        gen_dev.manual_seed(seed + 7919 * self.rank)
        E_r, H, I = self.epr, hidden, intermediate_hidden
        if self.family == "blackwell":
            self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group, num_experts, max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                num_shared_experts=0, mma_type="fp8xfp4")
            self.l1_raw, self.l2_raw = [], []
            l1_q = torch.empty((E_r, 2 * I, H // 2), dtype=torch.int8, device="cuda")
            l1_sf = torch.empty((E_r, 2 * I, H // 32), dtype=torch.float, device="cuda")
            l2_q = torch.empty((E_r, H, I // 2), dtype=torch.int8, device="cuda")
            l2_sf = torch.empty((E_r, H, I // 32), dtype=torch.float, device="cuda")
            for e in range(E_r):
                w1 = torch.randn((2 * I, H), dtype=torch.bfloat16, device="cuda", generator=gen_dev)
                l1_q[e], l1_sf[e] = per_token_cast_to_fp4(w1, use_ue8m0=True, gran_k=32)
                w2 = torch.randn((H, I), dtype=torch.bfloat16, device="cuda", generator=gen_dev)
                l2_q[e], l2_sf[e] = per_token_cast_to_fp4(w2, use_ue8m0=True, gran_k=32)
                del w1, w2
            self.l1_raw = (l1_q, l1_sf.clone())
            self.l2_raw = (l2_q, l2_sf.clone())
            t1 = deep_gemm.transform_sf_into_required_layout(l1_sf, 2 * I, H, (1, 32), E_r)
            t2 = deep_gemm.transform_sf_into_required_layout(l2_sf, H, I, (1, 32), E_r)
            self.l1, self.l2 = deep_gemm.transform_weights_for_mega_moe((l1_q, t1), (l2_q, t2))
        else:
            self.buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
                group, num_experts, max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
            ws1, sf1, ws2, sf2 = [], [], [], []
            for s0 in range(0, E_r, 16):
                cnt = min(16, E_r - s0)
                wb = torch.randn((cnt, 2 * I, H), dtype=torch.bfloat16, device="cuda", generator=gen_dev) * 0.05
                q, s = _quantize_grouped_fp8_block_128_128(wb)
                ws1.append(q); sf1.append(s); del wb
                wb = torch.randn((cnt, H, I), dtype=torch.bfloat16, device="cuda", generator=gen_dev) * 0.05
                q, s = _quantize_grouped_fp8_block_128_128(wb)
                ws2.append(q); sf2.append(s); del wb
            self.l1_raw = (torch.cat(ws1).contiguous(), torch.cat(sf1).contiguous())
            self.l2_raw = (torch.cat(ws2).contiguous(), torch.cat(sf2).contiguous())
            self.l1, self.l2 = deep_gemm.transform_weights_for_mega_moe_sm90(self.l1_raw, self.l2_raw)
        torch.cuda.empty_cache()
        self.max_tokens = int(self.buffer.num_max_tokens_per_rank)
        self.tokens = 0
        self.y = None
        self.routing_kind = None
        self.s_global = None

    # -- inputs ---------------------------------------------------------------
    def set_inputs(self, s_global: int, routing: str = "uniform", topk=None, weights=None):
        torch = self.torch
        from deep_gemm.utils import per_token_cast_to_fp8
        assert s_global % self.world == 0, "S_global must be a multiple of the EP size"
        tokens = s_global // self.world
        assert tokens <= self.max_tokens, f"S_global {s_global} exceeds the buffer capacity"
        if topk is None:
            topk, weights = make_routing(routing, s_global, self.E, self.K, self.world, self.seed)
        self.topk_global = topk
        lo, hi = self.rank * tokens, (self.rank + 1) * tokens
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed * 31 + s_global * 131 + self.rank)
        x = torch.randn((tokens, self.H), dtype=torch.bfloat16, device="cuda", generator=g)
        self.x_bf16 = x
        if self.family == "blackwell":
            from deep_gemm.utils import align
            x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            self.buffer.x[:tokens].copy_(x_fp8)
            self.buffer.x_sf[:tokens].copy_(x_sf)
        else:
            x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=False)
            self.buffer.x[:tokens].copy_(x_fp8)
            self.buffer.x_sf[:tokens].copy_(x_sf)
            self.x_deq = (x_fp8.float().view(tokens, self.H // 128, 128) * x_sf.float().unsqueeze(-1)).view(tokens, self.H)
        self.buffer.topk_idx[:tokens].copy_(topk[lo:hi].to(self.buffer.topk_idx.dtype))
        self.buffer.topk_weights[:tokens].copy_(weights[lo:hi].to(self.buffer.topk_weights.dtype))
        self.topk_local = topk[lo:hi].cuda()
        self.w_local = weights[lo:hi].cuda()
        self.tokens = tokens
        self.s_global = s_global
        self.routing_kind = routing
        self.y = torch.empty((tokens, self.H), dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        import torch.distributed as dist
        dist.barrier(group=self.group)

    def stats(self, bm2: int) -> RoutingStats:
        return routing_stats_from_topk(self.topk_global, self.E, self.world, bm2)

    # -- launches -------------------------------------------------------------
    def launch(self, use_weave_entry: bool = False):
        dg = self.dg
        if use_weave_entry:
            from deep_gemm.mega import weave_mega_moe
            kw = dict(activation_clamp=self.clamp, fast_math=True)
            if self.family == "hopper":
                kw["recipe"] = (128, 128, 128)
            return weave_mega_moe(self.y, self.l1, self.l2, self.buffer, **kw)
        if self.family == "blackwell":
            dg.fp8_fp4_mega_moe(y=self.y, l1_weights=self.l1, l2_weights=self.l2, sym_buffer=self.buffer,
                                activation_clamp=self.clamp, fast_math=True)
        else:
            dg.fp8_mega_moe(self.y, self.l1, self.l2, self.buffer, recipe=(128, 128, 128),
                            activation="swiglu", activation_clamp=self.clamp, fast_math=True)

    def compile_only(self):
        """Build (JIT) the kernel for the current env without launching it."""
        if self.family == "blackwell":
            os.environ["DG_JIT_COMPILE_ONLY"] = "1"
            try:
                self.launch()
            finally:
                os.environ.pop("DG_JIT_COMPILE_ONLY", None)
        else:
            from deep_gemm.utils import align
            self.dg._C.fp8_mega_moe_compile_only(self.world, self.E, self.max_tokens, self.K, self.H, self.I,
                                                 self.tokens, self.clamp, True)

    # -- reference ------------------------------------------------------------
    def reference_rows(self, rows):
        """FP32 reference of this rank's output rows `rows` from the dequantized operands."""
        torch, dist = self.torch, __import__("torch.distributed", fromlist=["x"])
        rows = rows.cuda()
        n = rows.numel()
        if self.family == "blackwell":
            xr = self.x_bf16[rows].float()
        else:
            xr = self.x_deq[rows]
        ti = self.topk_local[rows]
        tw = self.w_local[rows]
        xs = [torch.empty_like(xr) for _ in range(self.world)]
        tis = [torch.empty_like(ti) for _ in range(self.world)]
        tws = [torch.empty_like(tw) for _ in range(self.world)]
        dist.all_gather(xs, xr.contiguous(), group=self.group)
        dist.all_gather(tis, ti.contiguous(), group=self.group)
        dist.all_gather(tws, tw.contiguous(), group=self.group)
        X, TI, TW = torch.cat(xs), torch.cat(tis), torch.cat(tws)
        out = torch.zeros((X.shape[0], self.H), dtype=torch.float32, device="cuda")
        e0 = self.rank * self.epr
        I, H = self.I, self.H
        for le in range(self.epr):
            t, k = torch.where(TI == e0 + le)
            if t.numel() == 0:
                continue
            if self.family == "blackwell":
                w1 = _dequant_fp4(self.l1_raw[0][le], self.l1_raw[1][le])
                w2 = _dequant_fp4(self.l2_raw[0][le], self.l2_raw[1][le])
            else:
                w1 = (self.l1_raw[0][le].float().view(2 * I // 128, 128, H // 128, 128)
                      * self.l1_raw[1][le].view(2 * I // 128, 1, H // 128, 1)).reshape(2 * I, H)
                w2 = (self.l2_raw[0][le].float().view(H // 128, 128, I // 128, 128)
                      * self.l2_raw[1][le].view(H // 128, 1, I // 128, 1)).reshape(H, I)
            h = X[t] @ w1.t()
            gate = h[:, :I].clamp(max=self.clamp)
            up = h[:, I:].clamp(min=-self.clamp, max=self.clamp)
            a = torch.nn.functional.silu(gate) * up * TW[t, k].unsqueeze(1)
            out.index_add_(0, t, a @ w2.t())
        dist.all_reduce(out, group=self.group)
        return out[self.rank * n:(self.rank + 1) * n]


def _cos(a, b) -> float:
    import torch
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30))


def _all_reduce_scalar(value: float, op: str, group) -> float:
    import torch
    import torch.distributed as dist
    t = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.MIN, group=group)
    return float(t.item())


class Timer:
    """Paired palindromic ABBA timing, slowest rank per window.

    graph mode: CUDA-graph replays with an L2 flush before every replay.
    eager mode: barrier + spin + all-reduce aligned single launches (SM90 harness).
    """

    def __init__(self, wl: MoEWorkload, mode: str, iters: int = 0, rounds: int = 3):
        import torch
        self.wl = wl
        self.mode = mode
        self.rounds = rounds
        self.iters = iters
        self.graphs: Dict[str, Any] = {}
        l2 = max(wl.hw.l2_bytes, 64 << 20)
        self.flush = torch.empty(int(2 * l2 // 4), dtype=torch.int32, device="cuda")
        self.sync_tok = torch.ones((1,), dtype=torch.int32, device="cuda")

    def drop(self):
        self.graphs.clear()

    def _graph(self, env: Dict[str, str]):
        import torch
        key = env_key(env)
        g = self.graphs.get(key)
        if g is not None:
            return g
        with scoped_env(env):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.wl.launch()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.wl.launch()
            torch.cuda.synchronize()
        self.graphs[key] = g
        return g

    def _slot(self, env: Dict[str, str], iters: int) -> float:
        import torch
        import torch.distributed as dist
        if self.mode == "graph":
            g = self._graph(env)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for i in range(iters):
                self.flush.zero_()
                ev[i][0].record()
                g.replay()
                ev[i][1].record()
            torch.cuda.synchronize()
            return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)
        vals = []
        with scoped_env(env):
            for _ in range(iters):
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                dist.barrier(group=self.wl.group)
                torch.cuda._sleep(EAGER_HEAD_START_CYCLES)
                dist.all_reduce(self.sync_tok, group=self.wl.group)
                st.record()
                self.wl.launch()
                en.record()
                en.synchronize()
                vals.append(st.elapsed_time(en) * 1e3)
        return statistics.median(vals)

    def _iters_for(self, env: Dict[str, str]) -> int:
        if self.iters:
            return self.iters
        t = self._slot(env, 1)
        t = _all_reduce_scalar(t, "max", self.wl.group)
        if self.mode == "graph":
            return int(min(20, max(3, 15000.0 / max(t, 1.0))))
        return int(min(5, max(2, 8000.0 / max(t, 1.0))))

    def compare(self, env_a: Dict[str, str], env_b: Dict[str, str]) -> Tuple[float, float, List[float]]:
        """Returns (median_us_a, median_us_b, paired deltas b-a per round)."""
        import torch
        import torch.distributed as dist
        iters = self._iters_for(env_a)
        order_slots: List[Tuple[str, float]] = []
        for _ in range(self.rounds):
            for which in ("a", "b", "b", "a"):
                env = env_a if which == "a" else env_b
                order_slots.append((which, self._slot(env, iters)))
        t = torch.tensor([v for _, v in order_slots], dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.wl.group)
        vals = t.tolist()
        a = [v for (w, _), v in zip(order_slots, vals) if w == "a"]
        b = [v for (w, _), v in zip(order_slots, vals) if w == "b"]
        deltas = [(b[2 * i] + b[2 * i + 1] - a[2 * i] - a[2 * i + 1]) / 2.0 for i in range(self.rounds)]
        return statistics.median(a), statistics.median(b), deltas

    def measure_set(self, envs: List[Dict[str, str]], rounds: Optional[int] = None) -> List[float]:
        """Palindromic multi-arm measurement (A B C ... C B A per round); median per arm."""
        import torch
        import torch.distributed as dist
        rounds = rounds or self.rounds
        iters = self._iters_for(envs[0])
        slots = []
        for _ in range(rounds):
            for order in (list(range(len(envs))), list(reversed(range(len(envs))))):
                for i in order:
                    slots.append((i, self._slot(envs[i], iters)))
        t = torch.tensor([v for _, v in slots], dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.wl.group)
        vals = t.tolist()
        return [statistics.median(v for (j, _), v in zip(slots, vals) if j == i) for i in range(len(envs))]


class Calibrator:
    """Runs Part A on an initialised process group (every rank executes the same steps)."""

    def __init__(self, group, num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int,
                 mma_type: str = "auto", num_shared_experts: int = 0,
                 s_buckets: Sequence[int] = DEFAULT_S_BUCKETS, timing: str = "auto",
                 min_gain: float = 0.005, rounds: int = 3, ptxas_check: bool = True,
                 log=print, seed: int = 20261123):
        import torch
        if num_shared_experts:
            raise NotImplementedError("autotune measurement currently calibrates routed experts only "
                                      "(num_shared_experts=0); shared-expert layers use the bucket of the "
                                      "routed profile with the same shape")
        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        self.hw = query_fingerprint(self.world)
        self.family = self.hw.family
        if self.family is None:
            raise RuntimeError(f"no MegaMoE kernel family for sm_{self.hw.cc_major}{self.hw.cc_minor}")
        self.mma = default_mma_type(self.family) if mma_type in (None, "auto") else \
            __import__("deep_gemm.mega.precision", fromlist=["x"]).canonical_precision(mma_type)
        self.E, self.K, self.H, self.I = num_experts, num_topk, hidden, intermediate_hidden
        self.shared = num_shared_experts
        w = self.world
        self.s_buckets = sorted({int(s) for s in s_buckets if int(s) >= w})
        self.s_buckets = [((s + w - 1) // w) * w for s in self.s_buckets]
        self.timing = ("graph" if self.family == "blackwell" else "eager") if timing == "auto" else timing
        self.min_gain = min_gain
        self.rounds = rounds
        self.ptxas_check = ptxas_check
        self.log = log if self.rank == 0 else (lambda *a, **k: None)
        self.seed = seed
        self.skipped: List[Dict[str, Any]] = []
        self.verified: Dict[Tuple[int, str], bool] = {}
        self.timeline: List[Dict[str, Any]] = []
        self.t0 = time.time()
        max_tokens = max(self.s_buckets) // w
        self.wl = MoEWorkload(group, self.hw, num_experts, num_topk, hidden, intermediate_hidden,
                              self.mma, max_tokens, seed=seed)
        self.timer = Timer(self.wl, self.timing, rounds=rounds)
        self.ref_y = None

    # -- guarded evaluation -----------------------------------------------------
    def _compile_ok(self, env: Dict[str, str]) -> Tuple[bool, str]:
        reason = static_legal(env, self.family)
        if reason:
            return False, f"static: {reason}"
        ok, msg = 1.0, ""
        if self.ptxas_check:
            verdict = self._spill_verdict(env) if self.rank == 0 else "ok"
            if not verdict_allows(self.family, verdict):
                ok, msg = 0.0, f"ptxas: {verdict} (DG_JIT_PTXAS_CHECK=2 build, spill policy " \
                               f"{SPILL_POLICY.get(self.family, 'reject')})"
        if ok:
            try:
                with scoped_env(env):
                    self.wl.compile_only()
            except Exception as exc:  # host assert / nvcc failure
                ok, msg = 0.0, f"compile: {type(exc).__name__}: {str(exc)[:300]}"
        agreed = _all_reduce_scalar(ok, "min", self.group)
        if agreed < 1.0 and not msg:
            msg = "compile failed on another rank"
        return agreed >= 1.0, msg

    def _spill_verdict(self, env: Dict[str, str]) -> str:
        """'ok' | 'spill' | 'failed' for (current S, env), from the spill-verdict file written by
        the precompile check pass; a missing cell is checked now in a compile-only subprocess."""
        s = self.wl.s_global
        key = _verdict_key(s, env)
        path = spill_verdict_path(self.hw, self.E, self.K, self.H, self.I, self.mma)
        v = load_spill_verdicts(path).get(key)
        if v is None:
            precompile(self.hw, self.E, self.K, self.H, self.I, [s], jobs=1, gpus=[0], ptxas_check=True,
                       buffer_s_buckets=self.s_buckets, cells=[dict(S=s, env=env)], warm=False,
                       mma_type=self.mma, log=self.log,
                       work_dir=default_autotune_dir() / ("precompile_%d_ondemand" % os.getpid()))
            v = load_spill_verdicts(path).get(key, "failed")
        return v

    def _run_once(self, env: Dict[str, str]):
        import torch
        with scoped_env(env):
            self.wl.y.zero_()
            self.wl.launch()
        torch.cuda.synchronize()
        return self.wl.y.clone()

    def guard(self, env: Dict[str, str], tag: str) -> bool:
        """Compile (+0-spill check) and accuracy-check `env` at the current inputs."""
        import torch
        key = (self.wl.s_global, self.wl.routing_kind, env_key(env))
        if key in self.verified:
            return self.verified[key]
        ok, msg = self._compile_ok(env)
        if ok:
            y = self._run_once(env)
            finite = 1.0 if bool(torch.isfinite(y).all()) else 0.0
            cos = _cos(y, self.ref_y) if self.ref_y is not None else 1.0
            finite = _all_reduce_scalar(finite, "min", self.group)
            cos = _all_reduce_scalar(cos, "min", self.group)
            if finite < 1.0 or not cos > 0.99:
                ok, msg = False, f"accuracy: finite={finite >= 1.0} cos_vs_ref={cos:.5f}"
        if not ok:
            self.skipped.append(dict(S=self.wl.s_global, routing=self.wl.routing_kind, tag=tag,
                                     env=env, reason=msg))
            self.log(f"[autotune]   skip {tag}: {msg}")
        self.verified[key] = ok
        return ok

    def _set_inputs(self, s_global: int, routing: str, ref_env: Dict[str, str]) -> bool:
        """Switch inputs; the reference output is `ref_env`'s output, itself checked
        against an FP32 reference on a 64-row sample per rank."""
        import torch
        self.timer.drop()
        self.wl.set_inputs(s_global, routing)
        self.ref_y = None
        ok, msg = self._compile_ok(ref_env)
        if not ok:
            raise RuntimeError(f"reference config {ref_env} does not compile at S={s_global}: {msg}")
        y = self._run_once(ref_env)
        n = min(64, self.wl.tokens)
        rows = torch.arange(n, device="cuda")
        ref = self.wl.reference_rows(rows)
        cos = _all_reduce_scalar(_cos(y[:n], ref), "min", self.group)
        finite = _all_reduce_scalar(1.0 if bool(torch.isfinite(y).all()) else 0.0, "min", self.group)
        self.ref_y = y
        self.timeline.append(dict(event="reference", S=s_global, routing=routing, env=ref_env,
                                  fp32_sample_cos=cos, finite=finite >= 1.0))
        self.log(f"[autotune] S={s_global} routing={routing}: reference env {ref_env or '{}'} "
                 f"cos_vs_fp32(64 rows/rank)={cos:.5f} finite={finite >= 1.0}")
        return cos > 0.99 and finite >= 1.0

    def _better(self, inc: Dict[str, str], cand: Dict[str, str], tag: str, stage: str,
                trace: List[Dict[str, Any]]) -> Tuple[bool, float, float]:
        if not self.guard(cand, tag):
            trace.append(dict(stage=stage, tag=tag, env=cand, status="skipped"))
            return False, float("nan"), float("nan")
        a, b, deltas = self.timer.compare(inc, cand)
        gain = (a - b) / a
        won = gain >= self.min_gain and statistics.median(deltas) < 0
        trace.append(dict(stage=stage, tag=tag, env=cand, us=round(b, 3), incumbent_us=round(a, 3),
                          gain=round(gain, 5), paired_delta_us=[round(d, 3) for d in deltas],
                          status="accepted" if won else "rejected"))
        self.log(f"[autotune]   {stage:7s} {tag:28s} {b:10.2f} us vs incumbent {a:10.2f} us "
                 f"gain={gain * 100:+.2f}% {'ACCEPT' if won else ''}")
        return won, a, b

    # -- per bucket search --------------------------------------------------------
    def tune_bucket(self, s_global: int) -> Dict[str, Any]:
        hw, fam = self.hw, self.family
        cold = cold_start_env(fam, hw, s_global)
        trustworthy = self._set_inputs(s_global, "uniform", cold)
        m_exp = m_expected(s_global, self.K, self.E)
        trace: List[Dict[str, Any]] = []
        inc = dict(cold)
        # Stage 1: GEMM tuples (default co-knobs)
        tuple_scores: List[Tuple[float, Dict[str, str]]] = []
        for t in gemm_candidates(fam, m_exp):
            cand = {**cold, **t}
            won, a, b = self._better(inc, cand, f"{t['WEAVE_GEMM1_CONFIG']}/{t['WEAVE_GEMM2_CONFIG']}",
                                     "gemm", trace)
            if not math.isnan(b):
                tuple_scores.append((b / a, t))
            if won:
                inc = cand
        tuple_scores.sort(key=lambda x: x[0])
        # Stage 2: one co-knob at a time on the incumbent
        accepted: Dict[str, Tuple[float, Dict[str, str]]] = {}
        for group_name, name, delta in coknob_variants(fam, hw, s_global, inc):
            cand = {**inc, **delta}
            if env_key(cand) == env_key(inc):
                continue
            won, a, b = self._better(inc, cand, name, "coknob", trace)
            if won:
                g = (a - b) / a
                if group_name not in accepted or g > accepted[group_name][0]:
                    accepted[group_name] = (g, delta)
        # Stage 2b: combine the best delta of every improving group
        if accepted:
            ranked = sorted(accepted.values(), key=lambda x: -x[0])
            combo = dict(inc)
            for _, d in ranked:
                combo.update(d)
            single = {**inc, **ranked[0][1]}
            if len(ranked) > 1:
                won, _, _ = self._better(single, combo, "+".join(sorted(accepted)), "combine", trace)
                inc = combo if won else single
            else:
                inc = single
        # Stage 3: runner-up GEMM tuple with the accepted co-knobs
        cur_tuple = {k: inc[k] for k in ("WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG") if k in inc}
        for ratio, t in tuple_scores[:2]:
            if t == cur_tuple or ratio > 1.03:
                continue
            cand = {**inc, **t}
            won, _, _ = self._better(inc, cand, f"{t['WEAVE_GEMM1_CONFIG']}/{t['WEAVE_GEMM2_CONFIG']}+co",
                                     "refine", trace)
            if won:
                inc = cand
            break
        # Final: incumbent vs cold start vs the compiled default ({} = nothing injected), measured
        # fresh in one palindromic set.  A calibrated bucket is never slower than doing nothing:
        # the compiled default replaces the incumbent when it is faster by >= min_gain.
        finalists = [cold] if env_key(inc) == env_key(cold) else [cold, inc]
        with_default = env_key({}) not in {env_key(e) for e in finalists} and self.guard({}, "compiled-default")
        times = self.timer.measure_set(finalists + ([{}] if with_default else []))
        cold_us = times[0]
        best_us = times[1] if len(finalists) > 1 else cold_us
        if with_default:
            default_us = times[-1]
            won = default_us <= best_us * (1.0 - self.min_gain)
            trace.append(dict(stage="final", tag="compiled_default", env={}, us=round(default_us, 3),
                              incumbent_us=round(best_us, 3), gain=round((best_us - default_us) / best_us, 5),
                              status="accepted" if won else "rejected"))
            self.log(f"[autotune]   final   compiled_default             {default_us:10.2f} us vs incumbent "
                     f"{best_us:10.2f} us {'ACCEPT' if won else ''}")
            if won:
                inc, best_us = {}, default_us
        bm2 = gemm2_block_m(inc)
        st = self.wl.stats(bm2)
        bucket = dict(S=s_global, m_expected=round(m_exp, 3), env=normalize_env(inc), us=round(best_us, 3),
                      cold_start_us=round(cold_us, 3), gain_vs_cold_start=round((cold_us - best_us) / cold_us, 5),
                      uniform_stats=st.to_dict(), reference_trusted=bool(trustworthy),
                      measurements=sum(1 for t in trace if t.get("status") in ("accepted", "rejected")),
                      trace=trace)
        self.log(f"[autotune] bucket S={s_global}: {inc or '{}'} {best_us:.2f} us "
                 f"(cold start {cold_us:.2f} us, {bucket['gain_vs_cold_start'] * 100:+.2f}%)")
        return bucket

    # -- break-even probes ------------------------------------------------------------
    def _probe_pair(self, s_global: int, routing: str, base: Dict[str, str], alt: Dict[str, str],
                    tag: str, probes: List[Dict[str, Any]]) -> Optional[float]:
        if self.wl.s_global != s_global or self.wl.routing_kind != routing:
            self._set_inputs(s_global, routing, base)
        if not self.guard(base, "probe-base") or not self.guard(alt, tag):
            probes.append(dict(probe=tag, S=s_global, routing=routing, env=alt, status="skipped"))
            return None
        a, b, deltas = self.timer.compare(base, alt)
        gain = (a - b) / a
        bm2 = gemm2_block_m(base)
        st = self.wl.stats(bm2)
        probes.append(dict(probe=tag, S=s_global, routing=routing, base_env=base, env=alt,
                           base_us=round(a, 3), us=round(b, 3), gain=round(gain, 5),
                           paired_delta_us=[round(d, 3) for d in deltas], stats=st.to_dict()))
        self.log(f"[autotune]   probe {tag:22s} S={s_global:6d} {routing:12s} base {a:9.2f} alt {b:9.2f} "
                 f"gain={gain * 100:+.2f}% saved_rows={st.m_rem - st.d_in} skew={skew_ratio(st):.2f}")
        return gain

    def _skew_routing(self, s_global: int) -> str:
        for a in (1.0, 1.2, 1.5, 2.0, 2.5):
            idx, _ = make_routing(f"zipf:{a}", s_global, self.E, self.K, self.world, self.seed)
            st = routing_stats_from_topk(idx, self.E, self.world, 128)
            if st.m_max > 3.0 * st.mean_m:
                return f"zipf:{a}"
        return "zipf:2.5"

    def probes(self, buckets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        fam, hw = self.family, self.hw
        probes: List[Dict[str, Any]] = []
        by_s = {b["S"]: b for b in buckets}
        s_max = max(self.s_buckets)

        def base_for(s):
            return dict(select_bucket(dict(buckets=buckets), s)["env"])

        # (1) dedup break-even: multiplicity sweep at S=1024 and S=max
        points = []
        for s in sorted({min(1024, s_max), s_max}):
            for routing in ("spread", "uniform", "concentrated"):
                base = {k: v for k, v in base_for(s).items() if k != "WEAVE_DEDUP_MODE"}
                g = self._probe_pair(s, routing, {**base, "WEAVE_DEDUP_MODE": "0"},
                                     {**base, "WEAVE_DEDUP_MODE": "1"}, "dedup", probes)
                if g is not None:
                    st = RoutingStats.from_dict(probes[-1]["stats"])
                    points.append((st.m_rem - st.d_in, g))
        wins = sorted(saved for saved, g in points if g >= 0.015)
        dedup_thr = None
        if wins:
            # smallest saving that wins and is not contradicted by a losing point with more saving
            for cand in wins:
                if all(g >= 0.0 for saved, g in points if saved >= cand):
                    dedup_thr = int(cand)
                    break
        # (2) comm-owned scatter crossover from the bucket searches
        verdicts = []
        for b in buckets:
            comm = b["env"].get("WEAVE_SCATTER_OWNER") == "comm_only"
            verdicts.append((b["S"], comm, b["uniform_stats"]))
        if all(v[1] for v in verdicts):
            rows, tiles = 0, 0
        elif not any(v[1] for v in verdicts):
            rows, tiles = None, None
        else:
            rows = tiles = None
            for i, (s, comm, st) in enumerate(verdicts):
                if comm and all(v[1] for v in verdicts[i:]):
                    rows, tiles = int(st["d_out"]), int(st["k_tiles"])
                    break
        # (3) skewed routing: ready-group size and spatial tiers at S=4096 (or the largest <= it)
        s_skew = max([s for s in self.s_buckets if s <= 4096] or [s_max])
        skew_kind = self._skew_routing(s_skew)
        base = base_for(s_skew)
        rgs_best, rgs_gain = 0, 0.0
        for rgs in (2, 4):
            g = self._probe_pair(s_skew, skew_kind, base, {**base, "WEAVE_READY_GROUP_SIZE": str(rgs)},
                                 f"ready_group_{rgs}", probes)
            if g is not None and g >= 0.01 and g > rgs_gain:
                rgs_best, rgs_gain = rgs, g
        comp_heavy, comp_gain = {}, 0.0
        for delta in comp_heavy_candidates(fam, hw):
            alt = {**base, **delta}
            if env_key(alt) == env_key(base):
                continue
            g = self._probe_pair(s_skew, skew_kind, base, alt, "comp_heavy", probes)
            if g is not None and g >= 0.01 and g > comp_gain:
                comp_heavy, comp_gain = dict(delta), g
        comm_heavy, comm_gain = {}, 0.0
        for delta in comm_heavy_candidates(fam, hw):
            alt = {**base, **delta}
            if env_key(alt) == env_key(base):
                continue
            g = self._probe_pair(s_skew, "spread", base, alt, "comm_heavy", probes)
            if g is not None and g >= 0.01 and g > comm_gain:
                comm_heavy, comm_gain = dict(delta), g
        thresholds = dict(dedup_min_saved_rows=dedup_thr, comm_scatter_min_rows=rows,
                          comm_scatter_min_tiles=tiles, skew_rgs=int(rgs_best),
                          skew_comm_heavy=normalize_env(comm_heavy), skew_comp_heavy=normalize_env(comp_heavy),
                          skew_routing=skew_kind, comm_scatter_verdicts=[[s, c] for s, c, _ in verdicts],
                          dedup_points=[[int(s), round(g, 5)] for s, g in points])
        return thresholds, probes

    # -- rule variants: verify and reconcile with the bucket winners ------------------
    def finalize_rules(self, buckets: List[Dict[str, Any]], thresholds: Dict[str, Any]):
        kinds = ["uniform", "spread", "concentrated", thresholds.get("skew_routing") or "zipf:2.0"]
        for b in buckets:
            s = b["S"]
            enabled = {n: True for n in RULE_ORDER}
            b["rules_enabled"] = enabled
            b["verified_variants"] = []
            # (a) the rules must reproduce (or beat) the measured winner under uniform routing
            st_u = RoutingStats.from_dict(b["uniform_stats"])
            flips_u = rule_flips(b, thresholds, st_u)
            for n in RULE_ORDER:
                if n not in flips_u:
                    continue
                cand = compose_env(b["env"], flips_u, [n])
                if env_key(cand) == env_key(b["env"]):
                    continue
                self._set_inputs(s, "uniform", b["env"]) if (self.wl.s_global != s or
                                                              self.wl.routing_kind != "uniform") else None
                ok = self.guard(cand, f"rule-{n}@uniform")
                if not ok:
                    enabled[n] = False
                    continue
                a, c, _ = self.timer.compare(b["env"], cand)
                if c > a * (1.0 + self.min_gain):
                    enabled[n] = False   # the rule contradicts this bucket's own measurement
                    self.log(f"[autotune]   S={s}: rule {n} disabled (uniform {c:.2f} vs winner {a:.2f} us)")
            # (b) collect every env the enabled rules produce on the calibration routings
            variants = {}
            for kind in kinds:
                idx, _ = make_routing(kind, s, self.E, self.K, self.world, self.seed)
                st = routing_stats_from_topk(idx, self.E, self.world, gemm2_block_m(b["env"]))
                flips = rule_flips(b, thresholds, st)
                act = [n for n in RULE_ORDER if n in flips]
                for r in range(len(act), 0, -1):
                    env = compose_env(b["env"], flips, act[:r])
                    if env_key(env) != env_key(b["env"]):
                        variants[env_key(env)] = env
                for n in act:
                    env = compose_env(b["env"], flips, [n])
                    if env_key(env) != env_key(b["env"]):
                        variants[env_key(env)] = env
            if variants:
                if self.wl.s_global != s or self.wl.routing_kind != "uniform":
                    self._set_inputs(s, "uniform", b["env"])
                for env in variants.values():
                    if self.guard(env, "rule-variant"):
                        b["verified_variants"].append(normalize_env(env))
            self.log(f"[autotune] S={s}: rules_enabled={enabled} verified_variants={len(b['verified_variants'])}")

    # -- driver ------------------------------------------------------------------------
    def _state_identity(self) -> Dict[str, Any]:
        return dict(hw_id=self.hw.hw_id, model=[self.E, self.K, self.H, self.I, self.shared, self.mma],
                    s_buckets=list(self.s_buckets), rounds=self.rounds, min_gain=self.min_gain,
                    timing=self.timing, seed=self.seed)

    def _load_state(self, state_path: Optional[str]) -> Dict[str, Any]:
        if not state_path or not Path(state_path).exists():
            return dict(identity=self._state_identity(), buckets=[], skipped=[], timeline=[], wall=[])
        st = json.loads(Path(state_path).read_text(encoding="utf-8"))
        if st.get("identity") != self._state_identity():
            raise ProfileError(f"autotune state {state_path} was produced for {st.get('identity')}, "
                               f"not {self._state_identity()}; use a fresh --state file")
        self.skipped = list(st.get("skipped", []))
        self.timeline = list(st.get("timeline", []))
        st.setdefault("wall", [])
        return st

    def _save_state(self, state_path: Optional[str], st: Dict[str, Any]):
        if not state_path or self.rank != 0:
            return
        st["skipped"], st["timeline"] = self.skipped, self.timeline
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(st, indent=1, sort_keys=True))
        os.replace(tmp, path)

    def run(self, out_path: Optional[str] = None, probes: bool = True, stage: str = "all",
            state_path: Optional[str] = None, only_s: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """stage: all | buckets | probes | finalize.  With `state_path`, every finished bucket /
        stage is checkpointed, so one calibration can be split over several short GPU admissions
        (buckets already present in the state are not re-measured)."""
        if stage not in ("all", "buckets", "probes", "finalize"):
            raise ValueError(f"unknown stage {stage!r}")
        if stage != "all" and not state_path:
            raise ValueError("a stage other than 'all' needs a state file")
        t_start = time.time()
        st = self._load_state(state_path)
        by_s = {int(b["S"]): b for b in st.get("buckets", [])}
        if stage in ("all", "buckets"):
            w = self.world
            want = {((int(x) + w - 1) // w) * w for x in only_s} if only_s else None
            todo = [s for s in self.s_buckets if want is None or s in want]
            for s in todo:
                if state_path and s in by_s:
                    self.log(f"[autotune] bucket S={s}: already in state, skipped")
                    continue
                t_b = time.time()
                by_s[s] = self.tune_bucket(s)
                st["buckets"] = [by_s[k] for k in sorted(by_s)]
                st["wall"].append(dict(stage="bucket", S=s, seconds=round(time.time() - t_b, 1)))
                self._save_state(state_path, st)
            if stage == "buckets":
                return st
        missing = [s for s in self.s_buckets if s not in by_s]
        if missing:
            raise ProfileError(f"buckets {missing} not calibrated yet (run the buckets stage first)")
        buckets = [by_s[s] for s in self.s_buckets]
        t_buckets = time.time()
        if probes and stage in ("all", "probes"):
            t_p = time.time()
            thresholds, probe_rows = self.probes(buckets)
            st["thresholds"], st["probes"] = thresholds, probe_rows
            st["wall"].append(dict(stage="probes", seconds=round(time.time() - t_p, 1)))
            self._save_state(state_path, st)
            if stage == "probes":
                return st
        if probes:
            if "thresholds" not in st:
                raise ProfileError("probes not run yet (run the probes stage first)")
            thresholds, probe_rows = st["thresholds"], st["probes"]
            t_f = time.time()
            self.finalize_rules(buckets, thresholds)
            st["wall"].append(dict(stage="finalize", seconds=round(time.time() - t_f, 1)))
        else:
            thresholds = dict(dedup_min_saved_rows=None, comm_scatter_min_rows=None, comm_scatter_min_tiles=None,
                              skew_rgs=0, skew_comm_heavy={}, skew_comp_heavy={})
            probe_rows = []
            for b in buckets:
                b["verified_variants"] = []
                b["rules_enabled"] = {n: False for n in RULE_ORDER}
        doc = dict(
            schema=SCHEMA, tool_version=TOOL_VERSION,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            hw=self.hw.to_dict(),
            model=dict(E=self.E, K=self.K, H=self.H, I=self.I, shared=self.shared, mma_type=self.mma),
            method=dict(timing=self.timing, rounds=self.rounds, min_gain=self.min_gain,
                        ptxas_0spill_check=self.ptxas_check,
                        accuracy_guard="finite and cos_sim > 0.99 vs the reference config output at the same "
                                       "inputs (min over ranks); the reference itself is checked against an "
                                       "FP32 dequantized-operand reference on 64 rows per rank",
                        windows="paired palindromic ABBA, slowest rank per window"),
            buckets=buckets, thresholds=thresholds, probes=probe_rows, skipped=self.skipped,
            timeline=self.timeline,
            wall_seconds=dict(buckets=round(t_buckets - t_start, 1), total=round(time.time() - t_start, 1),
                              stages=list(st.get("wall", []))))
        validate_profile(doc)
        if self.rank == 0:
            path = Path(out_path) if out_path else profile_path(self.hw, self.E, self.K, self.H, self.I,
                                                                 self.shared, self.mma)
            save_profile(doc, path)
            doc["_path"] = str(path)
            self.log(f"[autotune] wrote {path} ({doc['wall_seconds']['total']} s)")
        self._save_state(state_path, st)
        get_runtime_selector().clear()
        return doc


def autotune_moe(group, num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int,
                 mma_type: str = "auto", num_shared_experts: int = 0,
                 s_buckets: Sequence[int] = DEFAULT_S_BUCKETS, out: Optional[str] = None,
                 probes: bool = True, stage: str = "all", state_path: Optional[str] = None,
                 only_s: Optional[Sequence[int]] = None, **kwargs) -> Dict[str, Any]:
    """Calibrate one MoE shape on the live GPUs of `group` and write the profile JSON.

    Must be called on every rank of `group` (it launches collective kernels).
    Explicit WEAVE_* tuning variables are cleared for the duration of the run.
    """
    saved = {k: os.environ.pop(k) for k in manual_env_keys()}
    try:
        cal = Calibrator(group, num_experts, num_topk, hidden, intermediate_hidden, mma_type,
                         num_shared_experts, s_buckets, **kwargs)
        return cal.run(out, probes=probes, stage=stage, state_path=state_path, only_s=only_s)
    finally:
        os.environ.update(saved)


# =============================================================================
# CLI
# =============================================================================

MODEL_PRESETS = {
    # name: (E, K, H, I)
    "qwen3.8-2.4t": (512, 10, 8192, 2048),
    "qwen3.5-35b": (256, 8, 2048, 512),
    "qwen3.5-122b": (256, 8, 3072, 1024),
    "qwen3.5-397b": (512, 10, 4096, 1024),
}


def _cli_worker(local_rank: int, nprocs: int, args):
    from deep_gemm.utils.dist import init_dist
    import torch.distributed as dist
    rank, world, group = init_dist(local_rank, nprocs)
    try:
        autotune_moe(group, args.num_experts, args.num_topk, args.hidden, args.intermediate_hidden,
                     mma_type=args.mma_type, s_buckets=tuple(args.s_buckets), out=args.out,
                     probes=not args.no_probes, rounds=args.rounds, min_gain=args.min_gain,
                     ptxas_check=not args.no_ptxas_check, timing=args.timing,
                     stage=args.stage, state_path=args.state, only_s=args.only_s)
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group()


def precompile_plan(hw: HardwareFingerprint, num_experts: int, num_topk: int,
                    s_buckets: Sequence[int]) -> List[Dict[str, Any]]:
    """Every (S, env) the staged search can reach without knowing a winner yet:
    cold start, each GEMM tuple, and each single co-knob variant on each tuple."""
    fam, w = hw.family, hw.ep
    cells, seen = [], set()
    for s in sorted({((max(int(x), w) + w - 1) // w) * w for x in s_buckets}):
        cold = cold_start_env(fam, hw, s)
        bases = [cold] + [{**cold, **t} for t in gemm_candidates(fam, m_expected(s, num_topk, num_experts))]
        envs = list(bases) + [{}]          # {} = compiled default, re-measured in the final step
        for b in bases:
            envs += [{**b, **d} for _, _, d in coknob_variants(fam, hw, s, b)]
        for env in envs:
            key = (s, env_key(env))
            if key not in seen and static_legal(env, fam) is None:
                seen.add(key)
                cells.append(dict(S=s, env=env))
    return cells


def _precompile_worker_main(spec_path: str) -> int:
    """Subprocess body: JIT-build a list of (S, env) cells for this GPU without launching."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    for k in [k for k in os.environ if k.startswith("WEAVE_")]:
        del os.environ[k]
    os.environ.pop("DG_JIT_PTXAS_CHECK", None)
    if spec.get("ptxas_check"):
        os.environ["DG_JIT_PTXAS_CHECK"] = "2"   # fail on register spills only (stack frames allowed)
    if spec.get("cache_dir"):
        os.environ["DG_JIT_CACHE_DIR"] = spec["cache_dir"]
    import torch
    import deep_gemm  # noqa: F401  (sets up the JIT include paths)
    from deep_gemm import _C
    fam, w = spec["family"], int(spec["world"])
    E, K, H, I = spec["E"], spec["K"], spec["H"], spec["I"]
    clamp = float(spec.get("activation_clamp", 10.0))
    out = open(spec["out"], "a", encoding="utf-8")
    if fam == "blackwell":
        align = _C.get_token_alignment_for_mega_moe("fp8xfp4")
        max_tok = ((int(spec["max_tokens"]) + align - 1) // align) * align
        nbytes, slicer = _C.get_symm_buffer_size_for_mega_moe(w, E, max_tok, K, H, I, "fp8xfp4", "swiglu", 0)
        buf = torch.empty(nbytes, dtype=torch.int8, device="cuda")
        ptrs = [buf.data_ptr()] * w
        el = E // w

        def mn_major_sf(n, k):
            return torch.empty((el, k // 128, n), dtype=torch.int32, device="cuda").transpose(1, 2)

        l1 = (torch.empty((el, 2 * I, H // 2), dtype=torch.int8, device="cuda"), mn_major_sf(2 * I, H))
        l2 = (torch.empty((el, H, I // 2), dtype=torch.int8, device="cuda"), mn_major_sf(H, I))
        os.environ["DG_JIT_COMPILE_ONLY"] = "1"
    else:
        align = _C.get_token_alignment_for_sm90_mega_moe()
        max_tok = ((max(int(spec["max_tokens"]), 1) + align - 1) // align) * align
    rc = 0
    for cell in spec["cells"]:
        tokens = int(cell["S"]) // w
        t0 = time.time()
        status = "ok"
        spill = None
        try:
            with scoped_env(cell["env"]):
                if fam == "blackwell":
                    y = torch.empty((tokens, H), dtype=torch.bfloat16, device="cuda")
                    _C.fp8_fp4_mega_moe(y, l1, l2, None, None, None, buf, ptrs, 0, max_tok, E, K,
                                        (1, 1, 32), "swiglu", clamp, True, None, None)
                else:
                    _C.fp8_mega_moe_compile_only(w, E, max_tok, K, H, I, tokens, clamp, True)
        except Exception as exc:
            msg = str(exc)
            pairs = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", msg)
            if pairs:
                spill = [max(int(a) for a, _ in pairs), max(int(b) for _, b in pairs)]
            status = ("spill: " if "spill" in msg.lower() else "failed: ") + "%s: %s" % (type(exc).__name__, msg[:240])
            rc = 1
        out.write(json.dumps(dict(S=cell["S"], env=cell["env"], status=status, spill_bytes=spill,
                                  seconds=round(time.time() - t0, 1))) + chr(10))
        out.flush()
    return rc


def jit_cache_dir() -> Path:
    env = os.environ.get("DG_JIT_CACHE_DIR")
    return Path(env) if env else Path.home() / ".deep_gemm"


def spill_verdict_path(hw: HardwareFingerprint, num_experts: int, num_topk: int, hidden: int,
                       intermediate_hidden: int, mma_type: str) -> Path:
    return default_autotune_dir() / "spill_verdicts" / (
        f"{hw.hw_id}__{model_key(num_experts, num_topk, hidden, intermediate_hidden, 0, mma_type)}.json")


def _verdict_key(s_global: int, env: Dict[str, str]) -> str:
    return f"{int(s_global)}|{env_key(env)}"


def load_spill_verdicts(path: Path) -> Dict[str, str]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _merge_spill_verdicts(path: Path, rows: List[Dict[str, Any]]):
    cur = load_spill_verdicts(path)
    for r in rows:
        st = r["status"]
        v = "ok" if st == "ok" else st.split(":", 1)[0]
        if v == "spill" and r.get("spill_bytes"):
            v = "spill:%d/%d" % tuple(r["spill_bytes"])
        cur[_verdict_key(r["S"], r["env"])] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cur, indent=0, sort_keys=True))
    os.replace(tmp, path)


def verdict_allows(family: Optional[str], verdict: Optional[str]) -> bool:
    """Whether a spill-check verdict admits the candidate under SPILL_POLICY[family]."""
    if verdict is None:
        return False
    if verdict == "ok":
        return True
    return verdict.startswith("spill") and SPILL_POLICY.get(family or "", "reject") == "record"


def precompile(hw: HardwareFingerprint, num_experts: int, num_topk: int, hidden: int,
               intermediate_hidden: int, s_buckets: Sequence[int], jobs: int = 0,
               gpus: Optional[Sequence[int]] = None, ptxas_check: bool = True,
               work_dir: Optional[Path] = None, log=print,
               buffer_s_buckets: Optional[Sequence[int]] = None,
               cells: Optional[List[Dict[str, Any]]] = None, warm: bool = True,
               mma_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parallel JIT of the stage-1/2 candidate space without kernel launches.

    With `ptxas_check`, pass 1 compiles every cell with DG_JIT_PTXAS_CHECK=2 (ptxas
    --warn-on-spills; build fails on any spill store/load, a non-spill stack frame is
    allowed) into a separate cache directory and records one verdict per (S, env) in
    the spill-verdict file.  Pass 2 (`warm`) then builds the non-spilling cells with the
    production compiler flags into DG_JIT_CACHE_DIR, so the calibration run, the e2e
    benchmark and the user's runtime all hit the same cubins (compiler flags are part of
    the JIT cache key).
    """
    if cells is None:
        cells = precompile_plan(hw, num_experts, num_topk, s_buckets)
    if not cells:
        return []
    work = Path(work_dir) if work_dir else default_autotune_dir() / ("precompile_%d" % os.getpid())
    rows: List[Dict[str, Any]] = []
    if ptxas_check:
        import shutil
        mma = mma_type or default_mma_type(hw.family)
        vpath = spill_verdict_path(hw, num_experts, num_topk, hidden, intermediate_hidden, mma)
        known = load_spill_verdicts(vpath)
        todo = [c for c in cells if _verdict_key(c["S"], c["env"]) not in known]
        if todo:
            # a JIT cache hit skips ptxas (and therefore the check): always build into a fresh dir
            check_cache = jit_cache_dir() / ("spill_check_%d_%d" % (os.getpid(), int(time.time() * 1e3)))
            try:
                rows = _precompile_pass(hw, num_experts, num_topk, hidden, intermediate_hidden, s_buckets, todo,
                                        jobs, gpus, True, work / "check", log, buffer_s_buckets,
                                        cache_dir=str(check_cache), tag="spill-check")
            finally:
                shutil.rmtree(check_cache, ignore_errors=True)
            _merge_spill_verdicts(vpath, rows)
            known = load_spill_verdicts(vpath)
        cells = [c for c in cells if verdict_allows(hw.family, known.get(_verdict_key(c["S"], c["env"])))]
    if warm and cells:
        rows_w = _precompile_pass(hw, num_experts, num_topk, hidden, intermediate_hidden, s_buckets, cells, jobs,
                                  gpus, False, work / "warm", log, buffer_s_buckets, cache_dir=None, tag="warm")
        if not ptxas_check:
            rows = rows_w
    return rows


def _precompile_pass(hw, num_experts, num_topk, hidden, intermediate_hidden, s_buckets, cells, jobs, gpus,
                     ptxas_check, work, log, buffer_s_buckets, cache_dir, tag):
    import subprocess
    jobs = jobs or max(1, min(32, (os.cpu_count() or 8) // 4, len(cells)))
    gpus = list(gpus) if gpus else [0]
    w = hw.ep
    max_tokens = max(((max(int(x), w) + w - 1) // w) for x in (buffer_s_buckets or s_buckets))
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("job*.jsonl"):
        f.unlink()
    procs = []
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = [x.strip() for x in vis.split(",")] if vis else None
    for j in range(jobs):
        mine = cells[j::jobs]
        if not mine:
            continue
        spec = dict(family=hw.family, world=w, E=num_experts, K=num_topk, H=hidden, I=intermediate_hidden,
                    max_tokens=max_tokens, cells=mine, out=str(work / ("job%d.jsonl" % j)),
                    ptxas_check=ptxas_check, cache_dir=cache_dir)
        sp = work / ("job%d.json" % j)
        sp.write_text(json.dumps(spec), encoding="utf-8")
        g = gpus[j % len(gpus)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(phys[g % len(phys)] if phys else g))
        lf = open(work / ("job%d.log" % j), "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, "-m", "deep_gemm.mega.autotune",
                                        "--precompile-worker", str(sp)], env=env, stdout=lf,
                                       stderr=subprocess.STDOUT), lf))
    t0 = time.time()
    for proc, lf in procs:
        proc.wait()
        lf.close()
    rows = []
    for f in sorted(work.glob("job*.jsonl")):
        rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r for r in rows if r["status"] != "ok"]
    n_spill = sum(1 for r in bad if r["status"].startswith("spill"))
    log("[autotune] precompile[%s]: %d/%d cells, %d spill, %d other failures, %d jobs, %.0f s"
        % (tag, len(rows), len(cells), n_spill, len(bad) - n_spill, jobs, time.time() - t0))
    for r in bad[:20]:
        log("[autotune]   precompile[%s] %s S=%s %s" % (tag, r["status"][:160], r["S"], r["env"]))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deep_gemm.mega.autotune",
                                 description="Weave turnkey MegaMoE autotune (writes the runtime profile JSON)")
    ap.add_argument("--model", choices=sorted(MODEL_PRESETS), help="shape preset (overridden by explicit dims)")
    ap.add_argument("--num-experts", type=int)
    ap.add_argument("--num-topk", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--intermediate-hidden", type=int)
    ap.add_argument("--mma-type", default="auto")
    ap.add_argument("--num-processes", type=int, default=0, help="EP size (default: all visible GPUs)")
    ap.add_argument("--s-buckets", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(DEFAULT_S_BUCKETS))
    ap.add_argument("--out", default=None, help="profile path (default: WEAVE_AUTOTUNE_FILE / _DIR / ~/.deep_gemm)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--timing", choices=("auto", "graph", "eager"), default="auto")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--no-ptxas-check", action="store_true")
    ap.add_argument("--precompile-only", action="store_true",
                    help="JIT the whole candidate space without launching kernels, then exit")
    ap.add_argument("--precompile-jobs", type=int, default=0, help="parallel JIT jobs (0 = auto, -1 = skip)")
    ap.add_argument("--precompile-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--print-path", action="store_true", help="print the profile path for this GPU and exit")
    ap.add_argument("--stage", choices=("all", "buckets", "probes", "finalize"), default="all",
                    help="run one calibration stage (needs --state); lets a GPU scheduler split the run")
    ap.add_argument("--state", default=None, help="checkpoint JSON for staged / resumable calibration")
    ap.add_argument("--only-s", type=lambda v: [int(x) for x in v.split(",")], default=None,
                    help="buckets stage: calibrate only these S (buffers are still sized by --s-buckets)")
    args = ap.parse_args(argv)
    if args.precompile_worker:
        return _precompile_worker_main(args.precompile_worker)
    if args.model:
        E, K, H, I = MODEL_PRESETS[args.model]
        args.num_experts = args.num_experts or E
        args.num_topk = args.num_topk or K
        args.hidden = args.hidden or H
        args.intermediate_hidden = args.intermediate_hidden or I
    for k in ("num_experts", "num_topk", "hidden", "intermediate_hidden"):
        if not getattr(args, k):
            ap.error(f"--{k.replace('_', '-')} (or --model) is required")
    import torch
    if not args.num_processes:
        args.num_processes = torch.cuda.device_count()
    hw = query_fingerprint(args.num_processes)
    mma = default_mma_type(hw.family) if args.mma_type == "auto" else args.mma_type
    if args.print_path:
        print(profile_path(hw, args.num_experts, args.num_topk, args.hidden, args.intermediate_hidden, 0, mma))
        return 0
    if args.precompile_jobs >= 0 and args.stage in ("all", "buckets"):
        saved = {k: os.environ.pop(k) for k in manual_env_keys()}
        try:
            precompile(hw, args.num_experts, args.num_topk, args.hidden, args.intermediate_hidden,
                       args.only_s or args.s_buckets, jobs=args.precompile_jobs,
                       gpus=list(range(args.num_processes)), ptxas_check=not args.no_ptxas_check,
                       buffer_s_buckets=args.s_buckets, mma_type=mma)
        finally:
            os.environ.update(saved)
    if args.precompile_only:
        return 0
    torch.multiprocessing.spawn(_cli_worker, args=(args.num_processes, args), nprocs=args.num_processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
