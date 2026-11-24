"""Weave 3-Tier framework -- offline calibration + (Phase,N,K) GEMM profiles.

This module calibrates a ``HardwareCalibrationProfile`` for each supported
hardware (SM103, H20, MI308X) from the *real verified measurements* recorded
in this repository and the sprint measurement artifacts.  Every stored
constant is either a transcribed measurement or a derivation from one via a
documented rule; nothing is invented.  The profile drives:

  * ``select_offline_regime(profile, S_global, num_shared_experts)`` -- the
    launch-time regime: an exact pure-21-knob structural ``WEAVE_*`` env dict
    (compile-time structure), a regime label, and the online CTA-0 thresholds
    consumed by ``weave_cost_model.cuh`` / ``cost_model.py``.
  * ``select_phase_envelope / select_phase_inner_profile`` -- the 3-Tier
    Tier-1 engine: independent (Hardware, Precision, Phase in {GEMM1,GEMM2},
    N, K) launch envelopes + per-M inner-tuple LUTs, solved by analytical
    legal pruning + integer physics scoring (zero blind grid search, zero
    floating point -> bit-exact with the C++ mirror by construction).
  * ``build_sm0_expert_order`` -- the Python mirror of the SM0 dedup-aware
    co-scheduling + hot/cold wave pairing helper in weave_policy_runtime.cuh.

3-Tier selection engine (docs/WEAVE_TWO_LEVEL_FRAMEWORK.md):

  Tier 0  hardware calibration       this module's measured profiles
  Tier 1  (Phase,N,K) offline        envelopes + per-M inner tuples here
  Tier 2  online per-step            CTA-0 ``OnlineRoutingPolicy`` (Eq.(1)-(3))
                                     + per-expert profile codes + SM0 ordering
                                     + ready-token queue (device headers)

Knob binding (see docs/TUNABLE_PARAMETERS.md, section 7): structural
compile-time ~13 knobs frozen per offline regime; launch-time regime selected
here from (S_global, shared); online per-step CTA-0 adaptation.  ZERO new
public WEAVE_* knobs are introduced by any of this.

Measurement sources (sha256 recorded in the emitted JSON provenance):

  SM103   results/sm103_ep4_prior_verified.json
          (148-SM SM103, EP4, qwen38max S_rank=1024, H=8192 I=2048 E=512 K=10;
          optimal B8_bm80_epi2_st8_8 710.53 us, skew-safe B6_bm96 716.67 us;
          WeaveProfiler/nsys HBM-TMA mean 76.0 % / median 38.0 % Tensor Active)
  H20     pure21_s1024.json (H20 S=1024 pure-21-knob calibration snapshot of a
          previous kernel revision; not checked in)
          (78-SM H20, world=4, qwen38max S_global=1024; 21-arm pure sweep;
          best p15_adapt_bm 2515.232 us = 1.28591x vs official 3234.352 us;
          per-rank routing aggregates + rank_hit_multiplicity)
  MI308X  S4_MI308X_FINAL.json (MI308X EP8 final sweep summary of a previous
          kernel revision; not checked in)
          (80-CU MI308X, EP8, qwen38max S_global in {1024,128}; winner
          allopt_c 2316.36/2341.10 us at S1024, 1301.0/1306.13 us at S128;
          PMC compute_util max 16.4 % of 606.2 T, memory_util max 70.72 % of
          3684 GB/s, comm max 2.19 % of 287.2 GB/s xGMI; 0 spills; the 8-0
          wave split runtime-FAILs and is never selected)
  routing data/routing_stats_3hw.json (recorded SM100-family routing replay,
          S=1024 cell: local 1286 + remote 8954 routes, 5304 dedup scatter rows)

CLI:
    python deep_gemm/mega/calibrate.py --hardware all \
        --out results/hardware_calibration_3hw.json
    python deep_gemm/mega/calibrate.py --gemm-profiles \
        --out results/gemm_profiles_3hw.json
    python deep_gemm/mega/calibrate.py --hardware sm103 --s-global 4096
    python deep_gemm/mega/calibrate.py --guided --hardware h20 \
        --sweep-json my_pure21_sweep.json --out my_profile.json

``--guided`` calibrates a *new* GPU from a user-supplied pure-21-style sweep
JSON (same schema as pure21_s1024.json) using the identical derivation rules.
The module imports no torch / CUDA and loads cost_model.py by file path, so it
runs on any machine (mirrors tests/validate_cost_model_3hw.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]


def _load_cost_model():
    spec = importlib.util.spec_from_file_location(
        "weave_cost_model_py_calibrate", _HERE / "cost_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = _load_cost_model()


def f32(x: float) -> float:
    """Round to IEEE binary32 (the C++ ``float`` storage format)."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def hexf(x: float) -> str:
    return struct.pack("f", f32(x)).hex()


def nine(x: float) -> float:
    """9 significant digits round-trip any binary32 exactly."""
    return float("%.9g" % f32(x))


# ---------------------------------------------------------------------------
# Transcribed real measurements (zero fabrication: every number below is
# copied verbatim from the cited source artifact).
# ---------------------------------------------------------------------------

SM103_SOURCE = "results/sm103_ep4_prior_verified.json"
SM103_SHA256 = "fad1b847d433cc8ddec083b3c7750ceae5c86c72abddd39468e73b86151721ff"
SM103_VERIFIED = {
    "platform": "NVIDIA SM103 (148 SM, sm_103a)",
    "ep": 4, "model": "qwen38max", "S_rank": 1024, "S_global": 4096,
    "E": 512, "K": 10, "H": 8192, "I": 2048, "num_shared_experts": 0,
    "optimal_arm": "B8_bm80_epi2_st8_8",
    "optimal_median_us": 710.53,
    "optimal_speedup_vs_official": 1.2405,
    "skew_safe_arm": "B6_bm96",
    "skew_safe_median_us": 716.67,
    # weaveprofiler_optimal_s1024.hw_curves_same_run_nsys.counters
    "hbm_tma_util_mean_pct": 76.0,
    "hbm_tma_util_median_pct": 81.0,
    "hbm_tma_util_max_pct": 94.0,
    "tensor_active_median_pct": 38.0,
    "tensor_active_mean_pct": 33.3,
    "nvlink_rx_mean_pct": 47.9,
    "nvlink_rx_median_pct": 43.0,
    "active_sms_median_pct": 100.0,
    "hw_dur_us": 702.407,
    "p_tc_full_tflops": 4200.0,   # Table I SM100 row (weave_cost_model.cuh)
}

H20_SOURCE = ("pure21_s1024.json (H20 S=1024 pure-21-knob calibration snapshot "
              "of a previous kernel revision; not checked in)")
H20_SHA256 = "dd6cdb5b84bb7d2a712fe656828ea9f78bd7a13ac44a997b9b5fbace975b8432"
H20_PURE21 = {
    "platform": "NVIDIA H20 (78 SM, sm_90a)",
    "model": "qwen38max", "S_global": 1024, "world": 4, "num_shared_experts": 0,
    "E": 512, "K": 10, "H": 8192, "I": 2048,
    "official_optimum_us": 3234.352,
    "fused_base_us": 3218.08,
    # per-rank routing_histogram + rank_hit_multiplicity (rank0..rank3)
    "ranks": [
        {"total": 2482, "max": 131, "nonzero": 123, "mean": 19.390625,
         "dedup_rows": 964, "mean_hits": 2.574688796680498,
         "scatter_rows": 1849, "dedup_scatter_rows": 704},
        {"total": 2540, "max": 157, "nonzero": 122, "mean": 19.84375,
         "dedup_rows": 991, "mean_hits": 2.5630676084762866,
         "scatter_rows": 1893, "dedup_scatter_rows": 721},
        {"total": 2937, "max": 346, "nonzero": 122, "mean": 22.9453125,
         "dedup_rows": 1002, "mean_hits": 2.931137724550898,
         "scatter_rows": 2181, "dedup_scatter_rows": 731},
        {"total": 2281, "max": 102, "nonzero": 119, "mean": 17.8203125,
         "dedup_rows": 943, "mean_hits": 2.4188759278897137,
         "scatter_rows": 1704, "dedup_scatter_rows": 666},
    ],
    # measured arm medians (us) and speedup_vs_official
    "arms": {
        "p15_adapt_bm": {"env": {"WEAVE_ADAPT_BM": "1"},
                         "median_us": 2515.232, "speedup": 1.28591},
        "p12_scatter_owner": {"env": {"WEAVE_SCATTER_OWNER": "comm_only"},
                              "median_us": 3171.168, "speedup": 1.01992},
        "p20_scatter_staging": {"env": {"WEAVE_SCATTER_STAGING": "smem"},
                                "median_us": 3172.24, "speedup": 1.01958},
        "p09_v_head": {"env": {"WEAVE_V_HEAD": "64"},
                       "median_us": 3197.776, "speedup": 1.01144},
        "p06_reduce_warps": {"env": {"WEAVE_REDUCE_WARPS": "4"},
                             "median_us": 3211.728, "speedup": 1.00704},
        "p19_dedup_mode": {"env": {"WEAVE_DEDUP_MODE": "1"},
                           "median_us": 3228.224, "speedup": 1.0019},
        "p02_scatter_active_sms": {"env": {"WEAVE_SCATTER_ACTIVE_SMS": "52"},
                                   "median_us": 3693.312, "speedup": 0.87573},
        "p21_local_scatter_dedup": {"env": {"WEAVE_LOCAL_SCATTER_DEDUP": "1"},
                                    "median_us": 3699.968, "speedup": 0.87409},
    },
}

MI308X_SOURCE = ("S4_MI308X_FINAL.json (MI308X EP8 final sweep summary of a "
                 "previous kernel revision; not checked in)")
MI308X_SHA256 = "887d9ece02eb0cdb3900fbcc969512db689e7607833f333d543bdf06e431ff02"
MI308X_S4 = {
    "platform": "AMD MI308X (80 CU, gfx942)",
    "ep": 8, "model": "qwen38max", "seqlens_global": [1024, 128],
    "E": 512, "K": 10, "H": 8192, "I": 2048, "num_shared_experts": 0,
    "winner_arm": "allopt_c",
    "winner_env": {
        "WEAVE_CU_WAVE_SPLIT": "0-8", "WEAVE_SM0_STREAM_SCHED": "1",
        "WEAVE_ADAPT_BM": "1", "WEAVE_DEDUP_MODE": "1",
        "WEAVE_BLOCK_M_L1": "64", "WEAVE_BLOCK_M_L2": "0",
        "WEAVE_V_HEAD": "16", "WEAVE_DISPATCH_ACTIVE_SMS": "0",
        "WEAVE_SCATTER_ACTIVE_SMS": "0",
    },
    "latency_us": {
        "S1024": {"allopt_c": [2316.36, 2341.10], "b0_base": [2347.31, 2388.19, 2339.19]},
        "S128": {"allopt_c": [1301.0, 1306.13], "b0_base": [1468.65, 1474.13, 1468.32]},
    },
    # pmc_capture.curve_ranges_pct (max over the captured window)
    "compute_util_max_pct": 16.4,
    "memory_util_max_pct": 70.72,
    "comm_util_max_pct": 2.19,
    "sm_active_max_pct": 96.16,
    "normalization": {"compute_tflops": 606.2, "memory_gbps": 3684.0,
                      "comm_gbps": 287.2, "cus": 80},
    "spill_regs_all_production_configs": 0,
    "wave_split_8_0": "runtime FAIL (meta_ready handshake timeout) -- never selected",
}

# V58: the 2-Node 16xMI308X EP16 6-model 5-arm measured ledger
# (results/ep16_mi308x_v58_full5arm.json).  The P3 arm is the model-gated champion: 1.4495x vs RCCL+AITER,
# 1.0913x vs the P0/v46C flat-cursor bank, per-rank heterogeneous bank map
# selected on the host from the live per-rank valid_rows ratio (the exact
# fields the v59 single-binary On-Device CTA-0 OnlineRoutingPolicy now
# computes on-GPU from the post-dispatch routing histogram).
MI308X_EP16_SOURCE = "results/ep16_mi308x_v58_full5arm.json"
MI308X_EP16_V58 = {
    "platform": "AMD MI308X (gfx942), 2-Node 16xMI308X EP16, 4xNIC fic2 per node",
    "ep": 16, "world": 16, "E": 512, "K": 10, "num_shared_experts": 0,
    # qwen38max (2.4T) cell: gflop_per_rank 257.69803776e9 / valid_rows_per_rank
    # 10240 = 6*H*I -> H*I = 2^22; representative split H=2048 I=2048.
    "H": 2048, "I": 2048,
    "seqlens_global": [8192, 16384],
    "winner_arm": "P3",
    # roofline_sum_c04_ep16.json: per-GPU achieved TFLOPS of the qwen38max
    # (2.4T) P3 arm -- the most compute-saturated verified V58 cell.
    "f_eff_tflops": 89.02279218814735,
    "memory_util_max_pct": 70.72,   # same MI308X silicon (S4_MI308X_FINAL PMC)
    "model_gated_geomean_vs_rccl": 1.449524157520611,
    "model_gated_geomean_vs_p0": 1.091264849227943,
    "bank_space": {
        "C_FLT_32": {"n08": 1, "n44": 32, "mode": "fleet"},
        "C_FLT_16": {"n08": 4, "n44": 16, "mode": "fleet"},
        "C_COAL_16": {"n08": 4, "n44": 16, "mode": "coalesce"},
        "C_COAL_68": {"n08": 4, "n44": 8, "mode": "coalesce"},
        "C_BASE_16": {"n08": 4, "n44": 16, "mode": "base"},
        "C_BASE_68": {"n08": 4, "n44": 8, "mode": "base"},
    },
    # host pick rule (V58 launcher preset 3): per-rank valid_rows/mean
    # ratio thresholds; all banks keep NPULL >= 30 (the V57 failure-mode guard)
    "pick_thresholds": {"comm_bound": 1.15, "coalesce": 1.18,
                        "qwen38max": 1.10, "other": 1.15},
    "latency_us": {
        "qwen35_35b_S8192": {"rccl_aiter": 1547.515, "P0": 1201.966, "P3": 1166.633},
        "qwen35_122b_S8192": {"rccl_aiter": 2361.523, "P0": 1598.928, "P3": 1556.523},
        "qwen35_397b_S16384": {"rccl_aiter": 5910.06, "P0": 3984.876, "P3": 3831.518},
        "qwen38max_S16384": {"rccl_aiter": 6902.37, "P0": 4764.286, "P3": 4485.619},
        "dskv4_flash_S16384": {"rccl_aiter": 7517.088, "P0": 5611.678, "P3": 5440.392},
        "minimax_m3_S16384": {"rccl_aiter": 6122.798, "P0": 4345.804, "P3": 4173.524},
    },
}

ROUTING_SOURCE = "data/routing_stats_3hw.json"
ROUTING_SHA256 = "97c6cae0d1eb52d52570ef386388afb613eb85194d4be57138c42499cb8e5316"
ROUTING_S1024 = {  # recorded SM100-family replay, qwen38max S=1024 (EP8) cell
    "local_contributions": 1286, "remote_contributions": 8954,
    "dedup_remote_scatter_rows": 5304,
}


# ---------------------------------------------------------------------------
# Deterministic reference histograms derived from the measured aggregates.
# ---------------------------------------------------------------------------

def max_anchored_histogram(total: int, max_n: int, nonzero: int,
                           experts: int) -> List[int]:
    """Deterministic proxy consistent with ALL measured aggregates.

    One expert carries the measured ``max_n``; the remaining ``nonzero-1``
    active experts share ``total-max_n`` as evenly as possible (floor, then
    +1 to the first remainder experts).  Sum == total, max == max_n,
    nonzero count == nonzero, every entry <= max_n.  Used for the H20
    reference point because pure21 stores aggregates, not the full m_e.
    """
    if nonzero <= 0 or total <= 0:
        return [0] * experts
    if nonzero == 1:
        return [total] + [0] * (experts - 1)
    rest = total - max_n
    base, rem = divmod(rest, nonzero - 1)
    hist = [max_n] + [base + 1] * rem + [base] * (nonzero - 1 - rem)
    hist += [0] * (experts - nonzero)
    assert sum(hist) == total and max(hist) == max_n
    return hist


def zipf_histogram(total: int, bins: int) -> List[int]:
    """Zipf(0.9) proxy -- the exact convention of validate_cost_model_3hw.py."""
    weights = [1.0 / ((i + 1) ** 0.9) for i in range(bins)]
    norm = sum(weights)
    out = [int(total * w / norm) for w in weights]
    out[-1] += total - sum(out)
    return out


def _ref_breakdown(arch: str, hist: Sequence[int], hidden: int, inter: int,
                   dedup_pulls: int, scatter: int):
    wl = M.WorkloadSpec(len(hist), hidden, inter)
    return M.select_policy(arch, wl, list(hist), dedup_pulls, scatter)


def _util_tail(arch: str, br) -> Dict[str, float]:
    """Eq.(2) middle HBM utilization and Eq.(3) tail fraction, binary32."""
    hw = M.hardware_for(arch)
    denom = f32(f32(f32(br.t_comp_us) * 1000.0) * hw.b_hbm_gbps)
    util = f32(f32(br.v_mem_bytes) / denom)
    tail = f32(f32(br.t_tail_reduce_us) / f32(br.t_total_us))
    return {"middle_hbm_util": util, "tail_combine_frac": tail}


# ---------------------------------------------------------------------------
# Profile / regime dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HardwareCalibrationProfile:
    hardware: str                       # sm103 | h20 | mi308x
    arch: str                           # cost-model arch key (b300|h20|mi308x)
    precision_mode: int                 # launched backend precision code
    # Roofline constants (binary32; mirrored bit-exactly in C++):
    f_eff_tflops: float
    b_hbm_gbps: float
    b_comm_gbps: float
    tau_sync_us: float
    mu_dedup_thresh: float
    hbm_headroom_thresh: float
    tail_ratio_thresh: float
    v_head_hot_budget: int
    bm_pad_slack: int
    structural_knobs: Dict[str, int]    # Tier-1 pure-21-knob env dict
    verified_max_s_global: int
    provenance: Dict = field(default_factory=dict)

    def online_thresholds(self) -> Dict[str, float]:
        return {
            "f_eff_tflops": self.f_eff_tflops,
            "b_hbm_gbps": self.b_hbm_gbps,
            "b_comm_gbps": self.b_comm_gbps,
            "tau_sync_us": self.tau_sync_us,
            "mu_dedup_thresh": self.mu_dedup_thresh,
            "hbm_headroom_thresh": self.hbm_headroom_thresh,
            "tail_ratio_thresh": self.tail_ratio_thresh,
            "v_head_hot_budget": self.v_head_hot_budget,
            "bm_pad_slack": self.bm_pad_slack,
        }


@dataclass
class OfflineRegime:
    hardware: str
    regime: str
    s_global: int
    num_shared_experts: int
    verified: bool
    structural_env: Dict[str, int]      # exact pure-21-knob WEAVE_* dict
    online_thresholds: Dict[str, float]
    anchor: str
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Calibration (derivation rules documented per constant)
# ---------------------------------------------------------------------------

def _sm103_structural() -> Dict[str, int]:
    # Verified B8/B6 winner mapped onto the unified 21-knob surface: the
    # per-expert BM=80/96 profile lookup becomes WEAVE_ADAPT_BM=1 with auto
    # BLOCK_M_L1/L2 (the host then locks the coupled TMEM/epilogue tuples).
    return {
        "WEAVE_PRECISION_MODE": 0,          # fp8xfp4 backend
        "WEAVE_ADAPT_BM": 1,                # per-expert M-bucketed BM lookup
        "WEAVE_BLOCK_M_L1": 0,              # auto envelope
        "WEAVE_BLOCK_M_L2": 0,
        "WEAVE_SCATTER_OWNER": 0,           # epilogue_only (verified env)
        "WEAVE_SCATTER_STAGING": 0,         # reg_2stage (verified env)
        "WEAVE_LOCAL_SCATTER_DEDUP": 0,     # verified env = 0
        "WEAVE_COMBINE_SYNC_CHUNK": 0,
        "WEAVE_DISPATCH_ACTIVE_SMS": 0,
        "WEAVE_SCATTER_ACTIVE_SMS": 0,
        "WEAVE_REG_BALANCE": 0,
        "WEAVE_REDUCE_WARPS": 4,            # R_max for the online gate
        "WEAVE_V_HEAD": 0,                  # auto (Tier-3 adapts)
    }


def _h20_structural() -> Dict[str, int]:
    # p15_adapt_bm (best, 1.28591x) + p12/p20 comm_only+smem scatter (best
    # single-knob scatter arms).  p02 scatter-SM subset and p21 local dedup
    # regressed (0.876x / 0.874x) and stay off.  REDUCE_WARPS stays 0: the
    # SM90 streaming-reduce path requires COMBINE_SYNC_CHUNK + hbm_dedup,
    # which the measured-best smem structure does not use (p21 regressed).
    return {
        "WEAVE_PRECISION_MODE": 2,          # fp8xfp8 backend (H20 native)
        "WEAVE_ADAPT_BM": 1,                # p15 winner
        "WEAVE_BLOCK_M_L1": 0,
        "WEAVE_BLOCK_M_L2": 0,
        "WEAVE_SCATTER_OWNER": 1,           # comm_only (p12 winner)
        "WEAVE_SCATTER_STAGING": 1,         # smem (p20 winner)
        "WEAVE_LOCAL_SCATTER_DEDUP": 0,     # p21 regressed
        "WEAVE_COMBINE_SYNC_CHUNK": 0,
        "WEAVE_DISPATCH_ACTIVE_SMS": 0,
        "WEAVE_SCATTER_ACTIVE_SMS": 0,      # p02 regressed
        "WEAVE_DISPATCH_WARPS": 0,
        "WEAVE_REDUCE_WARPS": 0,            # unavailable in smem structure
        "WEAVE_V_HEAD": 0,                  # auto; hot budget 64 = p09 win
    }


def _mi308x_structural() -> Dict[str, int]:
    # allopt_c winner verbatim (mapped to integer codes), 0 spills.  The 8-0
    # wave split runtime-FAILs (meta_ready timeout) and is never emitted.
    return {
        "WEAVE_PRECISION_MODE": 2,          # fp8 backend (MI308X native)
        "WEAVE_CU_WAVE_SPLIT": 1,           # 0-8 (winner; 8-0 forbidden)
        "WEAVE_SM0_STREAM_SCHED": 1,        # winner
        "WEAVE_ADAPT_BM": 1,                # winner
        "WEAVE_DEDUP_MODE": 1,              # winner
        "WEAVE_BLOCK_M_L1": 64,             # winner
        "WEAVE_BLOCK_M_L2": 0,              # winner (auto)
        "WEAVE_V_HEAD": 16,                 # winner (explicit pin)
        "WEAVE_DISPATCH_ACTIVE_SMS": 0,     # winner
        "WEAVE_SCATTER_ACTIVE_SMS": 0,      # winner
        "WEAVE_READY_GROUP_SIZE": 0,        # auto
        "WEAVE_REG_BALANCE": 0,
        "WEAVE_REDUCE_WARPS": 0,            # CUDA-only knob; HIP path n/a
    }


def _mi308x_ep16_structural() -> Dict[str, int]:
    # V58 P3 winner verbatim (mapped to integer codes), 0 spills: the v46-SP
    # single-binary regime whose (N08, N44, N80) CU role triple, Fleet-4XCD vs
    # Coalesce claim engine, active_v_head and Middle Streaming Reduce gate are
    # the runtime fields of the On-Device CTA-0 OnlineRoutingPolicy.
    return {
        "WEAVE_PRECISION_MODE": 2,          # fp8 backend (MI308X native)
        # The V58 P3 CU-role mix is a heterogeneous per-rank (N08, N44, N80)
        # triple chosen at runtime by the CTA-0 OnlineRoutingPolicy, not one
        # static split, so the structural env stays at 0 = auto (code 1 would
        # mean a uniform 0-8 split, code 2 a uniform 4-4 split).
        "WEAVE_CU_WAVE_SPLIT": 0,
        # The EP16 TU binds the CTA-0 comm waves statically (GDA wave 4,
        # combine waves 5..7) and fails closed on WEAVE_SM0_STREAM_SCHED != 0
        # (megamoe_amd.hip MI308X_EP16_KNOB_GATE), so the emitted env must be 0.
        "WEAVE_SM0_STREAM_SCHED": 0,
        "WEAVE_ADAPT_BM": 1,                # v46 bm 32/64 adaptive
        "WEAVE_DEDUP_MODE": 1,              # v46 dedup=1
        "WEAVE_BLOCK_M_L1": 64,             # v46 BM=64
        "WEAVE_BLOCK_M_L2": 0,              # auto
        "WEAVE_V_HEAD": 16,                 # hot budget pin (auto-shrink under hot-expert skew)
        "WEAVE_DISPATCH_ACTIVE_SMS": 0,
        "WEAVE_SCATTER_ACTIVE_SMS": 0,
        "WEAVE_READY_GROUP_SIZE": 0,        # auto (the per-expert group ladder is a runtime policy field)
        "WEAVE_REG_BALANCE": 0,
        "WEAVE_REDUCE_WARPS": 0,            # CUDA-only knob; HIP path n/a
    }


def calibrate_hardware_profile(hardware: str) -> HardwareCalibrationProfile:
    hardware = hardware.lower()
    if hardware in ("sm103", "sm100", "b300"):
        return _calibrate_sm103()
    if hardware == "h20":
        return _calibrate_h20()
    if hardware in ("mi308x", "gfx942"):
        return _calibrate_mi308x()
    if hardware in ("mi308x_ep16", "gfx942_ep16"):
        return _calibrate_mi308x_ep16()
    raise ValueError(f"unknown hardware {hardware!r}; expected sm103|h20|mi308x|mi308x_ep16")


def _calibrate_sm103() -> HardwareCalibrationProfile:
    s = SM103_VERIFIED
    hw = M.hardware_for("b300")
    # F_eff: sustained TC rate at the verified optimum = Table-I peak x the
    # measured Tensor-Active median (38.0 %) of the optimal S1024 run.
    f_eff = f32(s["p_tc_full_tflops"] * s["tensor_active_median_pct"] / 100.0)
    # hbm_headroom_thresh: measured HBM/TMA mean utilization (76.0 %) of the
    # optimal run -- Middle below this leaves HBM headroom for Reduce lanes.
    headroom = f32(s["hbm_tma_util_mean_pct"] / 100.0)
    # tail_ratio_thresh: Eq.(3) tail/total at the verified reference point
    # (S_rank=1024, EP4 -> M=10240 routes/rank, E_local=128, Zipf(0.9) proxy
    # histogram -- the validator's documented convention; dedup/scatter = M).
    m_ref = s["S_rank"] * s["K"]            # routes per rank = S_global*K/EP
    hist = zipf_histogram(m_ref, s["E"] // s["ep"])
    br = _ref_breakdown("b300", hist, s["H"], s["I"], m_ref, m_ref)
    ut = _util_tail("b300", br)
    # mu_dedup_thresh: recorded routing replay S=1024 cell pull multiplicity
    # (local+remote routes) / (local+dedup scatter rows); the source study
    # measured dedup shrinking remote scatter 8954 -> 5304 at this mu.
    r = ROUTING_S1024
    mu = f32((r["local_contributions"] + r["remote_contributions"]) /
             (r["local_contributions"] + r["dedup_remote_scatter_rows"]))
    prov = {
        "source": SM103_SOURCE, "sha256": SM103_SHA256,
        "routing_source": ROUTING_SOURCE, "routing_sha256": ROUTING_SHA256,
        "derivations": {
            "f_eff_tflops": "4200.0 TFLOPS (Table I) x Tensor Active median 38.0 % (nsys counters, optimal S1024 run)",
            "b_hbm_gbps": "Table I SM100 row (weave_cost_model.cuh get_hardware_spec)",
            "b_comm_gbps": "Table I SM100 NVLink cap",
            "tau_sync_us": "Table I SM100 t_head sync constant",
            "hbm_headroom_thresh": "HBM / TMA Bandwidth (%) mean = 76.0 (weaveprofiler_optimal_s1024.hw_curves_same_run_nsys)",
            "tail_ratio_thresh": f"Eq.(3) t_tail_reduce/t_total at reference M={m_ref}, E_local=128, Zipf(0.9) proxy (validator convention), dedup=scatter=M",
            "mu_dedup_thresh": "(1286+8954)/(1286+5304) from data/routing_stats_3hw.json S=1024 cell; dedup verified shrinking scatter 8954->5304 in the source study",
            "v_head_hot_budget": "32-row SF column quantum (select_expert_tile_bm minimum candidate, scheduler/mega_moe.cuh) -- structural",
            "bm_pad_slack": "DG_D3_BM_PAD_SLACK default 0 -- structural",
        },
        "reference": {"M": m_ref, "E_local": s["E"] // s["ep"],
                      "histogram": "zipf0.9_proxy", **ut},
        "verified_arms": {s["optimal_arm"]: s["optimal_median_us"],
                          s["skew_safe_arm"]: s["skew_safe_median_us"]},
    }
    return HardwareCalibrationProfile(
        hardware="sm103", arch="b300", precision_mode=0,
        f_eff_tflops=nine(f_eff), b_hbm_gbps=nine(hw.b_hbm_gbps),
        b_comm_gbps=nine(hw.b_link_gbps), tau_sync_us=nine(hw.t_head_us),
        mu_dedup_thresh=nine(mu), hbm_headroom_thresh=nine(headroom),
        tail_ratio_thresh=nine(ut["tail_combine_frac"]),
        v_head_hot_budget=32, bm_pad_slack=0,
        structural_knobs=_sm103_structural(),
        verified_max_s_global=s["S_global"], provenance=prov)


def _calibrate_h20() -> HardwareCalibrationProfile:
    p = H20_PURE21
    hw = M.hardware_for("h20")
    r0 = p["ranks"][0]
    # F_eff: sustained-rate lower bound at the verified optimum = Eq.(1)
    # unpadded F_gemm(rank0) / measured p15_adapt_bm median latency.
    f_gemm = r0["total"] * 6 * p["H"] * p["I"]
    f_eff = f32(f_gemm / p["arms"]["p15_adapt_bm"]["median_us"] / 1e6)
    # Reference point: rank0 max-anchored proxy histogram (consistent with
    # measured total/max/nonzero), dedup_pulls=964, scatter=1849 (rank0).
    hist = max_anchored_histogram(r0["total"], r0["max"], r0["nonzero"],
                                  p["E"] // p["world"])
    br = _ref_breakdown("h20", hist, p["H"], p["I"],
                        r0["dedup_rows"], r0["scatter_rows"])
    ut = _util_tail("h20", br)
    # mu_dedup_thresh: p19_dedup_mode regressed vs fused_base (3228.224 >
    # 3218.08) at every measured rank multiplicity (max 2.9311) -> threshold
    # is the smallest integer strictly above the observed range.
    mu_max = max(r["mean_hits"] for r in p["ranks"])
    mu = f32(math.floor(mu_max) + 1.0)
    prov = {
        "source": H20_SOURCE, "sha256": H20_SHA256,
        "derivations": {
            "f_eff_tflops": f"Eq.(1) unpadded F_gemm(rank0 total={r0['total']}, H=8192, I=2048) = {f_gemm} FLOP / p15_adapt_bm median 2515.232 us (sustained-rate lower bound)",
            "b_hbm_gbps": "Table I H20 row (weave_cost_model.cuh get_hardware_spec)",
            "b_comm_gbps": "Table I H20 NVLink cap",
            "tau_sync_us": "Table I H20 t_head sync constant",
            "hbm_headroom_thresh": f"Eq.(2) V_mem/(T_comp*1000*B_hbm) at rank0 max-anchored proxy (total={r0['total']}, max={r0['max']}, nonzero={r0['nonzero']}), dedup_pulls=964, scatter=1849; pure21 has no PMC, so the anchor is the Eq. mirror at the measured operating point",
            "tail_ratio_thresh": "Eq.(3) t_tail_reduce/t_total at the same rank0 reference (rw=0)",
            "mu_dedup_thresh": f"ceil(max measured mean_hits {mu_max:.6f}) = 3: p19_dedup_mode 3228.224 us > fused_base 3218.08 us at mu in [2.4189, 2.9311]",
            "v_head_hot_budget": "64 = p09_v_head measured best single-knob temporal arm (3197.776 us, 1.01144x)",
            "bm_pad_slack": "DG_D3_BM_PAD_SLACK default 0 -- structural",
        },
        "reference": {"rank": 0, "histogram": "max_anchored_proxy",
                      "dedup_pulls": r0["dedup_rows"],
                      "scatter": r0["scatter_rows"], **ut},
        "verified_arms": {k: v["median_us"] for k, v in p["arms"].items()},
    }
    return HardwareCalibrationProfile(
        hardware="h20", arch="h20", precision_mode=2,
        f_eff_tflops=nine(f_eff), b_hbm_gbps=nine(hw.b_hbm_gbps),
        b_comm_gbps=nine(hw.b_link_gbps), tau_sync_us=nine(hw.t_head_us),
        mu_dedup_thresh=nine(mu), hbm_headroom_thresh=nine(ut["middle_hbm_util"]),
        tail_ratio_thresh=nine(ut["tail_combine_frac"]),
        v_head_hot_budget=64, bm_pad_slack=0,
        structural_knobs=_h20_structural(),
        verified_max_s_global=p["S_global"], provenance=prov)


def _mi308x_mu() -> float:
    """Analytic rank-hit multiplicity of the S4 verified workload.

    Each token picks K distinct experts of E spread over ep ranks; the
    probability a given rank is missed is C(E-E/ep, K)/C(E, K).  Expected
    routes per rank = S*K/ep, unique tokens per rank = S*(1-p_miss), so
    mu = (K/ep)/(1-p_miss) independent of S.  Dedup=1 is part of the verified
    winner allopt_c, so the threshold is this workload's expected mu.
    """
    s = MI308X_S4
    E, K, ep = s["E"], s["K"], s["ep"]
    p_miss = math.comb(E - E // ep, K) / math.comb(E, K)
    return f32((K / ep) / (1.0 - p_miss))


def _calibrate_mi308x() -> HardwareCalibrationProfile:
    s = MI308X_S4
    hw = M.hardware_for("mi308x")
    norm = s["normalization"]
    # F_eff: PMC compute_util max 16.4 % of the 606.2 T FP8-matrix peak.
    f_eff = f32(norm["compute_tflops"] * s["compute_util_max_pct"] / 100.0)
    # hbm_headroom_thresh: PMC memory_util max 70.72 % of 3684 GB/s.
    headroom = f32(s["memory_util_max_pct"] / 100.0)
    mu = _mi308x_mu()
    # tail_ratio_thresh: Eq.(3) at the S1024 reference (S_global=1024, EP8 ->
    # M=1280 routes/rank, E_local=64).  S4 stores no routing histogram, so the
    # reference is uniform (mean_n=20); dedup_pulls = round(M/mu) (dedup=1 is
    # the verified winner), scatter = M.
    m_ref = s["seqlens_global"][0] * s["K"] // s["ep"]
    e_local = s["E"] // s["ep"]
    hist = [m_ref // e_local] * e_local
    din = round(m_ref / mu)
    br = _ref_breakdown("mi308x", hist, s["H"], s["I"], din, m_ref)
    ut = _util_tail("mi308x", br)
    prov = {
        "source": MI308X_SOURCE, "sha256": MI308X_SHA256,
        "derivations": {
            "f_eff_tflops": "PMC compute_util max 16.4 % x 606.2 TFLOPS FP8-matrix peak (pmc_capture.normalization)",
            "b_hbm_gbps": "Table I MI308X row = PMC normalization 3684 GB/s",
            "b_comm_gbps": "Table I MI308X xGMI cap = PMC normalization 287.2 GB/s",
            "tau_sync_us": "Table I MI308X t_head sync constant",
            "hbm_headroom_thresh": "PMC memory_util max 70.72 % (curve_ranges_pct)",
            "tail_ratio_thresh": f"Eq.(3) t_tail_reduce/t_total at uniform reference M={m_ref}, E_local={e_local}, dedup_pulls={din}, scatter={m_ref} (S4 stores no histogram; uniform labeled proxy)",
            "mu_dedup_thresh": f"(K/ep)/(1-C(E-E/ep,K)/C(E,K)) with E=512 K=10 ep=8 = {mu:.9g}; dedup=1 verified in winner allopt_c (S128 ~1.12x)",
            "v_head_hot_budget": "16 = winner allopt_c WEAVE_V_HEAD (explicit pin in the structural regime)",
            "bm_pad_slack": "DG_D3_BM_PAD_SLACK default 0 -- structural",
        },
        "reference": {"M": m_ref, "E_local": e_local, "histogram": "uniform_proxy",
                      "dedup_pulls": din, "scatter": m_ref, **ut},
        "verified_latency_us": s["latency_us"],
        "forbidden": s["wave_split_8_0"],
    }
    return HardwareCalibrationProfile(
        hardware="mi308x", arch="mi308x", precision_mode=2,
        f_eff_tflops=nine(f_eff), b_hbm_gbps=nine(hw.b_hbm_gbps),
        b_comm_gbps=nine(hw.b_link_gbps), tau_sync_us=nine(hw.t_head_us),
        mu_dedup_thresh=nine(mu), hbm_headroom_thresh=nine(headroom),
        tail_ratio_thresh=nine(ut["tail_combine_frac"]),
        v_head_hot_budget=16, bm_pad_slack=0,
        structural_knobs=_mi308x_structural(),
        verified_max_s_global=max(s["seqlens_global"]), provenance=prov)


def _mi308x_ep16_mu() -> float:
    """Analytic rank-hit multiplicity of the V58 verified workload at EP16.

    Same derivation as _mi308x_mu with ep=16 (E=512, K=10, 2-Node EP16):
    mu = (K/ep)/(1-C(E-E/ep,K)/C(E,K)) independent of S.
    """
    s = MI308X_EP16_V58
    E, K, ep = s["E"], s["K"], s["ep"]
    p_miss = math.comb(E - E // ep, K) / math.comb(E, K)
    return f32((K / ep) / (1.0 - p_miss))


def _calibrate_mi308x_ep16() -> HardwareCalibrationProfile:
    v = MI308X_EP16_V58
    hw = M.hardware_for("mi308x")
    # F_eff: the measured per-GPU achieved TFLOPS of the qwen38max (2.4T) P3
    # arm at S16384 (roofline_sum_c04_ep16.json) -- the most compute-saturated
    # verified V58 cell (the EP8 profile instead anchored at the PMC-derived
    # 16.4%-of-peak S4 number; the EP16 ledger carries the achieved number).
    f_eff = f32(v["f_eff_tflops"])
    # hbm_headroom_thresh: PMC memory_util max 70.72 % (same MI308X silicon).
    headroom = f32(v["memory_util_max_pct"] / 100.0)
    mu = _mi308x_ep16_mu()
    # tail_ratio_thresh: Eq.(3) at the S8192 reference (S_global=8192, EP16 ->
    # M=5120 routes/rank, E_local=32, uniform histogram; dedup_pulls =
    # round(M/mu), scatter = M -- the same validator convention as the EP8
    # profile at the EP16 expert count).
    m_ref = v["seqlens_global"][0] * v["K"] // v["ep"]
    e_local = v["E"] // v["ep"]
    hist = [m_ref // e_local] * e_local
    din = round(m_ref / mu)
    br = _ref_breakdown("mi308x", hist, v["H"], v["I"], din, m_ref)
    ut = _util_tail("mi308x", br)
    src = _ROOT / MI308X_EP16_SOURCE
    sha = hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else ""
    prov = {
        "source": MI308X_EP16_SOURCE, "sha256": sha,
        "derivations": {
            "f_eff_tflops": "roofline_sum_c04_ep16.json qwen38max (2.4T) P3 arm: 257.69803776 GFLOP/rank / 4485.619 us per GPU = 89.02279 TFLOPS achieved (the most compute-saturated verified V58 cell)",
            "b_hbm_gbps": "Table I MI308X row = PMC normalization 3684 GB/s (same silicon)",
            "b_comm_gbps": "Table I MI308X xGPI cap = PMC normalization 287.2 GB/s (per-node serial path; the 7-peer xGMI aggregate 346.5 GB/s and the 24.05 GB/s/GPU RDMA share are recorded in the source artifact)",
            "tau_sync_us": "Table I MI308X t_head sync constant",
            "hbm_headroom_thresh": "PMC memory_util max 70.72 % (curve_ranges_pct, same MI308X silicon)",
            "tail_ratio_thresh": f"Eq.(3) t_tail_reduce/t_total at uniform reference M={m_ref}, E_local={e_local}, H*I={v['H'] * v['I']} (qwen38max cell), dedup_pulls={din}, scatter={m_ref}",
            "mu_dedup_thresh": f"(K/ep)/(1-C(E-E/ep,K)/C(E,K)) with E=512 K=10 ep=16 = {mu:.9g} (the EP8 profile's identical derivation at ep=8 -> 1.68861693)",
            "v_head_hot_budget": "16 = the hot-expert opening-pull budget pin (the V58 all-wave opening window; auto-shrinks when n_max/mean >= 2)",
            "bm_pad_slack": "DG_D3_BM_PAD_SLACK default 0 -- structural",
        },
        "reference": {"M": m_ref, "E_local": e_local, "histogram": "uniform_proxy",
                      "dedup_pulls": din, "scatter": m_ref, **ut},
        "verified_arms": {k: sub["P3"] for k, sub in v["latency_us"].items()},
        "model_gated_geomean_vs_rccl": v["model_gated_geomean_vs_rccl"],
        "model_gated_geomean_vs_p0": v["model_gated_geomean_vs_p0"],
        "bank_space": v["bank_space"],
        "pick_thresholds": v["pick_thresholds"],
    }
    return HardwareCalibrationProfile(
        hardware="mi308x_ep16", arch="mi308x", precision_mode=2,
        f_eff_tflops=nine(f_eff), b_hbm_gbps=nine(hw.b_hbm_gbps),
        b_comm_gbps=nine(hw.b_link_gbps), tau_sync_us=nine(hw.t_head_us),
        mu_dedup_thresh=nine(mu), hbm_headroom_thresh=nine(headroom),
        tail_ratio_thresh=nine(ut["tail_combine_frac"]),
        v_head_hot_budget=16, bm_pad_slack=0,
        structural_knobs=_mi308x_ep16_structural(),
        verified_max_s_global=max(v["seqlens_global"]), provenance=prov)


# ---------------------------------------------------------------------------
# Level-1 regime selection (Tier 2)
# ---------------------------------------------------------------------------

_DECODE_MAX_S = 1024
_BALANCED_MAX_S = 8192


def select_offline_regime(profile: HardwareCalibrationProfile,
                          s_global: int,
                          num_shared_experts: int = 0) -> OfflineRegime:
    """Return the exact pure-21-knob structural env + regime + online thresholds.

    The structural env is the verified winner mapping (Tier 1); the regime
    label records the launch bucket and whether S_global is inside the
    verified anchor range.  Beyond the anchor range the same structure is
    extrapolated (flagged, never claimed as measured).  Shared experts add
    always-active local GEMM work; the routed-histogram online rules still
    apply to the routed experts (documented in notes).
    """
    if s_global <= 0:
        raise ValueError("s_global must be positive")
    if s_global <= _DECODE_MAX_S:
        bucket = "decode"
    elif s_global <= _BALANCED_MAX_S:
        bucket = "balanced"
    else:
        bucket = "dense"
    verified = s_global <= profile.verified_max_s_global
    regime = f"{profile.hardware}_{bucket}_{'verified' if verified else 'extrapolated'}"
    notes = [
        "Tier-1 structural env frozen from the verified winner mapping; "
        "Tier-3 CTA-0 adapts reduce warps / v_head / ready group / dedup verdict per step.",
    ]
    if not verified:
        notes.append(f"S_global={s_global} exceeds the verified anchor range "
                     f"(<= {profile.verified_max_s_global}); structure extrapolated, not measured.")
    if num_shared_experts > 0:
        notes.append(f"num_shared_experts={num_shared_experts}: shared experts add "
                     "always-active local GEMM; online skew rules apply to routed experts only.")
    if profile.hardware == "mi308x":
        notes.append("WEAVE_CU_WAVE_SPLIT=8-0 runtime-FAILs (meta_ready timeout) and is never selected.")
    if profile.hardware == "h20":
        notes.append("REDUCE_WARPS=0: the measured-best comm_only+smem scatter structure "
                     "does not enable the SM90 chunked streaming-reduce path (p21 local dedup regressed).")
    anchor = profile.provenance.get("source", "")
    return OfflineRegime(
        hardware=profile.hardware, regime=regime, s_global=s_global,
        num_shared_experts=num_shared_experts, verified=verified,
        structural_env=dict(profile.structural_knobs),
        online_thresholds=profile.online_thresholds(),
        anchor=anchor, notes=notes)


# ---------------------------------------------------------------------------
# Guided calibration for a NEW GPU (pure-21-style sweep JSON input)
# ---------------------------------------------------------------------------

def calibrate_from_sweep(hardware: str, sweep: Dict) -> HardwareCalibrationProfile:
    """Calibrate a new GPU from a user-supplied pure21-style sweep artifact.

    Required sweep keys (same schema as pure21_s1024.json):
      model, S (global), world, E/K/H/I (or hidden/intermediate/experts/topk),
      routing_histogram.rank0 {total,max,nonzero,dedup_rows,scatter_rows},
      arms: {name: {env: {...}, median_us: ...}}, official_optimum_us.
    Optional pmc: {compute_util_max_pct, memory_util_max_pct,
                   compute_tflops, memory_gbps, comm_gbps}.
    The best arm's env becomes the structural regime; roofline constants use
    the PMC when present, otherwise the Eq.-anchor derivations (as for H20).
    """
    arch = {"sm103": "b300", "sm100": "b300", "b300": "b300",
            "h20": "h20", "mi308x": "mi308x"}.get(hardware.lower())
    if arch is None:
        raise ValueError(f"unsupported hardware {hardware!r}")
    hw = M.hardware_for(arch)
    E = sweep.get("E", sweep.get("experts"))
    K = sweep.get("K", sweep.get("topk"))
    H = sweep.get("H", sweep.get("hidden"))
    I = sweep.get("I", sweep.get("intermediate"))
    world = sweep["world"]
    r0 = sweep["routing_histogram"]["rank0"]
    arms = sweep["arms"]
    best = min(arms.items(), key=lambda kv: kv[1]["median_us"])
    hist = max_anchored_histogram(r0["total"], r0["max"], r0["nonzero"], E // world)
    br = _ref_breakdown(arch, hist, H, I,
                        r0.get("dedup_rows", r0["total"]),
                        r0.get("scatter_rows", r0["total"]))
    ut = _util_tail(arch, br)
    pmc = sweep.get("pmc")
    if pmc:
        f_eff = f32(pmc["compute_tflops"] * pmc["compute_util_max_pct"] / 100.0)
        headroom = f32(pmc["memory_util_max_pct"] / 100.0)
        b_hbm = pmc.get("memory_gbps", hw.b_hbm_gbps)
        b_comm = pmc.get("comm_gbps", hw.b_link_gbps)
    else:
        f_eff = f32(r0["total"] * 6 * H * I / best[1]["median_us"] / 1e6)
        headroom = ut["middle_hbm_util"]
        b_hbm, b_comm = hw.b_hbm_gbps, hw.b_link_gbps
    mu = f32(r0["total"] / r0["dedup_rows"]) if r0.get("dedup_rows") else 1.0
    structural = {k: int(v) for k, v in best[1]["env"].items()}
    prov = {"source": sweep.get("source", "user-supplied sweep"),
            "sha256": sweep.get("sha256", ""),
            "best_arm": best[0], "best_median_us": best[1]["median_us"],
            "derivations": {"method": "calibrate_from_sweep (guided, same rules as H20)"}}
    return HardwareCalibrationProfile(
        hardware=hardware.lower(), arch=arch,
        precision_mode=sweep.get("precision_mode", 0),
        f_eff_tflops=nine(f_eff), b_hbm_gbps=nine(b_hbm),
        b_comm_gbps=nine(b_comm), tau_sync_us=nine(hw.t_head_us),
        mu_dedup_thresh=nine(mu), hbm_headroom_thresh=nine(headroom),
        tail_ratio_thresh=nine(ut["tail_combine_frac"]),
        v_head_hot_budget=sweep.get("v_head_hot_budget", 32), bm_pad_slack=0,
        structural_knobs=structural,
        verified_max_s_global=sweep["S"], provenance=prov)


# ---------------------------------------------------------------------------
# Tier 1 (3-Tier upgrade): offline (Hardware, Precision, Phase, N, K) GEMM
# envelope + per-M inner-tuple LUT selection.
#
# GEMM1 (N1 = 2*I, K1 = H, SwiGLU + quant epilogue) and GEMM2 (N2 = H, K2 = I,
# BF16 weight-mul + NVLink/xGMI scatter epilogue) are solved COMPLETELY
# independently: their (N, K) aspect ratios, K-loop lengths, L2 weight
# footprints and epilogue costs differ fundamentally.
#
# The engine is 100% analytical legal pruning + integer scoring -- zero blind
# grid search, zero floating point.  Every arithmetic step is integer, so the
# C++ mirror in weave_cost_model.cuh (select_phase_envelope /
# select_phase_inner_profile) is bit-exact by construction, not by rounding.
#
# Physics rules (each documented at its use site):
#   BN   : N % BN == 0 (zero tail waste), N % (2*BN) == 0 (2-CTA cluster
#          N-pairing; for GEMM1 this also keeps the SwiGLU gate/up halves
#          BN-aligned), accumulator register budget BM*BN <= acc_tile_limit
#          (zero-spill).
#   BK   : K % BK == 0; stages bounded by SMEM capacity, by the K-loop length
#          (stages <= K/BK iterations) and by the short-K rule (K <= 1024
#          never amortizes more than 4 prefetch stages).
#   L1   : intra-expert 2D L2-swizzle protection -- when M_blocks >= 2 and the
#          expert weight fits half of L2, the weight is read from HBM ONCE and
#          re-hit from L2 across M-blocks (interleave_mode 1, swizzle group =
#          M_blocks); otherwise tight wave-level interleave would evict W1[e]
#          and the weight is re-read tiles_m times.
#   score: max(wave-quantized compute, HBM streaming) + pipeline fill +
#          K-iteration TMA issue overhead + epilogue + (GEMM2) scatter,
#          summed over the decode/prefill design anchors M in {32, 192}.
#          Strict-< tie rule, fixed candidate order (BN, BK, BM ascending).
# ---------------------------------------------------------------------------

# Integer hardware constants.  smem_bytes = max per-CTA SMEM/LDS (SM100/SM90:
# 227 KB of the 228 KB per-SM file; MI308X: 64 KB LDS per CU).  l2_bytes from
# Table I (weave_cost_model.cuh get_hardware_spec l2_cache_mb).  b_comm_dgbps
# is the link cap in deci-GB/s (287.2 -> 2872) so every term stays integer.
# acc_tile_limit: BM*BN accumulator elements per 256-thread acc group (SM100/
# SM90: 128 regs/thread) or 512-lane wavefront (MI308X: 128 VGPRs/lane).
PHASE_NK_HW: Dict[str, Dict[str, int]] = {
    "b300": dict(num_sms=148, smem_bytes=232448, smem_reserve=16384,
                 l2_bytes=128 * 1024 * 1024, p_tc_tflops=4200,
                 b_hbm_gbps=7680, b_comm_dgbps=7800,
                 acc_tile_limit=32768, tma_m_cap=128),
    "h20": dict(num_sms=78, smem_bytes=232448, smem_reserve=16384,
                l2_bytes=60 * 1024 * 1024, p_tc_tflops=255,
                b_hbm_gbps=3600, b_comm_dgbps=3700,
                acc_tile_limit=32768, tma_m_cap=64),
    "mi308x": dict(num_sms=80, smem_bytes=65536, smem_reserve=16384,
                   l2_bytes=256 * 1024 * 1024, p_tc_tflops=360,
                   b_hbm_gbps=3684, b_comm_dgbps=2872,
                   acc_tile_limit=65536, tma_m_cap=64),
}

# Precision byte widths in 1/32-byte units (activation, weight) -- the exact
# Table-I / cost_model.py widths: fp8xfp4 (0.5625, 0.5), nvfp4 (0.5625,
# 0.5625), fp8 (1.03125, 1.03125), bf16 (2.03125, 2.0).
PHASE_NK_BYTES32: Dict[str, Tuple[int, int]] = {
    "fp8xfp4": (18, 16),
    "fp4xfp4": (18, 18),
    "fp8": (33, 33),
    "bf16": (65, 64),
}

# LEGAL tile ladders per backend, transcribed from the kernels' static asserts
# (an envelope the backend cannot instantiate is not a recommendation):
#   b300   sm100_fp8_fp4_mega_moe.cuh:1148-1150  BM%16==0, BN 64/128/256,
#                                                BK 128/256
#   h20    sm90_fp8_mega_moe.cuh:623-634         BM%64==0 (16/32 under swapAB),
#                                                BN 64/128/256/512, BK 128/256
#   mi308x megamoe_amd.hip:54-60 + weave_ck_bm_configs.hpp
#                                                BM 16/32/64/128, BN 64/128/256,
#                                                BK 64/128/256
# Walked DESCENDING: the largest surviving legal tile wins.
PHASE_NK_BN: Dict[str, Tuple[int, ...]] = {
    "b300": (256, 128, 64),
    "h20": (512, 256, 128, 64),
    "mi308x": (256, 128, 64),
}
PHASE_NK_BK: Dict[str, Tuple[int, ...]] = {
    "b300": (256, 128),
    "h20": (256, 128),
    "mi308x": (256, 128, 64),
}
PHASE_NK_BM: Dict[str, Tuple[int, ...]] = {
    "b300": (256, 240, 192, 128, 96, 80, 64, 32),
    "h20": (256, 192, 128, 64),
    "mi308x": (128, 64, 32, 16),
}
# MMA M atom: WGMMA/UMMA tile height 64, CDNA3 MFMA 32.
PHASE_NK_MMA_M: Dict[str, int] = {"b300": 64, "h20": 64, "mi308x": 32}
# Ascending BLOCK_M height table shared by the 4-bit envelope code field.
PHASE_NK_BM_HEIGHTS: Tuple[int, ...] = (16, 32, 64, 80, 96, 128, 192, 240, 256)
PHASE_NK_DESIGN_M = (32, 192)   # decode / prefill roofline reporting anchors
PNK_PREFILL_M = 240             # structural-verdict anchor (top of M ladder)
# Structural penalty weights (milli-units, inherited from an earlier tuned design arm).
PNK_WAVE_WEIGHT = 2
PNK_MMA_PAD_WEIGHT = 2
PNK_ROW_PAD_WEIGHT = 1
PNK_BK_BIAS = 96
PNK_NARROW_N = 128
PNK_DEPTH_BONUS = 24
M_LADDER: Tuple[int, ...] = (8, 16, 32, 48, 64, 96, 128, 192, 240)


def pnk_stage_ladder(k: int) -> int:
    """K-length stage preference (Tier-1 stage ladder); SMEM/K-iters still cap."""
    return 6 if k <= 512 else 5 if k <= 1024 else 4 if k <= 2048 else \
           3 if k <= 4096 else 2


def pnk_wg_layout(arch: str, bm: int) -> int:
    """Phase-level warpgroup layout verdict at the envelope's max_BM."""
    if arch == "h20":
        return 1 if bm < 64 else 0
    if arch == "b300":
        return 2 if bm <= 64 else 0
    return 1 if bm < 32 else 0

# The six standard models (baselines/run_b300_baselines.py MODELS geometry;
# qwen36_35b is the prompt key for the repo's Qwen3.5-35B-A3B cell).
STANDARD_MODELS: Dict[str, Dict[str, int]] = {
    "qwen38max": dict(hidden=8192, intermediate=2048, experts=512, topk=10),
    "dskv4_flash": dict(hidden=4096, intermediate=2048, experts=256, topk=6),
    "minimax_m3": dict(hidden=6144, intermediate=3072, experts=128, topk=4),
    "qwen36_35b": dict(hidden=2048, intermediate=512, experts=256, topk=8),
    "qwen35_122b": dict(hidden=3072, intermediate=1024, experts=256, topk=8),
    "qwen35_397b": dict(hidden=4096, intermediate=1024, experts=512, topk=10),
}


def _pnk_ceil_div(n: int, d: int) -> int:
    return 0 if d == 0 else (n + d - 1) // d


def _pnk_tflops(arch: str, precision_mode: str) -> int:
    """Integer tensor-core rate: NVFP4 doubles, BF16 halves (Table I)."""
    base = PHASE_NK_HW[arch]["p_tc_tflops"]
    mode = M.normalize_precision_mode(precision_mode)
    if mode == "fp4xfp4":
        return base * 2
    if mode == "bf16":
        return base // 2
    return base


def phase_geometry(phase: int, hidden: int, inter: int) -> Tuple[int, int]:
    """GEMM1: (N1=2I, K1=H); GEMM2: (N2=H, K2=I).  phase in {1, 2}."""
    if phase == 1:
        return 2 * inter, hidden
    if phase == 2:
        return hidden, inter
    raise ValueError(f"phase must be 1 (GEMM1) or 2 (GEMM2), got {phase}")


def m_bucket(m_tokens: int) -> int:
    """Smallest M_LADDER member >= m_tokens (capped at 240)."""
    for b in M_LADDER:
        if m_tokens <= b:
            return b
    return M_LADDER[-1]


@dataclass(frozen=True)
class PhaseEnvelope:
    bn: int
    bk: int
    max_bm: int
    num_stages: int
    l2_swizzle_group: int
    interleave_mode: int    # 0 tight, 1 expert-major 2D swizzle, 2 grouped
    wg_layout_mode: int = 0  # phase-level warpgroup layout at max_BM
    score_ns: int = 0       # integer roofline estimate at the design anchors
    epw: int = 1            # experts per wave (derived, JSON-only)
    penalty: int = 0        # structural objective that SELECTED this envelope

    def code(self) -> int:
        # 26-bit code, lossless for every structural field (only score_ns /
        # penalty diagnostics are dropped).  BN/BK are log2(v/64) (BN up to
        # 512 on SM90); BLOCK_M uses a 4-bit index into PHASE_NK_BM_HEIGHTS.
        tidx = {64: 0, 128: 1, 256: 2, 512: 3}
        return tidx[self.bn] | (tidx[self.bk] << 2) | \
               (PHASE_NK_BM_HEIGHTS.index(self.max_bm) << 4) | \
               ((self.num_stages - 2) << 8) | \
               ((self.l2_swizzle_group - 1) << 11) | \
               (self.interleave_mode << 14) | (self.wg_layout_mode << 16) | \
               ((self.epw & 0xFF) << 18)


@dataclass(frozen=True)
class GemmInnerProfile:
    block_m: int
    tma_load_m: int
    wg_layout_mode: int     # 0 split-M, 1 split-N, 2 SM103 sub-tile
    pipe_mode: int          # b300 TMEM permille 100/175/200; h20 depth 1..3;
                            # mi308x 1..2
    l2_swizzle_group: int
    store_block_m: int
    epi_reg_stages: int
    scatter_path: int       # GEMM2: 0 direct reg, 1 SMEM handoff,
                            # 2 lock-free local pre-reduce; GEMM1: 0

    def pipe_code(self, arch: str) -> int:
        if arch == "b300":
            return {100: 0, 175: 1, 200: 2}[self.pipe_mode]
        return self.pipe_mode - 1

    def code(self, arch: str) -> int:
        return ((self.block_m // 16) & 0x1f) | \
               (((self.tma_load_m // 16) & 0xf) << 5) | \
               (((self.store_block_m // 16) & 0x7) << 9) | \
               ((self.wg_layout_mode & 0x3) << 12) | \
               ((self.pipe_code(arch) & 0x3) << 14) | \
               (((self.l2_swizzle_group - 1) & 0x7) << 16) | \
               (((self.epi_reg_stages - 1) & 0x1) << 19) | \
               ((self.scatter_path & 0x3) << 20)


def _pnk_stage_bytes(max_bm: int, bn: int, bk: int,
                     act32: int, w32: int) -> int:
    return (max_bm * bk * act32 + bn * bk * w32) // 32


def _pnk_anchor_ns(arch: str, tflops: int, phase: int, n: int, k: int,
                   bn: int, bk: int, max_bm: int, stages: int, m: int,
                   act32: int, w32: int) -> int:
    """Integer time estimate (ns) of one envelope at one design anchor M.

    Terms (all documented physics, integer ns):
      compute : wave-quantized TC time.  Table-I TFLOPS is GPU-wide, so
                FLOP/(TFLOPS*1000) is already whole-GPU ns; multiplying by
                waves = ceil(CTAs/num_sms) charges the quantization rounds.
      hbm     : weight + activation streaming.  L1 intra-expert 2D L2-swizzle
                protection: with >= 2 M-blocks and the weight inside half of
                L2, W[e] is HBM-read ONCE and L2-hit across M-blocks; tight
                wave-level interleave would evict it (tiles_m reads).
                Sustained HBM needs >= 4 in-flight prefetch stages (the
                verified D3 pipeline depth); below that the utilization
                scales as stages/4.
      fill    : pipeline fill = stages * stage_bytes streamed at B_hbm.
      iter    : 2 ns TMA-pair issue overhead per K-iteration per wave.
      epi     : SwiGLU+quant (GEMM1) / bf16 write (GEMM2) at B_hbm.
      scat    : GEMM2 NVLink/xGMI scatter of M*N bf16 at the link cap.
    """
    hw = PHASE_NK_HW[arch]
    bm_eff = M.select_online_expert_bm(m, max_bm, 0)
    tiles_m = _pnk_ceil_div(m, bm_eff)
    tiles_n = n // bn
    ctas = tiles_m * tiles_n
    waves = _pnk_ceil_div(ctas, hw["num_sms"])
    m_pad = tiles_m * bm_eff
    flops = 2 * m_pad * n * k
    compute_ns = (flops // (tflops * 1000)) * waves
    weight_bytes = n * k * w32 // 32
    protected = tiles_m >= 2 and weight_bytes * 2 <= hw["l2_bytes"]
    hbm_bytes = weight_bytes * (1 if protected else tiles_m) + \
        m_pad * k * act32 // 32
    hbm_ns = hbm_bytes // hw["b_hbm_gbps"]
    if stages < 4:
        hbm_ns = hbm_ns * 4 // stages
    stage_bytes = _pnk_stage_bytes(max_bm, bn, bk, act32, w32)
    fill_ns = stages * stage_bytes // hw["b_hbm_gbps"]
    iter_ns = (k // bk) * 2 * waves
    if phase == 1:
        # SwiGLU: read the 2I gate/up activations, write I quantized rows.
        epi_bytes = m_pad * (n * act32 // 32) + m_pad * ((n // 2) * act32 // 32)
        scat_ns = 0
    else:
        # BF16 weight-mul output rows + NVLink/xGMI scatter of M*H bf16.
        epi_bytes = m_pad * (k * act32 // 32) + m_pad * n * 2
        scat_ns = (m_pad * n * 2 * 10) // hw["b_comm_dgbps"]
    epi_ns = epi_bytes // hw["b_hbm_gbps"]
    return max(compute_ns, hbm_ns) + fill_ns + iter_ns + epi_ns + scat_ns


def select_phase_envelope(arch: str, precision_mode: str, phase: int,
                          n: int, k: int) -> PhaseEnvelope:
    """Legal pruning + integer STRUCTURAL scoring; mirrors the C++ exactly.

    The envelope is a structural choice, so the objective is the candidate's
    summed structural waste (see weave_cost_model.cuh select_phase_envelope
    for the term-by-term physics).  A roofline-time objective is dominated by
    candidate-independent terms -- the runtime M ladder clamps BLOCK_M at the
    design anchors -- and collapses distinct geometries onto one envelope.
    The integer roofline estimate is still reported as score_ns.
    """
    if arch not in PHASE_NK_HW:
        raise ValueError(f"unknown arch {arch!r}")
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    hw = PHASE_NK_HW[arch]
    mode = M.normalize_precision_mode(precision_mode)
    act32, w32 = PHASE_NK_BYTES32[mode]
    tflops = _pnk_tflops(arch, mode)
    cluster_slots = hw["num_sms"] // 2
    mma_atom = PHASE_NK_MMA_M[arch]
    bk_min = PHASE_NK_BK[arch][-1]
    weight_bytes = n * k * w32 // 32
    stage_top = min(pnk_stage_ladder(k), 4 if k <= 1024 else 8)
    smem_budget = hw["smem_bytes"] - hw["smem_reserve"]
    best: Optional[PhaseEnvelope] = None
    for bn in PHASE_NK_BN[arch]:
        # Zero N-tail waste + 2-CTA cluster pairing (GEMM1: SwiGLU halves).
        if n % bn != 0 or n % (2 * bn) != 0:
            continue
        tiles_n = n // bn
        clusters_n = tiles_n // 2
        cw = _pnk_ceil_div(clusters_n, cluster_slots)
        col_pen = (cw * cluster_slots - clusters_n) * 1024 // clusters_n
        narrow = PNK_NARROW_N if (n >= 4096 and bn < 128) else 0
        for bk in PHASE_NK_BK[arch]:
            if k % bk != 0:
                continue
            bk_bias = PNK_BK_BIAS if ((k <= 1024 and bk != bk_min) or
                                      (k >= 4096 and bk != 128)) else 0
            st_cap = min(stage_top, max(2, k // bk))
            for stages in range(st_cap, 1, -1):
                # Largest surviving legal BLOCK_M (accumulator + SMEM budget).
                max_bm = 0
                for cand in PHASE_NK_BM[arch]:
                    if cand * bn > hw["acc_tile_limit"]:
                        continue
                    if stages * _pnk_stage_bytes(cand, bn, bk,
                                                 act32, w32) > smem_budget:
                        continue
                    max_bm = cand
                    break
                if max_bm == 0:
                    continue
                bm_d = M.select_online_expert_bm(PNK_PREFILL_M, max_bm, 0)
                tm_d = _pnk_ceil_div(PNK_PREFILL_M, bm_d)
                m_pad = tm_d * bm_d
                mma_pad = (_pnk_ceil_div(max_bm, mma_atom) * mma_atom -
                           max_bm) * 1024 // max_bm
                pad_pen = (m_pad - PNK_PREFILL_M) * 1024 // m_pad
                penalty = (PNK_WAVE_WEIGHT * col_pen +
                           PNK_MMA_PAD_WEIGHT * mma_pad +
                           PNK_ROW_PAD_WEIGHT * pad_pen + bk_bias + narrow)
                penalty = max(0, penalty - stages * PNK_DEPTH_BONUS)
                if best is None or penalty < best.penalty:
                    # L1 L2-swizzle / wave-granularity verdict at the prefill
                    # anchor, from L2_cap and the M distribution, using the
                    # SAME BM ladder the runtime inner tuples use.
                    ctas = tm_d * tiles_n
                    epw = max(1, hw["num_sms"] // ctas)
                    if tm_d >= 2 and weight_bytes * 2 <= hw["l2_bytes"]:
                        interleave, group = 1, min(tm_d, 8)
                        epw = 1
                    elif ctas <= hw["num_sms"]:
                        interleave, group = 0, 1
                    else:
                        interleave, group = 2, min(epw, 8)
                    roof = sum(_pnk_anchor_ns(arch, tflops, phase, n, k,
                                              bn, bk, max_bm, stages, m,
                                              act32, w32)
                               for m in PHASE_NK_DESIGN_M)
                    best = PhaseEnvelope(bn, bk, max_bm, stages, group,
                                         interleave,
                                         pnk_wg_layout(arch, max_bm),
                                         roof, epw, penalty)
    if best is None:
        raise ValueError(f"no legal envelope for {arch} {mode} phase{phase} "
                         f"N={n} K={k}")
    return best


def select_inner_from_envelope(arch: str, precision_mode: str, phase: int,
                               n: int, k: int, env: PhaseEnvelope,
                               m_tokens: int, mu_dedup: int,
                               is_tight_reg_arch: int) -> GemmInnerProfile:
    """Per-M inner tuple constrained to the phase envelope (BN*, BK*, max_BM*,
    stages*).  mu_dedup is the LIVE DEDUP VERDICT (0/1) reduced from mu vs
    mu_dedup_thresh before the call -- integer, so selection stays bit-exact.
    """
    hw = PHASE_NK_HW[arch]
    mode = M.normalize_precision_mode(precision_mode)
    _, w32 = PHASE_NK_BYTES32[mode]
    bm = M.select_online_expert_bm(m_tokens, env.max_bm, 0)
    tma = min(bm, hw["tma_m_cap"])
    if arch == "h20":
        # Small M: split-N keeps all 4 WGMMA warpgroups active; large M
        # splits M across the warpgroups.
        wg = 1 if bm < 64 else 0
    elif arch == "b300":
        wg = 2 if bm <= 64 else 0   # SM103 sub-tile mode for small M
    else:
        wg = 1 if bm < 32 else 0
    if arch == "b300":
        pipe = 200 if (bm >= 128 and env.num_stages >= 4) else \
               175 if bm >= 64 else 100
    elif arch == "h20":
        pipe = 3 if env.num_stages >= 5 else 2 if env.num_stages >= 3 else 1
    else:
        pipe = 2 if env.num_stages >= 4 else 1
    m_blocks = _pnk_ceil_div(m_tokens, bm)
    weight_bytes = n * k * w32 // 32
    if m_blocks >= 2 and weight_bytes * 2 <= hw["l2_bytes"]:
        group = min(m_blocks, 8)
    else:
        group = 1
    store = max(s for s in (16, 32, 64) if bm % s == 0)
    epi = 2 if (bm >= 128 or
                (arch == "b300" and bm in (80, 96) and pipe >= 175)) else 1
    if phase == 1:
        scat = 0
    elif mu_dedup:
        scat = 2    # lock-free local pre-reduce (dedup active)
    elif is_tight_reg_arch:
        scat = 1    # SMEM handoff (register-tight SM90)
    else:
        scat = 0    # direct register scatter
    return GemmInnerProfile(bm, tma, wg, pipe, group, store, epi, scat)


def select_phase_inner_profile(arch: str, precision_mode: str, phase: int,
                               n: int, k: int, m_tokens: int, mu_dedup: int,
                               is_tight_reg_arch: int) -> GemmInnerProfile:
    """Public Tier-1 lookup: envelope + per-M inner tuple (C++ bit-exact)."""
    env = select_phase_envelope(arch, precision_mode, phase, n, k)
    return select_inner_from_envelope(arch, precision_mode, phase, n, k, env,
                                      m_tokens, mu_dedup, is_tight_reg_arch)


def phase_inner_lut(arch: str, precision_mode: str, phase: int,
                    n: int, k: int, mu_dedup: int,
                    is_tight_reg_arch: int) -> Dict[int, GemmInnerProfile]:
    """The 9-bucket per-M LUT sharing one phase envelope."""
    env = select_phase_envelope(arch, precision_mode, phase, n, k)
    return {m: select_inner_from_envelope(arch, precision_mode, phase, n, k,
                                          env, m, mu_dedup, is_tight_reg_arch)
            for m in M_LADDER}


def build_gemm_profiles_report() -> Dict:
    """6 models x 3 hardware x 2 phases x M ladder, native launch precision."""
    arch_of = {"sm103": "b300", "h20": "h20", "mi308x": "mi308x"}
    prec_of = {"sm103": "fp8xfp4", "h20": "fp8", "mi308x": "fp8"}
    profiles: Dict[str, Dict] = {}
    for hwname, arch in arch_of.items():
        profiles[hwname] = {}
        for model, geo in STANDARD_MODELS.items():
            cell: Dict[str, Dict] = {}
            for phase in (1, 2):
                n, k = phase_geometry(phase, geo["hidden"], geo["intermediate"])
                env = select_phase_envelope(arch, prec_of[hwname], phase, n, k)
                lut = phase_inner_lut(arch, prec_of[hwname], phase, n, k,
                                      0, 1 if arch == "h20" else 0)
                cell[f"gemm{phase}"] = {
                    "n": n, "k": k,
                    "precision": prec_of[hwname],
                    "envelope": asdict(env),
                    "envelope_code": env.code(),
                    "inner_lut": {str(m): asdict(p) for m, p in lut.items()},
                    "inner_codes": {str(m): p.code(arch) for m, p in lut.items()},
                }
            profiles[hwname][model] = cell
    envelopes = {
        f"{hw}/{model}/gemm{ph}": profiles[hw][model][f"gemm{ph}"]["envelope_code"]
        for hw in profiles for model in profiles[hw] for ph in (1, 2)}
    tiles = {}
    for hw in profiles:
        for model in profiles[hw]:
            for ph in (1, 2):
                env = profiles[hw][model][f"gemm{ph}"]["envelope"]
                tiles[f"{hw}/{model}/gemm{ph}"] = tuple(
                    env[f] for f in ("bn", "bk", "max_bm", "num_stages"))
    independent = [
        f"{hw}/{model}" for hw in profiles for model in profiles[hw]
        if envelopes[f"{hw}/{model}/gemm1"] != envelopes[f"{hw}/{model}/gemm2"]]
    per_hw_variants = {
        hw: len({envelopes[f"{hw}/{model}/gemm{ph}"]
                 for model in profiles[hw] for ph in (1, 2)})
        for hw in profiles}
    lut_signatures = {
        f"{hw}/{model}/gemm{ph}": tuple(
            profiles[hw][model][f"gemm{ph}"]["inner_codes"][str(m)]
            for m in M_LADDER)
        for hw in profiles for model in profiles[hw] for ph in (1, 2)}
    return {
        "schema": "weave-gemm-profiles-3hw/v2",
        "framework": "3-Tier: (HW) -> (Phase,N,K) offline envelope + per-M "
                     "inner tuple -> (runtime M_e, mu) online CTA-0 + SM0 "
                     "dedup ordering, sharded ready rings & cluster DSMEM",
        "selector": "Legal backend tile ladders (transcribed from the kernels' "
                    "static asserts) + integer structural-waste objective "
                    "(N-cluster wave quantization, MMA M-atom padding, prefill "
                    "M-row padding, K-length BK preference, narrow-N tile "
                    "explosion, minus per-stage latency-hiding bonus). The "
                    "integer roofline estimate is reported as score_ns.",
        "reality_boundary": "Analytical integer physics (legal pruning + "
                            "structural waste scoring), not measured "
                            "latencies; bit-exact mirrored in "
                            "weave_cost_model.cuh and enforced by "
                            "tests/test_two_level_adaptive_policy.py.",
        "m_ladder": list(M_LADDER),
        "models": STANDARD_MODELS,
        "design_anchors": list(PHASE_NK_DESIGN_M),
        "structural_anchor": PNK_PREFILL_M,
        # Lists (not tuples) so a JSON round trip compares equal to the live
        # engine output -- the regeneration gate is a dict equality check.
        "legal_tiles": {
            "bn": {a: list(v) for a, v in PHASE_NK_BN.items()},
            "bk": {a: list(v) for a, v in PHASE_NK_BK.items()},
            "bm": {a: list(v) for a, v in PHASE_NK_BM.items()},
            "mma_m_atom": dict(PHASE_NK_MMA_M)},
        "profiles": profiles,
        "distinct_envelopes": len(set(envelopes.values())),
        "distinct_tile_tuples": len(set(tiles.values())),
        "distinct_lut_signatures": len(set(lut_signatures.values())),
        "independent_phase_pairs": len(independent),
        "independent_phase_pair_cells": independent,
        "per_hw_envelope_variants": per_hw_variants,
        "envelope_codes": envelopes,
    }


# ---------------------------------------------------------------------------
# SM0 dedup-aware co-scheduling + hot/cold wave pairing (Python mirror of
# weave::build_sm0_expert_order in weave_policy_runtime.cuh -- bit-exact,
# integer-only, deterministic).
#
# Input per expert: token_count, multi_hit_tokens, primary_partner (the expert
# id sharing the most multi-hit tokens; E means "none").  Output: a permutation
# expert_order[E].  Rules:
#   1. hot_thresh = max(13, 2*floor(M/active))  (M <= 12 is always cold).
#   2. Active experts are counting-sorted by token_count desc (tie: id asc).
#   3. dedup_active: union-find clusters experts linked by multi-hit partner
#      edges (root = min member id); cluster members stay ADJACENT so a
#      multi-hit token's completion span max(pos)-min(pos) collapses and
#      next_cnt == hit_count fires continuously from wave 1.
#   4. Clusters are classified hot/cold by their hottest member.  Hot clusters
#      are dealt round-robin across the ceil(active/wave_size) waves (into
#      waves with enough remaining capacity -- the last wave is partial), then
#      cold clusters over the same remaining capacities; inside each wave the
#      units are emitted [Hot, Cold, Hot, Cold, ...] (whole clusters,
#      first-appearance order = hottest member desc).  Dealing hot units across
#      ALL waves keeps any wave from being packed purely with M <= 12 cold
#      experts (the H20 Compute-valley waves) and lowers the per-wave token-sum
#      variance vs descending packing whenever the hot population spans waves.
#   5. Zero-token experts are appended last in id order.
# ---------------------------------------------------------------------------

SM0_ORDER_SENTINEL = 0xFFFFFFFF

# Ordering-mode verdicts (mirror of weave::kSm0Mode*).  The hot/cold pairing
# boundary is explicit: [Hot, Cold] interleaving happens ONLY in HOT_COLD
# (dedup inactive AND experts_per_wave >= 2).
SM0_MODE_HOT_COLD = 0
SM0_MODE_DEDUP_CLUSTER = 1
SM0_MODE_PENDULUM = 2
SM0_MODE_NO_PAIR = 3


def _sm0_pendulum_order(token_counts: Sequence[int], multi_hit: Sequence[int],
                        ordered: List[int]) -> List[int]:
    """Centre-out placement ranked by (multi_hit desc, count desc, id asc)."""
    ranked = sorted(ordered,
                    key=lambda e: (-min(multi_hit[e], 65535),
                                   -min(token_counts[e], 65535), e))
    n = len(ranked)
    out: List[int] = [0] * n
    center = (n - 1) // 2
    pos, right, left = center, center + 1, center - 1
    go_right = True
    for e in ranked:
        out[pos] = e
        if go_right and right < n:
            pos, right = right, right + 1
        elif left >= 0:
            pos, left = left, left - 1
        elif right < n:
            pos, right = right, right + 1
        go_right = not go_right
    return out


def build_sm0_expert_order(token_counts: Sequence[int],
                           multi_hit: Sequence[int],
                           partner: Sequence[int],
                           wave_size: int,
                           dedup_active: int) -> List[int]:
    return build_sm0_expert_order_with_mode(
        token_counts, multi_hit, partner, wave_size, dedup_active)[0]


def build_sm0_expert_order_with_mode(
        token_counts: Sequence[int], multi_hit: Sequence[int],
        partner: Optional[Sequence[int]], wave_size: int,
        dedup_active: int) -> Tuple[List[int], int]:
    """Bit-exact mirror of weave::build_sm0_expert_order (+ its mode verdict)."""
    e_count = len(token_counts)
    if e_count == 0:
        return [], SM0_MODE_NO_PAIR
    if len(multi_hit) != e_count:
        raise ValueError("multi_hit must match token_counts length")
    if partner is not None and len(partner) != e_count:
        raise ValueError("partner must match token_counts length")
    total = sum(token_counts)
    active = sum(1 for c in token_counts if c > 0)
    if active == 0:
        return list(range(e_count)), SM0_MODE_NO_PAIR
    hot_thresh = max(13, 2 * (total // active))
    # Stable counting-sort equivalent: count desc (16-bit clamp), id asc.
    ordered = sorted((e for e in range(e_count) if token_counts[e] > 0),
                     key=lambda e: (-min(token_counts[e], 65535), e))
    parent = list(range(e_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dedup = bool(dedup_active)
    have_edges = dedup and partner is not None
    if have_edges:
        for e in range(e_count):
            p = partner[e]
            if multi_hit[e] > 0 and p < e_count and p != e and token_counts[p] > 0:
                ra, rb = find(e), find(p)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = {e: (find(e) if have_edges else e) for e in range(e_count)}
    census: Dict[int, int] = {}
    max_hits = 0
    for e in range(e_count):
        if token_counts[e] == 0:
            continue
        census[roots[e]] = census.get(roots[e], 0) + 1
        if dedup:
            max_hits = max(max_hits, multi_hit[e])
    has_cluster = any(v >= 2 for v in census.values())
    tail = [e for e in range(e_count) if token_counts[e] == 0]
    if dedup and not has_cluster and max_hits > 0:
        return (_sm0_pendulum_order(token_counts, multi_hit, ordered) + tail,
                SM0_MODE_PENDULUM)
    mode = (SM0_MODE_DEDUP_CLUSTER if (dedup and has_cluster) else
            SM0_MODE_HOT_COLD if wave_size >= 2 else SM0_MODE_NO_PAIR)
    if mode == SM0_MODE_NO_PAIR:
        return ordered + tail, mode
    unit_members: Dict[int, List[int]] = {}
    hot_units: List[int] = []
    cold_units: List[int] = []
    for e in ordered:
        r = roots[e]
        if r not in unit_members:
            unit_members[r] = [e]
            (hot_units if token_counts[e] >= hot_thresh else cold_units).append(r)
        else:
            unit_members[r].append(e)
    wave_size = max(1, wave_size)
    num_waves = _pnk_ceil_div(active, wave_size)
    # Capacity-aware dealing: the last wave is partial (active % wave_size),
    # so units are dealt round-robin into waves with enough remaining slots
    # (whole clusters never straddle a wave).  Hot units are dealt first, then
    # cold units over the same remaining capacities.
    remaining = [wave_size] * num_waves
    remaining[-1] = active - wave_size * (num_waves - 1)
    wave_hot: List[List[int]] = [[] for _ in range(num_waves)]
    wave_cold: List[List[int]] = [[] for _ in range(num_waves)]

    def deal(units: List[int], dest: List[List[int]]) -> None:
        start = 0
        for u in units:
            sz = len(unit_members[u])
            found = -1
            for k in range(num_waves):
                w = (start + k) % num_waves
                if remaining[w] >= sz:
                    found = w
                    break
            if found < 0:
                # No wave has room for the whole cluster: keep adjacency and
                # overflow into the emptiest wave (lowest index on ties).
                found = max(range(num_waves), key=lambda w: (remaining[w], -w))
            dest[found].append(u)
            remaining[found] = max(0, remaining[found] - sz)
            start = (found + 1) % num_waves

    deal(hot_units, wave_hot)
    deal(cold_units, wave_cold)
    order: List[int] = []
    for w in range(num_waves):
        hs, cs = wave_hot[w], wave_cold[w]
        i = j = 0
        while mode == SM0_MODE_HOT_COLD and i < len(hs) and j < len(cs):
            order.extend(unit_members[hs[i]])
            i += 1
            order.extend(unit_members[cs[j]])
            j += 1
        while i < len(hs):
            order.extend(unit_members[hs[i]])
            i += 1
        while j < len(cs):
            order.extend(unit_members[cs[j]])
            j += 1
    order.extend(tail)
    assert len(order) == e_count and sorted(order) == list(range(e_count))
    return order, mode


def sm0_completion_span(order: Sequence[int],
                        hit_sets: Sequence[Sequence[int]]) -> int:
    """Sum of per-token completion spans max(pos)-min(pos) over hit experts."""
    pos = {e: i for i, e in enumerate(order)}
    span = 0
    for hits in hit_sets:
        ps = [pos[e] for e in hits]
        if ps:
            span += max(ps) - min(ps)
    return span


def sm0_wave_m_variance(order: Sequence[int], token_counts: Sequence[int],
                        wave_size: int) -> float:
    """Variance of per-wave token sums (wave = consecutive wave_size slots)."""
    sums = []
    for i in range(0, len(order), wave_size):
        sums.append(sum(token_counts[e] for e in order[i:i + wave_size]))
    if not sums:
        return 0.0
    mean = sum(sums) / len(sums)
    return sum((s - mean) ** 2 for s in sums) / len(sums)


# ---------------------------------------------------------------------------
# JSON emission + CLI
# ---------------------------------------------------------------------------

def profile_to_dict(p: HardwareCalibrationProfile) -> Dict:
    d = asdict(p)
    d["float_bits"] = {k: hexf(d[k]) for k in (
        "f_eff_tflops", "b_hbm_gbps", "b_comm_gbps", "tau_sync_us",
        "mu_dedup_thresh", "hbm_headroom_thresh", "tail_ratio_thresh")}
    return d


def build_report(profiles: Sequence[HardwareCalibrationProfile]) -> Dict:
    return {
        "schema": "weave-hardware-calibration-3hw/v1",
        "framework": "Two-Level: offline hardware calibration + online CTA-0 routing-adaptive selection",
        "reality_boundary": "Every constant is a transcribed measurement or a documented derivation from one; "
                            "reference histograms are labeled proxies (max-anchored / Zipf(0.9) / uniform) "
                            "consistent with the measured aggregates. No performance claim is made for "
                            "extrapolated regimes.",
        "profiles": {p.hardware: profile_to_dict(p) for p in profiles},
    }


_GUIDED_STEPS = """\
Guided offline calibration for a NEW GPU (Tier 1):
  1. Run the pure-21-knob sweep at one representative global sequence length
     (e.g. the 21 single-knob arms + fused base, 5 reps, median) and record
     per-rank routing aggregates {total,max,nonzero,dedup_rows,scatter_rows}.
  2. Capture PMC / profiler utilization (compute, HBM, comm) for the best arm.
  3. Save the artifact as a pure21-style JSON (see pure21_s1024.json schema).
  4. python deep_gemm/mega/calibrate.py --guided --hardware <name> \\
         --sweep-json <artifact.json> --out <profile.json>
  5. The emitted structural env + online thresholds feed select_offline_regime;
     the C++ constants in weave_cost_model.cuh must be regenerated to match
     (tests/test_two_level_adaptive_policy.py enforces bit-exact parity).
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hardware", default="all",
                    help="sm103 | h20 | mi308x | mi308x_ep16 | all")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--s-global", type=int, default=None,
                    help="also print select_offline_regime for this S_global")
    ap.add_argument("--shared-experts", type=int, default=0)
    ap.add_argument("--guided", action="store_true",
                    help="calibrate a new GPU from --sweep-json")
    ap.add_argument("--sweep-json", default=None)
    ap.add_argument("--gemm-profiles", action="store_true",
                    help="emit the Tier-1 (HW, Precision, Phase, N, K) GEMM "
                         "profile report (6 models x 3 hw x 2 phases x M ladder)")
    args = ap.parse_args(argv)

    if args.gemm_profiles:
        report = build_gemm_profiles_report()
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8", newline="\n")
            print(f"WROTE {out}")
        else:
            print(text, end="")
        return 0

    if args.guided:
        if not args.sweep_json or args.hardware == "all":
            print("--guided requires --hardware <name> and --sweep-json <path>")
            print(_GUIDED_STEPS)
            return 2
        sweep = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
        profiles = [calibrate_from_sweep(args.hardware, sweep)]
    else:
        names = (["sm103", "h20", "mi308x", "mi308x_ep16"] if args.hardware == "all"
                 else [args.hardware])
        profiles = [calibrate_hardware_profile(n) for n in names]

    report = build_report(profiles)
    if args.s_global is not None:
        report["regimes"] = [
            asdict(select_offline_regime(p, args.s_global, args.shared_experts))
            for p in profiles]
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"WROTE {out}")
    else:
        print(text, end="")
    if args.guided:
        print(_GUIDED_STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
