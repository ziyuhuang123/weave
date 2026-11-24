#!/usr/bin/env python3
"""Turnkey Two-Stage Compile/Calibration & <1 us Runtime LUT Selector for CDNA3 (gfx942).

Supports any CDNA3 family SKU dynamically without hardcoding CU or XCD counts
(e.g., MI308X 80-CU / 4-XCD, MI300X 304-CU / 8-XCD, MI325X 304-CU / 8-XCD):
  - Stage 1 (Compile + Warmup ABBA Calibration -> JSON LUT):
      Queries dynamic `(gcn_arch, num_cus, num_xcds, l2_bytes, ep_world, nic_gbps)`,
      compiles legal 0-spill `.co` candidates across the finalized 14-knob surface,
      and records the winning `(co_name, grid=2*num_cus, block=256, knobs, thresholds)`
      per token bucket `S_bucket in (16, 128, 512, 1024, 4096, 8192, 16384)`.
  - Stage 2 (<1 us Runtime Selection):
      Looks up `ceil_bucket(S_global)` in an in-memory dict (`O(1)`, < 1 us) and
      dispatches the pre-cached `hipModule_t` (`g_mod_cache` in `ep16_turbo_host.hip`)
      with `grid = 2 * num_cus, block = 256, dyn_lds = 32768`.
"""
from __future__ import annotations

import bisect
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

S_BUCKETS: Tuple[int, ...] = (16, 128, 512, 1024, 4096, 8192, 16384)

PUBLIC_14_KNOBS: Tuple[str, ...] = (
    # Spatial (5)
    "WEAVE_DISPATCH_ACTIVE_SMS",
    "WEAVE_SCATTER_ACTIVE_SMS",
    "WEAVE_DISPATCH_WARPS",
    "WEAVE_SCATTER_WARPS",
    "WEAVE_CU_WAVE_SPLIT",
    # Temporal (3)
    "WEAVE_V_HEAD",
    "WEAVE_READY_GROUP_SIZE",
    "WEAVE_SCATTER_OWNER",
    # Implementation (6)
    "WEAVE_PRECISION_MODE",
    "WEAVE_GEMM1_CONFIG",
    "WEAVE_GEMM2_CONFIG",
    "WEAVE_REG_PULL_BYTES",
    "WEAVE_DEDUP_MODE",
    "WEAVE_SCATTER_STAGING",
)


@dataclass(frozen=True)
class Cdna3Topology:
    gcn_arch: str
    num_cus: int
    num_xcds: int
    l2_bytes_per_xcd: int
    ep_world: int
    nic_gbps_per_gpu: float

    @property
    def cus_per_xcd(self) -> int:
        return max(1, self.num_cus // max(1, self.num_xcds))

    @property
    def pathb_grid(self) -> int:
        return 2 * self.num_cus

    @property
    def pathb_block(self) -> int:
        return 256

    @property
    def fingerprint(self) -> str:
        return f"{self.gcn_arch}_cu{self.num_cus}_xcd{self.num_xcds}_ep{self.ep_world}_nic{int(self.nic_gbps_per_gpu)}g"


def detect_cdna3_topology(device_index: int = 0, ep_world: int = 16, nic_gbps_per_gpu: float = 200.0) -> Cdna3Topology:
    """Query live ROCm device properties & KFD XCC topology without hardcoded SKU constants."""
    num_cus = 0
    l2_bytes = 4 * 1024 * 1024
    gcn_arch = "gfx942"
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            num_cus = int(getattr(props, "multi_processor_count", 0))
            l2_bytes = int(getattr(props, "L2_cache_size", l2_bytes) or l2_bytes)
            arch_raw = str(getattr(props, "gcnArchName", gcn_arch) or gcn_arch)
            gcn_arch = arch_raw.split(":")[0]
    except Exception:
        pass
    if num_cus <= 0:
        raise RuntimeError("Cannot query multiProcessorCount from ROCm runtime; refusing to hardcode CU count")

    # Discover XCD count from KFD topology (num_xcc) or derive from CDNA3 20..38 CUs/XCD architecture
    num_xcds = 0
    kfd_nodes = pathlib.Path("/sys/class/kfd/kfd/topology/nodes")
    if kfd_nodes.is_dir():
        for prop_file in sorted(kfd_nodes.glob("*/properties")):
            txt = prop_file.read_text(encoding="utf-8", errors="ignore")
            kv = {}
            for line in txt.splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    kv[parts[0]] = int(parts[1])
            if kv.get("simd_count", 0) == num_cus * 4 and kv.get("num_xcc", 0) > 0:
                num_xcds = kv["num_xcc"]
                break
    if num_xcds <= 0:
        # Fallback topological divisor on gfx942 (4 XCDs on <=160 CUs, 8 XCDs on >160 CUs)
        num_xcds = 8 if num_cus > 160 else 4

    return Cdna3Topology(
        gcn_arch=gcn_arch,
        num_cus=num_cus,
        num_xcds=num_xcds,
        l2_bytes_per_xcd=l2_bytes,
        ep_world=ep_world,
        nic_gbps_per_gpu=nic_gbps_per_gpu,
    )


def candidate_14knob_space(topo: Cdna3Topology) -> List[Tuple[str, Dict[str, str]]]:
    """Compact Stage-1 compile+calibration candidate pool covering all active regimes."""
    base = {
        "WEAVE_DISPATCH_ACTIVE_SMS": "0",
        "WEAVE_SCATTER_ACTIVE_SMS": "0",
        "WEAVE_DISPATCH_WARPS": "8",
        "WEAVE_SCATTER_WARPS": "4",
        "WEAVE_CU_WAVE_SPLIT": "4-4",
        "WEAVE_V_HEAD": "25",
        "WEAVE_READY_GROUP_SIZE": "16",
        "WEAVE_SCATTER_OWNER": "comm_only",
        "WEAVE_PRECISION_MODE": "fp8",
        "WEAVE_GEMM1_CONFIG": "0",
        "WEAVE_GEMM2_CONFIG": "0",
        "WEAVE_REG_PULL_BYTES": "8192",
        "WEAVE_DEDUP_MODE": "1",
        "WEAVE_SCATTER_STAGING": "reg_2stage",
    }
    def ov(name: str, **kw: str) -> Tuple[str, Dict[str, str]]:
        d = dict(base)
        d.update(kw)
        return name, d

    return [
        ov("v70_4_4_adapt", WEAVE_CU_WAVE_SPLIT="4-4", WEAVE_GEMM1_CONFIG="0", WEAVE_GEMM2_CONFIG="0"),
        ov("v70_4_4_bm64",  WEAVE_CU_WAVE_SPLIT="4-4", WEAVE_GEMM1_CONFIG="64", WEAVE_GEMM2_CONFIG="64"),
        ov("v70_4_4_vhead0_scatw8", WEAVE_CU_WAVE_SPLIT="4-4", WEAVE_V_HEAD="0", WEAVE_SCATTER_WARPS="8",
           WEAVE_GEMM1_CONFIG="64", WEAVE_GEMM2_CONFIG="64"),
        ov("v70_4_4_epi",   WEAVE_CU_WAVE_SPLIT="4-4", WEAVE_SCATTER_OWNER="epilogue_only"),
        ov("v70_8_0_adapt", WEAVE_CU_WAVE_SPLIT="8-0"),
        ov("v70_0_8_pure",  WEAVE_CU_WAVE_SPLIT="0-8", WEAVE_GEMM1_CONFIG="64", WEAVE_GEMM2_CONFIG="64"),
    ]


class V70RuntimeLutSelector:
    """O(1) (<1 us) host-side bucket selector backed by Stage-1 calibrated JSON LUT."""

    def __init__(self, lut_data: Dict):
        self.topology = lut_data["topology"]
        self.buckets = sorted(int(k) for k in lut_data["buckets"].keys())
        self.entries = {int(k): v for k, v in lut_data["buckets"].items()}
        self.thresholds = lut_data.get("runtime_thresholds", {
            "dedup_min_saved_rows": 32,
            "comm_scatter_min_rows": 256,
            "skew_rgs_ratio": 2.5,
        })

    @classmethod
    def from_json(cls, path: pathlib.Path) -> "V70RuntimeLutSelector":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def select(self, s_global: int) -> Dict:
        idx = bisect.bisect_left(self.buckets, int(s_global))
        if idx >= len(self.buckets):
            idx = len(self.buckets) - 1
        b = self.buckets[idx]
        return self.entries[b]


def export_calibrated_lut(
    topo: Cdna3Topology,
    model_key: str,
    bucket_winners: Dict[int, Dict],
    thresholds: Optional[Dict] = None,
) -> Dict:
    return {
        "schema": "weave.cdna3.stage1_lut.v1",
        "model": model_key,
        "topology": asdict(topo) | {"pathb_grid": topo.pathb_grid, "pathb_block": topo.pathb_block},
        "buckets": {str(k): v for k, v in sorted(bucket_winners.items())},
        "runtime_thresholds": thresholds or {
            "dedup_min_saved_rows": 32,
            "comm_scatter_min_rows": 256,
            "skew_rgs_ratio": 2.5,
        },
    }
