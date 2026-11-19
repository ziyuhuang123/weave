"""Pure-Python mirror of the C++ Weave Eq.(1)-(3) cost model.

This module intentionally has no torch / CUDA dependency.  It is used by the
repository validator to cross-check the C++ header against golden outputs and
to replay routing aggregates without importing the compiled DeepGEMM module.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Sequence, Tuple

POLICY_MAGIC = 0x57415645
POLICY_VERSION = 4


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    arch: str
    num_sms_or_cus: int
    max_units_per_sm: int
    p_tc_full_tflops: float
    p_tc_half_tflops: float
    b_hbm_gbps: float
    b_link_gbps: float
    b_pull_per_unit_gbps: float
    b_scat_comm_per_unit_gbps: float
    b_scat_epi_per_unit_gbps: float
    b_epi_write_per_unit_gbps: float
    l2_cache_mb: float
    t_head_us: float
    t_tail_us: float
    t_handoff_us: float
    weight_bytes: float
    act_in_bytes: float
    out_bytes: float
    head_tasks_per_comm_unit: int


HARDWARE: Dict[str, HardwareSpec] = {
    "b300": HardwareSpec(
        "b300", 148, 16, 4200.0, 3100.0, 7680.0, 780.0,
        1.35, 1.20, 0.35, 8.0, 128.0, 14.0, 8.0, 48.0, 0.5, 0.5625, 2.0, 64),
    "h20": HardwareSpec(
        "h20", 78, 20, 255.0, 210.0, 3600.0, 370.0,
        0.58, 0.48, 0.14, 4.0, 60.0, 18.0, 9.0, 0.8, 1.0, 1.03125, 2.0, 32),
    "mi308x": HardwareSpec(
        "mi308x", 80, 8, 360.0, 205.0, 3684.0, 287.2,
        0.92, 0.82, 0.28, 8.0, 256.0, 22.0, 14.0, 1.5, 1.0, 1.03125, 2.0, 16),
}


def encode_r_mask(comm: int, comp: int, groups: int, heterogeneous: bool = False) -> int:
    return ((comm & 0xFF) | ((comp & 0xFF) << 8) |
            ((groups & 0xFF) << 16) | (0x80000000 if heterogeneous else 0))


PRECISION_ALIASES = {
    "fp8xfp4": "fp8xfp4",
    "fp4xfp4": "fp4xfp4",
    "nvfp4": "fp4xfp4",
    "fp8": "fp8",
    "fp8xfp8": "fp8",
    "bf16": "bf16",
}


def normalize_precision_mode(precision_mode: str) -> str:
    """Map mma_type / CLI spellings onto the four canonical cost-model modes."""
    key = str(precision_mode).strip().lower()
    if key not in PRECISION_ALIASES:
        raise ValueError(f"unknown precision_mode {precision_mode!r}")
    return PRECISION_ALIASES[key]


def precision_byte_width(precision_mode: str, activation: bool) -> float:
    mode = normalize_precision_mode(precision_mode)
    if mode == "fp8xfp4":
        return 0.5625 if activation else 0.5
    if mode == "fp4xfp4":
        return 0.5625
    if mode == "fp8":
        return 1.03125
    return 2.03125 if activation else 2.0


def precision_tc_scale(precision_mode: str) -> float:
    mode = normalize_precision_mode(precision_mode)
    return 2.0 if mode == "fp4xfp4" else (0.5 if mode == "bf16" else 1.0)


def hardware_for(arch: str, precision_mode: str = "fp8xfp4") -> HardwareSpec:
    """Return the Table-I hardware spec adjusted by the precision knob.

    The default FP8xFP4 row is returned unchanged, so every existing prediction
    remains bit-identical.
    """
    hw = HARDWARE[arch]
    mode = normalize_precision_mode(precision_mode)
    if mode == "fp8xfp4":
        return hw
    scale = precision_tc_scale(mode)
    return dataclasses.replace(
        hw,
        p_tc_full_tflops=hw.p_tc_full_tflops * scale,
        p_tc_half_tflops=hw.p_tc_half_tflops * scale,
        weight_bytes=precision_byte_width(mode, False),
        act_in_bytes=precision_byte_width(mode, True),
    )


@dataclasses.dataclass(frozen=True)
class SpatialCandidate:
    r_mask: int
    comm_units: int
    comp_units: int
    block_m: int
    role_groups: int
    p_tc_tflops: float
    heterogeneous: bool


CANDIDATES: Dict[str, Tuple[SpatialCandidate, ...]] = {
    "b300": (
        SpatialCandidate(encode_r_mask(4, 8, 4, False), 4, 8, 128, 4, 4200.0, False),
        SpatialCandidate(encode_r_mask(8, 4, 4, True), 8, 4, 64, 4, 3100.0, True),
    ),
    "h20": (
        SpatialCandidate(encode_r_mask(2, 16, 5, False), 2, 16, 128, 5, 255.0, False),
        SpatialCandidate(encode_r_mask(10, 8, 5, True), 10, 8, 64, 5, 210.0, True),
    ),
    "mi308x": (
        SpatialCandidate(encode_r_mask(0, 8, 8, False), 0, 8, 128, 8, 360.0, False),
        SpatialCandidate(encode_r_mask(4, 4, 8, False), 4, 4, 64, 8, 205.0, False),
    ),
}


@dataclasses.dataclass(frozen=True)
class WorkloadSpec:
    local_experts: int
    hidden: int
    intermediate: int


@dataclasses.dataclass(frozen=True)
class Breakdown:
    r_mask: int
    scatter_mode: int
    total_tokens: int
    v_head: int
    f_gemm_flops: float
    v_mem_bytes: float
    v_epi_bytes: float
    t_mma_us: float
    t_epi_us: float
    t_comp_us: float
    t_pull_us: float
    t_scat_us: float
    t_tail_reduce_us: float
    t_mid_us: float
    t_total_us: float


def _ceil_div(n: int, d: int) -> int:
    return 0 if d == 0 else (n + d - 1) // d


def _align(n: int, a: int) -> int:
    return n if a == 0 else _ceil_div(n, a) * a


def choose_v_head(hw: HardwareSpec, c: SpatialCandidate, dedup_pulls: int) -> int:
    if dedup_pulls == 0:
        return 0
    comm = c.comm_units if c.comm_units else c.comp_units
    window = hw.num_sms_or_cus * comm * hw.head_tasks_per_comm_unit
    return _align(min(window, dedup_pulls), 32)


def evaluate(arch: str, wl: WorkloadSpec, c: SpatialCandidate,
             scatter_mode: int, m_e: Sequence[int],
             dedup_pulls: int, scatter_count: int,
             precision_mode: str = "fp8xfp4") -> Breakdown:
    hw = hardware_for(arch, precision_mode)
    p_tc_tflops = c.p_tc_tflops * precision_tc_scale(precision_mode)
    total = int(sum(m_e))
    h = float(wl.hidden)
    i = float(wl.intermediate)
    bm = float(c.block_m)
    f_gemm = 0.0
    weight_bytes = 0.0
    for count in m_e:
        k = _ceil_div(int(count), c.block_m)
        f_gemm += float(k) * bm * 6.0 * h * i
        weight_bytes += float(k) * 3.0 * h * i * hw.weight_bytes
    v_mem = weight_bytes + float(total) * hw.act_in_bytes * (h + i)
    t_mma = max(v_mem / (hw.b_hbm_gbps * 1000.0),
                f_gemm / (p_tc_tflops * 1_000_000.0))
    v_epi = float(total) * (hw.act_in_bytes * i + hw.out_bytes * h)
    total_epi_units = float(hw.num_sms_or_cus) * float(c.comp_units)
    b_epi = min(hw.b_hbm_gbps,
                max(1.0, total_epi_units * hw.b_epi_write_per_unit_gbps))
    t_epi = v_epi / (b_epi * 1000.0)
    t_comp = max(t_mma, t_epi) if arch == "b300" else t_mma + t_epi

    eff_comm_per_sm = c.comm_units if c.comm_units else c.comp_units
    gpu_comm = float(hw.num_sms_or_cus * eff_comm_per_sm)
    gpu_epi = float(hw.num_sms_or_cus * c.comp_units)
    b_pull = min(hw.b_link_gbps,
                 max(1.0, gpu_comm * hw.b_pull_per_unit_gbps))
    if scatter_mode == 0:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, gpu_epi * hw.b_scat_epi_per_unit_gbps))
    else:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, gpu_comm * hw.b_scat_comm_per_unit_gbps))
    t_pull = 0.0 if dedup_pulls == 0 else (
        float(dedup_pulls) * h * hw.act_in_bytes / (b_pull * 1000.0))
    t_scat = 0.0 if scatter_count == 0 else (
        float(scatter_count) * h * hw.out_bytes / (b_scat * 1000.0))
    if scatter_mode == 0:
        t_mid = max(t_pull, t_comp + t_scat)
    else:
        t_mid = max(t_pull + t_scat, t_comp + hw.t_handoff_us)
    b_tail = min(hw.b_hbm_gbps,
                 max(1.0, float(hw.num_sms_or_cus) *
                     float(hw.max_units_per_sm) * 0.55))
    t_tail_reduce = float(total) * h * hw.out_bytes * 2.0 / (b_tail * 1000.0)
    t_total = hw.t_head_us + t_mid + t_tail_reduce
    return Breakdown(c.r_mask, scatter_mode, total,
                     choose_v_head(hw, c, dedup_pulls), f_gemm, v_mem, v_epi,
                     t_mma, t_epi, t_comp, t_pull, t_scat, t_tail_reduce,
                     t_mid, t_total)


def select_policy(arch: str, wl: WorkloadSpec, m_e: Sequence[int],
                  dedup_pulls: int, scatter_count: int,
                  precision_mode: str = "fp8xfp4") -> Breakdown:
    best: Breakdown | None = None
    for c in CANDIDATES[arch]:
        for s in (0, 1):
            br = evaluate(arch, wl, c, s, m_e, dedup_pulls, scatter_count,
                          precision_mode)
            if best is None or br.t_total_us < best.t_total_us:
                best = br
    assert best is not None
    return best



