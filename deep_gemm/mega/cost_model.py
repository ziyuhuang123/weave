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

# Mirrors weave::kMaxRuntimeExperts: PolicyControl holds E_local <= 256 counts.
MAX_RUNTIME_EXPERTS = 256
# Mirrors weave::kMaxReduceWarps and weave::kReduceGbpsPerWarp.
MAX_REDUCE_WARPS = 8
REDUCE_GBPS_PER_WARP = 0.55


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
             precision_mode: str = "fp8xfp4",
             reduce_warps: int = 0) -> Breakdown:
    """Eq.(1)-(3) for one candidate/scatter pair.

    reduce_warps mirrors WEAVE_REDUCE_WARPS.  0 keeps the published equations
    bit-identical; N > 0 carves N streaming Reduce lanes out of the CTA
    communication group, overlapping the Combine reduction with t_mid on the
    spare Middle HBM bandwidth and shrinking the comm-queue scatter group to
    max(1, N_c - N) units.
    """
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

    rw = min(int(reduce_warps), MAX_REDUCE_WARPS)
    eff_comm_per_sm = c.comm_units if c.comm_units else c.comp_units
    scat_comm_per_sm = eff_comm_per_sm
    if rw > 0 and scatter_mode != 0:
        scat_comm_per_sm = max(1, eff_comm_per_sm - rw)
    gpu_comm = float(hw.num_sms_or_cus * eff_comm_per_sm)
    gpu_scat_comm = float(hw.num_sms_or_cus * scat_comm_per_sm)
    gpu_epi = float(hw.num_sms_or_cus * c.comp_units)
    b_pull = min(hw.b_link_gbps,
                 max(1.0, gpu_comm * hw.b_pull_per_unit_gbps))
    if scatter_mode == 0:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, gpu_epi * hw.b_scat_epi_per_unit_gbps))
    else:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, gpu_scat_comm * hw.b_scat_comm_per_unit_gbps))
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
                     float(hw.max_units_per_sm) * REDUCE_GBPS_PER_WARP))
    reduce_bytes = float(total) * h * hw.out_bytes * 2.0
    t_tail_reduce = reduce_bytes / (b_tail * 1000.0)
    if rw > 0:
        # Middle already streams V_mem through HBM inside T_comp; only the
        # spare bandwidth is available to the Reduce lanes, and an empty Middle
        # offers no overlap window at all.
        b_mid_used = (v_mem / (t_comp * 1000.0) if t_comp > 0.0
                      else hw.b_hbm_gbps)
        b_spare = max(0.0, hw.b_hbm_gbps - b_mid_used)
        b_reduce = min(b_spare, float(hw.num_sms_or_cus) * float(rw) *
                       REDUCE_GBPS_PER_WARP)
        t_tail_reduce = (max(0.0, reduce_bytes - b_reduce * 1000.0 * t_mid) /
                         (b_tail * 1000.0))
    t_total = hw.t_head_us + t_mid + t_tail_reduce
    return Breakdown(c.r_mask, scatter_mode, total,
                     choose_v_head(hw, c, dedup_pulls), f_gemm, v_mem, v_epi,
                     t_mma, t_epi, t_comp, t_pull, t_scat, t_tail_reduce,
                     t_mid, t_total)


def select_policy(arch: str, wl: WorkloadSpec, m_e: Sequence[int],
                  dedup_pulls: int, scatter_count: int,
                  precision_mode: str = "fp8xfp4",
                  reduce_warps: int = 0) -> Breakdown:
    best: Breakdown | None = None
    for c in CANDIDATES[arch]:
        for s in (0, 1):
            br = evaluate(arch, wl, c, s, m_e, dedup_pulls, scatter_count,
                          precision_mode, reduce_warps)
            if best is None or br.t_total_us < best.t_total_us:
                best = br
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Two-Level framework, Level 2: online CTA-0 routing-adaptive policy mirror.
#
# The C++ Tier-3 path (weave_cost_model.cuh cta0_online_policy) stores every
# equation term in binary32.  The mirror below reproduces that arithmetic one
# rounding per operation so the OnlineRoutingPolicy is bit-exact across the
# two languages (tests/test_two_level_adaptive_policy.py compares float fields
# by their binary32 bit patterns).  The double-precision evaluate/select_policy
# above are untouched: every existing golden prediction stays bit-identical.
# ---------------------------------------------------------------------------

import struct as _struct


def f32(x: float) -> float:
    """Round to IEEE binary32 (the C++ ``float`` storage format)."""
    return _struct.unpack("f", _struct.pack("f", float(x)))[0]


def f32_bits(x: float) -> int:
    return _struct.unpack("I", _struct.pack("f", float(x)))[0]


@dataclasses.dataclass(frozen=True)
class HardwareCalibration:
    f_eff_tflops: float
    b_hbm_gbps: float
    b_comm_gbps: float
    tau_sync_us: float
    mu_dedup_thresh: float
    hbm_headroom_thresh: float
    tail_ratio_thresh: float
    v_head_hot_budget: int
    bm_pad_slack: int


# Mirrors weave::get_hardware_calibration; calibrated by deep_gemm/mega/
# calibrate.py from the real verified 3-hardware measurements (sources and
# derivation rules: results/hardware_calibration_3hw.json).
HARDWARE_CALIBRATION: Dict[str, HardwareCalibration] = {
    "b300": HardwareCalibration(
        1596.0, 7680.0, 780.0, 14.0,
        1.55386949, 0.75999999, 0.295702934, 32, 0),
    "h20": HardwareCalibration(
        99.3333054, 3600.0, 370.0, 18.0,
        3.0, 0.455443323, 0.0238960907, 64, 0),
    "mi308x": HardwareCalibration(
        99.4168015, 3684.0, 287.200012, 22.0,
        1.68861687, 0.707199991, 0.0550832227, 16, 0),
}


@dataclasses.dataclass(frozen=True)
class OnlineRoutingPolicy:
    active_reduce_warps: int
    active_v_head: int
    active_ready_group_size: int
    enable_dedup_runtime: int
    bm_tier_mean: int
    bm_envelope: int
    active_experts: int
    max_n: int
    total_routes: int
    adaptive_active: int
    reduce_gate_hbm_ok: int
    reduce_gate_tail_ok: int
    mean_n: float
    rank_hit_multiplicity_mu: float
    middle_hbm_util: float
    tail_combine_frac: float


def select_online_expert_bm(num_tokens: int, envelope: int, slack: int) -> int:
    """Mirror of sched::select_expert_tile_bm / weave::select_online_expert_bm."""
    if num_tokens == 0 or envelope % 32 != 0:
        return envelope
    min_padded = _ceil_div(num_tokens, 32) * 32
    best = 32
    if envelope >= 64 and envelope % 64 == 0:
        if _ceil_div(num_tokens, 64) * 64 <= min_padded + slack:
            best = 64
    if envelope >= 128 and envelope % 128 == 0:
        if _ceil_div(num_tokens, 128) * 128 <= min_padded + slack:
            best = 128
    if envelope >= 256 and envelope % 256 == 0:
        if _ceil_div(num_tokens, 256) * 256 <= min_padded + slack:
            best = 256
    if envelope > 32 and (envelope & (envelope - 1)) != 0:
        if _ceil_div(num_tokens, envelope) * envelope <= min_padded + slack:
            best = envelope
    return best


def _hw_f32(arch: str, precision_mode: str) -> HardwareSpec:
    hw = hardware_for(arch, precision_mode)
    return dataclasses.replace(
        hw,
        p_tc_full_tflops=f32(hw.p_tc_full_tflops),
        p_tc_half_tflops=f32(hw.p_tc_half_tflops),
        b_hbm_gbps=f32(hw.b_hbm_gbps),
        b_link_gbps=f32(hw.b_link_gbps),
        b_pull_per_unit_gbps=f32(hw.b_pull_per_unit_gbps),
        b_scat_comm_per_unit_gbps=f32(hw.b_scat_comm_per_unit_gbps),
        b_scat_epi_per_unit_gbps=f32(hw.b_scat_epi_per_unit_gbps),
        b_epi_write_per_unit_gbps=f32(hw.b_epi_write_per_unit_gbps),
        l2_cache_mb=f32(hw.l2_cache_mb),
        t_head_us=f32(hw.t_head_us),
        t_tail_us=f32(hw.t_tail_us),
        t_handoff_us=f32(hw.t_handoff_us),
        weight_bytes=f32(hw.weight_bytes),
        act_in_bytes=f32(hw.act_in_bytes),
        out_bytes=f32(hw.out_bytes),
    )


def evaluate_f32(arch: str, wl: WorkloadSpec, c: SpatialCandidate,
                 scatter_mode: int, m_e: Sequence[int],
                 dedup_pulls: int, scatter_count: int,
                 precision_mode: str = "fp8xfp4",
                 reduce_warps: int = 0) -> Breakdown:
    """Binary32 step-by-step mirror of weave::evaluate_spatial_candidate."""
    hw = _hw_f32(arch, precision_mode)
    p_tc = f32(f32(c.p_tc_tflops) * f32(precision_tc_scale(precision_mode)))
    total = int(sum(m_e))
    h = f32(wl.hidden)
    i = f32(wl.intermediate)
    bm = f32(c.block_m)
    f_gemm = 0.0
    weight_bytes = 0.0
    for count in m_e:
        k = f32(_ceil_div(int(count), c.block_m))
        t = f32(f32(f32(f32(k * bm) * 6.0) * h) * i)
        f_gemm = f32(f_gemm + t)
        w = f32(f32(f32(f32(k * 3.0) * h) * i) * hw.weight_bytes)
        weight_bytes = f32(weight_bytes + w)
    activation = f32(f32(f32(total) * hw.act_in_bytes) * f32(h + i))
    v_mem = f32(weight_bytes + activation)
    t_mem = f32(v_mem / f32(hw.b_hbm_gbps * 1000.0))
    t_mma_f = f32(f_gemm / f32(p_tc * 1000000.0))
    t_mma = max(t_mem, t_mma_f)
    v_epi = f32(f32(total) * f32(f32(hw.act_in_bytes * i) + f32(hw.out_bytes * h)))
    total_epi_units = f32(f32(hw.num_sms_or_cus) * f32(c.comp_units))
    b_epi = min(hw.b_hbm_gbps,
                max(1.0, f32(total_epi_units * hw.b_epi_write_per_unit_gbps)))
    t_epi = f32(v_epi / f32(b_epi * 1000.0))
    t_comp = max(t_mma, t_epi) if arch == "b300" else f32(t_mma + t_epi)

    rw = min(int(reduce_warps), MAX_REDUCE_WARPS)
    eff_comm_per_sm = c.comm_units if c.comm_units else c.comp_units
    scat_comm_per_sm = eff_comm_per_sm
    if rw > 0 and scatter_mode != 0:
        scat_comm_per_sm = max(1, eff_comm_per_sm - rw)
    gpu_comm = f32(hw.num_sms_or_cus * eff_comm_per_sm)
    gpu_scat_comm = f32(hw.num_sms_or_cus * scat_comm_per_sm)
    gpu_epi = f32(hw.num_sms_or_cus * c.comp_units)
    b_pull = min(hw.b_link_gbps,
                 max(1.0, f32(gpu_comm * hw.b_pull_per_unit_gbps)))
    if scatter_mode == 0:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, f32(gpu_epi * hw.b_scat_epi_per_unit_gbps)))
    else:
        b_scat = min(hw.b_link_gbps,
                     max(1.0, f32(gpu_scat_comm * hw.b_scat_comm_per_unit_gbps)))
    t_pull = 0.0 if dedup_pulls == 0 else f32(
        f32(f32(f32(dedup_pulls) * h) * hw.act_in_bytes) / f32(b_pull * 1000.0))
    t_scat = 0.0 if scatter_count == 0 else f32(
        f32(f32(f32(scatter_count) * h) * hw.out_bytes) / f32(b_scat * 1000.0))
    if scatter_mode == 0:
        t_mid = max(t_pull, f32(t_comp + t_scat))
    else:
        t_mid = max(f32(t_pull + t_scat), f32(t_comp + hw.t_handoff_us))
    b_tail = min(hw.b_hbm_gbps,
                 max(1.0, f32(f32(f32(hw.num_sms_or_cus) *
                                  f32(hw.max_units_per_sm)) * REDUCE_GBPS_PER_WARP)))
    reduce_bytes = f32(f32(f32(f32(total) * h) * hw.out_bytes) * 2.0)
    t_tail_reduce = f32(reduce_bytes / f32(b_tail * 1000.0))
    if rw > 0:
        b_mid_used = (f32(v_mem / f32(t_comp * 1000.0)) if t_comp > 0.0
                      else hw.b_hbm_gbps)
        b_spare = max(0.0, f32(hw.b_hbm_gbps - b_mid_used))
        b_reduce = min(b_spare, f32(f32(f32(hw.num_sms_or_cus) * f32(rw)) *
                                    REDUCE_GBPS_PER_WARP))
        overlapped = f32(f32(b_reduce * 1000.0) * t_mid)
        t_tail_reduce = f32(max(0.0, f32(reduce_bytes - overlapped)) /
                            f32(b_tail * 1000.0))
    t_total = f32(f32(hw.t_head_us + t_mid) + t_tail_reduce)
    return Breakdown(c.r_mask, scatter_mode, total,
                     choose_v_head(hw, c, dedup_pulls), f_gemm, v_mem, v_epi,
                     t_mma, t_epi, t_comp, t_pull, t_scat, t_tail_reduce,
                     t_mid, t_total)


def select_policy_f32(arch: str, wl: WorkloadSpec, m_e: Sequence[int],
                      dedup_pulls: int, scatter_count: int,
                      precision_mode: str = "fp8xfp4",
                      reduce_warps: int = 0) -> Breakdown:
    """Same search order / strict-< tie rule as weave::cta0_select_policy."""
    best: Breakdown | None = None
    for c in CANDIDATES[arch]:
        for s in (0, 1):
            br = evaluate_f32(arch, wl, c, s, m_e, dedup_pulls, scatter_count,
                              precision_mode, reduce_warps)
            if best is None or br.t_total_us < best.t_total_us:
                best = br
    assert best is not None
    return best


def online_policy(arch: str, wl: WorkloadSpec, m_e: Sequence[int],
                  dedup_pulls: int, scatter_count: int,
                  precision_mode: str = "fp8xfp4",
                  reduce_warps: int = 0,
                  adapt_bm: int = 0,
                  v_head_override: int | None = None,
                  ready_group_size: int = 0,
                  dedup_mode: str | int = "auto",
                  unique_tokens: int = 0) -> Tuple[Breakdown, OnlineRoutingPolicy]:
    """Bit-exact mirror of weave::cta0_online_policy on the winning breakdown.

    v_head_override=None means auto (WEAVE_V_HEAD=0); an integer pins it.
    dedup_mode: 'auto' (unset), 0 (off) or 1 (on).  Explicit pins always
    override the routing-adaptive rules; the adaptive rules only run when
    adapt_bm=1 (the existing online-tier master switch).
    """
    hw = _hw_f32(arch, precision_mode)
    cal = HARDWARE_CALIBRATION[arch]
    rw = min(int(reduce_warps), MAX_REDUCE_WARPS)
    if v_head_override is not None:
        # C++ kOverrideVHead bypasses the search (make_override_decision):
        # the default r_mask candidate (items[0] on every arch) with its auto
        # scatter path, exactly as weave::cta0_select_policy does.
        c = CANDIDATES[arch][0]
        s = 0 if c.comm_units == 0 else 1
        br = evaluate_f32(arch, wl, c, s, m_e, dedup_pulls, scatter_count,
                          precision_mode, rw)
    else:
        br = select_policy_f32(arch, wl, m_e, dedup_pulls, scatter_count,
                               precision_mode, rw)
    total = int(sum(m_e))
    active = sum(1 for n in m_e if n > 0)
    max_n = max(m_e) if len(m_e) else 0
    mean_n = f32(f32(total) / f32(active)) if active > 0 else 0.0
    if unique_tokens > 0:
        mu = f32(f32(total) / f32(unique_tokens))
    elif 0 < dedup_pulls < total:
        mu = f32(f32(total) / f32(dedup_pulls))
    else:
        mu = 1.0
    denom = f32(f32(br.t_comp_us * 1000.0) * hw.b_hbm_gbps)
    util = f32(br.v_mem_bytes / denom) if br.t_comp_us > 0.0 else 0.0
    tail = f32(br.t_tail_reduce_us / br.t_total_us) if br.t_total_us > 0.0 else 0.0

    adaptive = 1 if adapt_bm != 0 else 0
    hbm_ok = 1 if util <= cal.hbm_headroom_thresh else 0
    tail_ok = 1 if tail >= cal.tail_ratio_thresh else 0
    if not adaptive or rw == 0:
        active_rw = rw
    else:
        active_rw = rw if (hbm_ok and tail_ok) else 0

    vh = v_head_override if v_head_override is not None else br.v_head
    if adaptive and v_head_override is None and active > 0:
        hot = max_n * active >= 2 * total
        if hot:
            capped = min(vh, cal.v_head_hot_budget)
            if capped == 0:
                capped = 1
            vh = _align(capped, 32)

    g = int(ready_group_size)
    if g == 0:
        if active == 0:
            g = 1
        else:
            g = 1
            if total >= active * 32:
                g = 2
            if total >= active * 128:
                g = 4
            if total >= active * 256:
                g = 8
            if active * 4 >= wl.local_experts * 3 and g < 8:
                g *= 2

    if dedup_mode == 0:
        en = 0
    elif dedup_mode == 1:
        en = 1
    else:
        en = 1 if mu >= cal.mu_dedup_thresh else 0

    tier = select_online_expert_bm(int(mean_n), br_block_m(arch, br), cal.bm_pad_slack)
    return br, OnlineRoutingPolicy(
        active_rw, vh, g, en, tier, br_block_m(arch, br), active, max_n, total,
        adaptive, hbm_ok, tail_ok, mean_n, mu, util, tail)


def br_block_m(arch: str, br: Breakdown) -> int:
    """The winning candidate's BLOCK_M envelope (selected_block_m in C++)."""
    for c in CANDIDATES[arch]:
        if c.r_mask == br.r_mask:
            return c.block_m
    raise AssertionError(f"unknown r_mask {br.r_mask} for {arch}")



