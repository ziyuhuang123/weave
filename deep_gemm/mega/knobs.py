"""Unified Weave knob space: Spatial / Temporal / Implementation.

This module is the single Python-side schema for the public tunable surface.
`tests/verify_tunable_params_doc.py` resolves every citation produced from these
entries against the working tree, and the pybind module `_C.weave_knobs`
exposes the same names and compiled defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Knob:
    name: str
    category: str
    values: str
    default: str
    mechanism: str
    platform: str
    host_file: str
    host_needle: str
    kernel_file: str
    kernel_needle: str


KNOBS: Tuple[Knob, ...] = (
    # ------------------------------------------------------------------ Spatial
    # Strictly SM-level & Warp/Wave-level hardware resource partitioning ratios
    Knob("WEAVE_DISPATCH_ACTIVE_SMS", "Spatial",
         "`0 = auto` (all SMs run dispatch pull); `D = 1..num_sms`: only D SMs pull remote tokens",
         "0", "SM-level dispatch allocation: assigns D SMs to remote token pulling while remaining SMs enter prefetch/GEMM immediately.",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_DISPATCH_ACTIVE_SMS\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_DISPATCH_ACTIVE_SMS"),
    Knob("WEAVE_SCATTER_ACTIVE_SMS", "Spatial",
         "`0 = auto` (all SMs/CUs participate); `K = 1..num_sms`: only K SMs drain the scatter queue",
         "0", "SM-level scatter allocation (independent from dispatch SMs): bounds async NVLink/xGMI scatter draining to K SMs while Combine uses all SMs.",
         "SM103, H20, MI308X",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "\"WEAVE_SCATTER_ACTIVE_SMS\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_SCATTER_ACTIVE_SMS"),
    Knob("WEAVE_DISPATCH_WARPS", "Spatial",
         "`0 = auto` (4 warps); `4`; `8`",
         "0", "Per-SM dispatch warp ratio: sets the CTA communication warp count dedicated to remote token/SF pulling (`4D+4N+8E` vs `8D+4N+8E`).",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_DISPATCH_WARPS\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_DISPATCH_WARPS == kNumDispatchWarps"),
    Knob("WEAVE_SCATTER_WARPS", "Spatial",
         "`0 = automatic`; `4`; `8`",
         "0", "Per-SM scatter warp ratio: sets the CTA communication warp budget dedicated to async FC2 scatter (independent from dispatch pull parameters).",
         "SM103 fp8xfp4",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "#define WEAVE_SCATTER_WARPS",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_SCATTER_WARPS"),
    Knob("WEAVE_CU_WAVE_SPLIT", "Spatial",
         "`0 = auto` (cost-model selection); `1 = 0-8` (0 comm + 8 compute waves); `2 = 4-4` (4 comm + 4 compute waves); `3 = 8-0` (8 comm + 0 compute waves on dedicated comm CUs; names or codes accepted)",
         "0", "Per-CU wavefront ratio on MI308X: selects `(comm_waves - compute_waves)` across `0-8`, `4-4`, and `8-0` out of 8 waves per CU.",
         "MI308X EP8/EP16",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_CU_WAVE_SPLIT\"",
         "deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip",
         "WEAVE_CU_WAVE_SPLIT"),
    Knob("WEAVE_REG_BALANCE", "Spatial",
         "`0 = default (96/72/168)`; `1 = rebalanced (64/56/192) at E/rank > 64`",
         "0", "Per-SM register file ratio across warp roles: shifts spare dispatch/non-epilogue registers to epilogue warps (`64/56/192` vs `96/72/168`).",
         "SM103",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_REG_BALANCE\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_REG_BALANCE != 0"),

    # ----------------------------------------------------------------- Temporal
    # Head / Middle / Tail bubble elimination, task-queue dispatch & sync granularity
    Knob("WEAVE_V_HEAD", "Temporal",
         "`0 = automatic`; `N = head-pull task quota (tokens) before helping warps exit back to GEMM`",
         "automatic", "Head schedule: all warps help pull by default at layer start; controls how many pull tasks (tokens) helping non-comm warps complete before finishing and switching to GEMM.",
         "SM103, H20, MI308X",
         "deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh",
         "#define WEAVE_V_HEAD 0xffffffffu",
         "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
         "kCompiledVHeadDefault = WEAVE_V_HEAD"),
    Knob("WEAVE_READY_GROUP_SIZE", "Temporal",
         "`0 = full-expert readiness`; `N = group-granularity readiness publication (in BLOCK_M tiles)`",
         "0", "Head schedule: fine-grained arrival readiness group size so GEMM1 starts as soon as N tiles arrive without waiting for full experts.",
         "SM103 fp8xfp4",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "#define WEAVE_READY_GROUP_SIZE",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_READY_GROUP_SIZE"),
    Knob("WEAVE_READY_QUEUE", "Temporal",
         "`0 = formula`: static heuristic formula over prefix-summed offsets; `1 = sm0_fifo`: SM 0 streaming scheduler dynamically pushes ready tiles across all regions into the global FIFO queue for CTA preemption (names or codes accepted)",
         "0", "Head/Middle task dispatch: selects static heuristic formula addressing vs SM 0 dedicated dynamic global FIFO work-stealing queue.",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_READY_QUEUE\", {\"formula\", \"sm0_fifo\"}",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_READY_QUEUE"),
    Knob("WEAVE_SCATTER_OWNER", "Temporal",
         "`0 = epilogue_only`: epilogue warps issue every remote FC2 store; `1 = comm_only`: epilogue warps only stage locally and communication warps own every remote store (names or codes accepted)",
         "0 (auto `comm_only` when a staged path or a scatter SM throttle is requested)",
         "Middle schedule: strict binary scatter role ownership (never mixes inline epilogue scatter with comm-warp scatter) to overlap FC2 scatter with ongoing GEMM.",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_SCATTER_OWNER\", {\"epilogue_only\", \"comm_only\"}",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "kScatterCommOwned = (WEAVE_SCATTER_OWNER == 1)"),
    Knob("WEAVE_COMBINE_SYNC_CHUNK", "Temporal",
         "`0 = global barrier before combine`; `16, 32, 64, 128 = chunked dedup combine sync granularity (tokens per chunk)`",
         "0", "Tail schedule: aggregates deduplicated sends into G-token chunks in local L2 and emits one cross-GPU release per chunk so all idle warps stream Combine reduction overlapping tail Scatter.",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_COMBINE_SYNC_CHUNK\"",
         "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh",
         "static_cast<uint32_t>(WEAVE_COMBINE_SYNC_CHUNK),"),

    # -------------------------------------------------------------- Implementation
    # Independent GEMM1 / GEMM2 / Dispatch / Scatter single-operator configurations:
    # M -> blockM synchronously locks (blockN, blockK, mainloop_stages, epi_M/N/K, epi_stages)
    Knob("WEAVE_PRECISION_MODE", "Implementation",
         "`0 = fp8xfp4`; `1 = fp4xfp4`; `2 = fp8xfp8`; `3 = bf16`",
         "0", "Single-op numeric precision and tensor-core instruction format across GEMM1/GEMM2.",
         "SM103 fp8/fp4/bf16; H20 fp8; MI308X fp8",
         "deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh",
         "#define WEAVE_PRECISION_MODE 0u",
         "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
         "kCompiledPrecisionMode = WEAVE_PRECISION_MODE"),
    Knob("WEAVE_ADAPT_BM", "Implementation",
         "`0 = fixed layer-wide profile`; `1 = per-expert M-bucketed offline profile lookup`",
         "0", "Runtime per-expert lookup from M_e -> blockM, synchronously binding the offline-calibrated (blockM/N/K, mainloop_stages, epi_M/N/K, epi_stages) group.",
         "SM103 fp8xfp4, H20, MI308X",
         "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp",
         "\"WEAVE_ADAPT_BM\"",
         "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh",
         "WEAVE_ADAPT_BM <= 1"),
    Knob("WEAVE_BLOCK_M_L1", "Implementation",
         "`0 = auto lookup from M`; `16, 32, 64, 80, 96, 128, 192, 240, 256`",
         "0", "Independent GEMM1 (FC1) profile selector: choosing blockM synchronously locks GEMM1's paired blockN/K, mainloop TMA stages, and epilogue M/N/K & TMEM/reg stages.",
         "SM103, H20, MI308X",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_BLOCK_M_L1\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_BLOCK_M_L1 == 0"),
    Knob("WEAVE_BLOCK_M_L2", "Implementation",
         "`0 = auto lookup from M`; `16, 32, 64, 80, 96, 128, 192, 240, 256`",
         "0", "Independent GEMM2 (FC2) profile selector (decoupled from GEMM1): choosing blockM synchronously locks GEMM2's paired blockN/K, mainloop stages, and epilogue M/N/K & stages.",
         "SM103, H20, MI308X",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_BLOCK_M_L2\"",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_BLOCK_M_L2 == 0"),
    Knob("WEAVE_REG_PULL_BYTES", "Implementation",
         "`0 = TMA SMEM staging`; `1024, 2048, 4096 = per-warp bytes per pull chunk` (= `32, 64, 128 B` or `8, 16, 32` 32-bit regs per thread)",
         "0", "Independent Dispatch operator parameter: per-warp (32-lane) chunk width for zero-SMEM register-direct remote pulling (`uint4 regs[2/4/8]` per thread, freeing SMEM for deeper GEMM mainloop stages).",
         "SM103 fp8xfp4",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "#define WEAVE_REG_PULL_BYTES",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_REG_PULL_BYTES"),
    Knob("WEAVE_DEDUP_MODE", "Implementation",
         "`0 = off`; `1 = unique (token, destination) buckets`; `2 = local pre-reduce plus 32-lane fanout`",
         "0", "Independent Dispatch operator deduplication of duplicate remote token pulls.",
         "SM103 fp8xfp4",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "#define WEAVE_DEDUP_MODE",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_DEDUP_MODE"),
    Knob("WEAVE_SCATTER_STAGING", "Implementation",
         "`0 = reg_2stage`: double-buffered register transfer; `1 = smem`: per-SM bank-conflict-free SMEM ring handoff; `2 = hbm_dedup`: L2-resident HBM buffer with bf16x2 atomic dedup (names or codes accepted)",
         "0 (auto `hbm_dedup` when `WEAVE_LOCAL_SCATTER_DEDUP=1`)",
         "Independent Scatter operator staging implementation (`reg_2stage`, XOR-swizzled `smem`, or `hbm_dedup`).",
         "SM103, H20",
         "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
         "\"WEAVE_SCATTER_STAGING\", {\"reg_2stage\", \"smem\", \"hbm_dedup\"}",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_SCATTER_STAGING != 2 or DG_D7_QUEUE_SCATTER != 0"),
    Knob("WEAVE_LOCAL_SCATTER_DEDUP", "Implementation",
         "`0 = direct per-hit scatter`; `1 = bf16x2 local merge of multi-expert hits before send`",
         "0", "Independent Scatter operator local deduplication merging duplicate destination hits on the same GPU before NVLink/xGMI send.",
         "SM103, H20, MI308X",
         "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
         "#define WEAVE_LOCAL_SCATTER_DEDUP",
         "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
         "WEAVE_LOCAL_SCATTER_DEDUP != 0"),
)

REQUIRED = tuple(k.name for k in KNOBS)


def by_name() -> Dict[str, Knob]:
    return {k.name: k for k in KNOBS}


def as_dicts() -> List[dict]:
    return [asdict(k) for k in KNOBS]
