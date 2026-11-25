"""Unified Weave knob space: Spatial / Temporal / Implementation.

The public tunable surface consists of 14 knobs in 3 categories
(Spatial 5, Temporal 3, Implementation 6). This module is the single
Python-side schema for the parameter reference and pybind surface.
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
    compact_values_en: str = ""
    compact_values_zh: str = ""
    compact_mech_en: str = ""
    compact_mech_zh: str = ""
    gain_en: str = ""
    gain_zh: str = ""


CATEGORIES: Tuple[str, ...] = ("Spatial", "Temporal", "Implementation")

EXPECTED_COUNTS: Dict[str, int] = {"Spatial": 5, "Temporal": 3, "Implementation": 6}

# Internal compile-time diagnostic macros kept at 0 by default (never public knobs).
RETIRED_INTERNAL: Tuple[str, ...] = (
    "WEAVE_REDUCE_WARPS",
    "WEAVE_COMBINE_SYNC_CHUNK",
    "WEAVE_REG_BALANCE",
    "WEAVE_LOCAL_SCATTER_DEDUP",
    "WEAVE_SM0_STREAM_SCHED",
    "WEAVE_READY_QUEUE",
)

# Silent backward-compatible aliases accepted by host parsers.
LEGACY_ALIASES: Dict[str, str] = {
    "WEAVE_BLOCK_M_L1": "WEAVE_GEMM1_CONFIG",
    "WEAVE_BLOCK_M_L2": "WEAVE_GEMM2_CONFIG",
}

INTERNAL_COMPAT_ENV: Tuple[str, ...] = (
    "WEAVE_ADAPT_GEMM_CONFIG",
    "WEAVE_ADAPT_BM",
)


KNOBS: Tuple[Knob, ...] = (
    # ------------------------------------------------------------------ Spatial (5)
    Knob(
        "WEAVE_DISPATCH_ACTIVE_SMS", "Spatial",
        "`0 = auto` (all SMs/CUs participate in remote token pull); `D = 1..N`: assigns `D` SMs/CUs to remote token pulling "
        "while the remaining `N - D` SMs/CUs immediately enter weight prefetch and GEMM1. Independent of `WEAVE_SCATTER_ACTIVE_SMS` (`D != K`).",
        "0",
        "Cross-SM/CU dispatch partitioning: controls how many SMs/CUs pull remote tokens so non-dispatch SMs start compute without waiting.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_DISPATCH_ACTIVE_SMS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "static_cast<uint32_t>(WEAVE_DISPATCH_ACTIVE_SMS) > kMaxPullSms",
        compact_values_en="`0=auto`, `D=1..N` (`D != K`)",
        compact_values_zh="`0=auto`，`D=1..N`（与 `K` 独立）",
        compact_mech_en="Cross-SM/CU active Dispatch subset; remaining SMs/CUs prefetch and start GEMM1 immediately",
        compact_mech_zh="跨 SM/CU 独立控制参与远端拉取的 SM/CU 数，其余单元立即预取并启动计算",
        gain_en="**1.41x** MI308X EP16 (`1431->1015 us`); **1.007x** SM103 (`294.9->292.9 us`)",
        gain_zh="**1.41x** MI308X EP16（`1431->1015 us`）；**1.007x** SM103（`294.9->292.9 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_ACTIVE_SMS", "Spatial",
        "`0 = auto` (all SMs/CUs may drain async scatter); `K = 1..N`: assigns `K` SMs/CUs to drain the cross-GPU scatter queue "
        "(independent of `WEAVE_DISPATCH_ACTIVE_SMS`), while tail Combine reduction still uses all SMs/CUs.",
        "0",
        "Cross-SM/CU scatter partitioning: bounds async cross-GPU FC2 scatter draining to `K` SMs/CUs independently of Dispatch.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_SCATTER_ACTIVE_SMS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_D7_ACTIVE_SMS WEAVE_SCATTER_ACTIVE_SMS",
        compact_values_en="`0=auto`, `K=1..N` (`K != D`)",
        compact_values_zh="`0=auto`，`K=1..N`（与 `D` 独立）",
        compact_mech_en="Cross-SM/CU active Scatter subset; tail Combine still uses all SMs/CUs",
        compact_mech_zh="跨 SM/CU 独立控制参与异步 Scatter 发出的 SM/CU 数，尾部 Combine 仍用全卡单元",
        gain_en="**1.015x** MI308X EP16 (`1030->1015 us`); **1.013x** H20 `S=4K` (`5299->5233 us`)",
        gain_zh="**1.015x** MI308X EP16（`1030->1015 us`）；**1.013x** H20 `S=4K`（`5299->5233 us`）",
    ),
    Knob(
        "WEAVE_DISPATCH_WARPS", "Spatial",
        "`0 = auto` (architecture default: 4 warps on SM103, 2 or 4 warps on H20); `4` or `8` communication warps per CTA.",
        "0",
        "Intra-SM dispatch warp ratio: sets the per-CTA warp budget assigned to remote token and scale pulling.",
        "SM103, H20",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_DISPATCH_WARPS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_DISPATCH_WARPS == kNumDispatchWarps",
        compact_values_en="`0=auto`, `4`, `8`",
        compact_values_zh="`0=auto`，`4`，`8`",
        compact_mech_en="Intra-SM communication warps per CTA assigned to remote token/scale pulling",
        compact_mech_zh="单 SM 内分配给远端 token 与缩放因子拉取的通信 warp 数",
        gain_en="**1.17x** H20 `S=16K` (`8` vs `4`: `1644->1403 us`)",
        gain_zh="**1.17x** H20 `S=16K`（`8` vs `4`：`1644->1403 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_WARPS", "Spatial",
        "`0 = auto`; `4` or `8` communication warps/wavefronts per CTA/CU dedicated to asynchronous FC2 scatter.",
        "0",
        "Intra-SM/CU scatter warp ratio: controls the per-CTA/CU communication warp/wavefront budget for async FC2 cross-GPU writes.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_SCATTER_WARPS\", \"DG_TUNE_D7_WARPS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_TUNE_D7_SCATTER_WARPS WEAVE_SCATTER_WARPS",
        compact_values_en="`0=auto`, `4`, `8`",
        compact_values_zh="`0=auto`，`4`，`8`",
        compact_mech_en="Intra-SM/CU communication warps/waves dedicated to async FC2 scatter",
        compact_mech_zh="单 SM/CU 内分配给异步 FC2 Scatter 跨卡写出的通信 warp/wave 数",
        gain_en="**1.17x** H20 `S=16K` (`1644->1405 us`); **1.04x** MI308X EP16 (`1015->976 us`)",
        gain_zh="**1.17x** H20 `S=16K`（`1644->1405 us`）；**1.04x** MI308X EP16（`1015->976 us`）",
    ),
    Knob(
        "WEAVE_CU_WAVE_SPLIT", "Spatial",
        "**MI308X**: `0 = auto`, `1 = 0-8` (8 compute waves/CU), `2 = 4-4` (4 comm + 4 compute waves co-resident per CU), `3 = 8-0` (8 comm waves/CU). "
        "**H20**: `0 = auto` (pipelined GEMM1/GEMM2 expert-lead task stream) or integer expert-group width `E`.",
        "0",
        "Per-CU wavefront split `(comm_waves - compute_waves)` out of 8 Wave64s on MI308X (including heterogeneous `(0,8)` + `(4,4)` CU mixing), and GEMM1/GEMM2 expert-wave interleaving on H20.",
        "MI308X, H20",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_CU_WAVE_SPLIT\"",
        "deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip",
        "(WEAVE_CU_WAVE_SPLIT == 3)",
        compact_values_en="MI308X: `auto, 0-8, 4-4, 8-0`; H20: `auto` or `E`",
        compact_values_zh="MI308X：`auto, 0-8, 4-4, 8-0`；H20：`auto` 或 `E`",
        compact_mech_en="MI308X per-CU 8-Wave64 comm/compute split; H20 GEMM1/GEMM2 expert-stream interleaving",
        compact_mech_zh="MI308X 单 CU 8-Wave64 通算配比；H20 GEMM1/GEMM2 专家任务流交错粒度",
        gain_en="**2.15x** MI308X EP16 (`4-4` vs `0-8`: `2185->1015 us`); **1.053x** H20 `S=8K` (`926->880 us`)",
        gain_zh="**2.15x** MI308X EP16（`4-4` vs `0-8`：`2185->1015 us`）；**1.053x** H20 `S=8K`（`926->880 us`）",
    ),

    # ----------------------------------------------------------------- Temporal (3)
    Knob(
        "WEAVE_V_HEAD", "Temporal",
        "`0 = auto` (cost-model head quota); `N = 1..65536`: explicit opening pull quota completed by helper compute/epilogue warps before switching into GEMM1.",
        "0",
        "Head temporal role transition: lends idle compute/epilogue warps to opening token pull for a bounded quota `V_head` to shrink first-tile GEMM1 startup latency.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_V_HEAD\", \"DG_V_HEAD\"",
        "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "kCompiledVHeadDefault = WEAVE_V_HEAD",
        compact_values_en="`0=auto`, `N=1..65536`",
        compact_values_zh="`0=auto`，`N=1..65536`",
        compact_mech_en="Opening pull quota for helper warps before transitioning into GEMM1",
        compact_mech_zh="层首阶段计算/尾声 warp 临时协助远端拉取的任务配额，缩短首块 GEMM1 启动等待",
        gain_en="**1.088x** H20 `S=8K` (`941.5->865.3 us`); **1.043x** MI308X (`1015->973 us`)",
        gain_zh="**1.088x** H20 `S=8K`（`941.5->865.3 us`）；**1.043x** MI308X（`1015->973 us`）",
    ),
    Knob(
        "WEAVE_READY_GROUP_SIZE", "Temporal",
        "`0 = full-expert readiness`; `N in {1, 2, 4, 8, 16}`: publishes token readiness every `N` `BLOCK_M` tiles so GEMM1 starts before the entire expert finishes pulling.",
        "0",
        "Head-to-GEMM1 handoff granularity: fine-grained `N`-tile arrival publication overlapping ongoing remote pull with GEMM1 on hot experts.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_READY_GROUP_SIZE\", \"DG_TUNE_D1_G\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D1_G WEAVE_READY_GROUP_SIZE",
        compact_values_en="`0=full expert`, `1, 2, 4, 8, 16`",
        compact_values_zh="`0=整专家`，`1, 2, 4, 8, 16`",
        compact_mech_en="Publishes readiness every `N` `BLOCK_M` tiles so GEMM1 starts before full expert arrives",
        compact_mech_zh="按 `N` 个 `BLOCK_M` tile 粒度发布到达就绪信号，提前启动热点专家 GEMM1",
        gain_en="**1.030x** MI308X EP16 (`1015->985 us`); **1.009x** H20 `S=4K` (`5337->5289 us`)",
        gain_zh="**1.030x** MI308X EP16（`1015->985 us`）；**1.009x** H20 `S=4K`（`5337->5289 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_OWNER", "Temporal",
        "`0 = epilogue_only` (epilogue warps issue remote FC2 writes directly); `1 = comm_only` (epilogue stages locally while freed communication warps perform all cross-GPU scatter writes).",
        "0",
        "Middle temporal role transition: switches freed communication warps from Dispatch pull to asynchronous FC2 scatter so epilogue warps immediately turn around the next GEMM tile.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_SCATTER_OWNER\", {\"epilogue_only\", \"comm_only\"}",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "kScatterCommOwned = (WEAVE_SCATTER_OWNER == 1)",
        compact_values_en="`0=epilogue_only`, `1=comm_only`",
        compact_values_zh="`0=epilogue_only`，`1=comm_only`",
        compact_mech_en="Assigns FC2 cross-GPU scatter either to epilogue warps directly or to freed comm warps",
        compact_mech_zh="中部 FC2 跨卡写出角色归属（尾声直写 vs 释放后的通信 warp 后台异步写出）",
        gain_en="**1.14x** H20 `S=16K` (`1244->1094 us`); **1.034x** MI308X (`1015->981 us`)",
        gain_zh="**1.14x** H20 `S=16K`（`1244->1094 us`）；**1.034x** MI308X（`1015->981 us`）",
    ),

    # -------------------------------------------------------------- Implementation (6)
    Knob(
        "WEAVE_PRECISION_MODE", "Implementation",
        "`0 = fp8xfp4`; `1 = fp4xfp4` (`nvfp4`); `2 = fp8xfp8` (`fp8`); `3 = bf16`; `auto` = backend default.",
        "0",
        "Selects the numeric precision and Tensor/Matrix Core instruction format across GEMM1 and GEMM2.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "std::getenv(\"WEAVE_PRECISION_MODE\")",
        "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "kCompiledPrecisionMode = WEAVE_PRECISION_MODE",
        compact_values_en="`0=fp8xfp4`, `1=fp4xfp4`, `2=fp8`, `3=bf16`",
        compact_values_zh="`0=fp8xfp4`，`1=fp4xfp4`，`2=fp8`，`3=bf16`",
        compact_mech_en="Tensor/Matrix Core numeric format across GEMM1 and GEMM2",
        compact_mech_zh="选择 GEMM1/GEMM2 的张量核数值精度与指令格式",
        gain_en="**1.59x** SM103 `S=16K` (`1038->654 us`); **1.010x** H20 (`5276->5224 us`)",
        gain_zh="**1.59x** SM103 `S=16K`（`1038->654 us`）；**1.010x** H20（`5276->5224 us`）",
    ),
    Knob(
        "WEAVE_GEMM1_CONFIG", "Implementation",
        "`0 = auto` (shape-adaptive tuple selected from `(M_expected, N1, K1)`); explicit `BM` in `{16, 32, 64, 80, 96, 128, 192, 240, 256}`, or full tuple `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]` on SM103 (`STORE_BM <= BM`, `BK in {128, 256}`, mainloop `STAGES in 2..16`, multi-stage epilogue register pipeline `EPI in 1..4`).",
        "0",
        "Independent GEMM1 (FC1 + SwiGLU) configuration: locks tile sizes `(BM, BN, BK)`, mainloop pipeline stages, epilogue store height, and multi-stage TMEM-to-register epilogue pipelining.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "gemm_phase_config_env(\"WEAVE_GEMM1_CONFIG\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_BLOCK_M_L1 == 0",
        compact_values_en="`0=auto` or `BM[:STORE_BM[:BK[:ST[:EPI]]]]` (`EPI=1..4`)",
        compact_values_zh="`0=auto` 或 `BM[:STORE_BM[:BK[:ST[:EPI]]]]`（`EPI=1..4`）",
        compact_mech_en="Independent FC1 tile & multi-stage mainloop/epilogue configuration",
        compact_mech_zh="独立配置 GEMM1（FC1）分块尺寸、主循环流水级数与多级尾声寄存器流水",
        gain_en="**1.51x** H20 `S=16K` (`BM=128`: `1651->1091 us`); **1.047x** SM103 (`96,16`: `766->731 us`)",
        gain_zh="**1.51x** H20 `S=16K`（`BM=128`：`1651->1091 us`）；**1.047x** SM103（`96,16`：`766->731 us`）",
    ),
    Knob(
        "WEAVE_GEMM2_CONFIG", "Implementation",
        "`0 = auto / follow GEMM1`; accepts the same `BM` or `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]` (`EPI in 1..4`) grammar as `WEAVE_GEMM1_CONFIG`, allowing asymmetric tile shapes and store heights tailored to `(N2, K2) = (H, I)`.",
        "0",
        "Independent GEMM2 (FC2) configuration decoupled from GEMM1 so each GEMM matches its own aspect ratio, L2 weight footprint, and epilogue demand.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "gemm_phase_config_env(\"WEAVE_GEMM2_CONFIG\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_BLOCK_M_L2 == 0",
        compact_values_en="`0=auto` or `BM[:STORE_BM[:BK[:ST[:EPI]]]]` (decoupled)",
        compact_values_zh="`0=auto` 或与 GEMM1 独立解耦的 `BM[...]` 元组",
        compact_mech_en="Independent FC2 tile & multi-stage configuration decoupled from GEMM1",
        compact_mech_zh="与 GEMM1 解耦的 GEMM2（FC2）独立分块与多级尾声流水配置",
        gain_en="**1.036x** MI308X EP16 (`64` vs `32`: `1051->1015 us`); **1.025x** H20 `32/64` (`2234->2178 us`)",
        gain_zh="**1.036x** MI308X EP16（`64` vs `32`：`1051->1015 us`）；**1.025x** H20 `32/64`（`2234->2178 us`）",
    ),
    Knob(
        "WEAVE_REG_PULL_BYTES", "Implementation",
        "`0 = SMEM/LDS staged pull`; `1024, 2048, 4096` bytes per pull chunk on 32-lane NVIDIA warps; `8192` bytes per pull chunk on 64-lane MI308X wavefronts.",
        "0",
        "Independent Dispatch operator parameter: per-warp/wavefront register-direct remote pull width, bypassing shared memory/LDS to preserve capacity for GEMM stages.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_REG_PULL_BYTES\", \"DG_TUNE_D2_REG_PULL_BYTES\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D2_REG_PULL_BYTES WEAVE_REG_PULL_BYTES",
        compact_values_en="`0=SMEM`, `1024/2048/4096` (NV), `8192` (MI308X)",
        compact_values_zh="`0=SMEM`，NVIDIA `1024/2048/4096`，MI308X `8192`",
        compact_mech_en="Dispatch register-direct pull chunk width per warp/wavefront (preserves SMEM/LDS for GEMM)",
        compact_mech_zh="Dispatch 算子单轮寄存器直传拉取字节宽度，省出片上内存供更深 GEMM 流水使用",
        gain_en="**1.19x** MI308X EP16 (`8192` vs `1024`: `1212->1015 us`); **1.024x** SM103 (`618.5->604.1 us`)",
        gain_zh="**1.19x** MI308X EP16（`8192` vs `1024`：`1212->1015 us`）；**1.024x** SM103（`618.5->604.1 us`）",
    ),
    Knob(
        "WEAVE_DEDUP_MODE", "Implementation",
        "`0 = off` (per-slot pull); `1 = on` (pulls each unique `(src_rank, token)` once across the interconnect and fans out locally across all hit experts on the GPU).",
        "unset (auto by routing multiplicity)",
        "Independent Dispatch operator deduplication: eliminates duplicate cross-GPU token transfers when multiple local experts select the same remote token.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "get_env<int>(\"WEAVE_DEDUP_MODE\", -1)",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D5_DEDUP WEAVE_DEDUP_MODE",
        compact_values_en="`0=off`, `1=on`",
        compact_values_zh="`0=off`，`1=on`",
        compact_mech_en="Dispatch remote-pull deduplication: pulls unique `(src_rank, token)` once and fans out locally",
        compact_mech_zh="Dispatch 跨卡拉取去重：同一远端 token 命中多个本地专家时只拉一次并片内扇出",
        gain_en="**1.067x** H20 `S=16K` (`1644->1540 us`); **1.023x** MI308X EP16 (`1038->1015 us`)",
        gain_zh="**1.067x** H20 `S=16K`（`1644->1540 us`）；**1.023x** MI308X EP16（`1038->1015 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_STAGING", "Implementation",
        "`0 = reg` (multi-stage register-direct scatter); `1 = smem` (on-chip shared-memory staging streamed out by freed communication warps); `2 = hbm` (on-device HBM scatter buffer accumulating duplicate local expert hits before sending once).",
        "0",
        "Independent Scatter operator staging path (`reg` multi-stage registers, `smem` shared-memory staging, or `hbm` local pre-accumulation buffer).",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_SCATTER_STAGING\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_SCATTER_STAGING != 2 or DG_D7_QUEUE_SCATTER != 0",
        compact_values_en="`0=reg`, `1=smem`, `2=hbm`",
        compact_values_zh="`0=reg`，`1=smem`，`2=hbm`",
        compact_mech_en="Scatter staging path: multi-stage register-direct (`reg`), shared memory (`smem`), or HBM buffer (`hbm`)",
        compact_mech_zh="Scatter 算子数据通路：多级寄存器直写（`reg`）、共享内存异步写出（`smem`）或显存缓冲预合并（`hbm`）",
        gain_en="**1.14x** H20 `S=16K` (`smem` vs `reg`: `1244->1094 us`); **1.010x** MI308X (`1025->1015 us`)",
        gain_zh="**1.14x** H20 `S=16K`（`smem` vs `reg`：`1244->1094 us`）；**1.010x** MI308X（`1025->1015 us`）",
    ),
)


def _assert_contract() -> None:
    """Import-time check of the 14-knob / 3-category public contract."""
    names = [k.name for k in KNOBS]
    if len(KNOBS) != 14:
        raise AssertionError(f"public knob surface must have 14 entries, got {len(KNOBS)}")
    if len(set(names)) != len(names):
        raise AssertionError("duplicate public knob names")
    for category, expected in EXPECTED_COUNTS.items():
        got = sum(1 for k in KNOBS if k.category == category)
        if got != expected:
            raise AssertionError(f"category {category}: expected {expected} knobs, got {got}")
    if {k.category for k in KNOBS} != set(CATEGORIES):
        raise AssertionError("unexpected knob category")
    order = [CATEGORIES.index(k.category) for k in KNOBS]
    if order != sorted(order):
        raise AssertionError("knobs must be listed Spatial, then Temporal, then Implementation")
    if set(RETIRED_INTERNAL) & set(names):
        raise AssertionError("a retired internal knob is still public")
    if len(set(RETIRED_INTERNAL)) != len(RETIRED_INTERNAL) or len(RETIRED_INTERNAL) != 6:
        raise AssertionError("RETIRED_INTERNAL must list exactly 6 distinct names")
    if set(LEGACY_ALIASES) & set(names):
        raise AssertionError("a legacy alias is also a public name")
    if not set(LEGACY_ALIASES.values()) <= set(names):
        raise AssertionError("every legacy alias must map to a public knob")
    if set(INTERNAL_COMPAT_ENV) & set(names):
        raise AssertionError("an internal compat env name is also public")
    if set(INTERNAL_COMPAT_ENV) & (set(LEGACY_ALIASES) | set(RETIRED_INTERNAL)):
        raise AssertionError("internal compat env names must be disjoint from aliases / retired pins")
    for k in KNOBS:
        if not (k.gain_en and k.gain_zh and k.compact_values_en and k.compact_values_zh):
            raise AssertionError(f"{k.name}: missing compact description or measured gain")


_assert_contract()

REQUIRED = tuple(k.name for k in KNOBS)


def by_name() -> Dict[str, Knob]:
    return {k.name: k for k in KNOBS}


def as_dicts() -> List[dict]:
    return [asdict(k) for k in KNOBS]


def legacy_aliases_of(name: str) -> Tuple[str, ...]:
    """Backward-compatible environment spellings accepted for public knob `name`."""
    return tuple(old for old, new in LEGACY_ALIASES.items() if new == name)


def canonical_name(name: str) -> str:
    """Map a backward-compatible alias to its public name; other names are returned unchanged."""
    return LEGACY_ALIASES.get(name, name)
