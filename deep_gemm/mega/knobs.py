"""Unified Weave knob space: Spatial / Temporal / Implementation.

The public tunable surface is exactly 14 knobs in 3 categories
(Spatial 5, Temporal 3, Implementation 6).  This module is the single
Python-side schema for it.  `tests/verify_tunable_params_doc.py` resolves every
citation produced from these entries against the working tree, and the pybind
module `_C.weave_knobs` exposes the same names, categories and compiled
defaults.

Retired from the public surface (``RETIRED_INTERNAL``): the six names below are
no longer documented tunables.  Their host parsing and kernel code are kept as
internal compile-time macros (every one defaults to 0 through an ``#ifndef``
fallback) and as internal environment diagnostics, so existing launch scripts
and the measured winner presets keep working; they are simply not part of the
14-knob contract.

Accepted but not public (``INTERNAL_COMPAT_ENV``): ``WEAVE_ADAPT_GEMM_CONFIG``
and its legacy alias ``WEAVE_ADAPT_BM`` (plus ``DG_ADAPT_BM``) are still parsed
by the host so legacy scripts do not crash.  When ``WEAVE_GEMM1_CONFIG`` and
``WEAVE_GEMM2_CONFIG`` are 0 (unset / auto), host-level shape-adaptive GEMM
tuple selection is ALWAYS ON; a pinned GEMM1/GEMM2 tuple wins over the adaptive
pick.  Setting ``WEAVE_ADAPT_GEMM_CONFIG=0`` / ``WEAVE_ADAPT_BM=0`` forces the
fixed canonical tuple and is an internal legacy/diagnostic override, not a
public tunable.

Renamed (``LEGACY_ALIASES``): ``WEAVE_BLOCK_M_L1`` and ``WEAVE_BLOCK_M_L2`` are
transparent environment aliases of the public ``WEAVE_GEMM1_CONFIG`` and
``WEAVE_GEMM2_CONFIG``.
The host reads the new name first and falls back to the legacy name (the
MI308X build scripts, EP8 `build_one.sh` and EP16 `build_port.py`, still read only the legacy spelling).
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


# Category strings are compared verbatim by other code (generator, verifier,
# C++ schema check); do not rename them.
CATEGORIES: Tuple[str, ...] = ("Spatial", "Temporal", "Implementation")

# Required per-category size of the public surface (5 + 3 + 6 = 14).
EXPECTED_COUNTS: Dict[str, int] = {"Spatial": 5, "Temporal": 3, "Implementation": 6}

# Retired from the public surface, kept as internal compile-time macros
# (default 0) and internal environment diagnostics.
RETIRED_INTERNAL: Tuple[str, ...] = (
    "WEAVE_REDUCE_WARPS",
    "WEAVE_COMBINE_SYNC_CHUNK",
    "WEAVE_REG_BALANCE",
    "WEAVE_LOCAL_SCATTER_DEDUP",
    "WEAVE_SM0_STREAM_SCHED",
    "WEAVE_READY_QUEUE",
)

# Legacy spelling -> public name.  The host accepts the legacy name wherever it
# accepts the public one (the public name wins when both are set).
LEGACY_ALIASES: Dict[str, str] = {
    "WEAVE_BLOCK_M_L1": "WEAVE_GEMM1_CONFIG",
    "WEAVE_BLOCK_M_L2": "WEAVE_GEMM2_CONFIG",
}

# Accepted-but-not-public environment names (backward compatibility).  They are
# NOT part of the 14-knob surface: `WEAVE_ADAPT_GEMM_CONFIG` (and its legacy
# alias `WEAVE_ADAPT_BM`, plus `DG_ADAPT_BM`) is still parsed by the host; `0`
# selects the fixed canonical GEMM tuple (internal legacy/diagnostic override),
# anything else / unset keeps the always-on shape-adaptive selection.
INTERNAL_COMPAT_ENV: Tuple[str, ...] = (
    "WEAVE_ADAPT_GEMM_CONFIG",
    "WEAVE_ADAPT_BM",
)


KNOBS: Tuple[Knob, ...] = (
    # ------------------------------------------------------------------ Spatial (5)
    Knob(
        "WEAVE_DISPATCH_ACTIVE_SMS", "Spatial",
        "`0 = auto` (every SM/CU takes part in the dispatch pull); `D = 1..N`: only D SMs pull remote tokens "
        "(host range 0..1024; 0..80 on MI308X). D is independent of `WEAVE_SCATTER_ACTIVE_SMS` "
        "(D != K is supported). SM103 clamps a D above the pull-capable SM count to all of them; on MI308X EP16 "
        "D in [48, 80] is a no-op (only 48 CTAs are dispatch-capable)",
        "0",
        "SM-level dispatch allocation: assigns D SMs to remote token pulling while remaining SMs enter prefetch/GEMM immediately.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_DISPATCH_ACTIVE_SMS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "static_cast<uint32_t>(WEAVE_DISPATCH_ACTIVE_SMS) > kMaxPullSms",
        compact_values_en="`0=auto`, `D=1..N` (`D != K`)",
        compact_values_zh="`0=auto`，`D=1..N`（与 `K` 独立）",
        compact_mech_en="Active SMs/CUs pulling remote tokens; remaining SMs prefetch/compute immediately",
        compact_mech_zh="限制参与跨卡拉取的活跃 SM/CU 数，其余 SM 立即进入预取/计算",
        gain_en="**1.41x** MI308X EP16 (`auto` vs `2`: `1431->1015 us`); **1.007x** SM103 (`D=40`: `294.9->292.9 us`)",
        gain_zh="**1.41x** MI308X EP16（`1431->1015 us`）；**1.007x** SM103（`D=40`：`294.9->292.9 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_ACTIVE_SMS", "Spatial",
        "`0 = auto` (all SMs/CUs may drain the scatter queue); `K = 1..N`: only K SMs drain it (host range 0..1024; "
        "0..80 on MI308X where K = 80 is a no-op and the SMs only assist). K is independent of "
        "`WEAVE_DISPATCH_ACTIVE_SMS`. K > 0 needs comm-owned scatter (selected automatically; an explicit "
        "`WEAVE_SCATTER_OWNER=epilogue_only` is rejected) and shapes the HBM-queue executor "
        "(`WEAVE_SCATTER_STAGING=hbm_dedup`); with `smem` staging every SM's own comm warps drain, so K is inert",
        "0",
        "SM-level scatter allocation (independent from dispatch SMs): bounds async NVLink/xGMI scatter draining to K SMs while Combine uses all SMs.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_SCATTER_ACTIVE_SMS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_D7_ACTIVE_SMS WEAVE_SCATTER_ACTIVE_SMS",
        compact_values_en="`0=auto`, `K=1..N` (`K != D`)",
        compact_values_zh="`0=auto`，`K=1..N`（与 `D` 独立）",
        compact_mech_en="Active SMs/CUs draining async scatter queue; tail Combine still uses all SMs",
        compact_mech_zh="限制参与异步 Scatter 排空的 SM/CU 数，尾部 Combine 仍用全卡 SM",
        gain_en="**1.015x** MI308X EP16 (`1030->1015 us`); **1.013x** H20 `S=4K` (`K=39`: `5299->5233 us`)",
        gain_zh="**1.015x** MI308X EP16（`1030->1015 us`）；**1.013x** H20 `S=4K`（`K=39`：`5299->5233 us`）",
    ),
    Knob(
        "WEAVE_DISPATCH_WARPS", "Spatial",
        "`0 = arch default` (SM103: 4 warps; H20: engine default of 2 or 4 warps, re-laddered so that 4 math "
        "warpgroups still fit); `4`; `8`. No other count is accepted (host `DG_HOST_ASSERT` and kernel `#error`)",
        "0",
        "Per-SM dispatch warp ratio: sets the CTA communication warp count dedicated to remote token/SF pulling (`4D+4N+8E` vs `8D+4N+8E`).",
        "SM103, H20",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_DISPATCH_WARPS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_DISPATCH_WARPS == kNumDispatchWarps",
        compact_values_en="`0=default`, `4`, `8`",
        compact_values_zh="`0=默认`，`4`，`8`",
        compact_mech_en="Per-CTA communication warps assigned to remote token/scale pulling",
        compact_mech_zh="单 CTA 内专职跨卡拉取 token/scale 的通信 warp 配比",
        gain_en="**1.17x** H20 `S=16K` (`8` vs `4`: `1644->1403 us`, `-14.6%`)",
        gain_zh="**1.17x** H20 `S=16K`（`8` vs `4`：`1644->1403 us`，`-14.6%`）",
    ),
    Knob(
        "WEAVE_SCATTER_WARPS", "Spatial",
        "`0 = arch default`; `4`; `8`. No other count is accepted. On SM103 the scatter warps are the dispatch "
        "warp group (the value is an alias of the communication warp count), so a value contradicting an explicit "
        "`WEAVE_DISPATCH_WARPS` is rejected; on H20 the value sizes the CTA communication warp group "
        "(it must fit beside the dispatch warps)",
        "0",
        "Per-SM scatter warp ratio: sets the CTA communication warp budget dedicated to async FC2 scatter (independent from dispatch pull parameters).",
        "SM103 fp8xfp4, H20",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_SCATTER_WARPS\", \"DG_TUNE_D7_WARPS\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_TUNE_D7_SCATTER_WARPS WEAVE_SCATTER_WARPS",
        compact_values_en="`0=default`, `4`, `8`",
        compact_values_zh="`0=默认`，`4`，`8`",
        compact_mech_en="Per-CTA communication warps/waves dedicated to async FC2 scatter",
        compact_mech_zh="单 CTA/CU 内专职异步 FC2 Scatter 写出的通信 warp/wave 数",
        gain_en="**1.17x** H20 `S=16K` (`1644->1405 us`); **1.04x** MI308X EP16 (`8` vs `4`: `1015->976 us`)",
        gain_zh="**1.17x** H20 `S=16K`（`1644->1405 us`）；**1.04x** MI308X EP16（`1015->976 us`）",
    ),
    Knob(
        "WEAVE_CU_WAVE_SPLIT", "Spatial",
        "Per-hardware syntax (the same name is parsed differently on each backend). "
        "**MI308X (EP8/EP16)**: `0 = auto` (cost-model selection); `1 = 0-8` (0 comm + 8 compute waves); `2 = 4-4` (4 comm + 4 compute waves); `3 = 8-0` (8 comm + 0 compute waves on dedicated comm CUs); names or codes accepted, anything else is rejected (EP8 additionally rejects `8-0` together with the internal SM0 streaming-scheduler pin; `8-0` hit a runtime handshake timeout in a previous-revision MI308X sweep and is never selected by the calibration profile). "
        "**H20 (SM90)**: `0` / `auto` = barrier-free `7-W` lag-W expert task queue with shape-default lead `W` (`max(4, 3*E_local/8)` for `I <= 512`, else `E_local`); `E` (integer, 4..1024) = L1/L2 wave interleave mode 5 with E experts per wave; `M-E` (M = 5..6 barrier-free K-block interleave mode, E = 1..1024 experts per wave) = interleave mode M with E experts per wave; `7-W` (W = 1..1024) = lag-W expert task queue: one flat statically interleaved task stream `L1(0..W-1), L2(0), L1(W), L2(1), ...` in exec order claimed by every SM with one atomicAdd (no expert waves, no L1/L2 grid sync; each L2 tile waits only on the L1 K-block bits it consumes), W is the L1 lead in experts and is not rounded; E must not exceed the experts per rank and a non-divisor E is rounded up to the next divisor of experts-per-rank; the interleave mode applies to the fused launch only (the split L1/L2 launch ignores it); legacy non-K-block modes `0..4`, MI308X codes `1`/`2`/`3` and `8-0` are rejected on fused H20. "
        "**SM103**: the value is parsed (`auto`, `0-8`, `4-4`, `8-0` or codes 0..3) and compiled as a macro, but no SM100 device code reads it, so it has no effect.",
        "0",
        "Per-unit wave split with a different meaning per backend: on MI308X the per-CU `(comm_waves - compute_waves)` wavefront ratio out of 8 waves per CU; on H20 the fused engine's L1/L2 expert-wave split (interleave mode and experts per wave); no effect on SM103. The default (`0`/`auto`) is used by every measured winner.",
        "MI308X EP8/EP16 (wavefront ratio); H20 (expert-wave split, fused launch); SM103 accepted but inert",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_CU_WAVE_SPLIT\"",
        "deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip",
        "(WEAVE_CU_WAVE_SPLIT == 3)",
        compact_values_en="MI308X: `auto, 0-8, 4-4, 8-0`; H20: `auto/7-W, 5-E, 6-E`",
        compact_values_zh="MI308X：`auto, 0-8, 4-4, 8-0`；H20：`auto/7-W, 5-E, 6-E`",
        compact_mech_en="MI308X per-CU 8-wave comm/compute split; H20 zero-`grid_sync` L1/L2 K-bitmap interleave",
        compact_mech_zh="MI308X 单 CU 8-wave 通算切分；H20 无全局栅栏 L1/L2 K-block 交错流",
        gain_en="**2.15x** MI308X EP16 (`4-4` vs `0-8`: `2185->1015 us`); **1.053x** H20 `S=8K` (`926->880 us`)",
        gain_zh="**2.15x** MI308X EP16（`4-4` vs `0-8`：`2185->1015 us`）；**1.053x** H20 `S=8K`（`926->880 us`）",
    ),

    # ----------------------------------------------------------------- Temporal (3)
    Knob(
        "WEAVE_V_HEAD", "Temporal",
        "`0` / unset = automatic (SM103: head helper off, only the dispatch warps pull; H20: auto quota = half of "
        "the one-token-per-warp head slot space; MI308X: build default); `N > 0` = head quota "
        "(host range 0..65536): SM103 runs N non-blocking helper rounds per SM, i.e. a head window of N x num_SMs "
        "tokens clipped to the first L1 ring window and to the received tokens; H20 pulls N head tokens with every "
        "warp before the first WGMMA; MI308X EP8 reads 0 < N < 64 as tokens per head local-plan claim (shrink-only) "
        "and the EP16 build accepts any positive decimal",
        "0 (automatic)",
        "Head schedule: at layer start the helper warps of every CTA (SM103: dispatch warps W00..W03 plus epilogue warps W08..W15, non-blocking; H20: every warp, static zero-CAS slot split) cooperatively pull the head window while the remaining tokens are pulled by the comm warps; GEMM starts after the head rounds. On SM103 the head path is compiled out together with `WEAVE_READY_GROUP_SIZE > 0` and `WEAVE_DEDUP_MODE=1`.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_V_HEAD\", \"DG_V_HEAD\"",
        "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "kCompiledVHeadDefault = WEAVE_V_HEAD",
        compact_values_en="`0=auto`, `N=1..65536`",
        compact_values_zh="`0=auto`，`N=1..65536`",
        compact_mech_en="Head cooperative helper-warp pull quota before compute warps enter GEMM1",
        compact_mech_zh="层启动阶段协助拉取远端 token 的首部配额，缩短首块 GEMM1 等待",
        gain_en="**1.088x** H20 `S=8K` (`4096`: `941.5->865.3 us`, `-8.1%`); **1.043x** MI308X (`1015->973 us`)",
        gain_zh="**1.088x** H20 `S=8K`（`941.5->865.3 us`，`-8.1%`）；**1.043x** MI308X（`1015->973 us`）",
    ),
    Knob(
        "WEAVE_READY_GROUP_SIZE", "Temporal",
        "`0 = full-expert readiness`; `N = 1, 2, 4, 8, 16` = group-granularity readiness publication (in BLOCK_M tiles). "
        "Other integers pass the SM103 host range check (`N >= 0`) but fail the kernel `static_assert`; the H20 host "
        "rejects them",
        "0",
        "Head schedule: fine-grained arrival readiness group size so GEMM1 starts as soon as N tiles arrive without waiting for full experts.",
        "SM103 fp8xfp4, H20",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_READY_GROUP_SIZE\", \"DG_TUNE_D1_G\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D1_G WEAVE_READY_GROUP_SIZE",
        compact_values_en="`0=full expert`, `1, 2, 4, 8, 16`",
        compact_values_zh="`0=整专家`，`1, 2, 4, 8, 16`",
        compact_mech_en="Fine-grained N-tile arrival publication so GEMM1 starts before full expert arrives",
        compact_mech_zh="按 `N` 个 tile 粒度发布到达信号，无需等整专家拉齐即启动 GEMM1",
        gain_en="**1.030x** MI308X EP16 (`1` vs `16`: `1015->985 us`); **1.009x** H20 `S=4K` (`5337->5289 us`)",
        gain_zh="**1.030x** MI308X EP16（`1` vs `16`：`1015->985 us`）；**1.009x** H20 `S=4K`（`5337->5289 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_OWNER", "Temporal",
        "`0 = epilogue_only`: epilogue warps issue every remote FC2 store; `1 = comm_only`: epilogue warps only stage locally and communication warps own every remote store (names or codes accepted; ownership is strictly binary, there is no third mode and `2` is rejected). With `WEAVE_SCATTER_STAGING=smem` the SM103 handoff is strict: the epilogue blocks on a free dedicated-SMEM slot and never scatters, W00..W03 drain the slots between pulled tokens (with `WEAVE_DEDUP_MODE=1` the handoff starts once the pull ends)",
        "0 (auto `comm_only` when a staged path or a scatter SM throttle is requested; MI308X: unset keeps the validated mixed drain)",
        "Middle schedule: strict binary scatter role ownership (never mixes inline epilogue scatter with comm-warp scatter) to overlap FC2 scatter with ongoing GEMM.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_SCATTER_OWNER\", {\"epilogue_only\", \"comm_only\"}",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "kScatterCommOwned = (WEAVE_SCATTER_OWNER == 1)",
        compact_values_en="`0=epilogue_only`, `1=comm_only`",
        compact_values_zh="`0=epilogue_only`，`1=comm_only`",
        compact_mech_en="Strict binary FC2 scatter ownership to overlap cross-GPU writes with ongoing GEMM",
        compact_mech_zh="严格二分的 FC2 跨卡发送归属（尾声直写 vs 通信 warp 全异步排空）",
        gain_en="**1.14x** H20 `S=16K` (`comm_only`: `1244->1094 us`, `-12.0%`); **1.034x** MI308X (`1015->981 us`)",
        gain_zh="**1.14x** H20 `S=16K`（`comm_only`：`1244->1094 us`，`-12.0%`）；**1.034x** MI308X（`1015->981 us`）",
    ),

    # -------------------------------------------------------------- Implementation (6)
    Knob(
        "WEAVE_PRECISION_MODE", "Implementation",
        "`0 = fp8xfp4`; `1 = fp4xfp4` (alias `nvfp4`); `2 = fp8xfp8` (alias `fp8`); `3 = bf16` (alias `bf16xbf16`); `auto` = backend default (names, aliases or codes accepted, case-insensitive; any other spelling is rejected; MI308X builds accept only `2` / `fp8` / `fp8xfp8`)",
        "0",
        "Single-op numeric precision and tensor-core instruction format across GEMM1/GEMM2: selects the MegaMoE backend in Python (`deep_gemm/mega/precision.py`); the host never overrides the precision of the backend actually launched.",
        "SM103 fp8/fp4/bf16; H20 fp8; MI308X fp8",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "std::getenv(\"WEAVE_PRECISION_MODE\")",
        "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "kCompiledPrecisionMode = WEAVE_PRECISION_MODE",
        compact_values_en="`0=fp8xfp4`, `1=fp4xfp4`, `2=fp8`, `3=bf16`",
        compact_values_zh="`0=fp8xfp4`，`1=fp4xfp4`，`2=fp8`，`3=bf16`",
        compact_mech_en="Selects Tensor/Matrix Core numeric format and backend across GEMM1 & GEMM2",
        compact_mech_zh="选择 GEMM1/GEMM2 张量核数值精度与对应硬件后端",
        gain_en="**1.59x** SM103 `S=16K` (`fp4xfp4` vs `fp8xfp4`: `1038->654 us`); **1.010x** H20 `fp8` (`5276->5224 us`)",
        gain_zh="**1.59x** SM103 `S=16K`（`fp4xfp4` vs `fp8xfp4`：`1038->654 us`）；**1.010x** H20（`5276->5224 us`）",
    ),
    Knob(
        "WEAVE_GEMM1_CONFIG", "Implementation",
        "`0 = auto`: host shape-adaptive whole tuple chosen per launch from M_expected = S_global*topk/num_experts and the N/K extents; always on (no knob needed). "
        "Adaptive ladder: SM103 walks the BM ladder {16, 32, 64, 80, 96, 128, 192, 240} with BK 128/256, asymmetric L1/L2 store heights, epilogue register stages 2 for BM 80/96/240 and the 64/56/192 setmaxnreg rebalance when pull dedup is on; H20 picks BM 128/128 (I <= 1024 and M_expected >= 48), a 32/64 split launch (I >= 1536 and M_expected <= 24, unless a fused-engine-only setting forces the fused launch, which keeps 64/64) or the canonical 64/64. "
        "A pinned GEMM1/GEMM2 tuple wins over the adaptive pick. On SM103 a manual `BM=128` + `EPI=2` request is clamped to `EPI=1` (0-spill safety clamp: V3 ptxas reports BM128 epi2 = 208 B spill stores / 304 B spill loads, while BM128 epi1 and BM80/BM96 epi2 = 128 regs / 0 spill). "
        "A legacy `WEAVE_ADAPT_GEMM_CONFIG=0` / `WEAVE_ADAPT_BM=0` is still accepted as an internal diagnostic override (fixed canonical tuple), not a public tunable. "
        "Explicit forms: a bare `BM` in `16, 32, 64, 80, 96, 128, 192, 240, 256`; SM103 "
        "additionally accepts the whole-tuple form `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]` (`:` or `,` separated; "
        "STORE_BM a multiple of 8 and <= BM, BK 128 or 256, STAGES 2..16, EPI 0..2; BK, STAGES and EPI are "
        "launch-wide and must agree with `WEAVE_GEMM2_CONFIG`). Legacy alias `WEAVE_BLOCK_M_L1` (bare BM only). "
        "H20 takes a bare BM only (a decoupled L1/L2 split with GEMM1 `BM=64` is rejected by the host); the MI308X "
        "builds (EP8, EP16 and V70) accept `WEAVE_GEMM1_CONFIG` (legacy alias `WEAVE_BLOCK_M_L1`) with `0, 16, 32, 64, 128`",
        "0",
        "Independent GEMM1 (FC1) config selector: choosing the tuple (blockM, optionally store height / BK / stages / epilogue register stages) synchronously locks GEMM1's paired blockN/K, mainloop TMA stages, and epilogue M/N/K & TMEM/reg stages.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "gemm_phase_config_env(\"WEAVE_GEMM1_CONFIG\", \"WEAVE_BLOCK_M_L1\")",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_BLOCK_M_L1 == 0",
        compact_values_en="`0=auto` or `BM[:STORE_BM[:BK[:ST[:EPI]]]]` (`16..256`; alias `WEAVE_BLOCK_M_L1`)",
        compact_values_zh="`0=自适应常开` 或 `BM[:STORE_BM[:BK[:ST[:EPI]]]]`（旧名 `WEAVE_BLOCK_M_L1`）",
        compact_mech_en="Independent FC1 tuple selector; locks paired `BN/BK/stages/epi_stages` (0-spill clamped)",
        compact_mech_zh="独立选择 FC1 配置元组并同步锁定 `BN/BK/stages/epi_stages`（自动防溢出）",
        gain_en="**1.51x** H20 `S=16K` (`BM=128`: `1651->1091 us`, `-33.9%`); **1.047x** SM103 (`96,16`: `766->731 us`)",
        gain_zh="**1.51x** H20 `S=16K`（`BM=128`：`1651->1091 us`，`-33.9%`）；**1.047x** SM103（`96,16`：`766->731 us`）",
    ),
    Knob(
        "WEAVE_GEMM2_CONFIG", "Implementation",
        "`0 = follow GEMM1 / auto` (host shape-adaptive whole tuple, always on, see `WEAVE_GEMM1_CONFIG`; a pinned tuple wins over the adaptive pick); same value grammar as `WEAVE_GEMM1_CONFIG` (bare `BM` in `16, 32, 64, 80, 96, 128, 192, 240, 256` on every backend, whole-tuple form on SM103), decoupled from GEMM1 (L1 and L2 may pick different BM: on SM103 fp8xfp4 the smaller must divide the larger and be a multiple of 32, on H20 the two must divide each other). Legacy alias `WEAVE_BLOCK_M_L2` (MI308X builds accept `WEAVE_GEMM2_CONFIG` and `WEAVE_BLOCK_M_L2` with `0, 16, 32, 64, 128`)",
        "0",
        "Independent GEMM2 (FC2) config selector (decoupled from GEMM1): choosing the tuple synchronously locks GEMM2's paired blockN/K, mainloop stages, and epilogue M/N/K & stages.",
        "SM103, H20, MI308X",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "gemm_phase_config_env(\"WEAVE_GEMM2_CONFIG\", \"WEAVE_BLOCK_M_L2\")",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_BLOCK_M_L2 == 0",
        compact_values_en="`0=follow/auto` or `BM[...]` (`16..256`; alias `WEAVE_BLOCK_M_L2`)",
        compact_values_zh="`0=跟随/自适应` 或显式元组（与 FC1 解耦；旧名 `WEAVE_BLOCK_M_L2`）",
        compact_mech_en="Independent FC2 tuple selector decoupled from GEMM1 (e.g. `32/64` or `96,16` + `96,8`)",
        compact_mech_zh="与 GEMM1 解耦的 FC2 独立配置元组（如 `32/64` 或非对称 `96,16` + `96,8`）",
        gain_en="**1.036x** MI308X EP16 (`64` vs `32`: `1051->1015 us`); **1.025x** H20 `32/64` (`2234->2178 us`)",
        gain_zh="**1.036x** MI308X EP16（`64` vs `32`：`1051->1015 us`）；**1.025x** H20 `32/64`（`2234->2178 us`）",
    ),
    Knob(
        "WEAVE_REG_PULL_BYTES", "Implementation",
        "`0 = TMA/LDS SMEM staging`; `1024, 2048, 4096` = per-warp bytes per pull chunk on 32-lane NVIDIA warps (= `32, 64, 128 B` or `8, 16, 32` 32-bit regs per thread); `8192` = per-wavefront bytes per pull chunk on 64-lane MI308X CDNA3 wavefronts (`128 B` per lane); every other value is rejected. Unset: SM103 behaves as `0`, H20 keeps the engine's register-direct fast pull (an explicit `0` turns it off), MI308X V70 uses `8192`",
        "0",
        "Independent Dispatch operator parameter: per-warp/wavefront chunk width for zero-SMEM register-direct remote pulling (`uint4 regs[2/4/8]` per lane, freeing SMEM/LDS for deeper GEMM mainloop stages).",
        "SM103 fp8xfp4, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "\"WEAVE_REG_PULL_BYTES\", \"DG_TUNE_D2_REG_PULL_BYTES\"",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D2_REG_PULL_BYTES WEAVE_REG_PULL_BYTES",
        compact_values_en="`0=SMEM`, `1024, 2048, 4096` (NV), `8192` (MI308X)",
        compact_values_zh="`0=SMEM`，NVIDIA `1024/2048/4096`，MI308X `8192`",
        compact_mech_en="Zero-SMEM register-direct remote pull width per warp/wavefront (frees SMEM/LDS stages)",
        compact_mech_zh="零 SMEM 寄存器直传单轮拉取字节宽度，腾出片上内存给更深 GEMM 流水",
        gain_en="**1.19x** MI308X EP16 (`8192` vs `1024`: `1212->1015 us`); **1.024x** SM103 (`4096`: `618.5->604.1 us`)",
        gain_zh="**1.19x** MI308X EP16（`8192` vs `1024`：`1212->1015 us`）；**1.024x** SM103（`4096`：`618.5->604.1 us`）",
    ),
    Knob(
        "WEAVE_DEDUP_MODE", "Implementation",
        "`0 = off` (per-slot pull); `1 = on` (pull each unique `(src_rank, token)` once over NVLink/xGMI, then fan out locally across all hit experts); any other value is rejected. Unset = backend default (SM103: off, the live-mu verdict is left to the CTA-0 policy; H20 unified engine: on; MI308X builds: on)",
        "unset (SM103 0; H20 / MI308X 1)",
        "Independent Dispatch operator remote-pull deduplication: pulls a multi-hit remote token once and broadcasts it into all hit local expert slots.",
        "SM103 fp8xfp4, H20, MI308X",
        "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
        "get_env<int>(\"WEAVE_DEDUP_MODE\", -1)",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "#define DG_ISOLATED_D5_DEDUP WEAVE_DEDUP_MODE",
        compact_values_en="`0=off`, `1=on` (unset=arch default)",
        compact_values_zh="`0=off`，`1=on`（未设时按架构默认）",
        compact_mech_en="Remote pull deduplication: pulls unique `(src_rank, token)` once and fans out locally",
        compact_mech_zh="跨卡拉取去重：同源多专家命中只跨卡拉一次并在片内广播扇出",
        gain_en="**1.067x** H20 `S=16K` (`1644->1540 us`); **1.023x** MI308X EP16 (`1` vs `0`: `1038->1015 us`)",
        gain_zh="**1.067x** H20 `S=16K`（`1644->1540 us`）；**1.023x** MI308X EP16（`1` vs `0`：`1038->1015 us`）",
    ),
    Knob(
        "WEAVE_SCATTER_STAGING", "Implementation",
        "`0 = reg_2stage`: double-buffered register transfer; `1 = smem`: per-SM bank-conflict-free SMEM ring handoff; `2 = hbm_dedup`: L2-resident HBM buffer with bf16x2 atomic dedup (names or codes accepted; `1` and `2` require comm-owned scatter)",
        "0 (auto `smem` on H20 when comm-owned scatter is requested; auto `hbm_dedup` when the internal local-scatter-dedup pin is 1, and on H20 also when a scatter SM throttle is set)",
        "Independent Scatter operator staging implementation (`reg_2stage`, XOR-swizzled `smem`, or `hbm_dedup`).",
        "SM103, H20",
        "csrc/jit_kernels/heuristics/weave_knob_config.hpp",
        "\"WEAVE_SCATTER_STAGING\", {\"reg_2stage\", \"smem\", \"hbm_dedup\"}",
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
        "WEAVE_SCATTER_STAGING != 2 or DG_D7_QUEUE_SCATTER != 0",
        compact_values_en="`0=reg_2stage`, `1=smem`, `2=hbm_dedup`",
        compact_values_zh="`0=reg_2stage`，`1=smem`，`2=hbm_dedup`",
        compact_mech_en="Scatter staging data path (2-stage regs, XOR-swizzled SMEM ring, or L2/HBM dedup)",
        compact_mech_zh="Scatter 暂存数据通路（双缓冲寄存器、无冲突 SMEM 环或 L2/HBM 去重缓冲）",
        gain_en="**1.14x** H20 `S=16K` (`smem` vs `reg_2stage`: `1244->1094 us`); **1.010x** MI308X (`1025->1015 us`)",
        gain_zh="**1.14x** H20 `S=16K`（`smem` vs `reg_2stage`：`1244->1094 us`）；**1.010x** MI308X（`1025->1015 us`）",
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
    """Legacy environment spellings that the host accepts for public knob `name`."""
    return tuple(old for old, new in LEGACY_ALIASES.items() if new == name)


def canonical_name(name: str) -> str:
    """Map a legacy alias to its public name; other names are returned unchanged."""
    return LEGACY_ALIASES.get(name, name)
