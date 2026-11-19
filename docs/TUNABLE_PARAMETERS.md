# Weave Tunable Parameters & Exact Code Locations

This document is machine-verified by `tests/verify_tunable_params_doc.py`. Every `path:line` citation is resolved against the working tree; the verifier requires the cited source line to contain the parameter or its documented kernel-side symbol.

Rows list both spellings when a host knob compiles down to a different kernel symbol (for example `DG_TUNE_D1_G / DG_ISOLATED_D1_G`).

## 0. Hardware and precision modes

| Hardware | Language | Precision | Entry |
|---|---|---|---|
| NVIDIA B300 (SM100/SM103) | CUDA C++ | `mma_type=fp8xfp4` or `fp4xfp4` (NVFP4) | `sm100_fp8_fp4_mega_moe.cuh` / `sm100_fp4_fp4_mega_moe.cuh` |
| NVIDIA H20 (SM90) | CUDA C++ | `fp8` | `sm90_fp8_mega_moe.cuh` |
| AMD MI308X (gfx942) | HIP C++ | `fp8` | `amd/megamoe_amd.hip` |

Precision byte widths and P_tc scaling are defined in `weave_cost_model.cuh`; the default FP8xFP4 row is bit-identical to Paper Table I.

## 1. Unified `WeaveKnobs`, cost model, and ablation

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | 0/1/2/3 | 0 | Preset 0 = full Weave; 1 = -S; 2 = -S-T; 3 = -S-T-C. Presets 1..3 disable the runtime search and force a fixed spatial/scatter candidate. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` |
| `WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL` / `enable_runtime_cost_model` | 0/1 | 1 when preset=0, else 0 | Master switch for the CTA-0 Eq.(1)-(3) search. An explicit -D...=1 overrides the preset default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:47` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:436` |
| `WEAVE_KNOB_COMM_UNITS` | uint32 | 4 | Spatial default N_c per SM/CU when the policy search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:51` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:258` |
| `WEAVE_KNOB_COMP_UNITS` | uint32 | 4 | Spatial default N_e per SM/CU when the policy search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:54` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:259` |
| `WEAVE_KNOB_DUALGROUP_MIN_TOKENS` | uint32 | 64 | AMD dual-group threshold: t_local at or above this value rebinds communication waves into a second compute group. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:57` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1211` |
| `WEAVE_KNOB_BM_TIER_MODE` | 0/1/2 | 2 | AMD BM tier selector. Mode 2 enables the five-tier {16,32,64,128,256} planner. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:60` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:439` |
| `WEAVE_KNOB_BN1` | 64/128/256 | 128 | First-layer N tile width default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:63` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:263` |
| `WEAVE_KNOB_BN2_MODE` | 0/64/128 | 0 | Second-layer N tile mode; 64 enables the BN64 small-M path. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:66` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:953` |
| `WEAVE_KNOB_BK` | 64/128/256 | 128 | K tile width default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:69` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:265` |
| `WEAVE_KNOB_S2_NGROUP_MODE` | 0/1/2/4 | 0 | AMD stage-2 N grouping width; 0 = auto. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:72` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:961` |
| `WEAVE_KNOB_HETERO_SM_MASK` | bitmask | 0 | Per-SM heterogeneous role mask seed. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:75` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:267` |
| `WEAVE_KNOB_OVERLAP_PIPELINE` | 0/1 | 1 (0 at preset >= 2) | Temporal overlap switch. 0 falls back to bulk-synchronous stage serialization. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:78` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:706` |
| `WEAVE_KNOB_SCATTER_MODE` / `WEAVE_SCATTER_MODE` | 0/1/2 | 1 | Scatter path selector: 0 = inline epilogue, 1 = comm-queue async, 2 = local dedup first. The AMD EP16 header consumes the WEAVE_SCATTER_MODE alias. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:126` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:52` |
| `WEAVE_KNOB_GEMM_STAGES` | 1..4 | 2 | Mainloop software pipeline depth. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:84` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:270` |
| `WEAVE_KNOB_ALL_WAVE_COMBINE` | 0/1 | 1 (0 at preset >= 2) | Use all waves for the cooperative tail combine; 0 restricts to four waves. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:87` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |
| `WEAVE_KNOB_SPIN_BEFORE_SLEEP` | uint32 | 256 | Polling spin count before issuing the first sleep/backoff. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:90` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:272` |
| `WEAVE_KNOB_SELECTIVE_DEDUP` | 0/1 | 1 (0 at preset >= 3) | Cross-rank selective token dedup. EP16 requires 1. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:93` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:506` |
| `WEAVE_KNOB_REGISTER_DIRECT_COMM` | 0/1 | 1 | Register-direct zero-copy gather (no dense communication staging). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:96` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:276` |
| `WEAVE_KNOB_PARALLEL_ROUTE_PLAN` / `WEAVE_PARALLEL_ROUTE_PLAN` | 0/1 | 1 | All-CTA parallel route histogram/prefix/scatter. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:99` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `WEAVE_KNOB_LOCAL_FIRST_ORDER` | 0/1 | 1 | LOCAL_READY -> REMOTE_PRIMARY -> LOCAL_COPY worklist priority. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:102` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:278` |
| `WEAVE_KNOB_FUSED_SWIGLU_REQUANT` | 0/1 | 1 | Fuse SwiGLU with per-128 blockscale requantization. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:105` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:280` |
| `WEAVE_KNOB_TAIL_FINE_GRAINED` | 0/1 | 1 (0 at preset >= 2) | Fine-grained tail reduce; 0 uses the constant t_tail_us. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:108` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:359` |
| `WEAVE_KNOB_V_HEAD` | uint32 | 0xffffffff (auto) | Manual V_head override; effective_v_head applies it after Eq.(3). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:111` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:274` |
| `WEAVE_KNOB_PRECISION_MODE` / `precision_mode` | 0/1/2/3 | 0 (FP8xFP4) | First-class B300 precision selector: 0=FP8xFP4, 1=NVFP4, 2=FP8xFP8, 3=BF16. Mirrored by Python mma_type. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:117` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:368` |
| `precision_mode` struct field | uint32 | 0 | Runtime field in WeaveKnobs; the CTA-0 selector reads it to pick byte widths and P_tc(r). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:229` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:368` |
| `weave_mega_moe(..., mma_type=...)` (Python) | fp8xfp4/fp4xfp4/nvfp4/fp8/bf16 | sym_buffer.mma_type | Unified `weave_mega_moe` entry point; dispatches to the four existing DeepGEMM wrappers without altering them. | `deep_gemm/mega/__init__.py:535` | `deep_gemm/mega/__init__.py:571` |
| `mma_type` (C++) | fp8xfp4/fp4xfp4/nvfp4/fp8/bf16 | fp8xfp4 | Host dispatch table mapping precision to backend, PrecisionMode, and kernel header. | `csrc/apis/mega.hpp:68` | `csrc/apis/mega.hpp:73` |
| `override_mask` | bit0..bit3 | 0 | Explicit override bits for spatial / scatter / V_head / temporal. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:195` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:287` |
| `override_r_mask` | packed uint32 | auto | Packed r=(N_c,N_e,groups,hetero); applied when bit0 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:196` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:300` |
| `override_scatter_mode` | 0/1/2 | auto | Forced scatter path; applied when bit1 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:197` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:293` |
| `override_v_head` | uint32 | auto | Forced V_head; applied when bit2 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:198` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:307` |
| `selected_r_mask` | packed uint32 | argmin | Published spatial decision consumed by every CTA. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:98` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:457` |
| `selected_scatter_mode` | 0/1/2 | argmin | Published temporal decision consumed by the scatter path. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:99` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:458` |
| `selected_v_head` | uint32 | argmin | Published head-pull budget consumed by the pull loops. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:100` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:459` |

## 2. B300 SM100/SM103 7-axis engineering knobs

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `DG_TUNE_D1_G` / `DG_ISOLATED_D1_G` | 0/1/2/4/8/16 | 0 | D1 ready-group granularity G; 0 disables grouped publication. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:129` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:520` |
| `DG_TUNE_D2_MODE` / `DG_ISOLATED_D2_MODE` | 0..15 | 0 | D2 dispatch/register-pull mode; >=1 enables warp reconfiguration. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:131` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:52` |
| `DG_TUNE_D2_REG_PULL_BYTES` / `DG_ISOLATED_D2_REG_PULL_BYTES` | 0/1024/2048/4096 | 0 | D2 register-direct pull width in bytes; 0 uses SMEM staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:132` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:62` |
| `DG_ISOLATED_D2_D8_SM_LIMIT` | 0..148 | 74 | Maximum SM id participating in the D8 role split. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:133` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:645` |
| `DG_ISOLATED_D2_DISP_REGS` | register count | 0 | Explicit register budget for dispatch warps under D2. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:134` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:851` |
| `DG_ISOLATED_D2_EPI_REGS` | register count | 0 | Explicit register budget for epilogue warps under D2. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:135` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:860` |
| `DG_TUNE_D3_BM` / `DG_D3_BM` | 32/64/96/128/192 | 0 (heuristic) | D3 block-M override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:137` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:80` |
| `DG_TUNE_D3_BN` / `DG_D3_BN` | 128/256 | 0 (heuristic) | D3 block-N override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:138` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:87` |
| `DG_TUNE_D3_BK` / `DG_D3_BK` | 64/128/256 | 0 (heuristic) | D3 block-K override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:139` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:94` |
| `DG_TUNE_D3_STORE_M` / `DG_D3_STORE_M` | 4/8/16/32 | 0 (heuristic) | TMA-store M granularity. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:140` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:102` |
| `DG_TUNE_D3_DYN_EPI` / `DG_D3_DYN_EPI` | 0/1/2 | 0 | Dynamic epilogue pass count; 2 expects STORE_BLOCK_M=16. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:141` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:111` |
| `DG_TUNE_D3_DYN_TMA_A` / `DG_D3_DYN_TMA_A` | 0/1/2 | 0 | Dynamic TMA-A descriptor mode for small M. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:142` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:121` |
| `DG_TUNE_D3_OVERLAP_SMEM` / `DG_D3_OVERLAP_EXPERT_SMEM` | 0/1 | 0 | Overlap expert metadata SMEM with the next expert tile. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:143` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:131` |
| `DG_TUNE_D3_EXPERT_STEP_MODE` / `DG_D3_EXPERT_STEP_MODE` | 0/1/2 | 0 | Expert-step scheduling mode; non-zero raises scheduler register demand. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:144` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:852` |
| `DG_D3_ACCUM_STAGES` | 2/3/4 | 2 | TMEM accumulator stage count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:145` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:519` |
| `DG_D3_TMA_STORE_STAGES` | 2/3/4/5/6 | 2 | TMA-store ring depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:521` |
| `DG_TUNE_D3_SCHED_STAGES` / `DG_D3_SCHED_STAGES` | 2/3 | 2 | Scheduler look-ahead depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:147` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:25` |
| `DG_TUNE_D4_EPW` / `DG_ISOLATED_D4_EPW` | 1/2/4/8/16/32/64 | 0 (heuristic) | Experts-per-wave override for the D4 dynamic queue. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:149` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:73` |
| `DG_TUNE_D4_STATIC` / `DG_ISOLATED_D4_STATIC` | 0/1 | 0 | Use a static expert-order table instead of the ready queue. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:150` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:73` |
| `DG_TUNE_D4_READYQ` / `DG_ISOLATED_D4_READYQ` | 0/1 | 0 | D4 ready-queue scheduling path. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:151` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2473` |
| `DG_TUNE_D4_MODE` / `DG_ISOLATED_D4_MODE` | 0..3 | 0 | D4 scheduling mode selector. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:152` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:664` |
| `DG_TUNE_D5_DEDUP` / `DG_ISOLATED_D5_DEDUP` | 0/1/2 | 0 | D5 dedup mode: 0 off, 1 dedup, 2 local-dedup pre-reduce. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:155` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:413` |
| `DG_TUNE_D5_SCHED` / `DG_ISOLATED_D5_SCHED` | 0/1 | 1 when dedup=2 | D5 dedup scheduler toggle. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:156` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:88` |
| `DG_TUNE_D6_G` / `DG_ISOLATED_D6_G` | 1/2/4/8/16 | 0 | D6 combine group tile count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:158` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:155` |
| `DG_TUNE_D6_REDUCE_WARPS` / `DG_ISOLATED_D6_REDUCE_WARPS` | 0/1/2/4/8/16 | 0 (all epilogue warps) | Static cap on combine warps; the kernel takes min(policy N_e, this cap). | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:159` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:162` |
| `DG_TUNE_D7_SCATTER_MODE` / `DG_ISOLATED_D7_SCATTER` | 0/1/2 | 0 | D7 scatter mode: 0 inline epilogue, 1 handoff, 2 local dedup. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:161` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:253` |
| `DG_TUNE_D7_SMEM_ALIAS` | 0/1 | 1 | Alias the D7 staging SMEM with reusable epilogue storage. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:163` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:574` |
| `DG_TUNE_D7_WARPS` / `DG_TUNE_D7_SCATTER_WARPS` | 0/2/4/8 | 0 (auto) | D7 scatter executor warp count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:164` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:217` |
| `DG_TUNE_D7_HANDOFF_WAIT` | 0/1 | 1 | Wait for the D6 ready epoch before the D7 handoff. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:165` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:229` |
| `DG_SHARED_L2_EARLY_BLOCKS` | 0..148 | 10 | Number of early blocks that seed the shared-L2 cache. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:264` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:170` |
| `kEnableMulticast` / `cluster_size` | true / 2 | true | 2-CTA multicast invariant: cluster_size=2 on the host; kEnableMulticast is the compiled kernel-side assertion. | `csrc/jit_kernels/heuristics/mega_moe.hpp:109` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:66` |

## 3. H20 SM90 knobs

Production delivery configuration (DELIVERY_20261116): `DG_MOE_SWAPAB_AUTO=1`, `DG_MOE_SWAPAB_EPI_WG=4`, `DG_MOE_SWAPAB_MAX_LOAD=100000`, `DG_MOE_SWAPAB_PAD_PCT=25`, `DG_MOE_SWAPAB_MIN_LOAD=8`; source defaults below stay backward-compatible with the official rule.

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `DG_MOE_SWAPAB_AUTO` / `kSwapABActive` | 0/1 | 0 | Two-layer AUTO gate. 1 makes the swapAB decision a superset of the official n<=24 decision and lets pad_frac widen the tile. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:616` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_EPI_WG` / `kNumEpilogueWarpgroups` | 2/4 | 2 | Requested epilogue warpgroups. 4 enables BN=256 + 4WG only when pad_ok; otherwise returns 2. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:612` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:370` |
| `DG_MOE_SWAPAB_MAX_LOAD` / `kSwapABActive` | int | 24 compact / 16 other | Explicit swapAB tokens-per-expert ceiling when AUTO=0. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:631` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_PAD_PCT` / `kSwapABActive` | percent | 25 | Minimum padding waste that justifies the wide tile. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:601` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_MIN_LOAD` / `kSwapABActive` | tokens/expert | 8 | Token floor below which the wide tile is rejected (n<=5 is a measured regression). | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:602` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `kSwapSFSubBlocks` | 1/2 | derived | One activation SF per 64 output columns. BLOCK_N=128 -> 1; BLOCK_N=256 -> 2. Eliminates the 0.366 -> 5.94e-04 drift. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:1095` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:5199` |
| `SWAPAB_FINE_LADDER` / `n_swap` | 8/16/24/32/40/48/56/64 | 8-token ladder | Both L1 and L2 dispatch points select the WGMMA N_SWAP template in 8-token steps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:1096` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:3599` |
| `AGY_MCAST` | 1/2 | 1 | Adaptive multicast: 2 = pair M blocks; 1 = single-tile stream. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:442` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:529` |
| `AGY_SWIZZLE_TILES` / `kAgySwizzleTiles` | 0/1 | 1 in AGY presets | Tile-order swizzle for L1/L2 expert caches. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:429` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:597` |
| `AGY_HOT_COLD_L1` / `kAgyHotColdMode` | 0/1/2 | 0 | Hot/cold expert pairing inside each experts-per-wave window. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:481` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:485` |
| `AGY_DEDUP_DISP` | 0/1 | 0 | Dedup at dispatch time; duplicate memberships reuse the primary payload. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:411` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:547` |
| `AGY_DEDUP_SCAT` | 0/1 | 0 | Dedup-aware scatter path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:413` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:559` |
| `AGY_REG_COMB` / `kAgyRegComb` | 0/1/2 | 0 | Flattened direct-register combine; avoids SMEM round trips. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:418` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:580` |
| `AGY_ALL_IDLE_COMB` | 0/1 | 0 | Let all idle epilogue warps join the tail reduce. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:515` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:489` |
| `AGY_DYN_SCHED` / `kAgyDynSched` | 0/1 | 0 | Dynamic queue scheduler instead of the static task cursor. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:403` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:494` |
| `AGY_INTERLEAVE_L1_L2` / `kAgyInterleaveMode` | 0..6 | 0 | Interleave L1/L2 tile execution; 4 = epw L1/L2, 5 = B300-style, 6 = wave-lagged. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:521` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:375` |
| `AGY_SCATTER_ROLE_DISP` | 0/1 | 0 | Dispatch warps execute scatter instead of dedicated epilogue warps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:528` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:381` |
| `AGY_D_MASK_MODE` / `kAgyHeteroD8D4` | 0..5 | 0 | Heterogeneous D8/D4 SM-role mask mode. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:464` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:576` |
| `AGY_PIPE_WGMMA` | 0/1 | 0 | Pipelined WGMMA issue across K blocks. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:405` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:537` |
| `AGY_SWAP_PIPE` | 0/1 | 0 | Pipelined swapAB WGMMA path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:407` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:542` |
| `AGY_FAST_PULL` | 0/1 | 0 | Fast dispatch pull path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:409` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:554` |
| `AGY_ADAPT_BM` | 0/1 | 0 | Per-expert adaptive BLOCK_M tiers. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:394` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:512` |

### 3.1 Host-only / currently inactive AGY knobs

| Knob | Valid values | Default | Status | Host parse |
|---|---|---|---|---|
| `AGY_FINE_COMB_G` | 0/1/2/4/8/16 | 0 | Fine-combine group size. Parsed and emitted; the current single-header kernel has no consumer yet. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:508` |
| `AGY_EARLY_COMB` | 0/1 | 0 | Early combine flag. Parsed and emitted; no consumer in this revision. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:511` |
| `AGY_COMB_WARPS` | 0/1/2/4/8 | 0 | Fixed combine warp count. Parsed and emitted; no consumer in this revision. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:513` |
| `AGY_META_GROUP_G` | 0/1/2/4/8/16 | 0 | Metadata ready-group size. Parsed and emitted; no consumer in this revision. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:451` |
| `AGY_TILE_SYNC_G` | 0/1/2/4/8/16 | 0 | Tile-level sync group. Parsed and emitted; no consumer in this revision. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:456` |
| `AGY_HOT_COLD_L2` | 0/1/2 | 0 | L2 hot/cold mode. Parsed and emitted; the kernel currently consumes only AGY_HOT_COLD_L1. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:483` |

## 4. MI308X gfx942 knobs

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | 0/1/2/3 | 0 | EP16 build-time ablation selector; preset 3 also switches WEAVE_SCATTER_MODE to 0. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:82` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1206` |
| `WEAVE_SCATTER_MODE` | 0/1/2 | 2 (0 at preset 3) | EP16 scatter and return mode; 2 enables the node-partial path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:106` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:52` |
| `WEAVE_KNOB_BM_TIER_MODE` | 0/2 | 2 | Five-tier BM planner ({16,32,64,128,256}) when non-zero. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:439` |
| `WEAVE_KNOB_SELECTIVE_DEDUP` | 1 | 1 | EP16 requires selective input dedup in every ablation arm. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:506` |
| `WEAVE_KNOB_DUALGROUP_MIN_TOKENS` | uint32 | 64 | Local-token threshold for rebinding comm waves into a second compute group. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:57` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1211` |
| `WEAVE_KNOB_OVERLAP_PIPELINE` | 0/1 | 1 | 0 disables the EP16 overlap pipeline and runs the bulk-synchronous path. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:78` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1264` |
| `WEAVE_KNOB_ALL_WAVE_COMBINE` | 0/1 | 1 | 0 limits combine to 64 workers instead of 96. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:87` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |
| `WEAVE_KNOB_BN2_MODE` | 0/64/128 | 0 | Stage-2 BN mode; 64 selects the small-M 1x4 BN64 kernel. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:66` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:953` |
| `WEAVE_KNOB_S2_NGROUP_MODE` | 0/1/2/4 | 0 | Stage-2 N-group width. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:72` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:961` |
| `WEAVE_TH_BM64` | uint32 | 16 | Small-M threshold for BM64 tiles. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:542` |
| `WEAVE_TH_BM128` | uint32 | 96 | Small-M threshold for BM128 tiles. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:545` |
| `WEAVE_PARALLEL_ROUTE_PLAN` | 0/1 | 1 | All-CTA parallel route histogram/prefix/scatter. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:101` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `V1864Roles` / `WEAVE_PARALLEL_ROUTE_PLAN` | struct | roles={80,0,80} | Wave-role descriptor: {non_epilogue_waves, comm_waves, mma_waves}. CTA-0 remaps it from selected_r_mask. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:101` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1200` |
| `EP16_SLOT_CHUNK` / `ep16t_slot_chunk` | 1/2/4/8/16/32/64 | adaptive | Bulk DISPATCH descriptor granularity; host clamps to 1..64 and the plan validates power-of-two. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:159` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:289` |
| `ep16t_ret_chunk` | derived 1/2/4 | derived | RETURN descriptor granularity, derived from slot_chunk and capped at 4. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:303` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:308` |
| `meta_chunk_slots` | >=1 | max(256, t_local) | Metadata slots per RDMA WR descriptor. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:55` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:338` |
| `ep16_ld2_agent_u4` | single-block asm volatile | enabled | Single-block 2x-MLP global_load_dwordx4 sc1 helper used by the EP16 bucket reduce (v28). | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:27` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:62` |
| `FI_LEVEL` | 0/1/2 | 0 | WeaveProfiler instrumentation level: 0 production, 1 open/close, 2 full events. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:100` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:32` |

## 5. External CK dependency

The 1100-file `aiter_meta` CK tree is intentionally not vendored. Builds pass `-I $AITER_META/3rdparty/ck_helper`, `-I $AITER_META/3rdparty/composable_kernel/include`, `-I $AITER_META/3rdparty/composable_kernel/library/include`, `-I $AITER_META/csrc/include`, and `-I $AITER_META/csrc/ck_gemm_moe_2stages_codegen`. Small custom overlay headers are vendored under `amd/third_party_overlay/`.
