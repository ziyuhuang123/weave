# Weave Tunable Parameters and Exact Code Locations

This document is machine-verified by tests/verify_tunable_params_doc.py. Every
path:line citation is resolved against the current working tree, and the cited
line must contain the parameter or its documented kernel-side binding. New
launch scripts use WEAVE_* only. Archived spellings remain internal C++ fallback
paths and are not part of the public surface.

Tier 1 values are compile-time or architecture-fixed engineering knobs. Tier 2
values are O(1) host-side shape and padding heuristics. Tier 3 decisions are
selected after dispatch by the in-kernel CTA-0 Eq.(1)-(3) model. See README.md
for the three-tier rationale and measured validation.

## 1. Unified core

| Parameter | Tier | Values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | 1 | fp8xfp4  /  fp4xfp4  /  fp8  /  bf16 | fp8xfp4 (0) | First-class precision switch; selects byte widths and tensor-core rate. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:269` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_ABLATION_PRESET` | 1 | 0/1/2/3 | 0 | Four-arm progressive ablation: Weave, -S, -S-T, -S-T-C. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:26` |
| `WEAVE_ENABLE_COST_MODEL` | 3 | 0/1 | 1 at preset 0 | Master switch for the CTA-0 Eq.(1)-(3) runtime selection. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:49` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:27` |
| `WEAVE_COMM_UNITS` | 1 | uint32 | 4 | Compile-time default communication units when search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:59` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:28` |
| `WEAVE_COMP_UNITS` | 1 | uint32 | 4 | Compile-time default compute units when search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:69` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:29` |
| `WEAVE_V_HEAD` | 3 | uint32 | auto (0xffffffff) | Head-pull budget override; auto value comes from Eq.(3). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:259` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_SCATTER_MODE` | 3 | 0/1/2 | 1 | Inline epilogue, communication-queue scatter, or local-dedup scatter. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:159` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:31` |
| `WEAVE_OVERLAP_PIPELINE` | 2 | 0/1 | 1 (0 at preset >=2) | Temporal overlap across dispatch, GEMM, scatter, and combine. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:149` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:32` |
| `WEAVE_SELECTIVE_DEDUP` | 2 | 0/1 | 1 (0 at preset 3) | Unique (token,destination) communication buckets. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:199` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:33` |
| `WEAVE_ALL_WAVE_COMBINE` | 2 | 0/1 | 1 (0 at preset >=2) | Use all available waves for cooperative tail combine. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:179` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:34` |
| `WEAVE_TAIL_FINE_GRAINED` | 2 | 0/1 | 1 (0 at preset >=2) | Fine-grained tail reduction instead of a monolithic tail. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:249` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:373` |

## 2. NVIDIA B300 (SM100/SM103)

| Parameter | Tier | Values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_READY_GROUP_SIZE` | 1 | 0/1/2/4/8/16 | 0 | Ready-group granularity for expert publication. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:129` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:7` |
| `WEAVE_WARP_SPLIT_MODE` | 1 | 0..15 | 0 | Dispatch warp reconfiguration and register split mode. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:131` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:23` |
| `WEAVE_REG_PULL_BYTES` | 1 | 0/1024/2048/4096 | 0 | Register-direct pull width; 0 uses SMEM staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:132` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:39` |
| `WEAVE_HETERO_SM_LIMIT` | 1 | 0..148 | 74 | SM-id limit for the heterogeneous role split. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:133` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:55` |
| `WEAVE_DISP_REGS` | 1 | register count | 0 | Explicit dispatch-warp register budget. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:134` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:66` |
| `WEAVE_EPI_REGS` | 1 | register count | 0 | Explicit epilogue-warp register budget. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:135` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:77` |
| `WEAVE_BLOCK_M` | 2 | 32/64/96/128/192 | shape heuristic | GEMM tile M override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:137` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:88` |
| `WEAVE_BLOCK_N` | 2 | 64/128/256 | shape heuristic | GEMM tile N override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:138` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:104` |
| `WEAVE_BLOCK_K` | 2 | 64/128/256 | shape heuristic | GEMM tile K override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:139` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:120` |
| `WEAVE_STORE_BLOCK_M` | 2 | 4/8/16/32 | shape heuristic | TMA-store M granularity. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:140` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:136` |
| `WEAVE_DYN_EPILOGUE` | 1 | 0/1/2 | 0 | Dynamic epilogue pass count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:141` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:152` |
| `WEAVE_DYN_TMA_A` | 1 | 0/1/2 | 0 | Dynamic TMA-A descriptor for small M. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:142` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:168` |
| `WEAVE_OVERLAP_EXPERT_SMEM` | 1 | 0/1 | 0 | Overlap expert metadata SMEM with the next tile. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:143` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:184` |
| `WEAVE_EXPERT_STEP_MODE` | 1 | 0/1/2 | 0 | Expert-step scheduling mode. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:144` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:200` |
| `WEAVE_ACCUM_STAGES` | 1 | 2/3/4 | 2 | TMEM accumulator ring depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:145` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:216` |
| `WEAVE_TMA_STORE_STAGES` | 1 | 2..6 | 2 | TMA-store ring depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:227` |
| `WEAVE_SCHED_STAGES` | 1 | 2/3 | 2 | Scheduler look-ahead depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:147` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:238` |
| `WEAVE_EXPERTS_PER_WAVE` | 2 | 1/2/4/8/16/32/64 | shape heuristic | Experts assigned to one scheduling wave. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:149` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:254` |
| `WEAVE_STATIC_EXPERT_ORDER` | 2 | 0/1 | 0 | Use a static expert-order table. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:150` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:270` |
| `WEAVE_READY_QUEUE` | 2 | 0/1 | 0 | Enable the dynamic ready queue. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:151` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:286` |
| `WEAVE_SCHED_MODE` | 2 | 0..3 | 0 | Scheduler mode selector. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:152` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:302` |
| `WEAVE_DEDUP_MODE` | 2 | 0/1/2 | 0 | Dedup mode: off, dedup, local pre-reduce. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:155` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:318` |
| `WEAVE_DEDUP_SCHED` | 2 | 0/1 | 1 when dedup=2 | Dedup scheduler toggle. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:156` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:334` |
| `WEAVE_COMBINE_GROUP_TILES` | 2 | 1/2/4/8/16 | 0 | Combine group tile count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:158` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:350` |
| `WEAVE_COMBINE_REDUCE_WARPS` | 2 | 0/1/2/4/8/16 | 0 (all epilogue warps) | Static combine-warp cap. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:159` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:366` |
| `WEAVE_SCATTER_SMEM_ALIAS` | 1 | 0/1 | 1 | Alias scatter staging SMEM with epilogue storage. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:163` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:393` |
| `WEAVE_SCATTER_WARPS` | 2 | 0/2/4/8 | 0 (auto) | Scatter executor warp count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:164` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:404` |
| `WEAVE_SCATTER_HANDOFF_WAIT` | 2 | 0/1 | 1 | Wait for the combine-ready epoch before handoff. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:165` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:420` |
| `WEAVE_SHARED_L2_EARLY_BLOCKS` | 1 | 0..148 | 10 | Early blocks used to seed shared-L2 reuse. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:203` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:431` |
| `WEAVE_ENABLE_MULTICAST` | 1 | 1 | 1 | Two-CTA TMA multicast required by the B300 FP8xFP4 kernel. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:166` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:442` |
| `WEAVE_SCATTER_MODE` | 3 | 0/1/2 | 0 | Scatter path override for the SM100 dataflow. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:161` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:382` |

## 3. NVIDIA H20 (SM90)

| Parameter | Tier | Values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_SWAPAB_AUTO` | 2 | 0/1 | 0 | Enable padding-aware automatic swap-AB selection. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:654` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:654` |
| `WEAVE_SWAPAB_EPI_WG` | 2 | 2/4 | 2 | Epilogue warpgroup count under swap-AB. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:650` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:650` |
| `WEAVE_SWAPAB_MAX_LOAD` | 2 | int | shape heuristic | Upper load bound for swap-AB. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:669` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:669` |
| `WEAVE_SWAPAB_PAD_PCT` | 2 | 0..100 | 25 | Padding-fraction gate for swap-AB. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:639` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:639` |
| `WEAVE_SWAPAB_MIN_LOAD` | 2 | int | 8 | Lower load bound for swap-AB. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:640` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:640` |
| `WEAVE_MCAST` | 1 | 0/1/2 | 0 | SM90 multicast mode. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:229` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:7` |
| `WEAVE_SWIZZLE_TILES` | 2 | 0/1 | 0 | Enable tile swizzling across expert tiles. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:230` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:18` |
| `WEAVE_HOT_COLD_MODE` | 2 | 0/1/2 | 0 | Hot/cold expert weight residency policy. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:231` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:29` |
| `WEAVE_DEDUP_DISPATCH` | 2 | 0/1 | 0 | Deduplicate dispatch rows. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:232` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:40` |
| `WEAVE_DEDUP_SCATTER` | 2 | 0/1 | 0 | Deduplicate scatter destinations. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:233` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:51` |
| `WEAVE_REG_COMBINE` | 1 | 0/1/2 | 0 | Register-direct combine path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:234` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:62` |
| `WEAVE_ALL_IDLE_COMBINE` | 2 | 0/1 | 0 | Use idle warps for the combine phase. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:235` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:73` |
| `WEAVE_DYN_SCHED` | 2 | 0/1 | 0 | Dynamic expert scheduling. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:236` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:84` |
| `WEAVE_INTERLEAVE_L1_L2` | 2 | 0..6 | 0 | Interleave layer-1 and layer-2 expert work. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:237` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:95` |
| `WEAVE_SCATTER_ROLE_DISP` | 2 | 0/1 | 0 | Assign scatter to dispatch warps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:238` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:106` |
| `WEAVE_HETERO_MASK_MODE` | 1 | 0..5 | 0 | Heterogeneous SM role mask mode. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:239` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:117` |
| `WEAVE_PIPE_WGMMA` | 1 | 0/1 | 0 | Pipeline WGMMA issue. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:240` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:128` |
| `WEAVE_SWAP_PIPE` | 1 | 0/1 | 0 | Swap producer/consumer pipeline roles. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:241` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:139` |
| `WEAVE_FAST_PULL` | 1 | 0/1 | 0 | Fast register pull path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:242` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:150` |
| `WEAVE_ADAPT_BM` | 2 | 0/1 | 0 | Per-expert adaptive block-M tiers. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:243` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:161` |

## 4. AMD MI308X (gfx942)

| Parameter | Tier | Values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_BM_TIER_MODE` | 2 | 0/2 | 2 | Five-tier block-M planner. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:105` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:71` |
| `WEAVE_DUALGROUP_MIN_TOKENS` | 2 | uint32 | 64 | Threshold for rebinding communication waves into a second compute group. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1159` |
| `WEAVE_BN2_MODE` | 2 | 0/64/128 | 0 | Second-stage BN mode. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:110` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:900` |
| `WEAVE_S2_NGROUP_MODE` | 2 | 0/1/2/4 | 0 | Second-stage N-group width. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:111` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:908` |
| `WEAVE_TH_BM64` | 2 | uint32 | 16 | Small-M threshold for BM64. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:113` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:544` |
| `WEAVE_TH_BM128` | 2 | uint32 | 96 | Small-M threshold for BM128. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:114` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:547` |
| `WEAVE_PARALLEL_ROUTE_PLAN` | 2 | 0/1 | 1 | All-CTA parallel route histogram and prefix. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:112` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `WEAVE_WAVE_ROLES` | 1 | struct | roles={80,0,80} | Wave-role descriptor remapped by CTA-0 from selected_r_mask. | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:397` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:397` |
| `WEAVE_SLOT_CHUNK` | 2 | 1/2/4/8/16/32/64 | adaptive | Dispatch descriptor slot granularity. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:170` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:54` |
| `WEAVE_RET_CHUNK` | 2 | 1/2/4/8/16 | derived | Return descriptor slot granularity. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:235` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:303` |
| `WEAVE_META_CHUNK_SLOTS` | 2 | >=1 | max(256,t_local) | Metadata slots per RDMA write descriptor. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:234` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:55` |
| `WEAVE_PROFILER_LEVEL` | 1 | 0/1/2 | 0 | Instrumentation level; bridged to FI_LEVEL for the HIP profiler. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:97` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:31` |
| `WEAVE_ABLATION_PRESET` | 1 | 0/1/2/3 | 0 | EP16 build-time ablation selector; preset 3 also forces scatter mode 0. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:93` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1154` |
| `WEAVE_SCATTER_MODE` | 3 | 0/1/2 | 2 (0 at preset 3) | EP16 scatter and return mode; mode 2 enables the node-partial path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:103` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:50` |
| `WEAVE_SELECTIVE_DEDUP` | 2 | 1 | 1 | Selective input dedup required by the EP16 path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:106` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:511` |
| `WEAVE_OVERLAP_PIPELINE` | 2 | 0/1 | 1 | Enable the EP16 overlap pipeline. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:108` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:709` |
| `WEAVE_ALL_WAVE_COMBINE` | 2 | 0/1 | 1 | Use all waves for the cooperative tail combine. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |

## 5. External CK dependency

The external AITER/CK tree is not vendored. AMD builds receive its include roots
through WEAVE_AITER_SITE. Small overlay headers remain under amd/third_party_overlay/.
