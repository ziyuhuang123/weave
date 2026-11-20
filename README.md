[English](README.md) | [简体中文](README_zh.md)

# Weave: Routing-Adaptive Runtime-Dynamic MoE Megakernel

Weave is a **routing-adaptive, runtime-dynamic MoE megakernel** implemented in a
unified C++/CUDA/HIP codebase. It is the artifact of
**Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for
Compute-Communication Overlap** ([arXiv:2609.21483](https://arxiv.org/abs/2609.21483)).
One scheduling contract spans **NVIDIA SM103 (148-SM)**, **NVIDIA H20**, and
**AMD MI308X**: the same kernel reads the live post-dispatch routing histogram
and changes its spatial split, scatter ownership, and opening head-pull budget
at runtime, inside the kernel, without a host round trip.

## Architecture

| Hardware | Language | Supported precision values | Entry point |
|---|---|---|---|
| NVIDIA SM103 (148-SM) | CUDA C++ | `0 = fp8xfp4` (default), `1 = fp4xfp4` (NVFP4), `3 = bf16`; `2 = fp8` is not supported | sm100_fp8_fp4_mega_moe.cuh / sm100_fp4_fp4_mega_moe.cuh via weave_mega_moe(..., mma_type=...) |
| NVIDIA H20 (SM90) | CUDA C++ | `2 = fp8` (fp8xfp8) only | sm90_fp8_mega_moe.cuh |
| AMD MI308X (gfx942) | HIP C++ | `2 = fp8` (fp8xfp8) only; EP8 and EP16 | gfx942_fp8_mega_moe.cuh and amd/megamoe_amd.hip |

The kernel uses a single device-side publication rail: CTA 0 reads the live
post-dispatch routing histogram, evaluates the cost model, publishes the
selected schedule with a release store, and every CTA acquires it with
`load_published_policy` or `wait_and_copy_policy`. At runtime, the scheduler
adapts the comm/compute warp split, scatter executor placement (epilogue inline
vs. async comm queue), and opening head-pull prefetch budget—with no host round
trip and no dynamic parallel launch.

## Partitioned hardware execution and global ready queues

Each SM/CU is partitioned into a **Communication Partition (`N_c`)** and a
**Compute/GEMM Partition (`N_e`)**. Each partition is backed by its own
**global ready queue**. Queue indexing is lightweight and implicit: arrival
counters, per-expert group bitmaps, and prefix sums replace expensive runtime
sorting.

![Weave Design Overview: Routing + Hardware Profile -> Cost Model -> Spatial Warp Ratio & Temporal Warp Roles](docs/figures/weave_design_overview.png)

![Weave Spatial Partitioning & Scatter Ownership vs MegaMoE](docs/figures/weave_design_rolegraph.png)

![Weave Temporal Scheduling: Register-Specialized All-Warp Head Pull & Fine-Grained Per-Token Tail Combine](docs/figures/weave_design_temporal.png)

- **GEMM Ready Queue** (consumed by the Compute Partition `N_e`): receives
  ready `(expert, m_block)` tiles as tokens arrive through the
  `LOCAL_READY -> REMOTE_PRIMARY -> LOCAL_COPY` flow. Entries are traversed in
  a deterministic global order (with L2 tile-swizzle / hot-cold window order
  parameterized at startup), or popped from the global FIFO when
  `WEAVE_READY_QUEUE=1`. When `s = epi` (`WEAVE_SCATTER_MODE=0`), Scatter
  executes inline inside the Compute/GEMM partition (epilogue).
- **Communication Ready Queue** (consumed by the Communication Partition
  `N_c`): drives remote Dispatch/Pull. When `s = comm` (`WEAVE_SCATTER_MODE=1`)
  or local-dedup (`WEAVE_SCATTER_MODE=2`), it receives ready FC2 scatter
  descriptors pushed by epilogue/local-combine warps, so `N_c` overlaps remote
  NVLink/xGMI scatter concurrently with ongoing GEMM tiles.

## Quick start

    git clone https://github.com/ziyuhuang123/weave.git
    cd weave
    python -m pip install -e .
    python -m py_compile baselines/*.py ablation/*.py
    python tests/verify_tunable_params_doc.py

Minimal Python dispatch:

    import torch
    from deep_gemm.mega import SymmBuffer, weave_mega_moe

    # Construct the symmetric buffer for your process group and model shape,
    # then call weave_mega_moe(y, l1_weights, l2_weights, sym_buffer,
    #                        mma_type="fp4xfp4").
    # NVIDIA SM103 (148-SM) supports fp8xfp4, fp4xfp4 and bf16;
    # NVIDIA H20 and AMD MI308X use fp8.

The public precision selector can also be set before import or before the
call:

    export WEAVE_PRECISION_MODE=fp4xfp4

## Measured performance

![3-Hardware End-to-End Speedup across 6 Models x 5 Batch Sizes](docs/figures/weave_eval_3hw_speedup.png)

Paper Fig. 7: measured 3-hardware MoE-layer speedup across 6 models x 5 batch
sizes.

![Bound Hardware Resource Utilization % across 6 Models x 5 Batch Sizes x 3 Hardware Platforms](docs/figures/weave_bound_utilization_3hw.png)

Bound hardware resource utilization: each point is the measured peak
utilization of the resource that actually bounds that cell (HBM, NVLink/xGMI,
or compute). Small models such as `Qwen3.5-35B` (`H=2048, I=512`) and
`Qwen3.5-122B` at `S=16..1024` across 8-GPU EP8 are strictly latency-floor
dominated (their entire ideal HBM/NVLink time is only `2-10 us` on 148-SM SM103
versus the `~35 us` 8-GPU barrier/launch floor), and climb back to `79.4%`
(SM103) / `60.8%` (H20) once `S=16K` amortizes the fixed floor. On 8-GPU-scale
models (`Qwen3.8-2.4T`, `MiniMax-M3`), Weave reaches up to `97.6%` on 148-SM
SM103 and `85.5%` on H20; at `S=1024` on H20, `Qwen3.8-2.4T` is HBM3-bound
(`T_hbm=755 us` vs `T_comp=437 us`), which physically caps FP8 Tensor Core
activity below `58%`.

All numbers below are measured on real hardware. Every TFLOPS value is a
single-GPU value, never a cluster aggregate. Full-output correctness is checked
on every rank before timing.

| Hardware / scale | Precision | Full-output correctness | Measured latency and throughput | Verdict |
|---|---|---|---|---|
| 8x NVIDIA SM103 (148-SM) installation; 4-GPU admissible verification window | fp4xfp4 (NVFP4) | EP4, S=16, H=6144, I=2048, E=128, K=8; 4/4 ranks finite, unified-vs-direct bit-exact, 512 routes; 4/4 full-FP32-reference rounds PASS; live CTA-0 readback magic 0x57415645, version 4, published_epoch=2, valid=1 | median-of-rank-medians 172.384 us; worst rank 172.480 us; 54.418 TFLOPS/GPU nominal (52.088--60.001). Pre-rail reference 143.552 us and 65.757 TFLOPS/GPU | PASS |
| Same 148-SM SM103 installation | fp8xfp4 | EP1 built the reference and printed the expected config; the first kernel launch did not return. No output or latency is claimed. Fresh clone reproduced the hang; compute-sanitizer memcheck found no invalid access before the hang | not measured | BLOCKED_RUNTIME_HANG |
| 8x NVIDIA H20 EP8 | fp8 | Qwen3.8-Max, E=512, K=10, H=8192, I=2048; worst-rank calc_diff 5.924e-4--5.948e-4, worst rel_l2 0.034483, cos_sim 0.999405; all ranks PASS | S=128 AUTO=1 1452.512 us versus AUTO=0 1942.784 us, a -25.24% latency change; S=1024 AUTO=1 4977.584 us versus AUTO=0 4981.344 us | PASS |
| 8x AMD MI308X EP8 | fp8 | 4/4 full-output cells PASS across all eight ranks; max rel_l2 0.001545--0.001908, max row rel_l2 0.001952--0.005731, device_error=0 | qwen38max S=16 687.422/713.942 us; S=128 1539.625/1554.185 us; DeepSeek-V4-Flash S=16 378.761/395.421 us; S=128 699.642/709.743 us; single-GPU TFLOPS 1.595--9.219 | PASS |
| 16x AMD MI308X EP16 (2 nodes x 8) | fp8 | 16/16 ranks full-element FP64 comparison PASS; qwen38max S=128 and DeepSeek-V4-Flash S=128; max rel_l2 0.012535/0.012292, max row rel_l2 0.019612/0.015528; all 96 launch-state records device_error=0 | qwen38max metric1 1594.079 -> 1010.679 us (1.5772x), metric2 1591.939 -> 1006.619 us (1.5815x); DeepSeek-V4-Flash metric1 1387.055 -> 735.300 us (1.8864x), metric2 1367.269 -> 732.679 us (1.8661x) | PASS |

The 148-SM SM103 fp8xfp4 path compiles and both 148-SM SM103 kernels carry the CTA-0 policy
rail, but the FP8xFP4 launch still waits in the FP8 GEMM transaction-barrier
protocol. Removing the policy rail, clamping the rank order, deleting the pull
wait, and changing the block shape all reproduced the wait. This is an open
runtime investigation, not a passing result. The NVFP4 sibling on the same
source line passes end-to-end with live policy readback.

## WeaveProfiler software + hardware timelines

![NVIDIA SM103 (148-SM) WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_sm103_timeline.png)

![NVIDIA H20 WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_h20_timeline.png)

![AMD MI308X WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_mi308x_timeline.png)

All three panels use the unified WeaveProfiler contract on the matched
`Qwen3.8-Max, EP8, S=1024` workload: hardware counters, all physical SM/CU
lanes, and a representative CTA pipeline. The exact tuned `WEAVE_*` presets
(organized by **Spatial**, **Temporal**, and **Implementation**) and
reproducible launch scripts live under [examples/](examples/README.md):

- NVIDIA SM103 (148-SM): [examples/sm103_qwen38max_ep8_s1024.sh](examples/sm103_qwen38max_ep8_s1024.sh)
- NVIDIA H20: [examples/h20_qwen38max_ep8_s1024.sh](examples/h20_qwen38max_ep8_s1024.sh)
- AMD MI308X: [examples/mi308x_qwen38max_ep8_s1024.sh](examples/mi308x_qwen38max_ep8_s1024.sh) (EP8) and [examples/mi308x_qwen38max_ep16_s128.sh](examples/mi308x_qwen38max_ep16_s128.sh) (2-node EP16)
- One-command runner & renderer: `python examples/profile_and_render_timeline.py --preset sm103` (supports `sm103`, `h20`, `mi308x_ep8`, `mi308x_ep16`, plus `--run` / `--render`)

Values come from the checked-in measured paired traces; missing counters remain
unavailable rather than being zero-filled. MI308X hardware counters are
per-dispatch aggregates; its trace explicitly records a window mismatch and
does not claim absolute clock alignment.

## Parameter taxonomy: Spatial, Temporal, and Implementation

Weave's scheduling and kernel parameters fall into three categories:

- **Spatial dimension:** how compute, communication, and tile
  resources are partitioned across SMs/CUs, warps, clusters, and numeric
  formats—including the comm/compute warp split, heterogeneous SM role masks,
  2-CTA multicast, numeric precision (`fp8xfp4` / `fp4xfp4` / `fp8` / `bf16`),
  and **runtime per-expert tile sizing** from live per-expert token counts
  (MI308X five-tier `BM_e in {16, 32, 64, 128, 256}` with EP16 floored at 32;
  H20 8-tier swapAB `{8..64}` gated by padding waste; 148-SM SM103 dynamic
  TMA-A `m8/m16/BM`, dynamic epilogue `store_m8/m16`, and per-expert tile
  ladders).
- **Temporal dimension:** execution order, compute-communication
  overlap, scatter ownership (epilogue inline vs. async comm queue vs. local
  pre-reduce dedup), opening head-pull prefetch budget, ready-queue / hot-cold
  / L2-swizzled ordering, L1/L2 interleaving, cooperative all-wave tail
  combine, and **runtime-derived pipeline stage depth** (`num_stages` is
  computed at launch from the tile shape and remaining SMEM budget rather than
  hardcoded at compile time).
- **Implementation & prior work:** standard GPU/MoE systems techniques from
  prior work (register-direct pull/combine, selective deduplication, RDMA
  descriptor chunking, SMEM aliasing, profiler instrumentation, and ablation
  presets).

At runtime, CTA 0 ranks candidate schedules with the lightweight on-device cost
model: shared fixed overheads cancel in the ranking, matching the exhaustive
oracle on 38/48 measured 148-SM SM103 cells (1.301x geomean speedup vs. 1.332x
oracle, 2.4% selection regret; 34.0% mean absolute latency error).

Every file:line citation below is checked by
`tests/verify_tunable_params_doc.py`. The full reference is in
[docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md).

## Unified core WEAVE_* parameters

| Parameter | Level | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | 1 | `0 = fp8xfp4` (MXFP8 activations x MXFP4 weights); `1 = fp4xfp4` (NVFP4 activations and weights); `2 = fp8` (fp8xfp8); `3 = bf16` | 0 (fp8xfp4) | Numeric precision selector and the source of every byte width and tensor-core rate in the cost model. Support is per hardware and NOT identical: NVIDIA SM103 (148-SM) accepts `0` (default), `1` and `3` and is the only platform with FP4; NVIDIA H20 (SM90) accepts `2` only; AMD MI308X (gfx942) accepts `2` only. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:269` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_ABLATION_PRESET` | 1 | `0 = Weave` (full: CTA-0 spatial search + temporal overlap + selective dedup + fine-grained tail); `1 = -S` (fixed per-architecture spatial split; overlap and dedup stay on); `2 = -S-T` (fixed split; temporal overlap off; dedup on); `3 = -S-T-C` (fixed split; overlap off; selective dedup and cooperative tail combine off) | 0 | Compile-time progressive ablation. The preset also flips defaults: ENABLE_COST_MODEL is 1 only at 0; OVERLAP_PIPELINE becomes 0 at preset >= 2; SELECTIVE_DEDUP becomes 0 at preset 3; ALL_WAVE_COMBINE and TAIL_FINE_GRAINED become 0 at preset >= 2. On MI308X EP16, preset 3 also forces SCATTER_MODE to 0. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:26` |
| `WEAVE_ENABLE_COST_MODEL` | 3 | `0 = off`: every CTA uses the fixed Level-1 candidate or an explicit override; `1 = on`: CTA 0 evaluates Eq. (1)-(3) on the live post-dispatch histogram and publishes (r, s, V_head) | 1 at preset 0; 0 at presets 1-3 | Master switch for the CTA-0 runtime selector. When off, the spatial split, scatter ownership and head budget come from compile-time defaults or an override and no histogram scan is performed. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:49` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:27` |
| `WEAVE_COMM_UNITS` | 1 | `N = compile-time default N_c communication ratio (per SM/CU) used when the runtime search is disabled`; `0 = compute-only split with no dedicated communication unit` | 4 | Default N_c written into the r_mask when the cost model is off. The cost model treats candidate comm/comp units as per-SM/CU ratios and multiplies them by the SM/CU count when it applies the link-bandwidth caps. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:59` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:28` |
| `WEAVE_COMP_UNITS` | 1 | `N = compile-time default N_e compute units`; `0 = no compute unit (degenerate, not a valid standalone schedule)` | 4 | Default N_e written into the r_mask when the cost model is off. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:69` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:29` |
| `WEAVE_V_HEAD` | 3 | `0xffffffff = auto`: the opening head-pull budget is computed by Eq. (3); `0 = disabled`: no opening head-pull; `V = explicit override`: pull exactly the first V distinct remote tokens at peak link bandwidth | auto (0xffffffff) | Opening Head-Pull Token Budget (the paper's V_head). At layer start, while GEMM has no ready tile yet, all available communication warps/waves pull the first V_head tokens at peak link bandwidth to eliminate the startup bubble; the pipeline then transitions to the steady-state N_c warps overlapping with compute. Auto value = align_up(min(num_SMs x N_c x head_tasks_per_comm_unit, D_in), 32), where D_in is the live deduplicated remote-pull count. Setting an integer overrides it. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:259` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_SCATTER_MODE` | 3 | `0 = epilogue-inline (s = epi)`: GEMM epilogue warps directly issue NVLink/xGMI scatter inside the Compute/GEMM partition (zero handoff overhead; optimal for small batches); `1 = comm-offloaded (s = comm)`: epilogue warps write FC2 tile outputs through SMEM/async handoff into the Communication partition Ready Queue, and communication warps drain NVLink/xGMI scatter asynchronously in parallel with subsequent GEMM tiles; `2 = local-dedup / node-partial pre-reduced`: when one token routes to multiple experts on the same GPU (or the same 8-GPU node in EP16), epilogue/local-combine warps first reduce and deduplicate those partial contributions into one merged HBM buffer, then enqueue one deduplicated transfer into the Communication partition Ready Queue so comm warps transmit once per destination token | 1 (148-SM SM103 dataflow 0; MI308X EP16 2) | Selects who owns the scatter step and which ready queue carries it. It is also one of the three routing-dependent decisions selected by CTA 0 (s). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:159` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:31` |
| `WEAVE_OVERLAP_PIPELINE` | 2 | `0 = disabled`: ablation toggle that forces bulk-synchronous stage-by-stage serialization (used by -S-T / -S-T-C); `1 = enabled` (default): compute-communication overlap is ON | 1 | Temporal overlap across dispatch, GEMM, scatter and combine. Default = 1: compute-communication overlap is ON by default. Setting 0 is the ablation toggle that forces stage-by-stage bulk-synchronous execution. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:149` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:32` |
| `WEAVE_SELECTIVE_DEDUP` | 2 | `0 = off`: one communication bucket per (token, expert) route; `1 = on`: unique (token, destination) buckets, so a token routed to several experts of the same destination rank is transmitted once | 1 (0 at preset 3) | Selective input dedup for the communication stage. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:199` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:33` |
| `WEAVE_ALL_WAVE_COMBINE` | 2 | `0 = only the compute-group epilogue warps perform the tail combine`; `1 = all otherwise-idle waves (dispatch and epilogue) join the cooperative tail combine` | 1 (0 at preset >= 2) | Width of the cooperative tail combine. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:179` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:34` |
| `WEAVE_TAIL_FINE_GRAINED` | 2 | `0 = monolithic tail`: a single coarse tail-reduction term; `1 = fine-grained tail`: bandwidth-bound reduction overlapped with the ready queue | 1 (0 at preset >= 2) | Fine-grained tail reduction instead of a monolithic tail. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:249` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:373` |

## NVIDIA SM103 (148-SM) WEAVE_* parameters

| Parameter | Level | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_READY_GROUP_SIZE` | 1 | `0 = expert-level readiness`: a pool tile is published only after the whole expert has arrived; `1, 2, 4, 8, 16 = group granularity`: publish every N x BLOCK_M tokens of an expert, in order | 0 | Fine-grained arrival publication. Larger groups trade earlier availability for fewer metadata updates. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:129` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:7` |
| `WEAVE_WARP_SPLIT_MODE` | 1 | `0 = baseline`: no warp split, 4 dispatch warps, default epilogue count; `1 = split on`: D2 pull loop on, 4 dispatch warps, 8 epilogue warps; `2 = 8 dispatch warps on every SM`; `3 = 8 dispatch warps on even physical SMIDs, 4 elsewhere`; `4 = 8 dispatch warps on the lower half of SMIDs, 4 elsewhere`; `5 = 8 warps + split hybrid pull (warps 0-3 TMA, warps 4-7 register-direct), all SMs`; `6 = mode 5 with the even-SMID mask`; `7 = mode 5 with the lower-half mask`; `8 = mode 5 with the ((smid >> 1) & 1) == 0 mask`; `9, 10 = reserved, behave as 1`; `11 = 8 dispatch warps, default epilogue count`; `12 = 8 warps on even SMIDs, default epilogue count`; `13 = 8 warps on the lower half, default epilogue count`; `14 = 8 warps + split hybrid pull, default epilogue count`; `15 = mode 14 with the ((smid >> 1) & 1) == 0 mask` | 0 | Dispatch-warp reconfiguration and register split. Modes 2-4 force 8 dispatch warps and a 2048-byte pull chunk; modes 5-8 and 14 use the split hybrid pull; modes 3, 4, 6, 7, 8, 12, 13 and 15 use physical SMID masks so only a subgroup runs the 8-warp front end. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:131` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:23` |
| `WEAVE_REG_PULL_BYTES` | 1 | `0 = SMEM-staged TMA pull`; `1024, 2048, 4096 = zero-SMEM register-direct pull, N bytes per warp iteration` | 0 | Register-direct pull width. A non-zero value activates the zero-SMEM pull path and disables TMA dispatch staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:132` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:39` |
| `WEAVE_HETERO_SM_LIMIT` | 1 | `N in 0..148 = physical SMID threshold`: SMIDs below N join the 8-warp subgroup; `0 = no SM qualifies, so the 8-warp subgroup is disabled` | 74 | SM-id limit for the heterogeneous role split. It takes effect only in the SMID-masked warp-split modes that use the threshold form. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:133` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:55` |
| `WEAVE_DISP_REGS` | 1 | `0 = automatic register budget`; `N = explicit maximum registers per dispatch warp` | 0 | Explicit dispatch-warp register budget applied with setmaxnreg. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:134` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:66` |
| `WEAVE_EPI_REGS` | 1 | `0 = automatic register budget`; `N = explicit maximum registers per epilogue warp` | 0 | Explicit epilogue-warp register budget applied with setmaxnreg. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:135` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:77` |
| `WEAVE_BLOCK_M` | 2 | `0 = shape heuristic`; `32, 64, 96, 128, 192 = explicit GEMM tile M` | shape heuristic | GEMM tile M override. The value must be one of the legal candidate tile heights. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:137` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:88` |
| `WEAVE_BLOCK_N` | 2 | `0 = automatic (128)`; `64, 128, 256 = explicit GEMM tile N` | shape heuristic | GEMM tile N override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:138` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:104` |
| `WEAVE_BLOCK_K` | 2 | `0 = automatic`; `64, 128, 256 = explicit GEMM tile K` | shape heuristic | GEMM tile K override. It must divide the layer reduction dimensions. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:139` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:120` |
| `WEAVE_STORE_BLOCK_M` | 2 | `0 = automatic`; `4, 8, 16, 32 = explicit TMA-store M granularity` | shape heuristic | TMA-store M granularity. DYN_EPILOGUE = 2 forces this value to 16 and uses two 8-row atoms per store. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:140` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:136` |
| `WEAVE_DYN_EPILOGUE` | 1 | `0 = static epilogue`: one fixed full-tile pass per warpgroup; `1 = runtime pass count`: ceil(valid_m / STORE_BLOCK_M) blocks are distributed across epilogue warpgroups from the live valid_m; `2 = atom-granular store`: two 8-row atoms per 16-row store plus an m8 tail descriptor | 0 | Dynamic epilogue pass count / store granularity. Mode 2 is the store_m8 / store_m16 dynamic epilogue and requires STORE_BLOCK_M = 16. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:141` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:152` |
| `WEAVE_DYN_TMA_A` | 1 | `0 = always load LOAD_BLOCK_M rows`; `1 = switch to a 16-row TMA-A descriptor when aligned_valid_m <= 32`; `2 = 8-row descriptor when aligned_valid_m <= 16, 16-row when <= 32` | 0 | Dynamic TMA-A descriptor for small M. Modes 1 and 2 remove the wasted A-tile traffic of the final partial tile. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:142` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:168` |
| `WEAVE_OVERLAP_EXPERT_SMEM` | 1 | `0 = expert metadata keeps its own SMEM phase`; `1 = overlay expert metadata SMEM with the next tile load` | 0 | Overlap expert metadata SMEM with the next tile. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:143` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:184` |
| `WEAVE_EXPERT_STEP_MODE` | 1 | `0 = fixed BLOCK_M for every m-block of an expert`; `1 = runtime per-expert BM ladder`: choose the smallest of 32, 64, 128, 256 whose padded token count does not exceed the 32-row padded count (plus optional pad slack); `2 = accepted alias of 1` | 0 | Expert-step scheduling mode. Any value > 0 enables select_expert_bm at runtime from the live per-expert token count; it additionally reserves more scheduler registers. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:144` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:200` |
| `WEAVE_ACCUM_STAGES` | 1 | `2 = two TMEM accumulator ring slots`; `3 = three slots`; `4 = four slots` | 2 | TMEM accumulator ring depth. Deeper rings allow more epilogue overlap at the cost of TMEM columns. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:145` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:216` |
| `WEAVE_TMA_STORE_STAGES` | 1 | `2, 3, 4, 5, 6 = TMA-store ring depth in stages` | 2 | TMA-store ring depth. It must fit the epilogue output SMEM budget. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:227` |
| `WEAVE_SCHED_STAGES` | 1 | `2 = two-slot scheduler look-ahead`; `3 = three-slot look-ahead` | 2 | Scheduler look-ahead depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:147` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:238` |
| `WEAVE_EXPERTS_PER_WAVE` | 2 | `0 = natural expert order, no window`; `1, 2, 4, 8, 16, 32, 64 = scheduling window of N experts`: within each window experts are sorted by descending m-block count (LPT) | 0 (natural order) | Experts assigned to one scheduling wave. Windowed LPT ordering starts the largest experts first while keeping the order local. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:149` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:254` |
| `WEAVE_STATIC_EXPERT_ORDER` | 2 | `0 = derive the expert order online`; `1 = load a per-rank expert order table from WEAVE_STATIC_EXPERT_ORDER_FILE (ignored when the file is absent)` | 0 | Use a static expert-order table. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:150` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:270` |
| `WEAVE_READY_QUEUE` | 2 | `0 = static prefix-summed block order`: every CTA derives its (expert, m_block) from the prefix sums; `1 = global FIFO ready queue`: dispatch pushes a descriptor when a pool tile becomes fully arrived and scheduler CTAs pop entries in arrival order | 0 | Enable the dynamic ready queue. The queue gives the GEMM partition live (expert, m_block) readiness without runtime sorting. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:151` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:286` |
| `WEAVE_SCHED_MODE` | 2 | `0 = descending m-block count, token-count tiebreak, reverse expert-index final tiebreak`; `1 = descending m-block count only, reverse expert-index tiebreak (cheaper key)`; `2 = mode 0 but the last (possibly partial) expert window stays in sorted order instead of hot/cold pairing`; `3 = reserved, currently identical to 0` | 0 | Scheduler ordering mode selector. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:152` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:302` |
| `WEAVE_DEDUP_MODE` | 2 | `0 = off`; `1 = dedup`: unique (token, destination) communication buckets; `2 = local pre-reduce`: dedup plus same-rank partial reduction before communication (4 or 8 ranks only; requires E/rank >= 18, topk <= 10, H % 512 == 0 and pull bytes % 1024 == 0) | 0 | Dedup mode: off, dedup, or local pre-reduce. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:155` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:318` |
| `WEAVE_DEDUP_SCHED` | 2 | `0 = dedup scheduler off`; `1 = dedup scheduler on` (the default when DEDUP_MODE = 2) | 1 when dedup=2, otherwise 0 | Dedup scheduler toggle. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:156` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:334` |
| `WEAVE_COMBINE_GROUP_TILES` | 2 | `0 = off`; `1, 2, 4, 8, 16 = group N consecutive L2 output tiles for one cooperative combine` (must be a power of two dividing L2_SHAPE_N / BLOCK_N) | 0 | Combine group tile count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:158` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:350` |
| `WEAVE_COMBINE_REDUCE_WARPS` | 2 | `0 = all epilogue warps`; `1, 2, 4, 8, 16 = cap the combine to N warps` | 0 (all epilogue warps) | Static combine-warp cap. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:159` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:366` |
| `WEAVE_SCATTER_SMEM_ALIAS` | 1 | `0 = dedicated scatter staging SMEM`; `1 = alias scatter staging with the epilogue store buffer` | 1 | Alias scatter staging SMEM with epilogue storage. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:163` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:393` |
| `WEAVE_SCATTER_WARPS` | 2 | `0 = automatic`: use the configured dispatch warps; `2, 4, 8 = N dedicated scatter executor warps` (also sets the dispatch thread count to 32N) | 0 (auto) | Scatter executor warp count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:164` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:404` |
| `WEAVE_SCATTER_HANDOFF_WAIT` | 2 | `0 = hand off immediately after the epilogue writes`; `1 = wait until the combine-ready epoch / pull-done flag before handoff` | 1 | Wait for the combine-ready epoch before handoff. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:165` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:420` |
| `WEAVE_SHARED_L2_EARLY_BLOCKS` | 1 | `0 = no early seeding`; `N in 1..148 = the first N CTAs seed shared-L2 reuse` | 10 | Early blocks used to seed shared-L2 reuse. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:203` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:431` |
| `WEAVE_ENABLE_MULTICAST` | 1 | `1 = two-CTA TMA multicast enabled` (required by the 148-SM SM103 FP8xFP4 kernel); `0 = compile the multicast path out` (not a validated schedule) | 1 | Two-CTA TMA multicast required by the 148-SM SM103 FP8xFP4 kernel. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:166` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:442` |
| `WEAVE_SCATTER_MODE` | 3 | `0 = epilogue-inline (s = epi)`: scatter inside the Compute/GEMM partition; `1 = comm-offloaded (s = comm)`: FC2 outputs handed to the Communication partition Ready Queue; `2 = local-dedup`: same-rank partial contributions are reduced into one merged buffer before one deduplicated transfer | 0 for the 148-SM SM103 dataflow | Scatter path override for the 148-SM SM103 dataflow. The three values have the same ownership semantics as the unified row. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:161` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:382` |

## NVIDIA H20 WEAVE_* parameters

For the Level 2 host-side rows below, the Kernel consumption column cites the
host decision site of the fixed compile-time kernel template. The kernel
header consumes that selected configuration, not the runtime environment
variable itself.

| Parameter | Level | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_SWAPAB_AUTO` | 2 | `0 = off`: official fixed tokens-per-expert ceiling, bit-identical to official DeepGEMM; `1 = on`: padding-aware gate that keeps the official decision and additionally admits shapes with pad_frac >= WEAVE_SWAPAB_PAD_PCT | 0 | Enable padding-aware automatic swap-AB selection. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:654` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:654` |
| `WEAVE_SWAPAB_EPI_WG` | 2 | `2 = official BN=128 tile with two epilogue warpgroups`; `4 = wide BN=256 tile with four epilogue warpgroups` | 2 | Epilogue warpgroup count under swap-AB. A request for 4 still falls back to 2 when SWAPAB_AUTO = 1 and the padding gate rejects the wide tile. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:650` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:650` |
| `WEAVE_SWAPAB_MAX_LOAD` | 2 | `N = tokens-per-expert ceiling when SWAPAB_AUTO = 0` (default 24 compact, 16 otherwise); `0 = disable swap-AB` | shape heuristic (24 compact / 16 otherwise) | Upper load bound for swap-AB in the official fixed rule. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:669` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:669` |
| `WEAVE_SWAPAB_PAD_PCT` | 2 | `0..100 = minimum padding-waste percentage that admits the wide tile`; `0 = always admit`; `100 = never admit` | 25 | Padding-fraction gate for swap-AB under SWAPAB_AUTO = 1. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:639` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:639` |
| `WEAVE_SWAPAB_MIN_LOAD` | 2 | `N = minimum non-zero tokens per expert` for swap-AB admission; `0 = no floor` | 8 | Lower load bound for swap-AB. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:640` | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:640` |
| `WEAVE_MCAST` | 1 | `0 = off, which compiles to the same single-CTA load path as 1`; `1 = explicit single-CTA weight load, no cross-CTA multicast`; `2 = two-CTA cluster TMA multicast` (auto-downgraded to 1 when the SM count or N-tile count is odd) | 0 | SM90 multicast mode. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:229` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:7` |
| `WEAVE_SWIZZLE_TILES` | 2 | `0 = linear tile order`; `1 = GroupM tile swizzle over L1 and L2` | 0 | Enable tile swizzling across expert tiles. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:230` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:18` |
| `WEAVE_HOT_COLD_MODE` | 2 | `0 = natural expert order`; `1 = LPT: sort experts by descending m-block count, hot experts first`; `2 = LPT plus hot/cold pairing`: pair the hottest remaining expert with the coldest remaining expert to balance each wave and L2 weight residency | 0 | Hot/cold expert weight residency policy. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:231` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:29` |
| `WEAVE_DEDUP_DISPATCH` | 2 | `0 = every route row is dispatched`; `1 = deduplicate dispatch rows` (active only when 3 x pool/8 >= ranks x max_tokens) | 0 | Deduplicate dispatch rows and enable the fast pull path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:232` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:40` |
| `WEAVE_DEDUP_SCATTER` | 2 | `0 = every partial row is scattered`; `1 = deduplicate identical scatter destinations` (requires the pool and N-tile capacity checks) | 0 | Deduplicate scatter destinations. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:233` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:51` |
| `WEAVE_REG_COMBINE` | 1 | `0 = SMEM-staged combine`; `1 = register-direct combine`; `2 = register-direct plus forced dual-vector chunk candidate even for large token counts` | 0 | Register-direct combine path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:234` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:62` |
| `WEAVE_ALL_IDLE_COMBINE` | 2 | `0 = epilogue warps only`; `1 = all idle dispatch and epilogue warps join the combine` (requires register combine and H % 512 == 0) | 0 | Use idle warps for the combine phase. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:235` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:73` |
| `WEAVE_DYN_SCHED` | 2 | `0 = static expert waves`; `1 = dynamic expert cursor with an expert cache` (auto-enabled when INTERLEAVE_L1_L2 > 0 or HOT_COLD_MODE > 0) | 0 | Dynamic expert scheduling. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:236` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:84` |
| `WEAVE_INTERLEAVE_L1_L2` | 2 | `0 = all L1 then all L2`; `1 = expert-completion interleave, L2 preferred`; `2 = balanced split: even CTAs prefer L2, odd CTAs prefer L1`; `3 = m-block granularity interleave`; `4 = expert-per-wave interleave: L1(w) then L2(w)`; `5 = EPW alternation plus L2 K-block dependency`; `6 = wave-lagged EPW: L1(w+1) issued before L2(w)` | 0 | Interleave layer-1 and layer-2 expert work. Modes 4-6 use the per-wave expert count. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:237` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:95` |
| `WEAVE_SCATTER_ROLE_DISP` | 2 | `0 = scatter runs on epilogue warps`; `1 = scatter is assigned to dispatch warps` | 0 | Assign scatter to dispatch warps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:238` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:106` |
| `WEAVE_HETERO_MASK_MODE` | 1 | `0 = uniform dispatch warps`; `1 = even physical SMIDs get 6 (or 2) warps, odd get 2 (or 0)`; `2 = lower half of SMIDs get 6, upper half 2`; `3 = lower half 6, upper half 0`; `4 = even SMIDs 6, odd 0`; `5 = three of every four SMIDs get 6, the fourth gets 2` | 0 | Heterogeneous SM role mask mode. It requires 2, 6 or 8 dispatch warps and runs in Linear1. The listed splits are the 6-warp mapping; a 2-warp front end scales them to 2/0, and for an 8-warp front end modes 2-5 reduce to the lower-half mask. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:239` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:117` |
| `WEAVE_PIPE_WGMMA` | 1 | `0 = issue WGMMA after the operand wait`; `1 = pipeline the next WGMMA issue with the current wait` | 0 | Pipeline WGMMA issue. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:240` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:128` |
| `WEAVE_SWAP_PIPE` | 1 | `0 = producer/consumer roles unchanged`; `1 = swap producer and consumer pipeline roles` | 0 | Swap producer/consumer pipeline roles. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:241` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:139` |
| `WEAVE_FAST_PULL` | 1 | `0 = regular pull`; `1 = fast register pull` (also enabled automatically by DEDUP_DISPATCH) | 0 | Fast register pull path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:242` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:150` |
| `WEAVE_ADAPT_BM` | 2 | `0 = fixed BLOCK_M = 64`; `1 = per-expert adaptive BM tiers` (requires swap-AB, BM 64, split-N and the expert cache) | 0 | Per-expert adaptive block-M tiers. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:243` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:161` |

## AMD MI308X WEAVE_* parameters

| Parameter | Level | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_BM_TIER_MODE` | 2 | `0 = legacy fixed BM 64 with 64-row padding`; `1 = compact 16/32/64 ladder` (EP8 overlay only); `2 = five-tier dynamic planner BM_e in {16, 32, 64, 128, 256} from the live m_e` (EP16 requires 2 and floors the ladder at 32) | 2 | Five-tier block-M planner. The chosen BM_e follows the live per-expert token count, so tile shapes adapt at runtime inside the kernel. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:105` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:71` |
| `WEAVE_DUALGROUP_MIN_TOKENS` | 2 | `N = local token count at which communication waves 4-7 are rebound into a second compute group`; `0 = dual group whenever the EP16 path allows it` (preset 0 also enables dual when t_local >= 4 and est_s1_tiles >= 192) | 64 | Threshold for rebinding communication waves into a second compute group. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1159` |
| `WEAVE_BN2_MODE` | 2 | `0 = auto`: BN64 only when there is no large tile, init_m_tiles >= 16 and init_m_tiles x n2_blocks <= 320; `64 = force second-stage BN 64`; `128 = force second-stage BN 128` | 0 | Second-stage BN mode. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:110` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:900` |
| `WEAVE_S2_NGROUP_MODE` | 2 | `0 = auto`: choose 4, 2 or 1 from the work estimate; `1 = one N block per claim`; `2 = pair two N blocks per claim`; `4 = quad four N blocks per claim` (a fixed value is used only when it divides n2_blocks, otherwise 1) | 0 | Second-stage N-group width. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:111` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:908` |
| `WEAVE_TH_BM64` | 2 | `N = declared small-M threshold for the BM64 tier` (default 16); the current EP16 greedy planner derives tiers from q32, so this value is ABI-retained | 16 | Small-M threshold for BM64. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:113` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:544` |
| `WEAVE_TH_BM128` | 2 | `N = declared small-M threshold for the BM128 tier` (default 96); the current EP16 greedy planner derives tiers from q32, so this value is ABI-retained | 96 | Small-M threshold for BM128. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:114` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:547` |
| `WEAVE_PARALLEL_ROUTE_PLAN` | 2 | `0 = single-CTA route histogram and prefix`; `1 = all-CTA parallel 80-CTA route histogram and prefix` | 1 | All-CTA parallel route histogram and prefix. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:112` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `WEAVE_WAVE_ROLES` | 1 | `struct {compute_ctas, comm_base, comm_ctas}`; default `{80, 0, 80}` = 80 compute CTAs and 80 communication CTAs; CTA 0 remaps it from the selected r_mask using the 64/16, 72/8 or 76/4 role split | roles={80,0,80} | Wave-role descriptor remapped by CTA 0 from selected_r_mask. | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:397` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:397` |
| `WEAVE_SLOT_CHUNK` | 2 | `0 = adaptive`: start at 4 and double while N x 8 < t_local, cap 64; `1, 2, 4, 8, 16, 32, 64 = fixed number of contiguous bucket slots per RDMA descriptor` | adaptive | Dispatch descriptor slot granularity. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:170` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:54` |
| `WEAVE_RET_CHUNK` | 2 | `0 = derive from SLOT_CHUNK` (cap 16); `1, 2, 4, 8, 16 = fixed node-return descriptor granularity` (cap 16) | derived | Return descriptor slot granularity for the node-partial path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:235` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:303` |
| `WEAVE_META_CHUNK_SLOTS` | 2 | `N >= 1 = metadata slots per RDMA write descriptor`; the effective value is max(N, t_local) | max(256, t_local) | Metadata slots per RDMA write descriptor. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:234` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:55` |
| `WEAVE_PROFILER_LEVEL` | 1 | `0 = production, no instrumentation`; `1 = phase open/close events only`; `2 = full per-stage events` | 0 | Instrumentation level, bridged to FI_LEVEL for the HIP profiler. build_port.py emits the FI2 variant only at level 2; level 1 is the open/close instrumentation level consumed by FI_LEVEL. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:97` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:31` |
| `WEAVE_ABLATION_PRESET` | 1 | `0 = Weave`; `1 = -S`; `2 = -S-T`; `3 = -S-T-C` (the EP16 build-time selector; preset 3 also forces scatter mode 0) | 0 | EP16 build-time ablation selector. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:93` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1154` |
| `WEAVE_SCATTER_MODE` | 3 | `0 = epilogue-inline (s = epi)`: GEMM epilogue warps issue xGMI scatter directly; `1 = comm-offloaded (s = comm)`: epilogue outputs are handed to the Communication partition Ready Queue and drained asynchronously; `2 = local-dedup / node-partial pre-reduced`: same-node partial contributions are first reduced into one merged buffer, then one deduplicated transfer is enqueued | 2 (0 at preset 3) | EP16 scatter and return mode; mode 2 enables the node-partial path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:103` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:50` |
| `WEAVE_SELECTIVE_DEDUP` | 2 | `1 = required selective input dedup for the EP16 path` (default); `0 = compile dedup out` (legacy comparison only, not a validated EP16 schedule) | 1 | Selective input dedup required by the EP16 path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:106` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:511` |
| `WEAVE_OVERLAP_PIPELINE` | 2 | `0 = disabled`: bulk-synchronous stage-by-stage serialization (used by -S-T / -S-T-C); `1 = enabled` (default): compute-communication overlap ON | 1 | Enable the EP16 overlap pipeline. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:108` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:709` |
| `WEAVE_ALL_WAVE_COMBINE` | 2 | `0 = compute-group waves only`; `1 = all waves join the cooperative tail combine` | 1 | Use all waves for the cooperative tail combine. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |

## Baselines

The complete installation and launch matrix for official DeepGEMM,
DeepEP plus DeepGEMM, FlashInfer, SGLang, vLLM, RCCL plus AITER (one-stage and
two-stage CK), and ROCm inference engines is in
[baselines/README.md](baselines/README.md) and
[baselines/README_zh.md](baselines/README_zh.md).

    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_h20_baselines.py --baseline all --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 16 --dry-run

The launcher file name `run_b300_baselines.py` is a code/path literal retained
for command compatibility; its public documentation and output labels use
NVIDIA SM103 (148-SM).

## Ablation and reproduction

[ablation/README.md](ablation/README.md) and
[ablation/README_zh.md](ablation/README_zh.md) document the four-arm
progressive ablation, the cost-model versus oracle study, EPLB compatibility,
and WeaveProfiler capture.

    python ablation/run_progressive_ablation.py --backend b300 --presets 0,1,2,3 --dry-run
    python ablation/eval_cost_model_vs_oracle.py
    python ablation/capture_weaveprofiler_timeline.py         --trace result.sw_hw.perfetto.json.gz --out-png result.png

Verified 48-cell geomean speedups over each official baseline (paper Table IV):

| Platform | Weave | -S | -S-T | -S-T-C |
|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | 1.301 | 1.192 | 1.010 | 0.996 |
| NVIDIA H20 | 1.162 | 1.137 | 1.053 | 0.947 |
| AMD MI308X EP16 | 2.147 | 1.808 | 1.537 | 1.362 |

EPLB leaves Weave orthogonal to placement (paper Table V): 1.274x over the baseline with
original routing, 1.128x with EPLB, and 1.144x over the baseline with EPLB.

## Build

Dependencies: CUDA 13.x or a ROCm 7.x stack, CMake, a C++20 compiler, PyTorch,
and the vendored CUTLASS tree. AMD builds additionally need the external AITER
site tree selected by WEAVE_AITER_SITE.

    python -m pip install -e .

On Windows, run the g++ host gates inside WSL. Run nvcc from a shell that has
`vcvars64.bat` loaded and add `-Xcompiler /Zc:preprocessor`; otherwise CCCL
headers fail to preprocess.

NVIDIA device front-end checks:

    nvcc -std=c++20 -ptx -arch=sm_90a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh -o h20.ptx

    nvcc -std=c++20 -ptx -arch=sm_100a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh -o sm103_fp8.ptx

    nvcc -std=c++20 -ptx -arch=sm_100a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh -o sm103_fp4.ptx

AMD EP16 build:

    cd deep_gemm/include/deep_gemm/impls/amd/ep16
    WEAVE_PROFILER_LEVEL=0 python build_port.py

## Verification gates

Run from the repository root:

    g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include         tests/test_cost_model_eq123.cpp -o test_cost_model
    ./test_cost_model

    for p in 0 1 2 3; do
      g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include           -DWEAVE_ABLATION_PRESET=$p tests/test_ablation_presets.cpp           -o test_ablation_$p
      ./test_ablation_$p
    done

    python tests/validate_cost_model_3hw.py
    python tests/check_kernel_policy_wiring.py
    python tests/verify_tunable_params_doc.py
    python -m py_compile baselines/*.py ablation/*.py
    python ablation/eval_cost_model_vs_oracle.py

## Citation

If you find Weave useful in your research, please cite our paper:

Ziyu Huang, Yangjie Zhou, Chenhao Zhu, Peng Yu, Zihan Liu, Jinyu Liu,
Shulai Zhang, Xingxun Tang, Hongzhe Yan, Xinhao Luo, Minyi Guo, Xiu Lin,
Yinghao Yu, Guodong Yang, Liping Zhang, Shixuan Sun, and Jingwen Leng.
Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for
Compute-Communication Overlap. arXiv:2609.21483, 2026.
<https://arxiv.org/abs/2609.21483>

    @misc{huang2026weavefinegraineddynamicsm,
          title={Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap},
          author={Ziyu Huang and Yangjie Zhou and Chenhao Zhu and Peng Yu and Zihan Liu and Jinyu Liu and Shulai Zhang and Xingxun Tang and Hongzhe Yan and Xinhao Luo and Minyi Guo and Xiu Lin and Yinghao Yu and Guodong Yang and Liping Zhang and Shixuan Sun and Jingwen Leng},
          year={2026},
          eprint={2609.21483},
          archivePrefix={arXiv},
          primaryClass={cs.DC},
          url={https://arxiv.org/abs/2609.21483},
    }
