[English](README.md) | [简体中文](README_zh.md)

# Weave: Routing-Adaptive Runtime-Dynamic MoE Megakernel

Weave is a **routing-adaptive, runtime-dynamic MoE megakernel** implemented in a
unified C++/CUDA/HIP codebase ([Weave: Fine-Grained Dynamic SM Scheduling in an
MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)),
supporting **NVIDIA SM103 (148-SM)**, **NVIDIA H20**, and **AMD MI308X**. Right
after dispatch finishes, the kernel reads live per-expert token counts and
performs **spatial partitioning and temporal scheduling** directly on the GPU at
runtime.

## Architecture and runtime scheduling

| Hardware | Language | Supported precision values | Entry point |
|---|---|---|---|
| NVIDIA SM103 (148-SM) | CUDA C++ | `0 = fp8xfp4` (default), `1 = fp4xfp4` (NVFP4), `3 = bf16`; `2 = fp8` is not supported | sm100_fp8_fp4_mega_moe.cuh / sm100_fp4_fp4_mega_moe.cuh via weave_mega_moe(..., mma_type=...) |
| NVIDIA H20 (SM90) | CUDA C++ | `2 = fp8` (fp8xfp8) only | sm90_fp8_mega_moe.cuh |
| AMD MI308X (gfx942) | HIP C++ | `2 = fp8` (fp8xfp8) only; EP8 and EP16 | gfx942_fp8_mega_moe.cuh and amd/megamoe_amd.hip |

![Weave System Design Overview](docs/figures/weave_design_overview.png)

![Weave Spatial Warp Ratio & Temporal Warp Roles vs MegaMoE](docs/figures/weave_design_rolegraph.png)

After dispatch completes, CTA 0 evaluates an on-device cost model over the live
per-expert token distribution and communication volume, selecting the layer's
**spatial partitioning** and **temporal schedule**, which all other CTAs read
from device memory:

- **Spatial partitioning:** chooses the per-SM/CU ratio between communication
  warps and compute/epilogue warps, and adapts per-expert tile sizes at runtime
  according to each expert's actual token count to cut padding waste.
- **Temporal scheduling (eliminating head, middle, and tail bubbles):**
  - **Head:** at layer start before any GEMM tile is ready, otherwise-waiting
    compute/TMA/epilogue warps join remote token pulling within their own
    register budgets to saturate link bandwidth, then switch to GEMM once the
    first tiles arrive;
  - **Middle:** based on the live comm-to-compute ratio, selects whether FC2
    Scatter runs inline in epilogue warps or is handed off to communication
    warps to overlap with subsequent GEMM tiles;
  - **Tail:** replaces the global cross-GPU barrier before Combine with
    **per-token readiness** (each token is reduced as soon as all `K` of its
    expert outputs arrive) and recruits **all finished warps (comm, TMA/MMA, and
    epilogue)** into cooperative weighted reduction.

## Hardware partitions and global ready queues

Warps on each SM/CU are divided into a **Communication Partition (`N_c`)** and a
**Compute/GEMM Partition (`N_e`)**, each pulling work from its own global task
queue:

- **GEMM Ready Queue** (consumed by `N_e`): holds `(expert_id, m_tile_idx)`
  compute tiles. As remote tokens land in local buffers, communication warps
  atomically increment per-expert (or per-group) arrival counters; instead of
  materializing and sorting a full per-tile array at runtime, compute CTAs claim
  ready tiles directly via prefix-summed expert offsets in a preset expert order
  (descending load / L2 tile-swizzle), or pop ready tile IDs from a global FIFO
  when `WEAVE_READY_QUEUE=1`. When `WEAVE_SCATTER_MODE=0`, epilogue warps send
  FC2 outputs directly over NVLink/xGMI inside the compute partition.
- **Communication Ready Queue** (consumed by `N_c`): holds cross-GPU Pull and
  async Scatter tasks. When `WEAVE_SCATTER_MODE=1`, epilogue warps hand off
  completed FC2 tile scatter descriptors into this queue; when
  `WEAVE_SCATTER_MODE=2`, duplicate partial outputs for the same source token on
  the local GPU (or node) are first merged in HBM before enqueuing a single
  transfer, allowing `N_c` warps to overlap NVLink/xGMI scatter with ongoing
  GEMM tiles.

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

## WeaveProfiler software + hardware timelines

![NVIDIA SM103 (148-SM) WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_sm103_timeline.png)

![NVIDIA H20 WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_h20_timeline.png)

![AMD MI308X WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_mi308x_timeline.png)

All three figures show aligned software + hardware timelines on the matched
`Qwen3.8-Max, EP8, S=1024` workload (hardware utilization curves, all physical
SM/CU lanes, and representative CTA warp-role pipelines). Full tuned `WEAVE_*`
presets (organized by **Spatial**, **Temporal**, and **Implementation**) and
reproducible launch scripts live under [examples/](examples/README.md):

- NVIDIA SM103 (148-SM): [examples/sm103_qwen38max_ep8_s1024.sh](examples/sm103_qwen38max_ep8_s1024.sh)
- NVIDIA H20: [examples/h20_qwen38max_ep8_s1024.sh](examples/h20_qwen38max_ep8_s1024.sh)
- AMD MI308X: [examples/mi308x_qwen38max_ep8_s1024.sh](examples/mi308x_qwen38max_ep8_s1024.sh) (EP8) and [examples/mi308x_qwen38max_ep16_s128.sh](examples/mi308x_qwen38max_ep16_s128.sh) (2-node EP16)
- One-command runner & renderer: `python examples/profile_and_render_timeline.py --preset sm103` (supports `sm103`, `h20`, `mi308x_ep8`, `mi308x_ep16`, plus `--run` / `--render`)

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
  overlap, opening all-warp prefetch token count, scatter ownership (epilogue
  inline vs. async comm queue vs. local pre-reduce dedup), ready-queue &
  L2-swizzled ordering, L1/L2 interleaving, per-token ready &
  cooperative all-warp tail reduction, and **runtime-derived pipeline stage
  depth** (`num_stages` is computed at launch from the tile shape and remaining
  SMEM budget rather than hardcoded at compile time).
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

The parameter surface is organized strictly as **Spatial / Temporal / Implementation**. Every row names both its host parse and its kernel consumption; `tests/verify_tunable_params_doc.py` resolves each `file:line` citation. The full value semantics are in [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md).

### Spatial

| Parameter | Category | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | Spatial | `0 = fp8xfp4`; `1 = fp4xfp4`; `2 = fp8xfp8`; `3 = bf16` | 0 | Numeric precision and tensor-core mode; the source of every byte width in the cost model. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:273` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_DISPATCH_WARPS` | Spatial | `0 = auto` (4 warps); `4`; `8` | 0 | Per-CTA communication warp budget; 8 warps double the dispatch pull width and raise the CTA to 256 dispatch threads. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:49` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:940` |
| `WEAVE_SCATTER_ACTIVE_SMS` | Spatial | `0 = auto` (all SMs/CUs participate); `K = 1..num_sms`: only K SMs drain the scatter queue | 0 | Decoupled throttled active scatter SMs: the epilogue only publishes descriptors and a bounded SM subset owns NVLink/xGMI scatter. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:173` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:499` |
| `WEAVE_LOCALITY_DOMAINS` | Spatial | `0 = auto`; `1 = single domain`; `2 = dual-die SM103`; `4 = XCD MI308X` | 0 | Splits every L1/L2 N-cluster queue across locality domains; each domain claims locally and steals from the peer at the tail. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:184` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:832` |
| `WEAVE_SM0_STREAM_SCHED` | Spatial | `0 = SM 0 communication warps pull`; `1 = SM 0 communication warps become the streaming scheduler lane` | 0 | Repurposes SM 0/CU 0 communication warps as a dedicated scheduler lane; the remaining SMs evenly own the full dispatch pull range. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:58` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2773` |
| `WEAVE_AMD_8WAVE_MIN_TOK` | Spatial | `N = minimum local tokens per CU-expert`; `0 = follow WEAVE_DUALGROUP_MIN_TOKENS` | 64 | MI308X 8-wave schedule admission floor: below N tokens the CU keeps the 4-wave schedule. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:67` | `deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip:1215` |
| `WEAVE_DUALGROUP_MIN_TOKENS` | Spatial | `N = second compute-group threshold in local tokens` | 64 | Historical MI308X dual-group threshold; the 8-wave knob defaults to this value. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:14` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1159` |
| `WEAVE_BM_TIER_MODE` | Spatial | `0 = five-tier`; `1 = four-tier`; `2 = EP16 four-tier {32,64,128,256}` | 2 | Per-expert block-M ladder used by the MI308X greedy tile planner. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:12` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:71` |

### Temporal

| Parameter | Category | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_L2_KBLOCK_DEP` | Temporal | `0 = default L1-first order`; `1 = 64-bit XOR L2KBlockDependency bitmask with a tight 1 L2 <-> 1 L1 alternating wave schedule` | 0 | Starts the L2 K block as soon as its matching L1 N subtile completes instead of waiting for the full L1 wave. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:247` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:833` |
| `WEAVE_RING_BLOCKS` | Temporal | `0 = auto capacity from get_num_max_live_pool_blocks`; `N >= 1 = forced ring block count` | 0 | Bounded ring-buffer reuse for the intermediate SwiGLU activations; the ring wraps after N live pool blocks instead of allocating the whole pool. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:209` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1057` |
| `WEAVE_EARLY_STREAM_REDUCE` | Temporal | `0 = global barrier before combine`; `1 = barrier-free per-token cnt==topk tail-interleaved early reduce` | 0 | Epilogue warps enter the per-token reduce as soon as their last L2 tile is handed off, overlapping the tail with the remaining scatter. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:210` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1368` |
| `WEAVE_TMEM_PIPELINE_MODE` | Temporal | `0 = auto` (2.0 at BM<=240 with one prefetch stage; 1.75 at BM=256); `100 = 1.0`; `175 = 1.75`; `200 = 2.0` | 0 | TMEM epilogue pipeline depth; the 1.75 mode overlaps the tail epilogue with the next accumulator column range at BM=256. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:211` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:3674` |
| `WEAVE_HEAD_PULL_CHUNK` | Temporal | `0 = pull the opening head budget as one prefix`; `N = tokens per head-pull chunk` | 0 | Bounds the all-warp opening head-pull prefix; after N tokens only the selected communication warps keep pulling. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:78` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2813` |
| `WEAVE_V_HEAD` | Temporal | `0 = automatic`; `N = opening head-pull budget in tokens` | automatic | Opening head-pull budget: all warps pull the first N remote tokens at peak link bandwidth before the first GEMM tile is ready. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:263` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_SCATTER_MODE` | Temporal | `0 = epilogue inline`; `1 = async communication queue`; `2 = local pre-reduce dedup first`; `3 = local bf16x2 dedup + HBM scatter queue` | 1 | Scatter executor ownership: which partition issues the FC2 scatter and whether a local dedup pass runs first. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:192` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:163` |
| `WEAVE_OVERLAP_PIPELINE` | Temporal | `0 = serialized`; `1 = compute-communication overlap` | 1 (forced 0 at preset 2 and above) | Turns compute-communication overlap on or off; the ablation presets derive it automatically. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:23` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:151` |
| `WEAVE_ALL_WAVE_COMBINE` | Temporal | `0 = compute-group waves only`; `1 = all waves join the cooperative tail combine` | 1 (forced 0 at preset 2 and above) | Cooperative all-wave tail combine; the ablation presets derive it automatically. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:24` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:181` |
| `WEAVE_READY_QUEUE` | Temporal | `0 = preset expert order`; `1 = global FIFO ready queue` | 0 | Selects the ready-tile source: prefix-summed expert offsets or a global FIFO. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:243` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:354` |
| `WEAVE_READY_GROUP_SIZE` | Temporal | `0 = per-expert readiness`; `N = group-granularity readiness publication` | 0 | Publishes readiness per expert or per group of N experts. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:222` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:58` |
| `WEAVE_SCHED_MODE` | Temporal | `0 = descending load`; `1 = L2 tile swizzle`; `2 = expert-major` | 0 | Tile ordering key used when the ready queue is not a FIFO. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:244` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:370` |
| `WEAVE_EXPERTS_PER_WAVE` | Temporal | `0 = auto`; `N = LPT scheduling window`; `100 + N = inline static order` | 0 | Longest-processing-time scheduling window across experts. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:241` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:322` |

### Implementation

| Parameter | Category | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | Implementation | `0 = full Weave`; `1 = -S`; `2 = -S-T`; `3 = -S-T-C` | 0 | Compile-time progressive ablation selector. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:41` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:26` |
| `WEAVE_ENABLE_COST_MODEL` | Implementation | `0 = static policy`; `1 = CTA-0 runtime cost model` | 1 at preset 0; 0 at presets 1-3 | Master switch for the on-device CTA-0 policy search. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:51` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:27` |
| `WEAVE_ADAPT_BM` | Implementation | `0 = fixed per-expert BM`; `1 = per-expert dynamic BM ladder` | 0 | Enables the per-expert adaptive block-M ladder (H20 swapAB tiers, SM103 expert step mode). | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:242` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:513` |
| `WEAVE_BLOCK_M_L1` | Implementation | `0 = auto`; `16, 32, 64, 96, 128, 192, 240, 256` | 0 | GEMM1 (FC1) per-expert block-M envelope; the adaptive ladder picks BM_e inside it. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:86` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1059` |
| `WEAVE_BLOCK_M_L2` | Implementation | `0 = auto`; `16, 32, 64, 96, 128, 192, 240, 256` | 0 | GEMM2 (FC2) per-expert block-M envelope; independent from GEMM1 on the dynamic-scheduler backends. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:89` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1061` |
| `WEAVE_STORE_BLOCK_M_L1` | Implementation | `0 = auto`; positive multiple of 8 (for example `24`) | 0 (auto 24 at BM=240) | GEMM1 epilogue store height; the asymmetric 24/8 split unlocks num_stages = 6. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:217` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1063` |
| `WEAVE_STORE_BLOCK_M_L2` | Implementation | `0 = auto`; positive multiple of 8 (for example `8`) | 0 (auto 8 at BM=240) | GEMM2 epilogue store height; the asymmetric 24/8 split unlocks num_stages = 6. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:218` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:204` |
| `WEAVE_LOCAL_SCATTER_DEDUP` | Implementation | `0 = full-precision local accumulation`; `1 = bf16x2 hardware L2 atomic dedup before send` | 0 | Fire-and-forget red.add.noftz.bf16x2 merge of duplicate expert contributions into a pre-zeroed L2-resident buffer; one descriptor is published per deduplicated row. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:213` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1371` |
| `WEAVE_REG_PULL_BYTES` | Implementation | `0 = SMEM staging`; `1024, 2048, 4096 = register-direct pull width in bytes` | 0 | Pulls remote token payloads directly into registers instead of staging them in SMEM. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:224` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:90` |
| `WEAVE_SCATTER_SMEM_ALIAS` | Implementation | `0 = dedicated scatter SMEM`; `1 = alias the scatter buffer into the dispatch dead zone` | 1 | Reuses the dispatch buffer's idle window for scatter staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:193` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:461` |
| `WEAVE_SCATTER_WARPS` | Implementation | `0 = automatic`; `4`; `8` | 0 | Scatter warp allocation inside the dispatch partition. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:194` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:472` |
| `WEAVE_DEDUP_MODE` | Implementation | `0 = off`; `1 = unique (token, destination) buckets`; `2 = local pre-reduce plus 32-lane fanout` | 0 | Selective input deduplication of duplicate routed reads. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:246` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:386` |
| `WEAVE_PROFILER_LEVEL` | Implementation | `0 = production`; `1 = phase open/close events`; `2 = full per-stage events` | 0 | WeaveProfiler instrumentation level for the AMD port. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:97` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:8` |

`num_shared_experts` is a Python/C++ API argument rather than an environment knob. Its default is `0` on every entry point; shared-expert buffers are allocated only when it is explicitly non-zero.

## Baselines

Installation and benchmark scripts for official DeepGEMM, DeepEP + DeepGEMM,
FlashInfer, SGLang, vLLM, and RCCL + AITER (1-stage and 2-stage CK) live under
[baselines/README.md](baselines/README.md) and
[baselines/README_zh.md](baselines/README_zh.md).

    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_h20_baselines.py --baseline all --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 16 --dry-run

## Ablation and reproduction

Progressive ablation scripts, cost-model vs. exhaustive-oracle evaluation, EPLB
experiments, and WeaveProfiler timeline capture live under
[ablation/README.md](ablation/README.md) and
[ablation/README_zh.md](ablation/README_zh.md).

    python ablation/run_progressive_ablation.py --backend b300 --presets 0,1,2,3 --dry-run
    python ablation/eval_cost_model_vs_oracle.py
    python ablation/capture_weaveprofiler_timeline.py --trace result.sw_hw.perfetto.json.gz --out-png result.png

Geomean speedup across the 48 evaluated configurations vs. official baselines
(paper Table IV):

| Platform | Weave | -S | -S-T | -S-T-C |
|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | 1.301 | 1.192 | 1.010 | 0.996 |
| NVIDIA H20 | 1.162 | 1.137 | 1.053 | 0.947 |
| AMD MI308X EP16 | 2.147 | 1.808 | 1.537 | 1.362 |

Combined with Expert-Parallel Load Balancing (EPLB, paper Table V): Weave
achieves 1.274x speedup without EPLB; enabling EPLB alone gives 1.128x over the
baseline, and enabling Weave on top of EPLB yields an additional 1.144x speedup.

## Build

Dependencies: CUDA 13.x or ROCm 7.x, CMake, a C++20 compiler, PyTorch, and the
vendored CUTLASS headers (AMD builds also use `WEAVE_AITER_SITE` to point to
external AITER headers).

    python -m pip install -e .

Standalone NVIDIA kernel compilation checks:

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
