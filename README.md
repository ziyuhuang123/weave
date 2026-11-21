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
**spatial resource ratios** and **temporal schedule**, executed on top of
independently tuned **single-operator and data-path implementations**:

- **Spatial partitioning (SM-level and Warp-level hardware ratios):**
  - **Across SMs/CUs:** independently assigns active SMs for Dispatch pulling
    (`D`), active SMs for async Scatter draining (`K`, $D \neq K$), and whether
    CTA 0's communication warps are dedicated as the streaming scheduler
    (`WEAVE_SM0_STREAM_SCHED`), while recruiting all SMs for tail Combine
    reduction.
  - **Within each SM/CU:** configures the warp/wave split between the
    communication partition (`N_c`: Dispatch / Scatter warps) and the compute
    partition (`N_e`: TMA/MMA + Epilogue warps)—including `4D+4N+8E` vs.
    `8D+4N+8E` on NVIDIA SMs and **`0-8` / `4-4` / `8-0`** `(comm - compute)`
    wavefront splits on AMD MI308X CUs—together with per-role register file
    budgets (e.g., `96/72/168` vs. `64/56/192`).
- **Temporal scheduling (eliminating head, middle, and tail bubbles):**
  - **Head:** at layer start all warps help pull remote tokens by default to
    saturate link bandwidth; `WEAVE_V_HEAD` sets how many pull tasks (tokens)
    those helping warps complete before exiting back to GEMM, while fine-grained
    group readiness (`WEAVE_READY_GROUP_SIZE`) starts GEMM1 without waiting for
    full experts;
  - **Task dispatch & Middle:** tile claiming switches between a **static
    heuristic formula** over prefix-summed offsets (`formula`) and **SM 0
    dynamic global-FIFO scheduling** (`sm0_fifo`, where SM 0's communication
    warps compute and push ready tiles across all regions into a global queue for
    dynamic CTA preemption); during interleaved GEMM1<->GEMM2 execution,
    **strict binary Scatter ownership** selects either inline epilogue stores
    (`epilogue_only`) or 100% dedicated communication-warp draining (`comm_only`,
    with zero mixed fallback) to overlap NVLink/xGMI scatter with ongoing GEMM;
  - **Tail:** replaces the global cross-GPU barrier with **chunked (`G`-token)
    cross-GPU readiness paired with local deduplication** (emitting a single
    `red.release.sys` per `G`-token destination chunk to avoid per-tile or
    per-token system-atomic storms), while all finished warps/waves
    automatically join cooperative Combine reduction overlapping tail Scatter.
- **Single-op optimizations & prior-art implementations (Implementation):**
  - **Independent 4-operator configuration & `M -> blockM` coupled profiles:**
    **GEMM1**, **GEMM2**, **Dispatch**, and **Scatter** configure their
    parameters independently. For GEMM1 and GEMM2, offline calibration maps each
    `M` bucket to `blockM`, which **synchronously fixes `(blockM, blockN,
    blockK, mainloop_stages)` and `(epi_M, epi_N, epi_K, epi_stages)`** (packing
    compact asymmetric store sub-blocks, `1.75/2.0`-stage TMEM accumulation, and
    2-stage register prefetch inside the profile tuple); at runtime each expert
    looks up its own profile from live `tokens[e]`, while in-flight `FC1 -> FC2`
    activations reuse a bounded modulo ring buffer (8-9x HBM compression);
  - **Independent Dispatch & Scatter transfer/staging/deduplication:** Dispatch
    independently configures per-warp register-direct pull chunk width (`1024 /
    2048 / 4096 B` per 32-lane warp = `8 / 16 / 32` 32-bit registers per thread)
    and pull deduplication, while Scatter independently selects among
    `reg_2stage`, bank-conflict-free `smem`, and `hbm_dedup` staging plus local
    `bf16x2` multi-hit deduplication.

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

Each figure shows aligned software + hardware timelines on `Qwen3.8-Max`
(hardware utilization curves, all physical SM/CU lanes, and representative CTA
warp-role pipelines). `S_rank` is the token count per rank and `S_global` the
total across all ranks:

- **NVIDIA SM103 (148-SM), 4x EP4, `S_rank=1024` (`S_global=4096`, `num_shared_experts=0`, `BM=80, store=8/8, stages=8, WEAVE_EPI_REG_STAGES=2`)**:
  same-run WeaveProfiler SW+HW capture (`3.52%` warm instrumentation overhead,
  `0` buffer overflow, bitwise-exact output across all 4 ranks) measures
  **Active SMs `100.0%`** median / max, **HBM bandwidth `81.0%`** median / `94.0%` max,
  **NVLink `43.0%`** median / `100.0%` max, and **Tensor Active `38.0%`**
  median / `42.0%` max (141 same-run `nsys` samples per curve, recorded in
  `weaveprofiler_optimal_s1024.hw_curves_same_run_nsys`).
- **NVIDIA H20 (78-SM), 8x EP8, `S_global=1024` (`Qwen3.8-Max`, `num_shared_experts=0`)**:
  aligned software + hardware timeline across all 78 physical SMs showing
  multi-wave `L1 GEMM <-> L2 GEMM <-> Scatter` interleaving and dedicated
  communication-warp scatter draining (`epi_scatter_spans = 0` on `Math+Epi
  WG0..WG3`).
- **AMD MI308X, 8x EP8 (`Qwen3.8-Max, S=1024`)**: aligned per-wavefront software
  events and hardware activity curves across all physical CUs.

### Same-precision EP4 comparison vs. upstream official DeepGEMM MegaMoE (`num_shared_experts = 0`)

Full raw receipts live in
[results/opus_sm103_ep4_verified.json](results/opus_sm103_ep4_verified.json),
[results/opus_h20_ep4_verified.json](results/opus_h20_ep4_verified.json), and
[results/gcmoe_same_precision_ep4.json](results/gcmoe_same_precision_ep4.json).

**NVIDIA SM103 (148-SM), 4x EP4 (`FP8 x MXFP4`, `num_shared_experts = 0`):**

| Workload | Upstream Official DeepGEMM MegaMoE | Weave | Speedup vs. Official | 4-GPU Accuracy |
|---|---:|---:|---:|---|
| `Qwen3.8-Max`, `S_rank=1024` (`S_global=4096`) | `881.38 us` (`BM=240`) / `748.88 us` (`BM=128`) | **`710.53 us`** | **`1.241x`** (`+24.1%`) / **`1.054x`** (`+5.4%`) | `rel_l2=2.91e-5`, `stats_exact` |
| `416E`, `S_rank=1024` (`S_global=4096`) | `450.26 us` (`BM=240`) / `411.66 us` (`BM=128`) | **`382.85 us`** | **`1.176x`** (`+17.6%`) / **`1.075x`** (`+7.5%`) | `rel_l2=5.37e-5`, `stats_exact` |

Under random multinomial routing against the same-precision `GC-MoE` baseline,
Weave also achieves **`1.055x`** (`779.0 us` vs. `824.7 us`) on `Qwen3.8-Max` and
**`1.042x`** (`393.8 us` vs. `412.9 us`) on `416E`, matching `GC-MoE` bit-for-bit
across all 4 ranks.

**NVIDIA H20 (78-SM), 4x EP4 (`FP8 x FP8`, `num_shared_experts = 0`, `Qwen3.8-Max`):**

| Global Sequence Length (`S_global`) | Upstream Official DeepGEMM MegaMoE | Weave | Speedup vs. Official |
|---|---:|---:|---:|
| `S=128` (`S_rank=32`) | `1477.84 us` | **`1328.26 us`** | **`1.113x`** (`+11.3%`) |
| `S=1024` (`S_rank=256`) | `3241.71 us` | **`2177.54 us`** | **`1.489x`** (`+48.9%`) |
| `S=4096` (`S_rank=1024`) | `6210.22 us` | **`5248.66 us`** | **`1.183x`** (`+18.3%`) |

Full tuned `WEAVE_*` presets (organized by **Spatial**, **Temporal**, and
**Implementation**) and reproducible launch scripts live under
[examples/](examples/README.md):

- NVIDIA SM103 (148-SM): [examples/sm103_qwen38max_ep8_s1024.sh](examples/sm103_qwen38max_ep8_s1024.sh)
- NVIDIA H20: [examples/h20_qwen38max_ep8_s1024.sh](examples/h20_qwen38max_ep8_s1024.sh)
- AMD MI308X: [examples/mi308x_qwen38max_ep8_s1024.sh](examples/mi308x_qwen38max_ep8_s1024.sh) (EP8) and [examples/mi308x_qwen38max_ep16_s128.sh](examples/mi308x_qwen38max_ep16_s128.sh) (2-node EP16)
- One-command runner & renderer: `python examples/profile_and_render_timeline.py --preset sm103` (supports `sm103`, `h20`, `mi308x_ep8`, `mi308x_ep16`, plus `--run` / `--render`)

## Parameter taxonomy: Spatial, Temporal, and Implementation

Weave's tunable surface is partitioned strictly across three axes:

- **Spatial dimension:** strictly **SM-level and per-SM/CU Warp/Wave-level
  hardware resource ratios**—active SMs for Dispatch
  (`WEAVE_DISPATCH_ACTIVE_SMS`), active SMs for Scatter
  (`WEAVE_SCATTER_ACTIVE_SMS`), whether CTA 0's communication warps are
  allocated as the streaming scheduler (`WEAVE_SM0_STREAM_SCHED`), per-SM/CU
  communication, compute, and streaming-reduce warp/wave ratios (`WEAVE_DISPATCH_WARPS`,
  `WEAVE_SCATTER_WARPS`, dedicated Middle zero-SMEM fine-grained reduce warps
  `WEAVE_REDUCE_WARPS`, and MI308X's 3-mode `0-8 / 4-4 / 8-0`
  `WEAVE_CU_WAVE_SPLIT`), and per-role register file ratios
  (`WEAVE_REG_BALANCE`).
- **Temporal dimension:** **cross-stage role switching, dynamic task dispatch,
  compute-communication overlap, and synchronization granularity**—how many
  opening pull tasks helping warps complete before switching to GEMM
  (`WEAVE_V_HEAD`) and fine-grained group arrival readiness
  (`WEAVE_READY_GROUP_SIZE`), static heuristic formula vs. SM 0 dynamic
  global-FIFO tile preemption (`WEAVE_READY_QUEUE`), strict binary Scatter role
  ownership (`WEAVE_SCATTER_OWNER`), and tail chunked `G`-token cross-GPU
  Combine readiness (`WEAVE_COMBINE_SYNC_CHUNK`, with all idle warps joining
  Combine by default).
- **Single-op optimizations & prior work (Implementation):** **independent
  GEMM1, GEMM2, Dispatch, and Scatter single-operator configurations**—numeric
  precision (`WEAVE_PRECISION_MODE`), runtime per-expert `M -> blockM` profile
  lookup (`WEAVE_ADAPT_BM`) and independent GEMM1 / GEMM2 profile selectors
  (`WEAVE_BLOCK_M_L1`, `WEAVE_BLOCK_M_L2`, where selecting `blockM`
  synchronously locks `blockN/K`, `mainloop_stages`, `epi_M/N/K`, and
  `epi_stages`), independent Dispatch pull width & deduplication
  (`WEAVE_REG_PULL_BYTES`, `WEAVE_DEDUP_MODE`), and independent Scatter staging
  & local deduplication (`WEAVE_SCATTER_STAGING`, `WEAVE_LOCAL_SCATTER_DEDUP`).

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
| `WEAVE_DISPATCH_ACTIVE_SMS` | Spatial | `0 = auto` (all SMs run dispatch pull); `D = 1..num_sms`: only D SMs pull remote tokens | 0 | SM-level dispatch allocation: assigns D SMs to remote token pulling while remaining SMs enter prefetch/GEMM immediately. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:100` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:49` |
| `WEAVE_SCATTER_ACTIVE_SMS` | Spatial | `0 = auto` (all SMs/CUs participate); `K = 1..num_sms`: only K SMs drain the scatter queue | 0 | SM-level scatter allocation (independent from dispatch SMs): bounds async NVLink/xGMI scatter draining to K SMs while Combine uses all SMs. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:203` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:531` |
| `WEAVE_SM0_STREAM_SCHED` | Spatial | `0 = CTA 0 communication warps run normal pull/scatter`; `1 = allocate CTA 0 communication warps as a dedicated streaming scheduler` | 0 | CTA-0 warp allocation: decides whether CTA 0's communication warps are dedicated to streaming tile scheduling while remaining SMs split the pull range. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:86` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:3031` |
| `WEAVE_DISPATCH_WARPS` | Spatial | `0 = auto` (4 warps); `4`; `8` | 0 | Per-SM dispatch warp ratio: sets the CTA communication warp count dedicated to remote token/SF pulling (`4D+4N+8E` vs `8D+4N+8E`). | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:73` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:972` |
| `WEAVE_SCATTER_WARPS` | Spatial | `0 = automatic`; `4`; `8` | 0 | Per-SM scatter warp ratio: sets the CTA communication warp budget dedicated to async FC2 scatter (independent from dispatch pull parameters). | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:224` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:504` |
| `WEAVE_REDUCE_WARPS` | Spatial | `0 = off` (tail-only Combine); `2, 4, 8 = dedicated fine-grained streaming Reduce warps` carved out of the CTA communication warp group during Middle | 0 | Per-SM Reduce warp allocation: assigns dedicated warps after opening Pull to zero-SMEM out-of-order fine-grained Reduce overlapping Middle GEMM/Scatter. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:76` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1269` |
| `WEAVE_CU_WAVE_SPLIT` | Spatial | `0 = auto` (cost-model selection); `1 = 0-8` (0 comm + 8 compute waves); `2 = 4-4` (4 comm + 4 compute waves); `3 = 8-0` (8 comm + 0 compute waves on dedicated comm CUs; names or codes accepted) | 0 | Per-CU wavefront ratio on MI308X: selects `(comm_waves - compute_waves)` across `0-8`, `4-4`, and `8-0` out of 8 waves per CU. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:94` | `deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip:4` |
| `WEAVE_REG_BALANCE` | Spatial | `0 = default (96/72/168)`; `1 = rebalanced (64/56/192) at E/rank > 64` | 0 | Per-SM register file ratio across warp roles: shifts spare dispatch/non-epilogue registers to epilogue warps (`64/56/192` vs `96/72/168`). | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:161` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1516` |

### Temporal

| Parameter | Category | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_V_HEAD` | Temporal | `0 = automatic`; `N = head-pull task quota (tokens) before helping warps exit back to GEMM` | automatic | Head schedule: all warps help pull by default at layer start; controls how many pull tasks (tokens) helping non-comm warps complete before finishing and switching to GEMM. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:263` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_READY_GROUP_SIZE` | Temporal | `0 = full-expert readiness`; `N = group-granularity readiness publication (in BLOCK_M tiles)` | 0 | Head schedule: fine-grained arrival readiness group size so GEMM1 starts as soon as N tiles arrive without waiting for full experts. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:270` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:90` |
| `WEAVE_READY_QUEUE` | Temporal | `0 = formula`: static heuristic formula over prefix-summed offsets; `1 = sm0_fifo`: SM 0 streaming scheduler dynamically pushes ready tiles across all regions into the global FIFO queue for CTA preemption (names or codes accepted) | 0 | Head/Middle task dispatch: selects static heuristic formula addressing vs SM 0 dedicated dynamic global FIFO work-stealing queue. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:85` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:386` |
| `WEAVE_SCATTER_OWNER` | Temporal | `0 = epilogue_only`: epilogue warps issue every remote FC2 store; `1 = comm_only`: epilogue warps only stage locally and communication warps own every remote store (names or codes accepted) | 0 (auto `comm_only` when a staged path or a scatter SM throttle is requested) | Middle schedule: strict binary scatter role ownership (never mixes inline epilogue scatter with comm-warp scatter) to overlap FC2 scatter with ongoing GEMM. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:177` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1444` |
| `WEAVE_COMBINE_SYNC_CHUNK` | Temporal | `0 = global barrier before combine`; `16, 32, 64, 128 = chunked dedup combine sync granularity (tokens per chunk)` | 0 | Tail schedule: aggregates deduplicated sends into G-token chunks in local L2 and emits one cross-GPU release per chunk so all idle warps stream Combine reduction overlapping tail Scatter. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:129` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:730` |

### Implementation

| Parameter | Category | Values (every accepted value is named) | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | Implementation | `0 = fp8xfp4`; `1 = fp4xfp4`; `2 = fp8xfp8`; `3 = bf16` | 0 | Single-op numeric precision and tensor-core instruction format across GEMM1/GEMM2. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:273` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_ADAPT_BM` | Implementation | `0 = fixed layer-wide profile`; `1 = per-expert M-bucketed offline profile lookup` | 0 | Runtime per-expert lookup from M_e -> blockM, synchronously binding the offline-calibrated (blockM/N/K, mainloop_stages, epi_M/N/K, epi_stages) group. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:283` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:591` |
| `WEAVE_BLOCK_M_L1` | Implementation | `0 = auto lookup from M`; `16, 32, 64, 80, 96, 128, 192, 240, 256` | 0 | Independent GEMM1 (FC1) profile selector: choosing blockM synchronously locks GEMM1's paired blockN/K, mainloop TMA stages, and epilogue M/N/K & TMEM/reg stages. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:146` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1091` |
| `WEAVE_BLOCK_M_L2` | Implementation | `0 = auto lookup from M`; `16, 32, 64, 80, 96, 128, 192, 240, 256` | 0 | Independent GEMM2 (FC2) profile selector (decoupled from GEMM1): choosing blockM synchronously locks GEMM2's paired blockN/K, mainloop stages, and epilogue M/N/K & stages. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:149` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1093` |
| `WEAVE_REG_PULL_BYTES` | Implementation | `0 = TMA SMEM staging`; `1024, 2048, 4096 = per-warp bytes per pull chunk` (= `32, 64, 128 B` or `8, 16, 32` 32-bit regs per thread) | 0 | Independent Dispatch operator parameter: per-warp (32-lane) chunk width for zero-SMEM register-direct remote pulling (`uint4 regs[2/4/8]` per thread, freeing SMEM for deeper GEMM mainloop stages). | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:272` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:122` |
| `WEAVE_DEDUP_MODE` | Implementation | `0 = off` (per-slot pull); `1 = on` (pull each unique `(src_rank, token)` once over NVLink/xGMI, then fan out locally across all hit experts) | 0 | Independent Dispatch operator remote-pull deduplication: pulls a multi-hit remote token once and broadcasts it into all hit local expert slots. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:294` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:418` |
| `WEAVE_SCATTER_STAGING` | Implementation | `0 = reg_2stage`: double-buffered register transfer; `1 = smem`: per-SM bank-conflict-free SMEM ring handoff; `2 = hbm_dedup`: L2-resident HBM buffer with bf16x2 atomic dedup (names or codes accepted) | 0 (auto `hbm_dedup` when `WEAVE_LOCAL_SCATTER_DEDUP=1`) | Independent Scatter operator staging implementation (`reg_2stage`, XOR-swizzled `smem`, or `hbm_dedup`). | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:169` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1450` |
| `WEAVE_LOCAL_SCATTER_DEDUP` | Implementation | `0 = direct per-hit scatter`; `1 = bf16x2 local merge of multi-expert hits before send` | 0 | Independent Scatter operator local deduplication merging duplicate destination hits on the same GPU before NVLink/xGMI send. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:247` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1433` |

`num_shared_experts` is a Python/C++ API argument rather than an environment knob. Its default is `0` on every MegaMoE entry point; shared-expert buffers are allocated only when it is explicitly non-zero. The gate API (`bf16_mega_gate`) takes it explicitly together with `use_shared_as_routed`.

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
