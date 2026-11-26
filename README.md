[English](README.md) | [简体中文](README_zh.md)

# Weave: Routing-Adaptive Runtime Scheduling for MoE Megakernels

Weave is a routing-adaptive fused MoE megakernel system implemented in unified C++/CUDA/HIP ([Weave: Routing-Adaptive Runtime Warp Scheduling for MoE Megakernels](https://arxiv.org/pdf/2611.27765)), supporting **NVIDIA SM100 / SM103**, **NVIDIA H20 (SM90)**, and **AMD MI308X (CDNA3 gfx942)**. To handle cross-GPU, cross-SM/CU, and cross-stage load imbalance under dynamic token routing, Weave jointly adapts multi-granularity spatial resource ratios, dynamic temporal role transitions, and independent per-operator configurations so cross-GPU communication overlaps tightly with Tensor / Matrix Core compute.

## Architecture and multi-granularity scheduling

| Hardware platform | Language | Supported precision (`WEAVE_PRECISION_MODE`) | Entry point |
|---|---|---|---|
| NVIDIA SM100 / SM103 (148-SM) | CUDA C++ | `0=fp8xfp4` (default), `1=fp4xfp4` (NVFP4), `3=bf16` | `sm100_fp8_fp4_mega_moe.cuh` / `sm100_fp4_fp4_mega_moe.cuh` |
| NVIDIA H20 (SM90, 78-SM) | CUDA C++ | `2=fp8` (`fp8xfp8`) | `sm90_fp8_mega_moe.cuh` |
| AMD MI308X (gfx942, 80-CU) | HIP C++ | `2=fp8` (`fp8xfp8`, single-node EP8 & 2-node EP16) | `gfx942_fp8_mega_moe.cuh` and `amd/ep16/` |

![Weave System Design Overview](docs/figures/weave_design_overview.png)

![Weave Multi-Granularity Spatial Partitioning and Dynamic Temporal Roles](docs/figures/weave_design_rolegraph.png)

Guided by one-time offline hardware profiling and a lightweight online cost model, Weave coordinates **14 unified parameters across 3 orthogonal axes (Spatial 5 / Temporal 3 / Implementation 6)**:

- **Multi-granularity spatial scheduling (Spatial, 5 knobs)**: spans three hardware levels simultaneously—
  1. **Across GPUs**: each GPU independently selects its spatial and temporal schedule $(r_g, s_g)$ from its own live per-step routing histogram (expert tokens $m_e$, unique remote pulls $D_{\text{in}}$, outgoing sends $M_{\text{out}}$), communicating via peer-to-peer readiness queues;
  2. **Across SMs/CUs on the same GPU**: different SMs/CUs on one GPU can run heterogeneous communication-to-compute ratios (mixing `SM 0..k-1` comm-rich SMs/CUs with `SM k..N-1` compute-rich SMs/CUs) and independent active SM/CU subsets for Dispatch ($D$) and Scatter ($K$, $D \neq K$), smoothly tuning GPU-wide communication bandwidth while keeping Tensor / Matrix Cores active on every SM/CU;
  3. **Within each SM/CU**: partitions resident warps/wavefronts across communication, async data movement, and matrix/epilogue execution on NVIDIA SM103 (16 warps/SM), NVIDIA H20 (20 warps/SM), and AMD MI308X (8 Wave64/CU, mixing `(0,8)` compute CUs with `(4,4)` hybrid comm+compute CUs).
- **Dynamic temporal role transitions (Temporal, 3 knobs)**: execution units switch roles across phases within a single kernel—in the **Head phase**, idle compute/epilogue warps assist opening remote pulls up to quota $V_{\text{head}}$ while `WEAVE_READY_GROUP_SIZE` starts GEMM1 every $N$ arrived tiles; in the **Middle phase**, freed communication warps transition into asynchronous FC2 scatter (`comm_only`) or leave direct writes to the epilogue (`epilogue_only`) while tile-ready bitmaps/rings pipeline `GEMM1 -> GEMM2`; in the **Tail phase**, per-token ($k/k$) readiness wakes all warps/waves for cooperative weighted Combine reduction.
- **Independent 4-operator implementation (Implementation, 6 knobs)**: **GEMM1, GEMM2, Dispatch, and Scatter** select configurations independently—`WEAVE_GEMM1_CONFIG` and `WEAVE_GEMM2_CONFIG` independently choose tile shapes, mainloop stages, and multi-stage epilogue register pipelines (`EPI=1..4`); Dispatch configures register-direct pull width (`WEAVE_REG_PULL_BYTES`) and remote-pull deduplication (`WEAVE_DEDUP_MODE`); Scatter selects multi-stage register-direct (`reg`), shared-memory (`smem`), or on-device HBM (`hbm`) staging (`WEAVE_SCATTER_STAGING`).

## Quick start and autotuning

```bash
git clone https://github.com/ziyuhuang123/weave.git
cd weave
python -m pip install -e .
```

On a new hardware deployment, run a one-time calibration pass for your model geometry; `weave_mega_moe` then loads the calibrated profile automatically during inference (see [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md)):

```bash
# 1. One-time offline calibration for the current GPU and model geometry
python -m deep_gemm.mega.autotune --model qwen3.8-2.4t --num-processes 4

# 2. Online inference with automatic schedule selection (or explicit WEAVE_* overrides)
python -c "from deep_gemm.mega import SymmBuffer, weave_mega_moe"
```

## End-to-end performance and bound-resource utilization

![3-Hardware End-to-End Speedup across 6 Models x 5 Batch Sizes](docs/figures/weave_eval_3hw_speedup.png)

![Bound Hardware Resource Utilization % across 6 Models x 5 Batch Sizes x 3 Hardware Platforms](docs/figures/weave_bound_utilization_3hw.png)

Across **90 evaluation cells** (3 hardware platforms x 6 MoE models x 5 global token counts `S in {16, 128, 1K, 8K, 16K}` under real ShareGPT routing; ledger in [results/bound_utilization_90cells_3hw.json](results/bound_utilization_90cells_3hw.json)), bound-resource utilization is `100 * T_bound / measured_latency`, where `T_bound` is the critical-rank roofline floor across HBM bandwidth, interconnect bandwidth, and peak compute. Weave achieves the lowest latency across all 90 cells:

| Platform | Baselines compared | 30-cell speedup (geomean / peak) | Bound-resource util. (geomean / peak) |
|---|---|---|---|
| **8x NVIDIA SM103 (148-SM)**, EP8 | DeepGEMM-MegaMoE, FlashInfer, SGLang, mKernel, FlashMoE, NanoFlow | **`1.219x`** / **`1.97x`** | `42.9%` / **`97.6%`** |
| **8x NVIDIA H20 (SM90)**, EP8 | DeepGEMM-MegaMoE, FlashInfer | **`1.212x`** / **`1.61x`** | `53.6%` / **`85.5%`** |
| **16x AMD MI308X (CDNA3)**, 2-node EP16 | AITER-1stage, vLLM (RCCL + 2-stage CK) | **`1.875x`** / **`2.87x`** | `21.6%` / **`92.0%`** |

## WeaveProfiler software + hardware timelines

![NVIDIA SM103 (148-SM) WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_sm103_timeline.png)

![NVIDIA H20 WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_h20_timeline.png)

![AMD MI308X 16x EP16 WeaveProfiler SW+HW Timeline](docs/figures/weaveprofiler_mi308x_timeline.png)

Each 4-panel timeline is captured from a single execution aligning cycle-accurate on-device software traces with hardware PMU curves (Panel A: aligned hardware utilization curves; Panel B: all physical SMs/CUs; Panel C: full warp/wavefront tracks of a representative SM/CU; Panel D: deterministic micro-zoom). On AMD MI308X, Panel A computes exact dynamic-clock Matrix Core pipe duty (`100 * dSQ_VALU_MFMA_BUSY_CYCLES / (80 * dGRBM_GUI_ACTIVE)`), eliminating static `398.7T` SKU denominator distortion:

- **NVIDIA SM103 (148-SM), 4x EP4, `Qwen3.8-2.4T S=1024` (`FP8 x NVFP4`)**: 200 kHz `nsys` capture across all 148 SMs and all 16 warps of SM 42. Helper pull (`V_HEAD`) populates initial tiles rapidly, sustaining **`92.0%`** peak HBM bandwidth (`72.8%` mean) and **`98.6%`** mean active SMs, while Panels C/D resolve per-warp `MEM d=0..19` and `MMA d=0..1` asynchronous pipeline depth strips and issue pins.
- **NVIDIA H20 (78-SM), 4x EP4, `Qwen3.5-35B S=16384` (`FP8`)**: 200 kHz `nsys` capture across all 78 SMs and all 20 warps of SM 54. Freed communication warps (`W00..W07`) transition directly from remote pull into `smem` async scatter, keeping Tensor Core activity at **`75%–96%`** while Panels C/D (`700–730 us`) show concurrent `MEM d=0..2`, `MMA d=0..1`, and `NVLink Scatter`.
- **AMD MI308X (80-CU), 2-node 16x EP16, `Qwen3.8-2.4T S=16384` (`FP8`)**: `rocprofiler-sdk` DCS hardware capture across all 80 physical CUs (`0–6,633.0 us`, `overflow 0`, `0 spill`) showing co-resident `(0,8)` compute-rich CUs and `(4,4)` hybrid CUs; on hybrid `CU 62` (`3520–3820 us` micro-zoom), `W00..W03` sustain Stage-1/Stage-2 MFMA compute while `W04..W07` concurrently overlap inter-node RDMA and intra-node `xGMI` transfers. Calibrated GPU-wide FP8 Matrix Core pipe duty reaches **`62.2%`** peak (`41.4%` mean; `73.7%` effective across the `540/640` compute wave pool), with **`33.1%`** peak HBM3 bandwidth (`20.5%` mean), **`60.5%`** peak intra-node `xGMI TX`, **`15.0%`** peak `xGMI RX`, and **`100.0%`** active CUs (inter-node RDMA NIC counters are omitted rather than zero-filled when bypassed by GPU-initiated rings).

## Unified core WEAVE_* parameters

Weave exposes **14 unified parameters across 3 axes** (Spatial 5 / Temporal 3 / Implementation 6) covering cross-SM/CU and intra-SM/CU partitioning, dynamic temporal role transitions, and independent per-operator configurations. Unset parameters (`0` / `auto`) are selected automatically from the calibrated hardware profile and live routing statistics; see [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md) and [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md).

### Spatial

| Parameter | Category | Values (`0` = auto) | Mechanism | Code (`host` / `kernel`) |
|---|---|---|---|---|
| `WEAVE_DISPATCH_ACTIVE_SMS` | Spatial | `0=auto`, `D=1..N` (`D != K`) | Cross-SM/CU active Dispatch subset; remaining SMs/CUs prefetch and start GEMM1 immediately | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:189` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:4073` |
| `WEAVE_SCATTER_ACTIVE_SMS` | Spatial | `0=auto`, `K=1..N` (`K != D`) | Cross-SM/CU active Scatter subset; tail Combine still uses all SMs/CUs | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:204` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:554` |
| `WEAVE_DISPATCH_WARPS` | Spatial | `0=auto`, `4`, `8` | Intra-SM communication warps per CTA assigned to remote token/scale pulling | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:162` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1134` |
| `WEAVE_SCATTER_WARPS` | Spatial | `0=auto`, `4`, `8` | Intra-SM/CU communication warps/waves dedicated to async FC2 scatter | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:195` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:529` |
| `WEAVE_CU_WAVE_SPLIT` | Spatial | MI308X: `auto, 0-8, 4-4, 8-0`; H20: `auto` or `E` | MI308X per-CU 8-Wave64 comm/compute split; H20 GEMM1/GEMM2 expert-stream interleaving | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:183` / `deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip:80` |

### Temporal

| Parameter | Category | Values (`0` = auto) | Mechanism | Code (`host` / `kernel`) |
|---|---|---|---|---|
| `WEAVE_V_HEAD` | Temporal | `0=auto`, `N=1..65536` | Opening pull quota for helper warps before transitioning into GEMM1 | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:193` / `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_READY_GROUP_SIZE` | Temporal | `0=full expert`, `1, 2, 4, 8, 16` | Publishes readiness every `N` `BLOCK_M` tiles so GEMM1 starts before full expert arrives | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:131` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:111` |
| `WEAVE_SCATTER_OWNER` | Temporal | `0=epilogue_only`, `1=comm_only` | Assigns FC2 cross-GPU scatter either to epilogue warps directly or to freed comm warps | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:276` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1780` |

### Implementation

| Parameter | Category | Values (`0` = auto) | Mechanism | Code (`host` / `kernel`) |
|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | Implementation | `0=fp8xfp4`, `1=fp4xfp4`, `2=fp8`, `3=bf16` | Tensor/Matrix Core numeric format across GEMM1 and GEMM2 | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:405` / `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_GEMM1_CONFIG` | Implementation | `0=auto` or `BM[:STORE_BM[:BK[:ST[:EPI]]]]` (`EPI=1..4`) | Independent FC1 tile & multi-stage mainloop/epilogue configuration | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:129` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1253` |
| `WEAVE_GEMM2_CONFIG` | Implementation | `0=auto` or `BM[:STORE_BM[:BK[:ST[:EPI]]]]` (decoupled) | Independent FC2 tile & multi-stage configuration decoupled from GEMM1 | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:132` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1255` |
| `WEAVE_REG_PULL_BYTES` | Implementation | `0=SMEM`, `1024/2048/4096` (NV), `8192` (MI308X) | Dispatch register-direct pull chunk width per warp/wavefront (preserves SMEM/LDS for GEMM) | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:135` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:143` |
| `WEAVE_DEDUP_MODE` | Implementation | `0=off`, `1=on` | Dispatch remote-pull deduplication: pulls unique `(src_rank, token)` once and fans out locally | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:170` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:443` |
| `WEAVE_SCATTER_STAGING` | Implementation | `0=reg`, `1=smem`, `2=hbm` | Scatter staging path: multi-stage register-direct (`reg`), shared memory (`smem`), or HBM buffer (`hbm`) | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:279` / `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1786` |

## Examples, baselines, and ablation studies

- **Launch presets**: [examples/README.md](examples/README.md) provides reproducible presets across SM103, H20, and MI308X plus the WeaveProfiler timeline driver.
- **Baselines**: [baselines/README.md](baselines/README.md) documents baseline setup, multi-rank verification, and matched-precision comparisons.
- **Ablation & EPLB**: [ablation/README.md](ablation/README.md) covers the 4-arm progressive ablation (`Weave`, `-S`, `-S-T`, `-S-T-C`) and EPLB compatibility; across 48 evaluation cells, full Weave reaches `1.301x / 1.162x / 2.147x` geomean speedup on SM103 / H20 / MI308X EP16 (`1.192x / 1.137x / 1.808x` without Spatial `-S`; `1.010x / 1.053x / 1.537x` without `-S-T`), and fresh 48-cell single-node EP8 receipts re-measured on the unified 14-knob codebase are archived in [results/ablation/](results/ablation/) (H20 48-cell `1.091x -> 1.036x -> 1.022x`, `S <= 128` `1.194x -> 1.111x -> 1.102x`; single-node MI308X EP8 `S <= 128` `1.769x -> 1.501x -> 1.387x -> 1.349x`, 48-cell `0.977x -> 0.930x -> 0.877x -> 0.828x`).

## Citation

Ziyu Huang, Yangjie Zhou, Chenhao Zhu, Hongzhe Yan, Tianwei Qiao, Sicheng He, Zihan Liu, Peng Yu, Shulai Zhang, Xingxun Tang, Xinhao Luo, Minyi Guo, Xiu Lin, Yinghao Yu, Guodong Yang, Liping Zhang, Shixuan Sun, Qingyun Chen, and Jingwen Leng. Weave: Routing-Adaptive Runtime Warp Scheduling for MoE Megakernels. arXiv:2611.27765, 2026. [PDF](https://arxiv.org/pdf/2611.27765) | [arXiv](https://arxiv.org/abs/2611.27765)

```bibtex
@misc{huang2026weaveroutingadaptiveruntimewarp,
      title={Weave: Routing-Adaptive Runtime Warp Scheduling for MoE Megakernels},
      author={Ziyu Huang and Yangjie Zhou and Chenhao Zhu and Hongzhe Yan and Tianwei Qiao and Sicheng He and Zihan Liu and Peng Yu and Shulai Zhang and Xingxun Tang and Xinhao Luo and Minyi Guo and Xiu Lin and Yinghao Yu and Guodong Yang and Liping Zhang and Shixuan Sun and Qingyun Chen and Jingwen Leng},
      year={2026},
      eprint={2611.27765},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/pdf/2611.27765},
}
```
