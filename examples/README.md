# Reproducible Weave Presets (`Spatial / Temporal / Implementation`)

This directory contains executable presets for NVIDIA SM103, NVIDIA H20, and AMD Instinct MI308X (`EP8` and 2-node `EP16`) together with the WeaveProfiler timeline driver, built over the unified **14 `WEAVE_*` parameters** in [../docs/TUNABLE_PARAMETERS.md](../docs/TUNABLE_PARAMETERS.md).

## Files

- `sm103_qwen38max_ep8_s1024.sh` — NVIDIA SM103 (148-SM), `fp8xfp4` / `fp4xfp4`
- `h20_qwen38max_ep8_s1024.sh` — NVIDIA H20 (78-SM), `fp8xfp8`
- `mi308x_qwen38max_ep8_s1024.sh` — AMD MI308X (80-CU), single-node EP8
- `mi308x_qwen38max_ep16_s128.sh` — AMD MI308X (80-CU), 2-node 16-GPU EP16
- `profile_and_render_timeline.py` — 3-axis `WEAVE_*` parameter table printer & WeaveProfiler timeline renderer driver

## 3-Axis Taxonomy

| Axis | Responsibility | Unified `WEAVE_*` Parameters |
|---|---|---|
| **Spatial** (5) | Cross-SM/CU heterogeneous mixing, independent Dispatch/Scatter SM subsets, and intra-SM/CU warp/wave ratios | `WEAVE_DISPATCH_ACTIVE_SMS`, `WEAVE_SCATTER_ACTIVE_SMS`, `WEAVE_DISPATCH_WARPS`, `WEAVE_SCATTER_WARPS`, `WEAVE_CU_WAVE_SPLIT` |
| **Temporal** (3) | Opening helper pull, fine-grained `N`-tile readiness, and middle async Scatter role transitions | `WEAVE_V_HEAD`, `WEAVE_READY_GROUP_SIZE`, `WEAVE_SCATTER_OWNER` |
| **Implementation** (6) | Independent GEMM1, GEMM2, Dispatch, and Scatter tile, multi-stage pipeline, and staging configurations | `WEAVE_PRECISION_MODE`, `WEAVE_GEMM1_CONFIG`, `WEAVE_GEMM2_CONFIG`, `WEAVE_REG_PULL_BYTES`, `WEAVE_DEDUP_MODE`, `WEAVE_SCATTER_STAGING` |

## Quick Start

```bash
python examples/profile_and_render_timeline.py --preset sm103 --no-render
python examples/profile_and_render_timeline.py --preset h20 --no-render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --no-render
```
