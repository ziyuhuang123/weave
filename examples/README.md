# Reproducible Weave Presets (`Spatial / Temporal / Implementation`)

This directory contains executable presets for NVIDIA SM103, NVIDIA H20, and AMD Instinct MI308X (`EP8` and 2-node `EP16`). Every preset uses the unified `WEAVE_*` surface: the 14 public parameters documented in [../docs/TUNABLE_PARAMETERS.md](../docs/TUNABLE_PARAMETERS.md).

## Files

- `sm103_qwen38max_ep8_s1024.sh` — NVIDIA SM103 (148-SM), `fp8xfp4` / `fp4xfp4`
- `h20_qwen38max_ep8_s1024.sh` — NVIDIA H20 (78-SM), `fp8xfp8`
- `mi308x_qwen38max_ep8_s1024.sh` — AMD MI308X (80-CU), single-node EP8
- `mi308x_qwen38max_ep16_s128.sh` — AMD MI308X (80-CU), 2-node 16-GPU EP16
- `profile_and_render_timeline.py` — 3-axis `WEAVE_*` parameter table printer & WeaveProfiler timeline renderer driver

## 3-Axis Taxonomy

| Axis | Responsibility | Unified `WEAVE_*` Parameters |
|---|---|---|
| **Spatial** (5) | Physical SM/CU-level and intra-SM/CU Warp/Wave-level hardware resource ratios | `WEAVE_DISPATCH_ACTIVE_SMS`, `WEAVE_SCATTER_ACTIVE_SMS`, `WEAVE_DISPATCH_WARPS`, `WEAVE_SCATTER_WARPS`, `WEAVE_CU_WAVE_SPLIT` |
| **Temporal** (3) | Cross-stage role handoff, compute-communication overlap, and sync granularity | `WEAVE_V_HEAD`, `WEAVE_READY_GROUP_SIZE`, `WEAVE_SCATTER_OWNER` |
| **Implementation** (6) | Independent GEMM1, GEMM2, Dispatch, and Scatter single-operator configurations (`M -> blockM` synchronously locks `blockN/K`, `mainloop_stages`, `epi_M/N/K`, `epi_stages`) | `WEAVE_PRECISION_MODE`, `WEAVE_GEMM1_CONFIG`, `WEAVE_GEMM2_CONFIG`, `WEAVE_REG_PULL_BYTES`, `WEAVE_DEDUP_MODE`, `WEAVE_SCATTER_STAGING` |

The tuned presets in this directory were measured together with the internal pins of the six knobs retired from the public surface (`WEAVE_SM0_STREAM_SCHED`, `WEAVE_REDUCE_WARPS`, `WEAVE_REG_BALANCE`, `WEAVE_READY_QUEUE`, `WEAVE_COMBINE_SYNC_CHUNK`, `WEAVE_LOCAL_SCATTER_DEDUP`), so the scripts still export them. `WEAVE_BLOCK_M_L1` and `WEAVE_BLOCK_M_L2` are the legacy spellings of `WEAVE_GEMM1_CONFIG` and `WEAVE_GEMM2_CONFIG`; `WEAVE_ADAPT_GEMM_CONFIG` (alias `WEAVE_ADAPT_BM`) is no longer public (host shape-adaptive tuple selection is always on when GEMM1/GEMM2 config are 0) but stays accepted as an internal legacy override, so the scripts may still export it; the NVIDIA host reads both, the MI308X build scripts read only the legacy spellings, which is why the scripts keep them.

## Quick Start

```bash
python examples/profile_and_render_timeline.py --preset sm103 --no-render
python examples/profile_and_render_timeline.py --preset h20 --no-render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --no-render
```
