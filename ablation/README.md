[English](README.md) | [简体中文](README_zh.md)

# Ablation and Cost-Model Reproduction

This directory contains executable scripts for the paper's progressive ablation, cost-model ranking study, EPLB compatibility evaluation, and WeaveProfiler timeline capture (all scripts support `--dry-run`).

## 1. Four-arm progressive ablation (Table IV)

The four arms progressively disable spatial scheduling (`-S`), temporal role overlap (`-S-T`), and selective remote-pull deduplication (`-S-T-C`):

| Preset | Arm | Spatial adaptation | Temporal overlap | Selective dedup |
|---:|---|---:|---:|---:|
| 0 | `Weave` | on | on | on |
| 1 | `-S` | fixed default ratio | on | on |
| 2 | `-S-T` | fixed default ratio | off | on |
| 3 | `-S-T-C` | fixed default ratio | off | off |

Key parameter deltas relative to full `Weave` (Arm 0):

| Arm | Parameter delta |
|---|---|
| `-S` | SM103: `WEAVE_SCATTER_WARPS=0`, `WEAVE_GEMM1_CONFIG=128`, `WEAVE_GEMM2_CONFIG=128`, `WEAVE_REG_PULL_BYTES=0`; H20: `WEAVE_GEMM1_CONFIG=0`, `WEAVE_GEMM2_CONFIG=0`; MI308X: `WEAVE_CU_WAVE_SPLIT=0-8` |
| `-T` | `WEAVE_V_HEAD=65536`, `WEAVE_READY_GROUP_SIZE=0` |
| `-C` | `WEAVE_DEDUP_MODE=0` |

Inspect or run all four arms:

```bash
python ablation/run_progressive_ablation.py --backend sm103 --presets 0,1,2,3 --dry-run
python ablation/run_progressive_ablation.py --backend h20 --presets 0,1,2,3 --dry-run
python ablation/run_progressive_ablation.py --backend mi308x --presets 0,1,2,3 --dry-run
```

Each arm uses an isolated `DG_JIT_CACHE_DIR` under `results/ablation`. Measured 48-cell geometric-mean speedups over each platform's official baseline:

| Platform | `S` range | `Weave` | `-S` | `-S-T` | `-S-T-C` |
|---|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | `S <= 128` (12 cells) | 1.165 | 1.076 | 0.915 | 0.913 |
| NVIDIA SM103 (148-SM) | `1K–4K` (18 cells) | 1.112 | 1.073 | 0.919 | 0.916 |
| NVIDIA SM103 (148-SM) | `S >= 8K` (18 cells) | 1.638 | 1.416 | 1.185 | 1.148 |
| NVIDIA SM103 (148-SM) | All (48 cells) | 1.301 | 1.192 | 1.010 | 0.996 |
| NVIDIA H20 | All (48 cells) | 1.162 | 1.137 | 1.053 | 0.947 |
| AMD MI308X EP16 | All (48 cells) | 2.147 | 1.808 | 1.537 | 1.362 |

## 2. Cost-model candidate ranking

Replay the analytical cost model across recorded routing statistics:

```bash
python ablation/eval_cost_model_vs_oracle.py
```

Writes `results/ablation/cost_model_vs_oracle.json`. Across the 48 SM103 cells, the cost model matches the recorded optimum on 38/48 cells with `2.4%` geomean regret (`1.301x` selected vs. `1.332x` exhaustive oracle).

## 3. EPLB compatibility (Table V)

Across 24 workloads (6 models x `S in {128, 1024, 4096, 16384}` with 32 redundant expert replicas), Weave composes cleanly with expert-parallel load balancing (EPLB):

| Metric | Measured value |
|---|---:|
| Weave over official baseline (original routing) | `1.274x` |
| Weave over official baseline (with EPLB) | `1.128x` |
| Weave + EPLB over official baseline + EPLB | `1.144x` |
| Max/mean token imbalance (original -> with EPLB) | `1.37 -> 1.27` |

## 4. WeaveProfiler SW+HW timeline capture

Capture paired cycle-accurate software traces and hardware counters with `WEAVE_PROFILER_LEVEL=2` and render the 4-panel overview figure:

```bash
# NVIDIA (nsys 200 kHz GPU metrics + WeaveProfiler)
WEAVE_PROFILER_LEVEL=2 nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=all \
    -o results/profiler/sm103_run1 python tests/bench_b300_ep8.py --dry-run

# AMD (rocprofiler-sdk PMC + WeaveProfiler)
WEAVE_PROFILER_LEVEL=2 rocprofv3 --hip-trace --kernel-trace \
    -o results/profiler/mi308x_run1 -- python tests/bench_mi308x_ep8_ep16.py --ep 16 --dry-run

# Render 4-panel SW+HW timeline PNG
python ablation/capture_weaveprofiler_timeline.py \
    --trace results/profiler/run.sw_hw.perfetto.json.gz \
    --out-png results/profiler/run.png
```

Checked-in publication timelines:
- `docs/figures/weaveprofiler_sm103_timeline.png`
- `docs/figures/weaveprofiler_h20_timeline.png`
- `docs/figures/weaveprofiler_mi308x_timeline.png`
