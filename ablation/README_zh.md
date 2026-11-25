[English](README.md) | [简体中文](README_zh.md)

# 消融实验与代价模型复现

本目录提供论文中渐进消融、代价模型评估、EPLB 兼容性与 WeaveProfiler 时间线采集的可执行脚本（所有脚本均支持 `--dry-run` 查看完整运行计划）。

## 1. 四臂渐进消融（Table IV）

四组实验臂依次移除空间调度（`-S`）、时间重叠（`-S-T`）与选择性跨卡去重（`-S-T-C`）：

| 预设编号 | 实验臂 | 空间自适应配比 | 时间角色重叠 | 选择性拉取去重 |
|---:|---|---:|---:|---:|
| 0 | `Weave` | 开 | 开 | 开 |
| 1 | `-S` | 固定默认配比 | 开 | 开 |
| 2 | `-S-T` | 固定默认配比 | 关 | 开 |
| 3 | `-S-T-C` | 固定默认配比 | 关 | 关 |

各实验臂相对于完整 `Weave` 预设的关键参数变化：

| 实验臂 | 参数增量（其余参数保持实验臂 0 取值） |
|---|---|
| `-S` | SM103：`WEAVE_SCATTER_WARPS=0`、`WEAVE_GEMM1_CONFIG=128`、`WEAVE_GEMM2_CONFIG=128`、`WEAVE_REG_PULL_BYTES=0`；H20：`WEAVE_GEMM1_CONFIG=0`、`WEAVE_GEMM2_CONFIG=0`；MI308X：`WEAVE_CU_WAVE_SPLIT=0-8` |
| `-T` | `WEAVE_V_HEAD=65536`、`WEAVE_READY_GROUP_SIZE=0` |
| `-C` | `WEAVE_DEDUP_MODE=0` |

查看或运行四臂消融：

```bash
python ablation/run_progressive_ablation.py --backend sm103 --presets 0,1,2,3 --dry-run
python ablation/run_progressive_ablation.py --backend h20 --presets 0,1,2,3 --dry-run
python ablation/run_progressive_ablation.py --backend mi308x --presets 0,1,2,3 --dry-run
```

每个实验臂自动使用独立的 `DG_JIT_CACHE_DIR` 编译缓存目录以保证宏配置隔离。48 组评测点上相对各平台官方基线的实测几何平均加速比汇总如下：

| 平台 | `S` 范围 | `Weave` | `-S` | `-S-T` | `-S-T-C` |
|---|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | `S <= 128`（12 格） | 1.165 | 1.076 | 0.915 | 0.913 |
| NVIDIA SM103 (148-SM) | `1K–4K`（18 格） | 1.112 | 1.073 | 0.919 | 0.916 |
| NVIDIA SM103 (148-SM) | `S >= 8K`（18 格） | 1.638 | 1.416 | 1.185 | 1.148 |
| NVIDIA SM103 (148-SM) | 全部（48 格） | 1.301 | 1.192 | 1.010 | 0.996 |
| NVIDIA H20 | 全部（48 格） | 1.162 | 1.137 | 1.053 | 0.947 |
| AMD MI308X EP16 | 全部（48 格） | 2.147 | 1.808 | 1.537 | 1.362 |

## 2. 代价模型选参精度验证

使用紧凑路由统计回放验证代价模型的候选排序精度：

```bash
python ablation/eval_cost_model_vs_oracle.py
```

输出文件 `results/ablation/cost_model_vs_oracle.json`。在 48 个 SM103 评测格上，代价模型在 38/48 格命中穷举最优候选，几何平均选择遗憾仅 `2.4%`（所选配置加速比 `1.301x` vs 穷举最优 `1.332x`）。

## 3. EPLB 专家负载均衡兼容性（Table V）

在 6 个模型 x `S in {128, 1024, 4096, 16384}`（共 24 格，配置 32 个冗余专家副本）下，Weave 与动态专家重布局（EPLB）正交叠加：

| 指标 | 实测值 |
|---|---:|
| Weave 相对官方基线（原始路由） | `1.274x` |
| Weave 相对官方基线（启用 EPLB） | `1.128x` |
| Weave + EPLB 相对官方基线 + EPLB | `1.144x` |
| 最大/平均 Token 不均衡度（原始 -> EPLB） | `1.37 -> 1.27` |

## 4. WeaveProfiler 软硬件对齐时间线采集与渲染

启用 `WEAVE_PROFILER_LEVEL=2` 同步采集片上周期级轨迹与硬件性能计数器，并渲染四面板总览图：

```bash
# NVIDIA (nsys 200 kHz GPU metrics + WeaveProfiler)
WEAVE_PROFILER_LEVEL=2 nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=all \
    -o results/profiler/sm103_run1 python tests/bench_b300_ep8.py --dry-run

# AMD (rocprofiler-sdk PMC + WeaveProfiler)
WEAVE_PROFILER_LEVEL=2 rocprofv3 --hip-trace --kernel-trace \
    -o results/profiler/mi308x_run1 -- python tests/bench_mi308x_ep8_ep16.py --ep 16 --dry-run

# 渲染四面板 SW+HW 时间线 PNG
python ablation/capture_weaveprofiler_timeline.py \
    --trace results/profiler/run.sw_hw.perfetto.json.gz \
    --out-png results/profiler/run.png
```

仓库内生成的三张硬件时间线图位于：
- `docs/figures/weaveprofiler_sm103_timeline.png`
- `docs/figures/weaveprofiler_h20_timeline.png`
- `docs/figures/weaveprofiler_mi308x_timeline.png`
