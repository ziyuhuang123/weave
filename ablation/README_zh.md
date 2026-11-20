[English](README.md) | [简体中文](README_zh.md)

# 消融与实验复现

本目录提供论文消融、代价模型、EPLB 与 profiler 研究的可执行入口。所有脚本都
支持 --dry-run，且不会伪造任何 measured 数字。

## 1. 四臂渐进消融

四个公开实验臂由同一个环境变量选择：

| 预设 | 实验臂 | 空间搜索 | 时间重叠 | 选择性去重 |
|---:|---|---:|---:|---:|
| 0 | Weave | 开 | 开 | 开 |
| 1 | -S | 固定默认配比 | 开 | 开 |
| 2 | -S-T | 固定默认配比 | 关 | 开 |
| 3 | -S-T-C | 固定默认配比 | 关 | 关 |

在单一平台上运行全部实验臂：

    python ablation/run_progressive_ablation.py --backend b300 --presets 0,1,2,3 --dry-run
    python ablation/run_progressive_ablation.py --backend h20 --presets 0,1,2,3 --dry-run
    python ablation/run_progressive_ablation.py --backend mi308x --presets 0,1,2,3 --dry-run

MI308X 实际执行需要指定独立 kernel driver：

    export WEAVE_MI308X_BENCH_RUNNER=/absolute/path/to/mi308x_runner
    python ablation/run_progressive_ablation.py --backend mi308x --presets 0,1,2,3

每个预设使用独立的 DG_JIT_CACHE_DIR，位于 results/ablation 下。预设会改变生成
宏，共享缓存可能掩盖实际执行的实验臂。

### 已验证的 48 格汇总（论文 Table IV）

下表是各平台相对官方基线的 measured 几何平均加速比，属于参考证据，不是
dry-run 脚本重新生成的数字。NVIDIA SM103 (148-SM) 数字来自论文的 fp8xfp4 实测 campaign；当前仓库
中 148-SM SM103 fp8xfp4 仍为 BLOCKED_RUNTIME_HANG，因此新跑的消融应通过 --runner 指向
NVFP4 测试入口 tests/test_nvfp4_mega_moe.py。

| 平台 | S 范围 | Weave | -S | -S-T | -S-T-C |
|---|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | S <= 128（12 格） | 1.165 | 1.076 | 0.915 | 0.913 |
| NVIDIA SM103 (148-SM) | 1K--4K（18 格） | 1.112 | 1.073 | 0.919 | 0.916 |
| NVIDIA SM103 (148-SM) | S >= 8K（18 格） | 1.638 | 1.416 | 1.185 | 1.148 |
| NVIDIA SM103 (148-SM) | 全部（48 格） | 1.301 | 1.192 | 1.010 | 0.996 |
| H20 | 全部（48 格） | 1.162 | 1.137 | 1.053 | 0.947 |
| MI308X EP16 | 全部（48 格） | 2.147 | 1.808 | 1.537 | 1.362 |

## 2. 代价模型与 Oracle 对比

紧凑回放直接读取 data/routing_stats_3hw.json，覆盖四种配置：148-SM SM103 EP8、
H20 EP8、MI308X EP8 和 MI308X EP16。

    python ablation/eval_cost_model_vs_oracle.py

输出：

    results/ablation/cost_model_vs_oracle.json

文件明确分成三部分：

1. local_replay：本次由六个紧凑路由聚合实际计算出的值；
2. paper_measured_reference：论文 48 格 measured 参考汇总；
3. oracle_file_metrics：由 --oracle-json 可选输入计算出的指标。

紧凑文件没有逐专家直方图，也没有全部候选的实测延迟，因此使用文档化的
Zipf(0.9) 代理生成 m_e，并把预测候选中的最小值称为 model oracle。它检查的
是模型内部一致性与候选排序，不是硬件最优性。

完整 measured oracle 扫描应提供如下 JSON：

    {
      "cells": [
        {
          "arch": "b300",
          "model": "qwen38max",
          "S": 128,
          "selected": "c4_8_bm128_s0",
          "predicted_us": 701.2,
          "measured_us": {
            "c4_8_bm128_s0": 712.4,
            "c4_8_bm128_s1": 726.8,
            "c8_4_bm64_s0": 748.1,
            "c8_4_bm64_s1": 733.0
          }
        }
      ]
    }

然后运行：

    python ablation/eval_cost_model_vs_oracle.py --oracle-json /absolute/path/to/oracle.json

### 已验证的论文汇总

- SM103 (148-SM) 48 格：38/48 格与已记录最优候选完全一致；
- 几何平均选择后悔：2.4%（算术平均 2.6%）；
- 所选配置相对基线 1.301x，穷举 oracle 为 1.332x；
- 平均绝对时延预测误差：34.0%。

解释必须明确：绝对时延预测并不完美，但共享开销项在 argmin 中相互抵消，因此
候选排序仍然准确。

## 3. EPLB 兼容性（论文 Table V）

EPLB 研究使用 24 格：六个模型乘以 S in {128, 1024, 4096, 16384}，增加 32 个
冗余专家。先启动启用 EPLB 的部署，再用 EPLB 路由表通过正常 runner 回放同一
组 24 格。

已验证汇总：

| 指标 | 实测值 |
|---|---:|
| Weave 相对官方基线，原始路由 | 1.274x |
| Weave 相对官方基线，启用 EPLB | 1.128x |
| Weave+EPLB 相对官方基线+EPLB | 1.144x |
| 最大/平均 token 不均衡，原始路由 | 1.37 |
| 最大/平均 token 不均衡，启用 EPLB | 1.27 |

EPLB 副本会改变路由直方图。不得复用原始路由的策略 trace 或 JIT 缓存。每次
结果都要保存 EPLB 放置映射与逐 rank 直方图快照。

## 4. WeaveProfiler SW+HW 时间线捕获

公开轨迹是软硬件配对时间线。完整插桩构建使用 WEAVE_PROFILER_LEVEL=2。

### NVIDIA 捕获

    WEAVE_PROFILER_LEVEL=2 nsys profile         --trace=cuda,nvtx,osrt --sample=cpu --cpuctxsw=none         -o results/profiler/sm103_run1         python tests/bench_b300_ep8.py --dry-run

### AMD 捕获

    WEAVE_PROFILER_LEVEL=2 rocprofv3         --hip-trace --rccl-trace --kernel-trace         -o results/profiler/mi308x_run1         -- python tests/bench_mi308x_ep8_ep16.py --ep 16 --dry-run

### 渲染四面板 PNG

    python ablation/capture_weaveprofiler_timeline.py         --trace results/profiler/run.sw_hw.perfetto.json.gz         --out-png results/profiler/run.png         --title "Weave: paired software and hardware timeline"

包装脚本从 WEAVE_PROFILER_ROOT 导入 weaveprof.preview 并调用官方渲染器。解压
后的 JSON 放在 PNG 同目录，不使用临时目录。

仓库内三张出版图：

    docs/figures/weaveprofiler_sm103_timeline.png
    docs/figures/weaveprofiler_h20_timeline.png
    docs/figures/weaveprofiler_mi308x_timeline.png

### 证据规则

- 插桩与未插桩稳态测量分别保存；
- 记录 trace SHA256、GPU 型号、rank 数、S 值与工具版本；
- 出现 HW/SW 窗口长度不一致时必须明确披露，不得静默缩放某一轴；
- 不可用的硬件计数器保持 unavailable，不能填零；
- 只渲染能被 Perfetto trace processor 接受的 trace。
