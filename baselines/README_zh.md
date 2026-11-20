[English](README.md) | [简体中文](README_zh.md)

# 基线复现

本目录说明 Weave 评测所使用的官方基线，覆盖 NVIDIA B300、NVIDIA H20 和
AMD MI308X 的 EP8 与 EP16 配置。所有平台遵循同一测量契约：

- 计时前，每个 rank 都检查完整的本地输出；
- 计时不包含模型加载、分词、路由生成与图捕获；
- TFLOPS 一律为单卡值，绝不报告集群聚合值；
- 缺少依赖或缺少任一 rank 回执时直接报错，不填零，不伪造结果。

启动脚本负责生成命令计划并执行基线，不会生成虚构的 measured 数字。任何
CPU 主机都可以用 --dry-run 查看完整命令。

## 1. 安装

统一安装入口：

    bash baselines/install_baselines.sh nvidia   # B300 或 H20
    bash baselines/install_baselines.sh amd      # MI308X

每个栈应使用独立的 Python 环境：上游 DeepGEMM 基线与本仓库都提供 `deep_gemm`
导入名，装进同一环境会互相遮蔽。建议把 BASELINE_ROOT 指向容量充足的持久化目录。
CUDA 主机需要匹配版本的
CUDA Toolkit、NCCL，以及 DeepEP 所需的 NVSHMEM。ROCm 主机需要可用的
RCCL、HIP 和 ROCm PyTorch。安装脚本中的推理引擎 wheel 只是起点；正式对比
时应把实验清单中的精确 wheel 版本固定下来。

环境自检：

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
    python -m pip show deep-gemm flashinfer sglang vllm 2>/dev/null || true

## 2. NVIDIA B300：8x EP8

Figure 7 的 B300 扫描使用 8 个 rank，每 rank 一张 GPU。官方 DeepGEMM 的两条
精度臂由 mma_type 选择：

| 基线 | 精度 | 命令 |
|---|---|---|
| 官方 DeepGEMM MegaMoE | fp8xfp4 | python baselines/run_b300_baselines.py --baseline official_deepgemm --precision fp8xfp4 |
| 官方 DeepGEMM MegaMoE | fp4xfp4 (NVFP4) | python baselines/run_b300_baselines.py --baseline official_deepgemm --precision fp4xfp4 |
| DeepEP + DeepGEMM 分离式 | 与激活精度一致 | python baselines/run_b300_baselines.py --baseline deepep_deepgemm --deepep-runner /absolute/path/to/deepep_bench.py |
| FlashInfer MoE | 由引擎选择 | python baselines/run_b300_baselines.py --baseline flashinfer --flashinfer-runner /absolute/path/to/flashinfer_bench.py |
| SGLang | 模型 checkpoint | python baselines/run_b300_baselines.py --baseline sglang --model-path /absolute/path/to/checkpoint |
| vLLM | 模型 checkpoint | python baselines/run_b300_baselines.py --baseline vllm --model-path /absolute/path/to/checkpoint |

DeepEP 分离式基线必须与 Weave 使用同一份真实路由张量。第一次计时迭代前，
要对所有 rank 的完整输出逐元素执行正确性门禁。不能只测 all-to-all 通信时间
就把它当作完整 MoE 基线。

## 3. NVIDIA H20：8x EP8

| 基线 | 命令 |
|---|---|
| 官方 DeepGEMM SM90 MegaMoE | python baselines/run_h20_baselines.py --baseline official_deepgemm |
| DeepEP + DeepGEMM 分离式 | python baselines/run_h20_baselines.py --baseline deepep_deepgemm --deepep-runner /absolute/path/to/deepep_bench.py |
| SGLang | python baselines/run_h20_baselines.py --baseline sglang --model-path /absolute/path/to/checkpoint |
| vLLM | python baselines/run_h20_baselines.py --baseline vllm --model-path /absolute/path/to/checkpoint |

官方臂使用 SM90 FP8 MegaMoE 及其对称缓冲区分配器，不是把 B300 调度器改编译
到旧架构。仓库自带的官方 wrapper 运行固定的 SM90 形状覆盖套件；若要复现论文的
六模型 x S 网格，请向外部官方 DeepGEMM 驱动，或向分离式与推理引擎 runner，
传入对应的形状。

## 4. AMD MI308X：8x EP8 与 16x EP16

EP8 为单节点 8 卡；EP16 为双节点、每节点 8 卡、全局 16 rank。

| 基线 | EP | 命令 |
|---|---:|---|
| 官方 RCCL + AITER | 8 或 16 | python baselines/run_mi308x_baselines.py --baseline rccl_aiter --ep 16 --rccl-runner /absolute/path/to/runner.py |
| AITER 一阶段融合 MoE | 8 或 16 | python baselines/run_mi308x_baselines.py --baseline aiter_1stage --ep 8 |
| AITER 两阶段 CK MoE | 8 或 16 | python baselines/run_mi308x_baselines.py --baseline aiter_2stage --ep 8 |
| vLLM ROCm | 8 或 16 | python baselines/run_mi308x_baselines.py --baseline vllm --ep 16 --model-path /absolute/path/to/checkpoint |
| SGLang ROCm | 8 或 16 | python baselines/run_mi308x_baselines.py --baseline sglang --ep 16 --model-path /absolute/path/to/checkpoint |

可通过以下环境变量显式指定测量驱动：

    WEAVE_AITER_1STAGE_BENCH=/absolute/path/to/one_stage_bench.py
    WEAVE_AITER_2STAGE_BENCH=/absolute/path/to/two_stage_bench.py
    WEAVE_RCCL_AITER_BENCH=/absolute/path/to/rccl_aiter_bench.py
    WEAVE_MI308X_BENCH_RUNNER=/absolute/path/to/kernel_runner

双节点实验应通过站点作业调度器启动，保证 rank 到节点的映射固定。必须保留
路由清单、rank 映射和逐 rank 计时回执。静默丢失 rank 的运行不构成有效 EP16
测量。

## 5. 一键查看命令计划

    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_h20_baselines.py --baseline all --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 8 --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 16 --dry-run

每个计划都是 JSON，列出每个基线对应的完整命令。实际执行遇到第一个非零
退出码即停止，并在日志中保留该命令。

## 6. 结果目录格式

每个基线、模型、S 值和 rank 数对应一个目录。一个合格 cell 至少包含：

    metadata.json       模型形状、精度、world size、版本与哈希
    rank_<r>.json       全输出正确性回执与延迟样本
    summary.json       跨 rank 中位数、最慢 rank 中位数、单卡 TFLOPS
    command.txt         实际 argv 与启动环境

聚合顺序固定：先检查全部 rank，再计算跨 rank 中位数和最慢 rank 中位数，两者
都保留。论文在具体位置会明确使用的是哪一种统计量。
