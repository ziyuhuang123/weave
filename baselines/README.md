[English](README.md) | [简体中文](README_zh.md)

# Baseline Reproduction

This directory documents the official baselines used by the Weave evaluation.
It covers NVIDIA B300, NVIDIA H20, and AMD MI308X for both EP8 and EP16. The
launchers preserve one measurement contract across all platforms:

- every rank checks the complete rank-local output before timing;
- timing excludes model loading, tokenization, routing generation, and graph
  capture;
- the reported TFLOPS value is per GPU, never a cluster aggregate;
- a missing dependency or missing per-rank receipt is an error, not a zero.

The scripts are command planners and launchers. They never invent a measured
number. Use --dry-run to inspect a plan on any CPU-only host.

## 1. Installation

The clean source installation helper is:

    bash baselines/install_baselines.sh nvidia   # B300 or H20 host
    bash baselines/install_baselines.sh amd      # MI308X host

Set BASELINE_ROOT to a persistent disk directory. CUDA hosts need a matching
CUDA toolkit, NCCL, and NVSHMEM where required by DeepEP. ROCm hosts need a
working RCCL installation, HIP, and a ROCm PyTorch build. The inference-engine
wheels in the helper are a starting point; reproduce the exact wheel versions
recorded in your experiment manifest when comparing against a published table.

Verify the environment:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
    python -m pip show deep-gemm flashinfer sglang vllm 2>/dev/null || true

## 2. NVIDIA B300: 8x EP8

The Figure 7 B300 sweep uses eight ranks, one GPU per rank. The two official
DeepGEMM precision arms are selected by mma_type:

| Arm | Precision | Command |
|---|---|---|
| Official DeepGEMM MegaMoE | fp8xfp4 | python baselines/run_b300_baselines.py --baseline official_deepgemm --precision fp8xfp4 |
| Official DeepGEMM MegaMoE | fp4xfp4 (NVFP4) | python baselines/run_b300_baselines.py --baseline official_deepgemm --precision fp4xfp4 |
| DeepEP + DeepGEMM split | same activation precision | python baselines/run_b300_baselines.py --baseline deepep_deepgemm --deepep-runner /absolute/path/to/deepep_bench.py |
| FlashInfer MoE | selected by the engine | python baselines/run_b300_baselines.py --baseline flashinfer --flashinfer-runner /absolute/path/to/flashinfer_bench.py |
| SGLang | model checkpoint | python baselines/run_b300_baselines.py --baseline sglang --model-path /absolute/path/to/checkpoint |
| vLLM | model checkpoint | python baselines/run_b300_baselines.py --baseline vllm --model-path /absolute/path/to/checkpoint |

The DeepEP split baseline must use the same real routing tensor as Weave. Its
correctness gate is applied to every element of every rank-local output before
the first timed iteration. Do not replace it with an all-to-all timing alone.

## 3. NVIDIA H20: 8x EP8

| Arm | Command |
|---|---|
| Official DeepGEMM SM90 MegaMoE | python baselines/run_h20_baselines.py --baseline official_deepgemm |
| DeepEP + DeepGEMM split | python baselines/run_h20_baselines.py --baseline deepep_deepgemm --deepep-runner /absolute/path/to/deepep_bench.py |
| SGLang | python baselines/run_h20_baselines.py --baseline sglang --model-path /absolute/path/to/checkpoint |
| vLLM | python baselines/run_h20_baselines.py --baseline vllm --model-path /absolute/path/to/checkpoint |

The official arm uses the SM90 FP8 MegaMoE implementation and its own symmetric
buffer allocator. It is not the B300 scheduler compiled for an older
architecture. H20 results should be collected with the same S values and the
same six model shapes used by the paper.

## 4. AMD MI308X: 8x EP8 and 16x EP16

EP8 is one node with eight visible GPUs. EP16 is two nodes with eight GPUs per
node and a 16-rank process group.

| Arm | EP | Command |
|---|---:|---|
| Official RCCL + AITER | 8 or 16 | python baselines/run_mi308x_baselines.py --baseline rccl_aiter --ep 16 --rccl-runner /absolute/path/to/runner.py |
| AITER one-stage fused MoE | 8 or 16 | python baselines/run_mi308x_baselines.py --baseline aiter_1stage --ep 8 |
| AITER two-stage CK MoE | 8 or 16 | python baselines/run_mi308x_baselines.py --baseline aiter_2stage --ep 8 |
| vLLM ROCm | 8 or 16 | python baselines/run_mi308x_baselines.py --baseline vllm --ep 16 --model-path /absolute/path/to/checkpoint |
| SGLang ROCm | 8 or 16 | python baselines/run_mi308x_baselines.py --baseline sglang --ep 16 --model-path /absolute/path/to/checkpoint |

Set the selected measurement driver explicitly with one of:

    WEAVE_AITER_1STAGE_BENCH=/absolute/path/to/one_stage_bench.py
    WEAVE_AITER_2STAGE_BENCH=/absolute/path/to/two_stage_bench.py
    WEAVE_RCCL_AITER_BENCH=/absolute/path/to/rccl_aiter_bench.py
    WEAVE_MI308X_BENCH_RUNNER=/absolute/path/to/kernel_runner

For the two-node arm, launch the runner with the site job scheduler so that
rank-to-node placement is deterministic. Preserve the route manifest, rank
mapping, and per-rank timing receipts. A benchmark that silently drops a rank
is not a valid EP16 measurement.

## 5. One-command plans

    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_h20_baselines.py --baseline all --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 8 --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 16 --dry-run

Each plan is a JSON object with one command per baseline. Execution stops at
the first nonzero exit code and preserves the command in the terminal log.

## 6. Result schema

Store one directory per baseline, model, S value, and rank count. A conforming
cell contains:

    metadata.json       model shape, precision, world size, versions, hashes
    rank_<r>.json       complete-output correctness receipt and latency samples
    summary.json        cross-rank median, slowest-rank median, per-GPU TFLOPS
    command.txt         exact argv and environment used for the launch

The aggregation rule is fixed: check all ranks first, then compute the
cross-rank median and the slowest-rank median. Keep both. The paper reports the
slowest-rank or cross-rank statistic explicitly wherever it is used.
