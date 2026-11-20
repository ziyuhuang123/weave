[English](README.md) | [简体中文](README_zh.md)

# Baseline Reproduction

This directory documents the official baselines used by the Weave evaluation.
It covers NVIDIA SM103 (148-SM), NVIDIA H20, and AMD MI308X for both EP8 and EP16. The
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

    bash baselines/install_baselines.sh nvidia   # 148-SM SM103 or H20 host
    bash baselines/install_baselines.sh amd      # MI308X host

Run each stack in a dedicated Python environment: the upstream DeepGEMM
baseline and this repository both provide a `deep_gemm` import, so installing
both into one environment shadows one of them. Set BASELINE_ROOT to a persistent
disk directory. CUDA hosts need a matching
CUDA toolkit, NCCL, and NVSHMEM where required by DeepEP. ROCm hosts need a
working RCCL installation, HIP, and a ROCm PyTorch build. The inference-engine
wheels in the helper are a starting point; reproduce the exact wheel versions
recorded in your experiment manifest when comparing against a published table.

Verify the environment:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
    python -m pip show deep-gemm flashinfer sglang vllm 2>/dev/null || true

## 2. NVIDIA SM103 (148-SM): 8x EP8

The Figure 7 SM103 (148-SM) sweep uses eight ranks, one GPU per rank. The two official
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
buffer allocator. It is not the 148-SM SM103 scheduler compiled for an older
architecture. The bundled official wrapper runs the repository's fixed SM90
shape-coverage suite; to reproduce the paper's six-model x S grid, pass the
matching shapes to an external official DeepGEMM driver or to the split and
serving runners.

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

## 7. Strict same-precision GC-MoE EP4 evaluation (`num_shared_experts = 0`)

To compare Weave against GC-MoE under matched numeric precision (`num_shared_experts = 0`,
5-round palindromic ABBA/ABCCBA, CUDA Graph, 256 MiB L2 flush), the SM90 native
FP8 weight patch is bundled at
[patches/gc_moe_sm90_native_fp8.patch](patches/gc_moe_sm90_native_fp8.patch)
and the full verified measurements live in
[../results/gcmoe_same_precision_ep4.json](../results/gcmoe_same_precision_ep4.json):

| Hardware & Precision | Workload (`num_shared_experts=0`) | GC-MoE | Weave (Unified) | Paired Weave / GC-MoE | Accuracy |
|---|---|---:|---:|---:|---|
| NVIDIA H20, `FP8 x FP8` | `Qwen3.8-Max`, `S_rank=32` (`S=128`) | `1956.7 us` | **`1839.2 us`** | **`1.059x`** | `cos >= 0.999988`, `stats_exact` |
| NVIDIA H20, `FP8 x FP8` | `Qwen3.8-Max`, `S_rank=128` (`S=512`) | `2190.8 us` | **`1988.3 us`** | **`1.102x`** | `cos >= 0.999989`, `stats_exact` |
| NVIDIA H20, `FP8 x FP8` | `Qwen3.8-Max`, `S_rank=256` (`S=1024`) | `3740.3 us` | **`2216.8 us`** | **`1.679x`** | `cos >= 0.999989`, `stats_exact` |
| NVIDIA H20, `FP8 x FP8` | `Qwen3.8-Max`, `S_rank=1024` (`S=4096`) | `11794.3 us` | **`5272.7 us`** | **`2.276x`** | `cos >= 0.999989`, `stats_exact` |
| NVIDIA SM103 (148-SM), `FP8 x MXFP4` | `Qwen3.8-Max`, `S_rank=1024` | `824.7 us` (`BM=96`) | **`779.0 us`** (auto `BM=96, st=16/8, epi2`) | **`1.0553x`** (5/5 rounds) | Bit-exact (`dL2=0.00`, `cos=1.000000`) |
| NVIDIA SM103 (148-SM), `FP8 x MXFP4` | `416E`, `S_rank=1024` | `412.9 us` (`BM=96`) | **`393.8 us`** (auto `BM=80, st=8/8, epi2`) | **`1.0423x`** (5/5 rounds) | Bit-exact (`dL2=0.00`, `cos=1.000000`) |
