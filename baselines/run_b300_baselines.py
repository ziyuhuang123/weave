#!/usr/bin/env python3
"""Launch the official NVIDIA B300 baselines through one stable CLI.

The script uses only the Python standard library. It validates the host,
materialises a command plan, and executes the selected baseline. Use
--dry-run to inspect every command without importing CUDA libraries.

Example:
    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_b300_baselines.py --baseline official_deepgemm \
        --model qwen38max --seqlen 1024 --precision fp8xfp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
MODELS: Dict[str, Dict[str, int]] = {
    "minimax_m3": {"experts": 128, "topk": 4, "hidden": 6144, "intermediate": 3072},
    "qwen38max": {"experts": 512, "topk": 10, "hidden": 8192, "intermediate": 2048},
    "DeepSeek-V4-Flash": {"experts": 256, "topk": 6, "hidden": 4096, "intermediate": 2048},
    "Qwen3.5-35B-A3B": {"experts": 256, "topk": 8, "hidden": 2048, "intermediate": 512},
    "Qwen3.5-122B-A10B": {"experts": 256, "topk": 8, "hidden": 3072, "intermediate": 1024},
    "Qwen3.5-397B-A17B": {"experts": 512, "topk": 10, "hidden": 4096, "intermediate": 1024},
}


@dataclass(frozen=True)
class Baseline:
    name: str
    description: str
    command: Callable[[argparse.Namespace], List[str]]


def _torchrun(nproc: int, module: str, extra: List[str]) -> List[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={nproc}", module, *extra]


def _official_deepgemm(args: argparse.Namespace) -> List[str]:
    spec = MODELS[args.model]
    return _torchrun(args.num_processes, "tests/test_mega_moe.py", [
        "--num-processes", str(args.num_processes),
        "--num-max-tokens-per-rank", str(args.seqlen),
        "--num-tokens", str(args.seqlen),
        "--hidden", str(spec["hidden"]),
        "--intermediate-hidden", str(spec["intermediate"]),
        "--num-experts", str(spec["experts"]),
        "--num-topk", str(spec["topk"]),
        "--num-shared-experts", "0",
        "--mma-type", args.precision,
    ])


def _deepep_deepgemm(args: argparse.Namespace) -> List[str]:
    runner = args.deepep_runner or os.environ.get("WEAVE_DEEPEP_BENCH", "")
    if not runner:
        return ["<DEEPEP_DEEPGEMM_RUNNER>", "--model", args.model,
                "--seqlen", str(args.seqlen), "--world-size", str(args.num_processes)]
    return [sys.executable, runner, "--model", args.model,
            "--seqlen", str(args.seqlen), "--world-size", str(args.num_processes),
            "--precision", args.precision]


def _flashinfer(args: argparse.Namespace) -> List[str]:
    runner = args.flashinfer_runner or os.environ.get("WEAVE_FLASHINFER_BENCH", "")
    if not runner:
        return ["<FLASHINFER_RUNNER>", "--model", args.model,
                "--seqlen", str(args.seqlen), "--world-size", str(args.num_processes)]
    return [sys.executable, runner, "--model", args.model,
            "--seqlen", str(args.seqlen), "--world-size", str(args.num_processes),
            "--dtype", "nvfp4" if args.precision == "fp4xfp4" else "fp8"]


def _sglang(args: argparse.Namespace) -> List[str]:
    return [sys.executable, "-m", "sglang.bench_latency",
            "--model-path", args.model_path, "--tp", str(args.num_processes),
            "--input-len", str(args.seqlen), "--output-len", "1"]


def _vllm(args: argparse.Namespace) -> List[str]:
    return [sys.executable, "-m", "vllm.benchmarks.benchmark_latency",
            "--model", args.model_path, "--tensor-parallel-size", str(args.num_processes),
            "--input-len", str(args.seqlen), "--output-len", "1"]


BASELINES = {
    b.name: b for b in (
        Baseline("official_deepgemm", "Official DeepGEMM MegaMoE, fp8xfp4 or NVFP4", _official_deepgemm),
        Baseline("deepep_deepgemm", "DeepEP dispatch/combine plus grouped DeepGEMM", _deepep_deepgemm),
        Baseline("flashinfer", "FlashInfer fused-MoE baseline", _flashinfer),
        Baseline("sglang", "SGLang serving baseline", _sglang),
        Baseline("vllm", "vLLM serving baseline", _vllm),
    )
}


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="official_deepgemm",
                    choices=(*BASELINES, "all"))
    ap.add_argument("--model", default="qwen38max", choices=tuple(MODELS))
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--precision", default="fp8xfp4",
                    choices=("fp8xfp4", "fp4xfp4"))
    ap.add_argument("--model-path", default="<MODEL_PATH>")
    ap.add_argument("--deepep-runner", default="")
    ap.add_argument("--flashinfer-runner", default="")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "baselines" / "b300")
    ap.add_argument("--allow-other-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _check_gpu(allow_other: bool) -> None:
    import torch
    name = torch.cuda.get_device_name(0)
    print(json.dumps({"event": "gpu", "name": name}))
    if "b300" not in name.lower() and not allow_other:
        raise SystemExit(f"expected a B300 device, got {name!r}; pass --allow-other-gpu")


def main() -> int:
    args = _args()
    names = list(BASELINES) if args.baseline == "all" else [args.baseline]
    plans = {name: BASELINES[name].command(args) for name in names}
    print(json.dumps({"event": "baseline_plan", "plans": plans}, indent=2))
    if args.dry_run:
        return 0
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    _check_gpu(args.allow_other_gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, cmd in plans.items():
        if cmd and cmd[0].startswith("<"):
            raise SystemExit(f"{name}: no runnable harness configured; "
                             "see baselines/README.md")
        print(json.dumps({"event": "run", "baseline": name, "cmd": cmd}))
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
