#!/usr/bin/env python3
"""Launch MI308X EP8/EP16 baselines through one stable CLI.

Dry-run mode is CPU-only. Execution requires a ROCm PyTorch build and the
selected baseline; hardware checks reject non-MI308X devices by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Baseline:
    name: str
    description: str
    command: Callable[[argparse.Namespace], List[str]]


def _aiter_script(args: argparse.Namespace, stage: str) -> List[str]:
    key = "WEAVE_AITER_1STAGE_BENCH" if stage == "1stage" else "WEAVE_AITER_2STAGE_BENCH"
    runner = os.environ.get(key, "")
    if not runner:
        return [f"<AITER_{stage.upper()}_BENCH>", "--ep", str(args.ep),
                "--model", args.model, "--seqlen", str(args.seqlen)]
    return [sys.executable, runner, "--ep", str(args.ep),
            "--model", args.model, "--seqlen", str(args.seqlen),
            "--world-size", str(args.ep)]


def _aiter_1stage(args: argparse.Namespace) -> List[str]:
    return _aiter_script(args, "1stage")


def _aiter_2stage(args: argparse.Namespace) -> List[str]:
    return _aiter_script(args, "2stage")


def _rccl_aiter(args: argparse.Namespace) -> List[str]:
    runner = args.rccl_runner or os.environ.get("WEAVE_RCCL_AITER_BENCH", "")
    if not runner:
        return ["<RCCL_AITER_RUNNER>", "--ep", str(args.ep),
                "--model", args.model, "--seqlen", str(args.seqlen)]
    extra = shlex.split(args.runner_args) if args.runner_args else []
    return [sys.executable, runner, *extra]


def _vllm(args: argparse.Namespace) -> List[str]:
    return [sys.executable, "-m", "vllm.benchmarks.benchmark_latency",
            "--model", args.model_path, "--tensor-parallel-size", str(args.ep),
            "--input-len", str(args.seqlen), "--output-len", "1"]


def _sglang(args: argparse.Namespace) -> List[str]:
    return [sys.executable, "-m", "sglang.bench_latency",
            "--model-path", args.model_path, "--tp", str(args.ep),
            "--input-len", str(args.seqlen), "--output-len", "1"]


BASELINES = {
    b.name: b for b in (
        Baseline("rccl_aiter", "RCCL all-to-all plus AITER fused MoE", _rccl_aiter),
        Baseline("aiter_1stage", "AITER one-stage fused MoE", _aiter_1stage),
        Baseline("aiter_2stage", "AITER two-stage CK MoE", _aiter_2stage),
        Baseline("vllm", "vLLM ROCm serving baseline", _vllm),
        Baseline("sglang", "SGLang ROCm serving baseline", _sglang),
    )
}


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="rccl_aiter",
                    choices=(*BASELINES, "all"))
    ap.add_argument("--ep", type=int, choices=(8, 16), default=16)
    ap.add_argument("--model", default="qwen38max")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--model-path", default="<MODEL_PATH>")
    ap.add_argument("--rccl-runner", default="")
    ap.add_argument("--runner-args", default="")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "baselines" / "mi308x")
    ap.add_argument("--allow-other-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _check_gpu(allow_other: bool) -> None:
    import torch
    name = torch.cuda.get_device_name(0)
    hip = getattr(torch.version, "hip", None)
    print(json.dumps({"event": "gpu", "name": name, "hip": hip}))
    if hip is None:
        raise SystemExit("ROCm PyTorch is required")
    if "mi308" not in name.lower() and not allow_other:
        raise SystemExit(f"expected an MI308X device, got {name!r}; pass --allow-other-gpu")


def main() -> int:
    args = _args()
    names = list(BASELINES) if args.baseline == "all" else [args.baseline]
    plans = {name: BASELINES[name].command(args) for name in names}
    print(json.dumps({"event": "baseline_plan", "ep": args.ep, "plans": plans}, indent=2))
    if args.dry_run:
        return 0
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is not available")
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
