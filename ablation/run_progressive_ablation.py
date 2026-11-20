#!/usr/bin/env python3
"""Run the four-arm progressive ablation on B300, H20, or MI308X.

Preset mapping:
  0  Weave (full spatial search + temporal overlap + cost model)
  1  -S
  2  -S-T
  3  -S-T-C

Each preset receives a distinct JIT cache directory so a cached object from a
previous arm cannot silently supply the next arm. With --dry-run the script is
CPU-only and prints every environment/command pair.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
PRESET_NAMES = {0: "Weave", 1: "-S", 2: "-S-T", 3: "-S-T-C"}


def _runner(args: argparse.Namespace, preset: int) -> List[str]:
    if args.runner:
        return shlex.split(args.runner)
    if args.backend == "b300":
        return [sys.executable, str(ROOT / "tests" / "bench_b300_ep8.py"),
                "--model", args.model, "--seqlen", str(args.seqlen),
                "--num-processes", str(args.num_processes)]
    if args.backend == "h20":
        return [sys.executable, str(ROOT / "tests" / "bench_h20_ep8.py"),
                "--num-processes", str(args.num_processes)]
    runner = args.mi308x_runner or os.environ.get("WEAVE_MI308X_BENCH_RUNNER", "")
    if not runner:
        return ["<MI308X_RUNNER>", "--ep", str(args.ep)]
    return [sys.executable, str(ROOT / "tests" / "bench_mi308x_ep8_ep16.py"),
            "--ep", str(args.ep), "--runner", runner]


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("b300", "h20", "mi308x"))
    ap.add_argument("--presets", default="0,1,2,3")
    ap.add_argument("--model", default="qwen38max")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--ep", type=int, choices=(8, 16), default=16)
    ap.add_argument("--runner", default="",
                    help="optional full runner command; overrides the default")
    ap.add_argument("--mi308x-runner", default="")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "ablation")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _args()
    presets = [int(x) for x in args.presets.split(",") if x.strip()]
    if any(p not in PRESET_NAMES for p in presets):
        raise SystemExit("--presets must be a comma-separated subset of 0,1,2,3")
    plan = []
    for preset in presets:
        cmd = _runner(args, preset)
        env = {
            "WEAVE_ABLATION_PRESET": str(preset),
            "WEAVE_PROFILER_LEVEL": "0",
            "DG_JIT_CACHE_DIR": str(args.output_dir / args.backend /
                                    f"jit_p{preset}"),
        }
        plan.append({
            "preset": preset,
            "name": PRESET_NAMES[preset],
            "env": env,
            "cmd": cmd,
        })
    print(json.dumps({"backend": args.backend, "plan": plan}, indent=2))
    if args.dry_run:
        return 0
    if any(row["cmd"] and row["cmd"][0].startswith("<") for row in plan):
        raise SystemExit("no MI308X runner configured; pass --mi308x-runner")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in plan:
        run_env = dict(os.environ)
        run_env.update(row["env"])
        print(json.dumps({"event": "run", **row}))
        proc = subprocess.run(row["cmd"], cwd=ROOT, env=run_env)
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
