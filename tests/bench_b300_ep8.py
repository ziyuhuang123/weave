#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {
    "minimax_m3": dict(experts=128, topk=4, hidden=6144, intermediate=3072),
    "qwen38max": dict(experts=512, topk=10, hidden=8192, intermediate=2048),
    "DeepSeek-V4-Flash": dict(experts=256, topk=6, hidden=4096, intermediate=2048),
    "Qwen3.5-35B-A3B": dict(experts=256, topk=8, hidden=2048, intermediate=512),
    "Qwen3.5-122B-A10B": dict(experts=256, topk=8, hidden=3072, intermediate=1024),
    "Qwen3.5-397B-A17B": dict(experts=512, topk=10, hidden=4096, intermediate=1024),
}

def check_gpu(expected: str, allow_other: bool) -> None:
    import torch
    name = torch.cuda.get_device_name(0)
    print(json.dumps({"event": "gpu", "name": name}))
    if expected.lower() not in name.lower() and not allow_other:
        raise SystemExit(f"expected a {expected} GPU, got {name!r}; use --allow-other-gpu to override")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38max", choices=tuple(MODELS))
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--allow-other-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = MODELS[args.model]
    test = Path(__file__).with_name("test_mega_moe.py")
    cmd = [
        sys.executable, str(test),
        "--num-processes", str(args.num_processes),
        "--num-max-tokens-per-rank", str(args.seqlen),
        "--num-tokens", str(args.seqlen),
        "--hidden", str(spec["hidden"]),
        "--intermediate-hidden", str(spec["intermediate"]),
        "--num-experts", str(spec["experts"]),
        "--num-topk", str(spec["topk"]),
        "--num-shared-experts", "0",
        "--mma-type", "fp8xfp4",
    ]
    print(json.dumps({"event": "command", "cmd": cmd}))
    if args.dry_run:
        return
    check_gpu("B300", args.allow_other_gpu)
    env = dict(os.environ)
    env.setdefault("DG_PRINT_CONFIGS", "1")
    raise SystemExit(subprocess.call(cmd, cwd=Path(__file__).parents[1], env=env))

if __name__ == "__main__":
    main()
