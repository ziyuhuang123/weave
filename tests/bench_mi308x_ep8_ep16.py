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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, choices=(8, 16), default=16)
    ap.add_argument("--runner", required=False, default="",
                    help="path to the standalone MI308X HIP benchmark driver")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runner = args.runner or os.environ.get("WEAVE_MI308X_BENCH_RUNNER", "")
    cmd = [runner or "<MI308X_RUNNER>", "--ep", str(args.ep)]
    print(json.dumps({"event": "command", "cmd": cmd}))
    if args.dry_run:
        return
    import torch
    name = torch.cuda.get_device_name(0)
    hip = getattr(torch.version, "hip", None)
    print(json.dumps({"event": "gpu", "name": name, "hip": hip, "ep": args.ep}))
    if hip is None:
        raise SystemExit("bench_mi308x_ep8_ep16.py requires a ROCm PyTorch build")
    if not runner:
        raise SystemExit(
            "No MI308X runner supplied.  The public archive does not contain "
            "official_m16_m32_configs_v1262.hpp, gemm2_configs_v1010_hip.hpp, "
            "v956_stage1_body.hpp, m16_packet_support.hpp, or fi_lite_hip.h, "
            "so this repository cannot honestly claim a buildable MI308X "
            "end-to-end harness.  Supply --runner <path> or "
            "WEAVE_MI308X_BENCH_RUNNER=<path> after installing the missing "
            "AITER/CK snapshot.")
    raise SystemExit(subprocess.call(cmd, cwd=Path(__file__).parents[1]))

if __name__ == "__main__":
    main()
