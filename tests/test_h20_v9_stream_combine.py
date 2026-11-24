#!/usr/bin/env python3
"""Host model gate for the H20 V9 per-token streaming-Combine protocol.

Compiles and runs tests/test_h20_v9_stream_combine.cpp, which replays the exact
integer logic of the device protocol added to sm90_fp8_mega_moe.cuh under
kSmemStreamCombineActive: sender block counters -> per-token NVLink arrival
counts -> readiness == popc(top-k mask) -> per-token claim bitmask -> bounded
middle pass plus a whole-CTA leftover tail.

No GPU and no compiled _C module are needed.  On Windows the build/run is
delegated to WSL g++ (fresh native binaries are blocked by application control).

Usage: python tests/test_h20_v9_stream_combine.py [--build-dir DIR]
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tests/test_h20_v9_stream_combine.cpp"
WINDOWS = platform.system() == "Windows"


def to_wsl(path: Path) -> str:
    s = str(path)
    if WINDOWS and len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def toolchain() -> tuple[list[str], bool]:
    if not WINDOWS:
        return ["g++"], False
    proc = subprocess.run(["wsl", "-e", "bash", "-lc", "command -v g++"],
                          capture_output=True, text=True, timeout=120)
    found = (proc.stdout or "").strip()
    if proc.returncode == 0 and found:
        print(f"[INFO] delegating to WSL: {found}")
        return ["wsl", "-e", found], True
    raise SystemExit("no host C++ compiler (install WSL g++ or run on Linux)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", default=None)
    args = ap.parse_args()

    cc, wsl = toolchain()
    build_dir = Path(args.build_dir) if args.build_dir else Path(
        tempfile.mkdtemp(prefix="weave_v9_stream_"))
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "weave_v9_stream_combine"

    cmd = [*cc, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter",
           to_wsl(SRC) if wsl else str(SRC), "-o",
           to_wsl(binary) if wsl else str(binary)]
    print("[BUILD] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        print(f"[FAIL] build exited with {proc.returncode}")
        return 1

    run_cmd = ["wsl", "-e", to_wsl(binary)] if wsl else [str(binary)]
    proc = subprocess.run(run_cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.rstrip())
    ok = (proc.returncode == 0
          and "ALL H20 V9 STREAM-COMBINE PROTOCOL CHECKS PASS" in out
          and "FAILURES=0" in out)
    print("ALL H20 V9 STREAM-COMBINE PROTOCOL GATES PASS" if ok
          else "H20 V9 STREAM-COMBINE PROTOCOL GATES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
