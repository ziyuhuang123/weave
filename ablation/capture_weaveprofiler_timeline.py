#!/usr/bin/env python3
"""Capture or render a WeaveProfiler software + hardware timeline.

Two modes are supported:

1. --capture-command "nsys profile ..." runs an existing benchmark with
   WEAVE_PROFILER_LEVEL=2 and writes a paired trace.
2. --trace trace.sw_hw.perfetto.json.gz --out-png timeline.png renders the
   four-panel publication view through the official preview module.

The script never writes to /tmp. Intermediate files stay beside the requested
output path.
"""
from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _preview_module():
    extra = os.environ.get("WEAVE_PROFILER_ROOT", "")
    if extra:
        sys.path.insert(0, extra)
    try:
        from weaveprof import preview  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "cannot import weaveprof.preview; set WEAVE_PROFILER_ROOT to the "
            f"WeaveProfiler checkout root ({exc})")
    return preview


def _load_trace(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _materialise_trace(path: Path, out_png: Path) -> Path:
    if path.suffix != ".gz":
        return path
    target = out_png.with_suffix(".trace.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rb") as src, target.open("wb") as dst:
        dst.write(src.read())
    return target


def _render(args: argparse.Namespace) -> int:
    preview = _preview_module()
    trace_path = _materialise_trace(args.trace, args.out_png)
    aligned = _load_trace(trace_path)
    verify_doc = None
    if args.verify_json:
        verify_doc = json.loads(args.verify_json.read_text(encoding="utf-8"))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"event": "render_plan", "trace": str(trace_path),
                          "out_png": str(args.out_png),
                          "title": args.title}, indent=2))
        return 0
    fn = getattr(preview, "render_timeline_png", None)
    if fn is None:
        raise SystemExit("weaveprof.preview has no render_timeline_png")
    kwargs = {}
    if "title_override" in inspect.signature(fn).parameters:
        kwargs["title_override"] = args.title
    fn(aligned, verify_doc, str(args.out_png), **kwargs)
    print(f"WROTE {args.out_png}")
    return 0


def _capture(args: argparse.Namespace) -> int:
    if not args.capture_command:
        print(json.dumps({
            "event": "capture_template",
            "cuda": "WEAVE_PROFILER_LEVEL=2 nsys profile --trace=cuda,nvtx,osrt -o <run> <benchmark>",
            "rocm": "WEAVE_PROFILER_LEVEL=2 rocprofv3 --hip-trace --rccl-trace -o <run> -- <benchmark>",
        }, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({"event": "capture_plan",
                          "cmd": args.capture_command,
                          "env": {"WEAVE_PROFILER_LEVEL": "2"}}, indent=2))
        return 0
    env = dict(os.environ)
    env["WEAVE_PROFILER_LEVEL"] = "2"
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(shlex.split(args.capture_command), cwd=ROOT, env=env)
    return proc.returncode


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--verify-json", type=Path)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--title", default="Weave MegaMoE: WeaveProfiler SW+HW Timeline")
    ap.add_argument("--capture-command", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _args()
    if args.trace:
        return _render(args)
    return _capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
