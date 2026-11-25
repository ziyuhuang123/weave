#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the unified 3-axis WEAVE_* preset table and drive a preset / timeline render.

Every render is gated by tests/validate_weaveprofiler_trace.py. The trace is rejected
before drawing when its Panel A curves are not real hardware-counter samples (nsys GPU
metrics on NVIDIA, rocprof PMC on AMD; "Model estimate" / synthetic / analytic curves
are refused), when a curve is missing, empty or leaves 0..100 %, when the NVLink / xGMI
curve is flat zero, when a Combine / Reduce slice would be dropped or hidden, or when
any drawn string would contain a forbidden display string. With WEAVE_REDUCE_WARPS > 0
the Panel C comm tracks that reduce during Middle are labeled, e.g.
"W02..W03 (Dispatch -> Streaming Reduce)". --validate-only stops after the gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent
RENDERER = ROOT / "ablation" / "render_weaveprofiler_moe_timeline.py"
VALIDATOR = ROOT / "tests" / "validate_weaveprofiler_trace.py"
DEFAULT_OUT = ROOT / "docs" / "figures"

BANNERS = {
    "sm103": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=40 (S=1024); all other spatial knobs at their defaults.\n"
        "Temporal: all knobs at their defaults.\n"
        "Implementation: WEAVE_PRECISION_MODE=0 (fp8xfp4); all other implementation knobs at their defaults."
    ),
    "h20": (
        "Spatial: all knobs at their defaults; 78 SMs.\n"
        "Temporal: WEAVE_SCATTER_OWNER=comm_only.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_GEMM1_CONFIG=32, WEAVE_GEMM2_CONFIG=64, "
        "WEAVE_SCATTER_STAGING=smem; all other implementation knobs at their defaults."
    ),
    "mi308x": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=0, WEAVE_SCATTER_ACTIVE_SMS=0, "
        "WEAVE_CU_WAVE_SPLIT=4-4 (80 CUs; 4/4 hybrid CUs run 4 compute + 4 comm waves).\n"
        "Temporal: WEAVE_V_HEAD=0 (auto), WEAVE_SCATTER_OWNER=comm_only.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_GEMM1_CONFIG=0 (auto), "
        "WEAVE_GEMM2_CONFIG=0 (auto), WEAVE_DEDUP_MODE=1, WEAVE_SCATTER_STAGING=hbm."
    ),
}

PRESETS = {
    "sm103": dict(
        platform="sm103",
        title="NVIDIA SM103 (148-SM) / 4x EP4 / E=256 top-8 H=7168 I=2048 / S=1024",
        script="sm103_qwen38max_ep4_s1024.sh",
        trace_env="WEAVE_SM103_TRACE / WEAVE_SM103_RESULT_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_DISPATCH_ACTIVE_SMS", "40", "40 SMs pull remote tokens (S=4096 -> 96, S=8192 -> auto)"),
            ("Implementation", "WEAVE_PRECISION_MODE", "0", "fp8xfp4 (default)"),
            ("-", "all other knobs", "0 (auto)", "selected automatically"),
        ],
    ),
    "h20": dict(
        platform="h20",
        title="NVIDIA H20 (SM90, 78 SMs) / Qwen3.8-Max / 4x EP4 / S=1024",
        script="h20_qwen38max_ep4_s1024.sh",
        trace_env="WEAVE_H20_TRACE / WEAVE_H20_RESULT_ROOT",
        render=True,
        rows=[
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "communication warps own cross-GPU scatter"),
            ("Implementation", "WEAVE_PRECISION_MODE", "2", "fp8xfp8"),
            ("Implementation", "WEAVE_GEMM1_CONFIG", "32", "independent GEMM1 blockM=32"),
            ("Implementation", "WEAVE_GEMM2_CONFIG", "64", "independent GEMM2 blockM=64"),
            ("Implementation", "WEAVE_SCATTER_STAGING", "smem", "per-SM shared-memory async scatter"),
        ],
    ),
    "mi308x_ep8": dict(
        platform="mi308x",
        title="AMD MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP8 / S=1024",
        script="mi308x_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_MI308X_TRACE / WEAVE_MI308X_TRACE_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_SCATTER_ACTIVE_SMS", "0", "all 80 CUs participate"),
            ("Spatial", "WEAVE_CU_WAVE_SPLIT", "4-4", "4 communication waves + 4 compute waves per CU"),
            ("Temporal", "WEAVE_V_HEAD", "0", "automatic opening pull quota"),
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "communication waves own remote xGMI scatter"),
            ("Implementation", "WEAVE_PRECISION_MODE", "2", "fp8xfp8"),
            ("Implementation", "WEAVE_GEMM1_CONFIG", "0", "auto per-expert tile selection"),
            ("Implementation", "WEAVE_GEMM2_CONFIG", "0", "auto per-expert tile selection"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "unique (src_rank, token) pull + local fanout"),
            ("Implementation", "WEAVE_SCATTER_STAGING", "hbm", "local pre-accumulation before xGMI send"),
        ],
    ),
    "mi308x_ep16": dict(
        platform=None,
        title="AMD MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP16 / S=128 (2 nodes x 8 ranks)",
        script="mi308x_qwen38max_ep16_s128.sh",
        trace_env="WEAVE_EP16_ROUTE_MANIFEST / WEAVE_EP16_WEIGHT_CACHE",
        render=False,
        rows=[
            ("Spatial", "WEAVE_CU_WAVE_SPLIT", "4-4", "4 communication waves + 4 compute waves per CU"),
            ("Temporal", "WEAVE_V_HEAD", "0", "automatic opening pull quota"),
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "communication waves own remote GDA/xGMI scatter"),
            ("Implementation", "WEAVE_PRECISION_MODE", "2", "fp8xfp8"),
            ("Implementation", "WEAVE_GEMM1_CONFIG", "0", "auto per-expert tile selection"),
            ("Implementation", "WEAVE_GEMM2_CONFIG", "0", "auto per-expert tile selection"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "unique (src_rank, token) pull + local fanout"),
            ("Implementation", "WEAVE_SCATTER_STAGING", "hbm", "node-local pre-accumulation before inter-node send"),
        ],
    ),
}


def print_table(preset_key: str) -> None:
    p = PRESETS[preset_key]
    print("=" * 108)
    print(f"Preset: {p['title']}")
    print(f"Script: {EXAMPLES_DIR / p['script']}")
    print("-" * 108)
    print(f"{'Axis':<16}{'Parameter':<30}{'Value':<14}Mechanism")
    print("-" * 108)
    for axis, param, value, note in p["rows"]:
        print(f"{axis:<16}{param:<30}{value:<14}{note}")
    print("-" * 108)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_renderer(path: Path):
    return _load_module(path, "weave_timeline_renderer_v3")


def load_validator(path: Path):
    return _load_module(path, "weave_trace_validator")


def run_preset(preset_key: str) -> None:
    p = PRESETS[preset_key]
    script = EXAMPLES_DIR / p["script"]
    subprocess.run(["bash", str(script)], cwd=str(EXAMPLES_DIR), check=True)


def validate_preview(preset_key: str, trace: Path | None, renderer: Path, validator: Path,
                     report_json: Path | None = None, no_banner: bool = False) -> Path:
    """Resolve the preset trace exactly as the renderer does and run every red line on it."""
    p = PRESETS[preset_key]
    platform = p["platform"]
    if platform is None:
        raise SystemExit("ERROR: this preset has no matching 3-panel preview renderer")
    module = load_renderer(renderer)
    if module.SHARED_BANNERS.get(platform) != BANNERS[platform]:
        raise SystemExit(f"ERROR: driver banner and renderer SHARED_BANNERS disagree for {platform}")
    resolved = Path(module.resolve_trace(module.platform_cfg(platform, trace=trace)))
    vmod = load_validator(validator)
    rep = vmod.validate_trace(resolved, platform=platform,
                              config_banner="" if no_banner else BANNERS[platform],
                              repro_script="" if no_banner else f"examples/{p['script']}", renderer=module)
    vmod.print_report(rep)
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps([vmod._json_ready(rep)], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"validation report -> {report_json}")
    if not rep["ok"]:
        rules = ", ".join(sorted({e["rule"] for e in rep["errors"]}))
        raise SystemExit(f"ERROR: trace rejected by WeaveProfiler red line(s): {rules}")
    return resolved


def render_preview(preset_key: str, trace: Path | None, out_dir: Path, renderer: Path,
                   no_banner: bool = False, title: str = "") -> None:
    p = PRESETS[preset_key]
    platform = p["platform"]
    if platform is None:
        raise SystemExit("ERROR: this preset has no matching 3-panel preview renderer")
    module = load_renderer(renderer)
    if module.SHARED_BANNERS.get(platform) != BANNERS[platform]:
        raise SystemExit(f"ERROR: driver banner and renderer SHARED_BANNERS disagree for {platform}")
    cmd = [
        sys.executable, str(renderer),
        "--platform", platform,
        "--out-dir", str(out_dir),
    ]
    if no_banner:
        cmd += ["--no-banner"]
    else:
        cmd += ["--config-banner", BANNERS[platform], "--repro-script", f"examples/{p['script']}"]
    if title:
        cmd += ["--title", title]
    if trace is not None:
        cmd += ["--trace", str(trace)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=tuple(PRESETS))
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--render", dest="render", action="store_true", default=None)
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--renderer", type=Path, default=RENDERER)
    ap.add_argument("--validator", type=Path, default=VALIDATOR)
    ap.add_argument("--validate-only", action="store_true",
                    help="run the WeaveProfiler red-line gate on the preset trace and stop")
    ap.add_argument("--validation-json", type=Path, default=None, help="write the red-line report as JSON")
    ap.add_argument("--no-banner", action="store_true",
                    help="draw no parameter banner (how the delivered README figures were rendered)")
    ap.add_argument("--title", default="",
                    help="explicit figure title (the sm103 / h20 renderer presets already title the delivered 4x EP4 figures)")
    args = ap.parse_args(argv)
    print_table(args.preset)
    if args.run:
        run_preset(args.preset)
    p = PRESETS[args.preset]
    do_render = p["render"] if args.render is None else args.render
    if not args.validate_only and (args.preset == "mi308x_ep16" or not do_render):
        return 0
    if args.trace is not None and not args.trace.is_file():
        raise SystemExit(f"ERROR: trace not found: {args.trace}")
    resolved = validate_preview(args.preset, args.trace, args.renderer, args.validator, args.validation_json,
                                no_banner=args.no_banner)
    if args.validate_only:
        return 0
    render_preview(args.preset, resolved, args.out_dir, args.renderer, no_banner=args.no_banner, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
