#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the 3-axis tuned-parameter table and drive a preset / timeline render.

The script has two independent actions:
  * --run      launch one of the reproducible preset shell scripts
  * --render   render the annotated 3-panel preview from an archived trace

Examples
  python examples/profile_and_render_timeline.py --preset sm103
  python examples/profile_and_render_timeline.py --preset h20 --render
  python examples/profile_and_render_timeline.py --preset mi308x_ep8 --trace /abs/trace.sw_hw.perfetto.json.gz
  python examples/profile_and_render_timeline.py --preset mi308x_ep16 --no-render
  python examples/profile_and_render_timeline.py --preset sm103 --run --render

Axis taxonomy
  Spatial        : how work is partitioned across SMs/CUs and tiles
                   (block M/N/K, store granularity, wave roles, tier mode).
  Temporal       : when stages overlap and how readiness/order is managed
                   (scatter mode, interleave, ready groups,
                   overlap pipeline, combine grouping).
  Implementation : code-path / precision choices that are neither a spatial
                   split nor a schedule (precision, dedup, register pull,
                   SMEM alias, profiler level, warp counts).

Evidence classes
  trace-metadata       literal field in the archived trace (including the H20
                       selected_cfg string and the MI308X grid/wave metadata)
  archived-script      literal value in the archived capture/build script
  current-build-default active default in the current unified build script
  heuristic-derived    value produced by the documented heuristic branch
  capture-only         set only while capturing, not a production tuning value
  documented-behavior  documented semantics of a reserved or auto value
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent
RENDERER = ROOT / "ablation" / "render_weaveprofiler_moe_timeline.py"
DEFAULT_OUT = ROOT / "docs" / "figures"

# The three banners are identical to SHARED_BANNERS in the v2 renderer. The
# driver asserts equality before rendering, so the table and the figure cannot
# silently drift apart.
BANNERS = {
    "sm103": (
        "Spatial: WEAVE_BLOCK_M=64, WEAVE_BLOCK_N=128, WEAVE_BLOCK_K=128, "
        "WEAVE_STORE_BLOCK_M=32; 148 SMs; 4 dispatch + 4 TMA/UMMA + 8 epilogue warps.\n"
        "Temporal: WEAVE_SCATTER_MODE=1, WEAVE_READY_GROUP_SIZE=2, "
        "WEAVE_COMBINE_GROUP_TILES=2, WEAVE_COMBINE_REDUCE_WARPS=16, "
        "WEAVE_SHARED_L2_EARLY_BLOCKS=77 (capture only).\n"
        "Implementation: WEAVE_PRECISION_MODE=0 (fp8xfp4), WEAVE_DEDUP_MODE=1, "
        "WEAVE_EXPERTS_PER_WAVE=2, WEAVE_REG_PULL_BYTES=4096, "
        "WEAVE_SCATTER_SMEM_ALIAS=1, WEAVE_SCHED_MODE=3 (reserved = mode 0)."
    ),
    "h20": (
        "Spatial: WEAVE_H20_CONFIG exact spec bm=32, bn=256, WEAVE_BLOCK_K=128, "
        "wgs=4, swap=1; 78 SMs; 4 epilogue warp groups.\n"
        "Temporal: WEAVE_INTERLEAVE_L1_L2=5, "
        "WEAVE_DYN_SCHED=1, stages=0 = runtime auto pipeline depth, "
        "WEAVE_SWAP_PIPE=1.\n"
        "Implementation: WEAVE_DEDUP_DISPATCH=1 + dedup_tier=1 (scatter dedup off), "
        "WEAVE_REG_COMBINE=1, WEAVE_ALL_IDLE_COMBINE=1, WEAVE_FAST_PULL=1; "
        "exact string in WEAVE_H20_CONFIG."
    ),
    "mi308x": (
        "Spatial: 80 CUs, 8 waves/CU; WEAVE_BM_TIER_MODE=2, fixed BN=128, BK=128.\n"
        "Temporal: WEAVE_SCATTER_MODE=2, WEAVE_OVERLAP_PIPELINE=1, "
        "WEAVE_DUALGROUP_MIN_TOKENS=64 (S=1024 local 128 -> 8-wave dual compute).\n"
        "Implementation: WEAVE_ABLATION_PRESET=0, WEAVE_SELECTIVE_DEDUP=1, "
        "WEAVE_ALL_WAVE_COMBINE=1, WEAVE_PARALLEL_ROUTE_PLAN=1, "
        "WEAVE_PROFILER_LEVEL=2 capture / 0 production."
    ),
}

PRESETS = {
    "sm103": dict(
        platform="sm103",
        title="SM103 (148-SM) / Qwen3.8-Max / EP8 / S=1024",
        script="sm103_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_SM103_TRACE / WEAVE_SM103_RESULT_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_BLOCK_M", "64", "heuristic-derived", "expected tokens/expert = 20.0 -> <= 32.5 branch"),
            ("Spatial", "WEAVE_BLOCK_N", "128", "heuristic-derived", "fixed BN of the archived block-shape branch"),
            ("Spatial", "WEAVE_BLOCK_K", "128", "heuristic-derived", "K tile; fp8 element width"),
            ("Spatial", "WEAVE_STORE_BLOCK_M", "32", "heuristic-derived", "TMA-store M granularity"),
            ("Spatial", "WEAVE_ENABLE_MULTICAST", "1", "current-build-default", "required by the 148-SM dataflow"),
            ("Temporal", "WEAVE_SCATTER_MODE", "1", "archived-script", "dispatch-warp scatter / comm-offloaded"),
            ("Temporal", "WEAVE_READY_GROUP_SIZE", "2", "archived-script", "ready-group granularity"),
            ("Temporal", "WEAVE_COMBINE_GROUP_TILES", "2", "archived-script", "combine tile grouping"),
            ("Temporal", "WEAVE_COMBINE_REDUCE_WARPS", "16", "archived-script", "16 reduce warps x 128 tiles"),
            ("Temporal", "WEAVE_SHARED_L2_EARLY_BLOCKS", "77", "capture-only", "production default is 10"),
            ("Implementation", "WEAVE_PRECISION_MODE", "0", "archived-script", "fp8xfp4; optional 1 = fp4xfp4"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "archived-script", "(token, destination) dedup"),
            ("Implementation", "WEAVE_EXPERTS_PER_WAVE", "2", "archived-script", "two experts per wave"),
            ("Implementation", "WEAVE_REG_PULL_BYTES", "4096", "archived-script", "4 KiB register pull"),
            ("Implementation", "WEAVE_SCATTER_SMEM_ALIAS", "1", "archived-script", "alias scatter staging SMEM"),
            ("Implementation", "WEAVE_SCHED_MODE", "3", "documented-behavior", "reserved; identical to mode 0"),
            ("Implementation", "WEAVE_ABLATION_PRESET", "0", "archived-script", "full Weave preset"),
        ],
    ),
    "h20": dict(
        platform="h20",
        title="H20 (SM90, 78 SMs) / Qwen3.8-Max / EP8 / S=1024",
        script="h20_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_H20_TRACE / WEAVE_H20_RESULT_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_H20_CONFIG.bm", "32", "trace-metadata", "spec field; no standalone env"),
            ("Spatial", "WEAVE_H20_CONFIG.bn", "256", "trace-metadata", "spec field; no standalone env"),
            ("Spatial", "WEAVE_BLOCK_K", "128", "trace-metadata", "bk=128"),
            ("Spatial", "WEAVE_H20_CONFIG.wgs", "4", "trace-metadata", "4 epilogue warp groups"),
            ("Spatial", "WEAVE_H20_CONFIG.swap", "1", "trace-metadata", "swap-AB layout"),
            ("Temporal", "WEAVE_INTERLEAVE_L1_L2", "5", "trace-metadata", "interleave=5: EPW + L2 K-dependency"),
            ("Temporal", "WEAVE_DYN_SCHED", "1", "trace-metadata", "dyn=1: dynamic expert cursor"),
            ("Temporal", "WEAVE_H20_CONFIG.stages", "0", "documented-behavior", "0 = runtime auto pipeline depth"),
            ("Temporal", "WEAVE_SWAP_PIPE", "1", "trace-metadata", "swap_pipe=1"),
            ("Implementation", "WEAVE_H20_CONFIG.epw", "8", "trace-metadata", "spec field; no standalone env"),
            ("Implementation", "WEAVE_SCATTER_ROLE_DISP", "1", "trace-metadata", "scat_disp=1"),
            ("Implementation", "WEAVE_HETERO_MASK_MODE", "1", "trace-metadata", "d_mask_mode=1"),
            ("Implementation", "WEAVE_DEDUP_DISPATCH", "1", "trace-metadata", "dedup_disp=1"),
            ("Implementation", "WEAVE_DEDUP_SCATTER", "0", "trace-metadata", "dedup_scat absent -> off"),
            ("Implementation", "WEAVE_REG_COMBINE", "1", "trace-metadata", "reg_comb=1"),
            ("Implementation", "WEAVE_ALL_IDLE_COMBINE", "1", "trace-metadata", "all_idle_comb=1"),
            ("Implementation", "WEAVE_FAST_PULL", "1", "trace-metadata", "fast_pull=1"),
            ("Implementation", "WEAVE_H20_CONFIG.dedup_tier", "1", "trace-metadata", "dedup_tier=1"),
            ("Implementation", "WEAVE_H20_CONFIG.*", "exact string", "trace-metadata", "fast_host,fused,compact,alias,fast_swap_epi,rank_cache,fast_clean,fast_comb"),
        ],
    ),
    "mi308x_ep8": dict(
        platform="mi308x",
        title="MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP8 / S=1024",
        script="mi308x_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_MI308X_TRACE / WEAVE_MI308X_TRACE_ROOT",
        render=True,
        rows=[
            ("Spatial", "gridX / waves_per_cu", "80 / 8", "trace-metadata", "80 unique physical CUs"),
            ("Spatial", "BN / BK", "128 / 128", "current-build-default", "fixed tile constants"),
            ("Spatial", "WEAVE_BM_TIER_MODE", "2", "current-build-default", "M tiers 16/32/64/128/256"),
            ("Spatial", "WEAVE_TH_BM64", "16", "current-build-default", "small-M tier threshold"),
            ("Spatial", "WEAVE_TH_BM128", "96", "current-build-default", "large-M tier threshold"),
            ("Temporal", "WEAVE_SCATTER_MODE", "2", "current-build-default", "local-dedup / node-partial pre-reduce"),
            ("Temporal", "WEAVE_OVERLAP_PIPELINE", "1", "current-build-default", "compute-communication overlap"),
            ("Temporal", "WEAVE_DUALGROUP_MIN_TOKENS", "64", "current-build-default", "local tokens 128 >= 64 -> 4+4 dual compute"),
            ("Implementation", "WEAVE_ABLATION_PRESET", "0", "current-build-default", "full Weave preset"),
            ("Implementation", "WEAVE_SELECTIVE_DEDUP", "1", "current-build-default", "required by the EP8 path"),
            ("Implementation", "WEAVE_ALL_WAVE_COMBINE", "1", "current-build-default", "all waves join the tail combine"),
            ("Implementation", "WEAVE_BN2_MODE", "0", "current-build-default", "auto second-stage BN"),
            ("Implementation", "WEAVE_S2_NGROUP_MODE", "0", "current-build-default", "auto N-group width"),
            ("Implementation", "WEAVE_PARALLEL_ROUTE_PLAN", "1", "current-build-default", "80-CTA parallel route plan"),
            ("Implementation", "WEAVE_PROFILER_LEVEL", "2", "capture-only", "production is 0"),
        ],
    ),
    "mi308x_ep16": dict(
        platform=None,
        title="MI308X (gfx942) / Qwen3.8-Max / EP16 / S=128 (2 nodes x 8 ranks)",
        script="mi308x_qwen38max_ep16_s128.sh",
        trace_env="WEAVE_EP16_ROUTE_MANIFEST / WEAVE_EP16_WEIGHT_CACHE",
        render=False,
        rows=[
            ("Spatial", "BN / BK", "128 / 128", "current-build-default", "fixed tile constants"),
            ("Spatial", "WEAVE_BM_TIER_MODE", "2", "current-build-default", "M tiers 16/32/64/128/256"),
            ("Spatial", "WEAVE_TH_BM64", "16", "current-build-default", "small-M tier threshold"),
            ("Spatial", "WEAVE_TH_BM128", "96", "current-build-default", "large-M tier threshold"),
            ("Temporal", "WEAVE_SCATTER_MODE", "2", "current-build-default", "node-partial pre-reduced scatter"),
            ("Temporal", "WEAVE_OVERLAP_PIPELINE", "1", "current-build-default", "mode-2 overlap pipeline"),
            ("Temporal", "WEAVE_DUALGROUP_MIN_TOKENS", "64", "documented-behavior", "local tokens 8 < 64 -> not triggered"),
            ("Implementation", "WEAVE_ABLATION_PRESET", "0", "current-build-default", "full Weave preset"),
            ("Implementation", "WEAVE_SELECTIVE_DEDUP", "1", "current-build-default", "required by the EP16 path"),
            ("Implementation", "WEAVE_ALL_WAVE_COMBINE", "1", "current-build-default", "all waves join the tail combine"),
            ("Implementation", "WEAVE_BN2_MODE", "0", "current-build-default", "auto second-stage BN"),
            ("Implementation", "WEAVE_S2_NGROUP_MODE", "0", "current-build-default", "auto N-group width"),
            ("Implementation", "WEAVE_PARALLEL_ROUTE_PLAN", "1", "current-build-default", "80-CTA parallel route plan"),
            ("Implementation", "WEAVE_PROFILER_LEVEL", "0", "current-build-default", "2 for a profiler build"),
            ("Implementation", "WEAVE_TURBO_MODES", "2", "current-build-default", "runtime mode-2 overlap arm"),
        ],
    ),
}


def print_table(preset_key: str) -> None:
    p = PRESETS[preset_key]
    print("=" * 118)
    print(f"Preset: {p['title']}")
    print(f"Script: {EXAMPLES_DIR / p['script']}")
    print("=" * 118)
    print("Evidence classes: trace-metadata | archived-script | current-build-default |")
    print("                  heuristic-derived | capture-only | documented-behavior")
    print("-" * 118)
    print(f"{'Axis':<15}{'Parameter / field':<40}{'Value':<20}{'Evidence':<23}Note")
    print("-" * 118)
    for axis, param, value, evidence, note in p["rows"]:
        print(f"{axis:<15}{param:<40}{value:<20}{evidence:<23}{note}")
    print("-" * 118)
    print(f"Trace resolution: {p['trace_env']} (or pass --trace)")
    print()


def load_renderer(path: Path):
    if not path.is_file():
        raise SystemExit(f"ERROR: renderer not found: {path}")
    spec = importlib.util.spec_from_file_location("weave_timeline_renderer_v2", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_preset(preset_key: str) -> None:
    p = PRESETS[preset_key]
    script = EXAMPLES_DIR / p["script"]
    if not script.is_file():
        raise SystemExit(f"ERROR: preset script not found: {script}")
    print(f"[run] bash {script}")
    subprocess.run(["bash", str(script)], cwd=str(EXAMPLES_DIR), check=True)


def render_preview(preset_key: str, trace: Path | None, out_dir: Path, renderer: Path) -> None:
    p = PRESETS[preset_key]
    platform = p["platform"]
    if platform is None:
        raise SystemExit("ERROR: this preset has no matching 3-panel preview renderer")
    module = load_renderer(renderer)
    expected = module.SHARED_BANNERS.get(platform)
    if expected != BANNERS[platform]:
        raise SystemExit(
            "ERROR: driver banner and renderer SHARED_BANNERS disagree for "
            f"{platform}; update both files together"
        )
    cmd = [
        sys.executable, str(renderer),
        "--platform", platform,
        "--out-dir", str(out_dir),
        "--config-banner", BANNERS[platform],
        "--repro-script", f"examples/{p['script']}",
    ]
    if trace is not None:
        cmd += ["--trace", str(trace)]
    print("[render] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=tuple(PRESETS))
    ap.add_argument("--trace", type=Path, default=None,
                    help="explicit .sw_hw.perfetto.json.gz trace for the selected platform")
    ap.add_argument("--run", action="store_true", help="launch the preset shell script")
    ap.add_argument("--render", dest="render", action="store_true", default=None,
                    help="force rendering of the annotated preview")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="skip rendering")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--renderer", type=Path, default=RENDERER)
    args = ap.parse_args(argv)

    print_table(args.preset)

    if args.run:
        run_preset(args.preset)
        if args.preset != "mi308x_ep16":
            print("[note] the preset run does not emit a paired SW+HW Perfetto trace;")
            print("       rendering still uses the archived trace resolved by the environment.")

    p = PRESETS[args.preset]
    do_render = p["render"] if args.render is None else args.render
    if args.preset == "mi308x_ep16":
        print("[render] EP16 has no matching 3-panel preview; skipping.")
        return 0
    if not do_render:
        print("[render] skipped (--no-render).")
        return 0
    if args.trace is not None and not args.trace.is_file():
        raise SystemExit(f"ERROR: trace not found: {args.trace}")
    render_preview(args.preset, args.trace, args.out_dir, args.renderer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
