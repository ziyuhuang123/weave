#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the unified 3-axis WEAVE_* preset table and drive a preset / timeline render."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent
RENDERER = ROOT / "ablation" / "render_weaveprofiler_moe_timeline.py"
DEFAULT_OUT = ROOT / "docs" / "figures"

BANNERS = {
    "sm103": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=0, WEAVE_SCATTER_ACTIVE_SMS=0, WEAVE_SM0_STREAM_SCHED=0, "
        "WEAVE_DISPATCH_WARPS=4, WEAVE_SCATTER_WARPS=4, WEAVE_REDUCE_WARPS=4, WEAVE_REG_BALANCE=0.\n"
        "Temporal: WEAVE_V_HEAD=0 (auto), WEAVE_READY_GROUP_SIZE=2, WEAVE_READY_QUEUE=formula, "
        "WEAVE_SCATTER_OWNER=comm_only, WEAVE_COMBINE_SYNC_CHUNK=32.\n"
        "Implementation: WEAVE_PRECISION_MODE=0 (fp8xfp4), WEAVE_ADAPT_BM=0, WEAVE_BLOCK_M_L1=64, "
        "WEAVE_BLOCK_M_L2=64, WEAVE_REG_PULL_BYTES=4096, WEAVE_DEDUP_MODE=1, WEAVE_SCATTER_STAGING=smem, WEAVE_LOCAL_SCATTER_DEDUP=0."
    ),
    "h20": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=0, WEAVE_SCATTER_ACTIVE_SMS=0, WEAVE_SM0_STREAM_SCHED=0, "
        "WEAVE_DISPATCH_WARPS=0 (auto), WEAVE_REG_BALANCE=0; 78 SMs.\n"
        "Temporal: WEAVE_V_HEAD=0 (auto), WEAVE_READY_GROUP_SIZE=0, WEAVE_READY_QUEUE=sm0_fifo, "
        "WEAVE_SCATTER_OWNER=comm_only, WEAVE_COMBINE_SYNC_CHUNK=0.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_ADAPT_BM=0, WEAVE_BLOCK_M_L1=32, "
        "WEAVE_BLOCK_M_L2=32, WEAVE_DEDUP_MODE=1, WEAVE_SCATTER_STAGING=smem, WEAVE_LOCAL_SCATTER_DEDUP=0."
    ),
    "mi308x": (
        "Spatial: WEAVE_SCATTER_ACTIVE_SMS=0, WEAVE_SM0_STREAM_SCHED=0, WEAVE_CU_WAVE_SPLIT=4-4 "
        "(80 CUs, 4 comm + 4 compute waves/CU).\n"
        "Temporal: WEAVE_V_HEAD=0 (auto), WEAVE_SCATTER_OWNER=comm_only.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_ADAPT_BM=1, WEAVE_BLOCK_M_L1=0 (auto), "
        "WEAVE_BLOCK_M_L2=0 (auto), WEAVE_DEDUP_MODE=1, WEAVE_LOCAL_SCATTER_DEDUP=1."
    ),
}

PRESETS = {
    "sm103": dict(
        platform="sm103",
        title="NVIDIA SM103 (148-SM) / Qwen3.8-Max / EP8 / S=1024",
        script="sm103_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_SM103_TRACE / WEAVE_SM103_RESULT_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_DISPATCH_ACTIVE_SMS", "0", "all SMs participate in dispatch pull"),
            ("Spatial", "WEAVE_SCATTER_ACTIVE_SMS", "0", "all SMs participate in scatter"),
            ("Spatial", "WEAVE_SM0_STREAM_SCHED", "0", "CTA 0 comm warps pull/scatter normally"),
            ("Spatial", "WEAVE_DISPATCH_WARPS", "4", "4 dispatch comm warps per CTA"),
            ("Spatial", "WEAVE_SCATTER_WARPS", "4", "4 scatter comm warps per CTA"),
            ("Spatial", "WEAVE_REDUCE_WARPS", "4", "dedicated fine-grained streaming Reduce warps in Middle"),
            ("Spatial", "WEAVE_REG_BALANCE", "0", "default 96/72/168 register file ratio"),
            ("Temporal", "WEAVE_V_HEAD", "0", "automatic head-pull quota before helping warps switch to GEMM"),
            ("Temporal", "WEAVE_READY_GROUP_SIZE", "2", "2-tile group arrival readiness"),
            ("Temporal", "WEAVE_READY_QUEUE", "formula", "static heuristic formula over prefix-summed offsets"),
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "strict communication-warp scatter ownership"),
            ("Temporal", "WEAVE_COMBINE_SYNC_CHUNK", "32", "fine-grained per-chunk arrival readiness"),
            ("Implementation", "WEAVE_PRECISION_MODE", "0", "fp8xfp4"),
            ("Implementation", "WEAVE_ADAPT_BM", "0", "fixed profile"),
            ("Implementation", "WEAVE_BLOCK_M_L1", "64", "GEMM1 blockM=64 profile"),
            ("Implementation", "WEAVE_BLOCK_M_L2", "64", "GEMM2 blockM=64 profile"),
            ("Implementation", "WEAVE_REG_PULL_BYTES", "4096", "4096B per-warp (32 regs/thread) zero-SMEM pull"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "remote pull dedup + local multi-expert fanout"),
            ("Implementation", "WEAVE_SCATTER_STAGING", "smem", "per-SM SMEM handoff"),
            ("Implementation", "WEAVE_LOCAL_SCATTER_DEDUP", "0", "direct per-hit scatter"),
        ],
    ),
    "h20": dict(
        platform="h20",
        title="NVIDIA H20 (SM90, 78 SMs) / Qwen3.8-Max / EP8 / S=1024",
        script="h20_qwen38max_ep8_s1024.sh",
        trace_env="WEAVE_H20_TRACE / WEAVE_H20_RESULT_ROOT",
        render=True,
        rows=[
            ("Spatial", "WEAVE_DISPATCH_ACTIVE_SMS", "0", "all 78 SMs participate in dispatch"),
            ("Spatial", "WEAVE_SCATTER_ACTIVE_SMS", "0", "all 78 SMs participate in scatter"),
            ("Spatial", "WEAVE_SM0_STREAM_SCHED", "0", "CTA 0 comm warps participate normally"),
            ("Spatial", "WEAVE_DISPATCH_WARPS", "0", "auto dispatch warp ratio"),
            ("Spatial", "WEAVE_REG_BALANCE", "0", "default register balance"),
            ("Temporal", "WEAVE_V_HEAD", "0", "automatic head-pull quota"),
            ("Temporal", "WEAVE_READY_GROUP_SIZE", "0", "expert arrival readiness"),
            ("Temporal", "WEAVE_READY_QUEUE", "sm0_fifo", "dynamic work-stealing FIFO queue"),
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "communication warps own remote stores"),
            ("Temporal", "WEAVE_COMBINE_SYNC_CHUNK", "0", "barrier combine sync"),
            ("Implementation", "WEAVE_PRECISION_MODE", "2", "fp8xfp8"),
            ("Implementation", "WEAVE_ADAPT_BM", "0", "fixed profile"),
            ("Implementation", "WEAVE_BLOCK_M_L1", "32", "GEMM1 blockM=32 (locks BN=256, BK=128, swapAB, 4 WGs)"),
            ("Implementation", "WEAVE_BLOCK_M_L2", "32", "GEMM2 blockM=32 (locks BN=256, BK=128, swapAB, 4 WGs)"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "remote pull dedup + local multi-expert fanout"),
            ("Implementation", "WEAVE_SCATTER_STAGING", "smem", "per-SM XOR-swizzled SMEM handoff"),
            ("Implementation", "WEAVE_LOCAL_SCATTER_DEDUP", "0", "direct per-hit scatter"),
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
            ("Spatial", "WEAVE_SM0_STREAM_SCHED", "0", "CTA 0 waves participate normally"),
            ("Spatial", "WEAVE_CU_WAVE_SPLIT", "4-4", "4 communication waves + 4 compute waves per CU"),
            ("Temporal", "WEAVE_V_HEAD", "0", "automatic opening pull quota"),
            ("Temporal", "WEAVE_SCATTER_OWNER", "comm_only", "communication waves own remote xGMI scatter"),
            ("Implementation", "WEAVE_PRECISION_MODE", "2", "fp8xfp8"),
            ("Implementation", "WEAVE_ADAPT_BM", "1", "per-expert M-bucketed profile lookup"),
            ("Implementation", "WEAVE_BLOCK_M_L1", "0", "auto from M_e"),
            ("Implementation", "WEAVE_BLOCK_M_L2", "0", "auto from M_e"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "unique (src_rank, token) pull + local fanout"),
            ("Implementation", "WEAVE_LOCAL_SCATTER_DEDUP", "1", "local bf16x2 merge before xGMI send"),
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
            ("Implementation", "WEAVE_ADAPT_BM", "1", "per-expert M-bucketed profile lookup"),
            ("Implementation", "WEAVE_DEDUP_MODE", "1", "unique (src_rank, token) pull + local fanout"),
            ("Implementation", "WEAVE_LOCAL_SCATTER_DEDUP", "1", "node-partial local merge before inter-node send"),
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


def load_renderer(path: Path):
    spec = importlib.util.spec_from_file_location("weave_timeline_renderer_v2", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_preset(preset_key: str) -> None:
    p = PRESETS[preset_key]
    script = EXAMPLES_DIR / p["script"]
    subprocess.run(["bash", str(script)], cwd=str(EXAMPLES_DIR), check=True)


def render_preview(preset_key: str, trace: Path | None, out_dir: Path, renderer: Path) -> None:
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
        "--config-banner", BANNERS[platform],
        "--repro-script", f"examples/{p['script']}",
    ]
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
    args = ap.parse_args(argv)
    print_table(args.preset)
    if args.run:
        run_preset(args.preset)
    p = PRESETS[args.preset]
    do_render = p["render"] if args.render is None else args.render
    if args.preset == "mi308x_ep16" or not do_render:
        return 0
    if args.trace is not None and not args.trace.is_file():
        raise SystemExit(f"ERROR: trace not found: {args.trace}")
    render_preview(args.preset, args.trace, args.out_dir, args.renderer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
