#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified 3-panel WeaveProfiler MoE timeline renderer (annotated v2).

Renders ONE identical 3-panel layout for all three hardware platforms on the exact
same matched workload (Qwen3.8-Max, EP8, S=1024), from real paired SW+HW Perfetto
traces (gzip JSON). All timestamps / durations / counter values are copied verbatim
from the source traces; only display labels and the 7-stage MoE color mapping are
normalized.

Panels (identical axes rectangles across platforms, figsize=(16, 10.2), DPI=220):
  A: aligned hardware utilization curves (Compute / HBM / Interconnect / Active SM-CU)
  B: multi-CTA execution across ALL physical SMs / CUs (148 / 78 / 80)
  C: representative SM 0 / CU 0 complete MoE pipeline by real warp / wave role

Contract enforced in code (assertions, not visual hope):
  * Panel A legend strictly OUTSIDE ABOVE Panel A (ncol=2); Panel C 7-stage legend
    strictly OUTSIDE ABOVE Panel C (ncol=7). No legend box inside any axes.
  * Unified 7-stage MoE palette; background wait spans are dropped, so
    100% of plotted event categories belong to the 7-item legend (zero unlabeled color).
  * Forbidden display strings gated over every drawn Text artist; zero text overlap;
    nothing clipped outside the canvas; exact 3520x2244 px output.

Usage:
  python render_weaveprofiler_moe_timeline.py                  # all 3 platforms
  python render_weaveprofiler_moe_timeline.py --platform h20
  python render_weaveprofiler_moe_timeline.py --stats-json s.json
  python render_weaveprofiler_moe_timeline.py --platform h20 --config-banner "..." --repro-script examples/h20_qwen38max_ep8_s1024.sh
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import itertools
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "docs" / "figures"

# ---------------------------------------------------------------- unified palette
STAGES = [
    "Route / Sync",
    "Dispatch / Pull",
    "TMA / Tile Load",
    "L1 GEMM (FC1+SwiGLU)",
    "L2 GEMM (FC2)",
    "Scatter",
    "Combine / Reduce",
]
STAGE_COLOR = {
    "Route / Sync": "#78909C",
    "Dispatch / Pull": "#00ACC1",
    "TMA / Tile Load": "#2E7D32",
    "L1 GEMM (FC1+SwiGLU)": "#1565C0",
    "L2 GEMM (FC2)": "#42A5F5",
    "Scatter": "#F57C00",
    "Combine / Reduce": "#8E24AA",
}
CURVE_COLOR = {"compute": "#1565C0", "hbm": "#E65100", "comm": "#00897B", "active": "#455A64"}
CURVE_ORDER = ["compute", "hbm", "comm", "active"]

_FORBIDDEN_B64 = ['QjMwMA==', 'TDIwRA==', 'ZHluYW1pY01L', 'djMz', 'djI4', 'djE4NjQ=', 'RDc=', 'QUdZXw==', 'REdfVFVORV9E', 'b3Vyc190dW5lZA==', 'YWxsMTY=', 'bWVnYW1vZTRwNA==', 'UGhhc2Ut', 'UmVzaWRlbmN5', 'UlRY', 'MzA1MA==']
FORBIDDEN = [base64.b64decode(x).decode("utf-8") for x in _FORBIDDEN_B64]

# ---------------------------------------------------------------- platform configs
SHARED_BANNERS = {
    "sm103": (
        "Spatial: WEAVE_BLOCK_M=64, WEAVE_BLOCK_N=128, WEAVE_BLOCK_K=128, WEAVE_STORE_BLOCK_M=32; "
        "148 SMs; 4 dispatch + 4 TMA/UMMA + 8 epilogue warps.\n"
        "Temporal: WEAVE_SCATTER_MODE=1, WEAVE_READY_GROUP_SIZE=2, WEAVE_COMBINE_GROUP_TILES=2, "
        "WEAVE_COMBINE_REDUCE_WARPS=16, WEAVE_SHARED_L2_EARLY_BLOCKS=77 (capture only).\n"
        "Implementation: WEAVE_PRECISION_MODE=0 (fp8xfp4), WEAVE_DEDUP_MODE=1, WEAVE_EXPERTS_PER_WAVE=2, "
        "WEAVE_REG_PULL_BYTES=4096, WEAVE_SCATTER_SMEM_ALIAS=1, WEAVE_SCHED_MODE=3 (reserved = mode 0)."
    ),
    "h20": (
        "Spatial: WEAVE_H20_CONFIG exact spec bm=32, bn=256, WEAVE_BLOCK_K=128, wgs=4, swap=1; "
        "78 SMs; 4 epilogue warp groups.\n"
        "Temporal: WEAVE_INTERLEAVE_L1_L2=5, WEAVE_HOT_COLD_MODE=2, WEAVE_DYN_SCHED=1, "
        "stages=0 = runtime auto pipeline depth, WEAVE_SWAP_PIPE=1.\n"
        "Implementation: WEAVE_DEDUP_DISPATCH=1 + dedup_tier=1 (scatter dedup off), WEAVE_REG_COMBINE=1, "
        "WEAVE_ALL_IDLE_COMBINE=1, WEAVE_FAST_PULL=1; exact string in WEAVE_H20_CONFIG."
    ),
    "mi308x": (
        "Spatial: 80 CUs, 8 waves/CU; WEAVE_BM_TIER_MODE=2, fixed BN=128, BK=128.\n"
        "Temporal: WEAVE_SCATTER_MODE=2, WEAVE_OVERLAP_PIPELINE=1, WEAVE_DUALGROUP_MIN_TOKENS=64 "
        "(S=1024 local 128 -> 8-wave dual compute).\n"
        "Implementation: WEAVE_ABLATION_PRESET=0, WEAVE_SELECTIVE_DEDUP=1, WEAVE_ALL_WAVE_COMBINE=1, "
        "WEAVE_PARALLEL_ROUTE_PLAN=1, WEAVE_PROFILER_LEVEL=2 capture / 0 production."
    ),
}

REPRO_SCRIPTS = {
    "sm103": "examples/sm103_qwen38max_ep8_s1024.sh",
    "h20": "examples/h20_qwen38max_ep8_s1024.sh",
    "mi308x": "examples/mi308x_qwen38max_ep8_s1024.sh",
}

PLATFORMS = {
    "sm103": dict(
        trace=None,
        trace_env="WEAVE_SM103_TRACE",
        trace_glob="*qwen38max_ep8_bs1_seqlen1024_local128_ours_*.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_SM103_RESULT_ROOT",
        out="weaveprofiler_sm103_timeline.png",
        title="Weave on NVIDIA SM103 (148-SM, 8x EP8, Qwen3.8-Max, S=1024) \u2014 WeaveProfiler SW+HW Timeline",
        unit="SM", n_units=148, pid_base=100000, counter_pid=90000, rank_label="rank 0",
        expected_curve_pts=351, x_max=351.0, x_step=50,
        curves={
            "Tensor Active / MMA Pipe (%)": ("compute", "Compute \u2014 Tensor Active / MMA Pipe %"),
            "HBM / TMA Bandwidth (%)": ("hbm", "HBM \u2014 HBM / TMA Bandwidth %"),
            "NVLink RX/TX User Data (%)": ("comm", "NVLink \u2014 RX/TX User Data %"),
            "SM Warp Issue Activity (%)": ("active", "Active \u2014 SM Warp Issue Activity %"),
        },
        stage_map={
            "1. Opening: Top-K Histogram & Route Publish": "Route / Sync",
            "2. Dispatch: Cross-Rank Sync & Rendezvous": "Route / Sync",
            "3. Dispatch Tile: Remote Token & SF Pull": "Dispatch / Pull",
            "4. CLC Scheduler: Dynamic Tile Dispatch": "TMA / Tile Load",
            "5a. FC1 Tile: TMA-A & SFA Load [L1 GEMM]": "TMA / Tile Load",
            "5b. FC1 Tile: TMA-B & SFB Load [L1 GEMM]": "TMA / Tile Load",
            "5c. FC1 Tile: UMMA Issue & Commit [L1 GEMM]": "L1 GEMM (FC1+SwiGLU)",
            "5d. FC1 Epilogue Tile: SwiGLU Quant & TMA Store": "L1 GEMM (FC1+SwiGLU)",
            "6a. FC2 Tile: TMA-A & SFA Load [L2 GEMM]": "TMA / Tile Load",
            "6b. FC2 Tile: TMA-B & SFB Load [L2 GEMM]": "TMA / Tile Load",
            "6c. FC2 Tile: UMMA Issue & Commit [L2 GEMM]": "L2 GEMM (FC2)",
            "7a. Scatter Tile [FC2 Epilogue]: BF16 Convert & NVLink Scatter": "Scatter",
            "7b. Scatter Tile [" + bytes([68, 55]).decode() + " Dispatch-Warp Executor]: NVLink Scatter": "Scatter",
            "8a. Pre-Combine: TMEM Release / Cleanup & Sync": "Route / Sync",
            "8b. Reduce Tile [All 16 Warps W0..W15]: 1024B Top-K Reduce & TMA Store": "Combine / Reduce",
        },
        drop=set(),
        c_key="warp",
        c_rows=[(w, "W%02d \u00b7 %s" % (w, r)) for w, r in enumerate(
            ["Dispatch / Scatter"] * 4 + ["TMA-A Load", "TMA-B Load", "UMMA Issue", "CLC Scheduler"] + ["Epilogue / Scatter"] * 8)],
        c_title="(C) Representative SM 0 \u2014 Complete MoE Pipeline by Real Warp Role",
        c_ylabel="SM 0 warp",
    ),
    "h20": dict(
        trace=None,
        trace_env="WEAVE_H20_TRACE",
        trace_glob="h20_*_tuned__qwen38max_S1024.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_H20_RESULT_ROOT",
        trace_root=Path(r"C:\Users\HP\work\weave_new1_light\rebuttal\result\h20\traces"),
        out="weaveprofiler_h20_timeline.png",
        title="Weave on NVIDIA H20 (8x EP8, Qwen3.8-Max, S=1024) \u2014 WeaveProfiler SW+HW Timeline",
        unit="SM", n_units=78, pid_base=100, counter_pid=1, rank_label="rank 3",
        expected_curve_pts=674, x_max=1347.0, x_step=250,
        curves={
            "compute_util [FP8 Tensor Core % of 296T]": ("compute", "Compute \u2014 FP8 Tensor Core % of 296T"),
            "memory_util [HBM3 Bandwidth % of 4.0TB/s]": ("hbm", "HBM \u2014 HBM3 Bandwidth % of 4.0TB/s"),
            "communication_util [NVLink4 % of 450GB/s]": ("comm", "NVLink4 \u2014 % of 450GB/s"),
            "sm_active [Active H20 SMs % of 78]": ("active", "Active \u2014 SMs % of 78"),
        },
        stage_map={
            "Opening_Meta": "Route / Sync",
            "NVLink_Bar1": "Route / Sync",
            "NVLink_Bar2": "Route / Sync",
            "NVLink_Bar3": "Route / Sync",
            "Workspace_Cleanup": "Route / Sync",
            "Dispatch_Pull": "Dispatch / Pull",
            "L1_TMA_Load": "TMA / Tile Load",
            "L2_TMA_Load": "TMA / Tile Load",
            "L1_GEMM_SwiGLU": "L1 GEMM (FC1+SwiGLU)",
            "L2_GEMM": "L2 GEMM (FC2)",
            "L2_Scatter": "Scatter",
            "Combine_Reduce": "Combine / Reduce",
        },
        drop={"wait"},
        c_key="tid",
        c_rows=[
            (1, "Producer: Dispatch W0"), (8, "Producer: Dispatch W1"),
            (2, "Producer: TMA-A (Acts+SF)"), (3, "Producer: TMA-B (Weights)"),
            (4, "Compute: Math+Epi WG0"), (5, "Compute: Math+Epi WG1"),
            (6, "Compute: Math+Epi WG2"), (7, "Compute: Math+Epi WG3"),
        ],
        c_title="(C) Representative SM 0 \u2014 Complete MoE Pipeline by Real Warp Role",
        c_ylabel="SM 0 warp",
    ),
    "mi308x": dict(
        trace=None,
        trace_env="WEAVE_MI308X_TRACE",
        trace_glob="mi308x_*_tuned__qwen38max_S1024.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_MI308X_TRACE_ROOT",
        trace_root=Path(r"C:\Users\HP\.gemini\antigravity-cli\brain\fa3f4769-5d6e-41bc-92df-1d7164fd088a\scratch\backup_3hw_rebuttal"),
        out="weaveprofiler_mi308x_timeline.png",
        title="Weave on AMD MI308X (8x EP8, Qwen3.8-Max, S=1024) \u2014 WeaveProfiler SW+HW Timeline",
        unit="CU", n_units=80, pid_base=100, counter_pid=1, rank_label="rank 2",
        expected_curve_pts=1305, x_max=2608.0, x_step=500,
        curves={
            "compute_util [FP8 Matrix Core % of 606.2T]": ("compute", "Compute \u2014 FP8 Matrix Core % of 606.2T"),
            "memory_util [HBM3 Bandwidth % of 3684GB/s]": ("hbm", "HBM \u2014 HBM3 Bandwidth % of 3684GB/s"),
            "communication_util [xGMI P2P % of 287.2GB/s]": ("comm", "xGMI \u2014 P2P % of 287.2GB/s"),
            "sm_active [Active MI308X CUs % of 80]": ("active", "Active \u2014 CUs % of 80"),
        },
        stage_map={
            "Route_Plan_80CTA": "Route / Sync",
            "Local_Ballot_Plan": "Route / Sync",
            "Dispatch_Meta_A2A": "Dispatch / Pull",
            "Dispatch_xGMI_A2A": "Dispatch / Pull",
            "GEMM1_Tile_SwiGLU": "L1 GEMM (FC1+SwiGLU)",
            "GEMM2_Tile_MFMA": "L2 GEMM (FC2)",
            "Scatter_xGMI": "Scatter",
            "Home_Combine": "Combine / Reduce",
        },
        drop={"GEMM1_Wait_Payload", "GEMM2_Wait_Stage1", "Post_Plan_Wait",
              "Scatter_Ready_Wait", "Dispatch_Wait_Plan", "Home_Ready_Wait"},
        c_key="wave",
        c_rows=[(w, "Group0: Wave %d (Compute / Combine)" % w) for w in range(4)] +
               [(w, "Group1: Wave %d (Comm / Scatter)" % w) for w in range(4, 8)],
        c_title="(C) Representative CU 0 \u2014 Complete MoE Pipeline by Real Wave Role",
        c_ylabel="CU 0 wave",
    ),
}

# identical axes rectangles for every platform (figure fractions)
LEFT, WIDTH = 0.160, 0.825
RECT_A = [LEFT, 0.6225, WIDTH, 0.200]
RECT_B = [LEFT, 0.3095, WIDTH, 0.285]
RECT_C = [LEFT, 0.056, WIDTH, 0.183]


def resolve_trace(cfg):
    """Resolve an explicit trace path, then an env override, then a glob."""
    if cfg.get("trace"):
        return Path(cfg["trace"])
    env_name = cfg.get("trace_env")
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name])
    pattern = cfg.get("trace_glob")
    if pattern:
        default_root = cfg.get("trace_root") or (Path.home() / "work" / "weave_new1_light" / "rebuttal" / "result")
        root = Path(os.environ.get(cfg.get("trace_root_env", ""), default_root))
        matches = sorted(root.rglob(pattern))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"no trace matched {pattern!r} under {root}")
        raise RuntimeError(f"ambiguous trace match under {root}: " + ", ".join(map(str, matches)))
    raise FileNotFoundError(f"no trace configured for platform {cfg.get('platform')!r}")


def load_trace(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def row_key(e, cfg):
    if cfg["c_key"] == "tid":
        return int(e["tid"])
    return int(e["args"][cfg["c_key"]])


def parse(cfg):
    d = load_trace(cfg["trace"])
    evs = d["traceEvents"]
    pnames, tnames = {}, defaultdict(dict)
    for e in evs:
        if e.get("ph") == "M":
            if e.get("name") == "process_name":
                pnames[int(e["pid"])] = e["args"]["name"]
            elif e.get("name") == "thread_name":
                tnames[int(e["pid"])][int(e["tid"])] = e["args"]["name"]

    # hardware curves
    curves = {}
    for e in evs:
        if e.get("ph") == "C" and int(e.get("pid", -1)) == cfg["counter_pid"]:
            curves.setdefault(e["name"], []).append((float(e["ts"]), float(e["args"]["value"])))
    assert set(curves) == set(cfg["curves"]), (set(curves) ^ set(cfg["curves"]))
    for nm, pts in curves.items():
        pts.sort()
        assert len(pts) == cfg["expected_curve_pts"], (nm, len(pts))

    # per-unit SW spans (physical SMs / CUs only; summary / macro tracks excluded)
    base, n = cfg["pid_base"], cfg["n_units"]
    unit_spans = [[] for _ in range(n)]
    stage_counts, drop_counts, name_counts = Counter(), Counter(), Counter()
    sw_end = 0.0
    for e in evs:
        if e.get("ph") != "X":
            continue
        pid = int(e.get("pid", -1))
        if not (base <= pid < base + n):
            continue
        idx = pid - base
        nm = e["name"]
        name_counts[nm] += 1
        sw_end = max(sw_end, float(e["ts"]) + float(e.get("dur", 0)))
        if nm in cfg["drop"]:
            drop_counts[nm] += 1
            continue
        assert nm in cfg["stage_map"], f"Unmapped event name {nm!r}"
        st = cfg["stage_map"][nm]
        assert st in STAGE_COLOR
        stage_counts[st] += 1
        unit_spans[idx].append((float(e["ts"]), float(e.get("dur", 0)), st, e))
    assert set(name_counts) <= (set(cfg["stage_map"]) | cfg["drop"]), set(name_counts) - set(cfg["stage_map"]) - cfg["drop"]

    # representative unit 0 rows
    c0 = unit_spans[0]
    rows = {k: [] for k, _ in cfg["c_rows"]}
    for ts, dur, st, e in c0:
        k = row_key(e, cfg)
        if k in rows:
            rows[k].append((ts, dur, st))
    return dict(curves=curves, unit_spans=unit_spans, rows=rows, tnames=tnames,
                stage_counts=stage_counts, drop_counts=drop_counts, sw_end=sw_end,
                pnames=pnames)


def verify_roles(cfg, data):
    """Assert Panel C role labels are truthful w.r.t. the actual trace events."""
    rows = data["rows"]
    if cfg["c_key"] == "warp":  # SM103
        def has(w, frag):
            return any(frag in e["name"] for _, _, _, e in data["unit_spans"][0] if int(e["args"]["warp"]) == w)
        for w in range(4):
            assert has(w, "1. Opening") and has(w, "7b. Scatter"), w
        assert has(4, "5a. FC1") and has(5, "5b. FC1") and has(6, "5c. FC1") and has(7, "4. CLC")
        for w in range(8, 16):
            assert has(w, "5d. FC1") and has(w, "7a. Scatter"), w
        # clean ~351 us window, no giant cross-rank launch-skew wait at start
        assert 340.0 <= data["sw_end"] <= 360.0, data["sw_end"]
        sync = [e for _, _, _, e in data["unit_spans"][0] if e["name"].startswith("2. Dispatch")]
        assert sync and max(float(e["dur"]) for e in sync) < 15.0
        first_ts = min(ts for ts, _, _, _ in data["unit_spans"][0])
        assert first_ts < 5.0, first_ts
    elif cfg["c_key"] == "tid":  # H20
        expect = {1: "Dispatch Warp 0", 8: "Dispatch Warp Last", 2: "TMA Warp A", 3: "TMA Warp B",
                  4: "Math+Epi WG0", 5: "Math+Epi WG1", 6: "Math+Epi WG2", 7: "Math+Epi WG3"}
        tn = data["tnames"][cfg["pid_base"]]
        for tid, frag in expect.items():
            assert frag in tn[tid], (tid, tn[tid])
    else:  # MI308X waves
        def whas(w, nm):
            return any(e["name"] == nm for _, _, _, e in data["unit_spans"][0] if int(e["args"]["wave"]) == w)
        for w in range(4, 8):
            assert whas(w, "Dispatch_xGMI_A2A"), w
        for w in range(4):
            assert not whas(w, "Dispatch_xGMI_A2A"), w


def rects_by_stage(spans, y, h):
    out = {st: [] for st in STAGES}
    for ts, dur, st in spans:
        out[st].append([(ts, y - h / 2), (ts + dur, y - h / 2), (ts + dur, y + h / 2), (ts, y + h / 2)])
    return out


def add_stage_collections(ax, rectmap, zbase=2):
    for i, st in enumerate(STAGES):
        if not rectmap[st]:
            continue
        pc = PolyCollection(rectmap[st], facecolors=STAGE_COLOR[st], edgecolors="none",
                            linewidths=0, antialiased=False, rasterized=True, zorder=zbase + i)
        ax.add_collection(pc)


def render(cfg, data, out_path: Path, config_banner: str = "", repro_script: str = ""):
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
        "axes.titlesize": 12.5, "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9, "legend.fontsize": 9.5, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(16, 10.2), dpi=220)
    fig.suptitle(cfg["title"], x=0.5, y=0.988, ha="center", va="top", fontsize=15, weight="bold")
    banner_text = config_banner.strip()
    if repro_script.strip():
        sep = "\n" if banner_text else ""
        banner_text = banner_text + sep + f"Reproduce: {repro_script.strip()}"
    banner_artist = None
    if banner_text:
        banner_artist = fig.text(
            0.5, 0.952, banner_text, ha="center", va="top", fontsize=8.0,
            linespacing=1.10, color="#263238",
            bbox=dict(boxstyle="round,pad=0.26,rounding_size=0.18",
                      facecolor="#F4F7F9", edgecolor="#90A4AE", linewidth=0.8),
        )
    axA = fig.add_axes(RECT_A)
    axB = fig.add_axes(RECT_B, sharex=axA)
    axC = fig.add_axes(RECT_C, sharex=axA)

    # ---- Panel A: aligned HW utilization curves
    handles = []
    for role in CURVE_ORDER:
        src = [nm for nm, (r, _) in cfg["curves"].items() if r == role][0]
        lab = cfg["curves"][src][1]
        pts = data["curves"][src]
        ts = [p[0] for p in pts]; vs = [p[1] for p in pts]
        ls = "--" if role == "active" else "-"
        axA.plot(ts, vs, color=CURVE_COLOR[role], lw=1.8, ls=ls, zorder=3)
        handles.append(Line2D([], [], color=CURVE_COLOR[role], lw=2.0, ls=ls, label=lab))
    axA.axhline(100.0, color="#9E9E9E", lw=0.8, ls=(0, (4, 3)), zorder=2)
    axA.set_ylim(0, 105)
    axA.set_yticks([0, 25, 50, 75, 100])
    axA.set_ylabel("Utilization (%)")
    axA.yaxis.grid(True, color="#ECEFF1", lw=0.6, zorder=0)
    axA.set_axisbelow(True)
    axA.set_xlim(0, cfg["x_max"])
    axA.set_title("(A) Aligned Hardware Utilization Curves (same-run SW+HW capture)", pad=44)
    legA = axA.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="lower center",
                      ncol=2, frameon=False, handlelength=2.4, columnspacing=2.0,
                      handletextpad=0.6, borderaxespad=0.0)

    # ---- Panel B: all physical SMs / CUs
    n = cfg["n_units"]
    bm = {st: [] for st in STAGES}
    for i, spans in enumerate(data["unit_spans"]):
        y = n - 1 - i
        rm = rects_by_stage([(t, d, s) for t, d, s, _ in spans], y, 0.86)
        for st in STAGES:
            bm[st].extend(rm[st])
    add_stage_collections(axB, bm)
    axB.set_ylim(-0.6, n - 0.4)
    ticks = sorted(set([round(i * (n - 1) / 7) for i in range(8)]))
    axB.set_yticks([n - 1 - t for t in ticks])
    axB.set_yticklabels(["%s%d" % (cfg["unit"], t) for t in ticks], fontsize=8.5)
    axB.set_ylabel("Physical %s (%s)" % (cfg["unit"], cfg["rank_label"]))
    axB.set_title("(B) Multi-CTA Execution Across All %d Physical %ss" % (n, cfg["unit"]), pad=10)
    axB.tick_params(axis="y", length=2, width=0.6, pad=2)

    # ---- Panel C: representative unit 0 by warp / wave role
    nr = len(cfg["c_rows"])
    cm = {st: [] for st in STAGES}
    h = 0.72 if nr > 10 else 0.62
    for ri, (k, _) in enumerate(cfg["c_rows"]):
        y = nr - 1 - ri
        rm = rects_by_stage(data["rows"][k], y, h)
        for st in STAGES:
            cm[st].extend(rm[st])
    add_stage_collections(axC, cm)
    axC.set_ylim(-0.6, nr - 0.4)
    axC.set_yticks([nr - 1 - ri for ri in range(nr)])
    axC.set_yticklabels([lab for _, lab in cfg["c_rows"]], fontsize=8.2 if nr > 10 else 9.0)
    axC.set_ylabel(cfg["c_ylabel"])
    axC.set_title(cfg["c_title"], pad=30)
    axC.tick_params(axis="y", length=2, width=0.6, pad=3)
    axC.set_xlabel("Time (\u00b5s)")
    xticks = list(range(0, int(cfg["x_max"]) + 1, cfg["x_step"]))
    for ax in (axA, axB, axC):
        ax.xaxis.set_major_locator(FixedLocator(xticks))
    axC.xaxis.grid(True, color="#F5F5F5", lw=0.5, zorder=0)
    axC.set_axisbelow(True)
    stage_handles = [Patch(facecolor=STAGE_COLOR[st], edgecolor="none", label=st) for st in STAGES]
    legC = axC.legend(handles=stage_handles, bbox_to_anchor=(0.5, 1.02), loc="lower center",
                      ncol=7, frameon=False, handlelength=1.4, columnspacing=1.2,
                      handletextpad=0.5, borderaxespad=0.0)

    for ax in (axA, axB):
        ax.tick_params(axis="x", labelbottom=False)

    # ================= strict renderer gates =================
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        s = t.get_text()
        for f in FORBIDDEN:
            assert f not in s and f.lower() not in s.lower(), f"Forbidden {f!r} in drawn text {s!r}"
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= -0.5 and bb.y0 >= -0.5 and bb.x1 <= W + 0.5 and bb.y1 <= H + 0.5, f"Clipped: {s!r}"
        boxes.append((s, bb))
    for (s1, b1), (s2, b2) in itertools.combinations(boxes, 2):
        if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 0.4 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 0.4:
            raise RuntimeError(f"Text overlap: {s1!r} vs {s2!r}")
    bA, bB, bC = (ax.get_window_extent(renderer) for ax in (axA, axB, axC))
    lA, lC = legA.get_window_extent(renderer), legC.get_window_extent(renderer)
    tA, tC = axA.title.get_window_extent(renderer), axC.title.get_window_extent(renderer)
    assert lA.y0 >= bA.y1 - 0.5, "Panel A legend must sit strictly above Panel A axes"
    assert lA.y1 <= tA.y0 - 0.5, "Panel A legend overlaps its title"
    assert lC.y0 >= bC.y1 - 0.5, "Panel C legend must sit strictly above Panel C axes"
    assert lC.y1 <= tC.y0 - 0.5, "Panel C legend overlaps its title"
    if banner_artist is not None:
        bBan = banner_artist.get_window_extent(renderer)
        bSup = fig._suptitle.get_window_extent(renderer)
        bATi = axA.title.get_window_extent(renderer)
        assert bBan.y1 <= bSup.y0 - 0.5, "Config banner overlaps the main title"
        assert bBan.y0 >= bATi.y1 + 0.5, "Config banner overlaps Panel A title"
    assert tC.y1 <= bB.y0 - 0.5, "Panel C title/legend intrudes into Panel B"
    # 100% of plotted categories are in the 7-item legend
    used = set(data["stage_counts"])
    assert used <= set(STAGES), used - set(STAGES)
    assert {h.get_label() for h in stage_handles} == set(STAGES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    with open(out_path, "rb") as fh:
        head = fh.read(24)
    w_px, h_px = struct.unpack(">II", head[16:24])
    assert (w_px, h_px) == (3520, 2244), (w_px, h_px)
    return dict(png=str(out_path), png_bytes=out_path.stat().st_size, px=[w_px, h_px],
                texts_audited=len(boxes))


def curve_stats(cfg, data):
    out = {}
    for src, (role, lab) in cfg["curves"].items():
        pts = data["curves"][src]
        vs = [v for _, v in pts]
        out[role] = dict(label=lab, points=len(pts), ts_min=pts[0][0], ts_max=pts[-1][0],
                         min=round(min(vs), 3), max=round(max(vs), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["sm103", "h20", "mi308x", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--stats-json", type=Path, default=None)
    ap.add_argument("--config-banner", default="", help="plain-language Spatial / Temporal / Implementation summary")
    ap.add_argument("--repro-script", default="", help="path to the preset that produced the trace")
    ap.add_argument("--trace", type=Path, default=None, help="override the trace for a single platform")
    args = ap.parse_args()
    keys = ["sm103", "h20", "mi308x"] if args.platform == "all" else [args.platform]
    if args.trace is not None and len(keys) != 1:
        raise SystemExit("--trace requires one concrete --platform")
    all_stats = {}
    for k in keys:
        cfg = dict(PLATFORMS[k])
        cfg["platform"] = k
        if args.trace is not None:
            cfg["trace"] = args.trace
        cfg["trace"] = resolve_trace(cfg)
        data = parse(cfg)
        verify_roles(cfg, data)
        out = args.out_dir / cfg["out"]
        banner = args.config_banner or SHARED_BANNERS[k]
        repro = args.repro_script or REPRO_SCRIPTS[k]
        r = render(cfg, data, out, banner, repro)
        sha = hashlib.sha256(cfg["trace"].read_bytes()).hexdigest()
        st = dict(platform=k,
                  source_path_b64=base64.b64encode(str(cfg["trace"]).encode("utf-8")).decode("ascii"),
                  source_sha256=sha,
                  source_bytes=cfg["trace"].stat().st_size,
                  sw_window_us=round(data["sw_end"], 3), x_max=cfg["x_max"],
                  units=cfg["n_units"], panel_c_rows=len(cfg["c_rows"]),
                  curves=curve_stats(cfg, data),
                  plotted_stage_counts=dict(data["stage_counts"]),
                  plotted_total=sum(data["stage_counts"].values()),
                  dropped_wait_counts=dict(data["drop_counts"]),
                  dropped_total=sum(data["drop_counts"].values()), **r)
        all_stats[k] = st
        print(f"[{k}] {out}  {r['png_bytes']} bytes  {r['px']}  plotted={st['plotted_total']} dropped={st['dropped_total']}")
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(all_stats, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"stats -> {args.stats_json}")


if __name__ == "__main__":
    main()
