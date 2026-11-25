#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the two repository architecture schematics:
  1. docs/figures/weave_design_overview.png
     Per-GPU live routing + hardware profile -> Spatial (cross-GPU, cross-SM/CU &
     intra-SM/CU), Temporal (head/middle/tail dynamic roles), and Independent
     4-Operator configurations across SM100/SM103, SM90 H20, and CDNA3 MI308X.
  2. docs/figures/weave_design_rolegraph.png
     Multi-granularity spatial partitioning & phase-aware temporal schedules:
       (a) Static Megakernel (uniform ratio + head/middle/tail bubbles),
       (b) Cross-GPU & Cross-SM/CU Spatial Mixing on one GPU (SM 0..k-1 vs SM k..N-1),
       (c) NVIDIA SM103 (16w/SM) & H20 (20w/SM) Dynamic Temporal Roles,
       (d) AMD MI308X CDNA3 (8 Wave64/CU: (0,8) Compute CU vs (4,4) Hybrid CU).

Programmatic gates enforce 0 text overlaps, 0 canvas clipping, 0 forbidden
tokens, and Type 42 vector PDF fonts.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Text

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

FORBIDDEN = [
    "B300", "L20D", "dynamicMK", "v33", "v28", "v70", "v62", "v58",
    "v1864", "D7", "AGY", "Residency", "§", "ABBA", "0-spill",
    "7-W", "reg_2stage", "hbm_dedup", "grid_sync", "Oracle",
]

FONT = next(
    n for n in ("Arial", "Helvetica", "DejaVu Sans")
    if any(f.name == n for f in font_manager.fontManager.ttflist)
)
plt.rcParams.update({
    "font.family": [FONT],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "text.usetex": False,
})


def verify_canvas(fig: plt.Figure, label: str) -> None:
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    vis = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if not t.get_visible() or not s:
            continue
        for f in FORBIDDEN:
            assert f.lower() not in s.lower(), f"{label}: forbidden {f!r} in {s!r}"
        bb = t.get_window_extent(rnd)
        assert (
            bb.x0 >= -0.5 and bb.y0 >= -0.5
            and bb.x1 <= fig.bbox.width + 0.5 and bb.y1 <= fig.bbox.height + 0.5
        ), f"{label}: text clipped {s!r} -> {bb} vs {fig.bbox}"
        vis.append((s, bb))
    overlaps = [
        (a, b) for (a, ba), (b, bb) in itertools.combinations(vis, 2)
        if min(ba.x1, bb.x1) - max(ba.x0, bb.x0) > 0.6 and min(ba.y1, bb.y1) - max(ba.y0, bb.y0) > 0.6
    ]
    assert not overlaps, f"{label}: text overlaps detected: {overlaps[:5]}"
    print(f"[{label}] gates PASS: {len(vis)} text items, 0 overlaps, 0 clipped, 0 forbidden")


def rbox(ax, x0, y0, x1, y1, ec="#2B3A4A", fc="white", lw=0.85, ls="-", r=0.045, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        ec=ec, fc=fc, lw=lw, ls=ls, zorder=z,
    ))


def arrow(ax, x0, y0, x1, y1, color="#334155", lw=0.95, z=8):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>,head_length=3.6,head_width=2.1",
        mutation_scale=1.0, lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def badge(ax, cx, cy, txt, fc="#0072B2", r=0.088, fs=6.4, z=15):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec="white", lw=0.7, zorder=z))
    ax.text(cx, cy, str(txt), fontsize=fs, fontweight="bold", color="white",
            ha="center", va="center", zorder=z + 1)


# ==============================================================================
# 1. Weave System Design Overview
# ==============================================================================
def render_overview() -> None:
    W, H = 9.2, 3.45
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(
        W / 2, 3.31,
        "Weave Architecture: Routing-Adaptive Spatial, Temporal, and Operator Scheduling Across GPUs & SMs",
        fontsize=8.5, fontweight="bold", color="#0F172A", ha="center", va="center",
    )

    # ---------------- Column 1: Hardware Profile + Per-GPU Live Routing ----------------
    c1_x0, c1_x1 = 0.10, 2.52
    rbox(ax, c1_x0, 0.12, c1_x1, 3.14, ec="#B4691E", fc="#FFF9F2", lw=1.0, r=0.06, z=1)
    badge(ax, c1_x0 + 0.15, 2.97, "1", fc="#C96A1B")
    ax.text(c1_x0 + 0.29, 3.01, "Inputs: Hardware & Live Routing", fontsize=7.2, fontweight="bold", color="#8C4308", va="center")
    ax.text(c1_x0 + 0.29, 2.86, "Offline calibration + online per-step stats", fontsize=5.8, color="#6B4F35", va="center")

    # Subbox 1A: Offline Hardware Profile Theta
    rbox(ax, 0.20, 1.78, 2.42, 2.72, ec="#D99B52", fc="white", lw=0.8, r=0.04, z=2)
    ax.text(1.31, 2.58, "Offline Hardware Profile (autotune)", fontsize=6.3, fontweight="bold", color="#7C3A06", ha="center", va="center")
    ax.text(1.31, 2.40, "• SM100 / SM103: TMEM, UMMA, NVLink", fontsize=5.7, color="#334155", ha="center", va="center")
    ax.text(1.31, 2.24, "• SM90 (H20): WGMMA, TMA, NVLink", fontsize=5.7, color="#334155", ha="center", va="center")
    ax.text(1.31, 2.08, "• CDNA3 (MI308X): MFMA, xGMI + RDMA", fontsize=5.7, color="#334155", ha="center", va="center")
    ax.text(1.31, 1.90, "Measures peak FLOPs, HBM/link BW & legal tiles", fontsize=5.45, color="#64748B", ha="center", va="center")

    # Subbox 1B: Live Per-GPU Routing Counts
    rbox(ax, 0.20, 0.22, 2.42, 1.66, ec="#0284C7", fc="white", lw=0.8, r=0.04, z=2)
    ax.text(1.31, 1.52, "Per-GPU Live Routing Histogram", fontsize=6.3, fontweight="bold", color="#075985", ha="center", va="center")
    ax.text(1.31, 1.35, "Per-expert tokens m_e, unique pulls D_in, sends M_out", fontsize=5.55, color="#334155", ha="center", va="center")

    # Mini GPU 0 vs GPU 1 skew illustration
    rbox(ax, 0.28, 0.78, 1.27, 1.22, ec="#0072B2", fc="#F0F9FF", lw=0.65, r=0.03, z=3)
    ax.text(0.775, 1.11, "GPU 0 (Comm-Skewed)", fontsize=5.4, fontweight="bold", color="#075985", ha="center", va="center")
    ax.text(0.775, 0.92, "High D_in -> more Comm SMs", fontsize=5.1, color="#1E293B", ha="center", va="center")

    rbox(ax, 1.35, 0.78, 2.34, 1.22, ec="#009E73", fc="#ECFDF5", lw=0.65, r=0.03, z=3)
    ax.text(1.845, 1.11, "GPU 1 (Compute-Heavy)", fontsize=5.4, fontweight="bold", color="#065F46", ha="center", va="center")
    ax.text(1.845, 0.92, "Hot experts -> more Math SMs", fontsize=5.1, color="#1E293B", ha="center", va="center")

    ax.text(1.31, 0.56, "Each GPU picks its own schedule (r_g, s_g)", fontsize=5.6, fontweight="bold", color="#0F172A", ha="center", va="center")
    ax.text(1.31, 0.37, "Decoupled across GPUs via peer-to-peer queues", fontsize=5.35, color="#475569", ha="center", va="center")

    # ---------------- Column 2: 3-Axis Schedule Selection (14 Knobs) ----------------
    c2_x0, c2_x1 = 2.74, 6.08
    rbox(ax, c2_x0, 0.12, c2_x1, 3.14, ec="#0072B2", fc="#F4F9FD", lw=1.0, r=0.06, z=1)
    badge(ax, c2_x0 + 0.15, 2.97, "2", fc="#0072B2")
    ax.text(c2_x0 + 0.29, 3.01, "Routing-Adaptive Schedule (14 Knobs)", fontsize=7.2, fontweight="bold", color="#075985", va="center")
    ax.text(c2_x0 + 0.29, 2.86, "Cost model selects Spatial + Temporal + Operator configs", fontsize=5.75, color="#334155", va="center")

    arrow(ax, 2.43, 2.25, 2.73, 2.25, color="#8C4308")
    arrow(ax, 2.43, 0.98, 2.73, 0.98, color="#0284C7")

    # Card 2A: Spatial (with explicit Cross-SM + Intra-SM visual bar!)
    rbox(ax, 2.84, 1.80, 5.98, 2.72, ec="#0072B2", fc="#EBF5FB", lw=0.85, r=0.04, z=2)
    ax.text(2.92, 2.58, "Spatial (5 Knobs) — Cross-SM/CU Mix & Intra-SM/CU Ratio", fontsize=6.15, fontweight="bold", color="#0072B2", va="center")
    # Visual Cross-SM strip on one GPU
    ax.add_patch(Rectangle((2.92, 2.22), 1.38, 0.23, fc="#0284C7", ec="white", lw=0.5, zorder=4))
    ax.text(3.61, 2.335, "SM 0..k-1: Comm-Rich (8c:4e)", fontsize=5.2, fontweight="bold", color="white", ha="center", va="center", zorder=5)
    ax.add_patch(Rectangle((4.32, 2.22), 1.58, 0.23, fc="#009E73", ec="white", lw=0.5, zorder=4))
    ax.text(5.11, 2.335, "SM k..N-1: Comp-Rich (4c:8e)", fontsize=5.2, fontweight="bold", color="white", ha="center", va="center", zorder=5)
    ax.text(2.92, 2.07, "• Cross-SM: DISPATCH_ACTIVE_SMS (D), SCATTER_ACTIVE_SMS (K)", fontsize=5.5, color="#1E293B", va="center")
    ax.text(2.92, 1.90, "• Intra-SM: DISPATCH_WARPS, SCATTER_WARPS, CU_WAVE_SPLIT", fontsize=5.5, color="#1E293B", va="center")

    # Card 2B: Temporal
    rbox(ax, 2.84, 1.02, 5.98, 1.72, ec="#009E73", fc="#EAFAF4", lw=0.85, r=0.04, z=2)
    ax.text(2.92, 1.58, "Temporal (3 Knobs) — Dynamic Phase Role Transitions", fontsize=6.15, fontweight="bold", color="#047857", va="center")
    ax.text(2.92, 1.39, "• Head: V_HEAD helper pull + READY_GROUP_SIZE N-tile start", fontsize=5.5, color="#1E293B", va="center")
    ax.text(2.92, 1.23, "• Middle: SCATTER_OWNER (epilogue direct vs. comm staged)", fontsize=5.5, color="#1E293B", va="center")
    ax.text(2.92, 1.09, "• Tail: per-token ready (k/k) all-warp cooperative Combine", fontsize=5.5, color="#1E293B", va="center")

    # Card 2C: Implementation
    rbox(ax, 2.84, 0.22, 5.98, 0.94, ec="#D55E00", fc="#FEF3EC", lw=0.85, r=0.04, z=2)
    ax.text(2.92, 0.81, "Implementation (6 Knobs) — Independent 4-Operator Configs", fontsize=6.15, fontweight="bold", color="#C2410C", va="center")
    ax.text(2.92, 0.62, "• GEMM1_CONFIG, GEMM2_CONFIG (decoupled tile & multi-stage)", fontsize=5.5, color="#1E293B", va="center")
    ax.text(2.92, 0.46, "• PRECISION_MODE  • REG_PULL_BYTES (1K-4K / 8K)", fontsize=5.5, color="#1E293B", va="center")
    ax.text(2.92, 0.30, "• DEDUP_MODE (pull once)  • SCATTER_STAGING (reg / smem / hbm)", fontsize=5.5, color="#1E293B", va="center")

    # ---------------- Column 3: Fused Megakernel Execution ----------------
    c3_x0, c3_x1 = 6.30, 9.10
    rbox(ax, c3_x0, 0.12, c3_x1, 3.14, ec="#1E293B", fc="#F8FAFC", lw=1.0, r=0.06, z=1)
    badge(ax, c3_x0 + 0.15, 2.97, "3", fc="#0F766E")
    ax.text(c3_x0 + 0.29, 3.01, "Fused MoE Megakernel", fontsize=7.3, fontweight="bold", color="#0F172A", va="center")
    ax.text(c3_x0 + 0.29, 2.86, "Tensor/Matrix Cores active on all SMs/CUs", fontsize=5.8, color="#0F766E", fontweight="bold", va="center")

    for yc in (2.26, 1.37, 0.58):
        arrow(ax, 5.99, yc, 6.29, yc, color="#334155")

    # Concurrent pipeline box inside Column 3
    rbox(ax, 6.40, 1.46, 9.00, 2.72, ec="#CBD5E1", fc="white", lw=0.75, r=0.04, z=2)
    ax.text(7.70, 2.59, "Concurrent Cross-Engine Execution", fontsize=6.25, fontweight="bold", color="#0F172A", ha="center", va="center")

    tx0, tx1 = 6.92, 8.92
    X = lambda t: tx0 + t * (tx1 - tx0)
    tracks = [
        ("Epi",  2.34, [(0.00, 0.14, "#0072B2", "V_h"), (0.14, 0.80, "#56B4E9", "GEMM1/2 epi"), (0.80, 1.00, "#475569", "comb")]),
        ("Math", 2.11, [(0.00, 0.10, "#0072B2", ""), (0.10, 0.78, "#009E73", "UMMA/WGMMA/MFMA"), (0.78, 1.00, "#475569", "")]),
        ("Load", 1.88, [(0.00, 0.08, "#0072B2", ""), (0.08, 0.75, "#CC79A7", "Async Weight/Act"), (0.75, 1.00, "#475569", "")]),
        ("Comm", 1.65, [(0.00, 0.36, "#0072B2", "reg pull"), (0.36, 0.83, "#E69F00", "smem/reg scatter"), (0.83, 1.00, "#475569", "")]),
    ]
    rh = 0.175
    for name, yc, segs in tracks:
        ax.text(tx0 - 0.04, yc, name, fontsize=5.7, fontweight="bold", color="#334155", ha="right", va="center")
        for s0, s1, col, lab in segs:
            ax.add_patch(Rectangle((X(s0), yc - rh / 2), X(s1) - X(s0), rh, fc=col, ec="white", lw=0.4, zorder=4))
            if lab:
                tc = "#111111" if col in ("#56B4E9", "#E69F00") else "white"
                ax.text((X(s0) + X(s1)) / 2, yc, lab, fontsize=5.0, fontweight="bold", color=tc, ha="center", va="center", zorder=6)

    # Fine-grained atomic handoff box
    rbox(ax, 6.40, 0.22, 9.00, 1.35, ec="#0F766E", fc="#ECFDF5", lw=0.8, r=0.04, z=2)
    ax.text(7.70, 1.22, "Fine-Grained Tile & Token Handoffs", fontsize=6.25, fontweight="bold", color="#065F46", ha="center", va="center")
    ax.text(7.70, 1.02, "• Dispatch -> GEMM1: N-tile arrival counters", fontsize=5.55, color="#1E293B", ha="center", va="center")
    ax.text(7.70, 0.84, "• GEMM1 <-> GEMM2: tile-ready bitmap / ring", fontsize=5.55, color="#1E293B", ha="center", va="center")
    ax.text(7.70, 0.66, "• FC2 Scatter: multi-stage reg / smem / hbm", fontsize=5.55, color="#1E293B", ha="center", va="center")
    ax.text(7.70, 0.48, "• Combine: per-token (k/k) acquire readiness", fontsize=5.55, color="#1E293B", ha="center", va="center")
    ax.text(7.70, 0.30, "1.22x SM103 · 1.21x H20 · 1.88x MI308X GM", fontsize=5.7, fontweight="bold", color="#0F766E", ha="center", va="center")

    verify_canvas(fig, "weave_design_overview")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "weave_design_overview.png"
    out_pdf = FIG_DIR / "weave_design_overview.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    raw = out_pdf.read_bytes()
    assert b"/Type3" not in raw and (b"/CIDFontType2" in raw or b"/TrueType" in raw)
    plt.close(fig)
    print(f"wrote {out_png}")


# ==============================================================================
# 2. Multi-Granularity Spatial & Temporal Role Graph (4-Panel)
# ==============================================================================
def render_rolegraph() -> None:
    W, H = 9.6, 3.35
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    COL = {
        "pull":    "#0072B2",
        "scatter": "#E69F00",
        "load":    "#CC79A7",
        "mma":     "#009E73",
        "epi":     "#56B4E9",
        "combine": "#475569",
        "idle":    "#E2E8F0",
    }

    ax.text(
        W / 2, 3.21,
        "Multi-Granularity Spatial Partitioning (Cross-GPU, Cross-SM/CU & Intra-SM/CU) and Dynamic Temporal Roles",
        fontsize=8.2, fontweight="bold", color="#0F172A", ha="center", va="center",
    )

    panels = [
        (0.08, 2.32, "(a) Static Megakernel", "Uniform SM split + head/middle/tail bubbles", "#991B1B", "#FEF2F2"),
        (2.44, 4.82, "(b) Cross-GPU & Cross-SM Mixing", "Different SMs run different warp ratios", "#075985", "#F0F9FF"),
        (4.94, 7.22, "(c) NVIDIA SM103 & H20 Roles", "16w (SM103) / 20w (H20) temporal handoff", "#065F46", "#ECFDF5"),
        (7.34, 9.52, "(d) AMD MI308X (CDNA3)", "8 Wave64/CU · (0,8) vs (4,4) CU mix", "#7C2D12", "#FFF7ED"),
    ]
    for x0, x1, title, sub, ec, fc in panels:
        rbox(ax, x0, 0.36, x1, 3.04, ec=ec, fc=fc, lw=0.9, r=0.05, z=1)
        ax.text((x0 + x1) / 2, 2.91, title, fontsize=6.8, fontweight="bold", color=ec, ha="center", va="center")
        ax.text((x0 + x1) / 2, 2.74, sub, fontsize=5.5, color="#334155", ha="center", va="center")

    def draw_tracks(px0, px1, y_top, y_bot, tracks, time_end=1.0, ref_end_line=False):
        lx = px0 + 0.49
        rx = px1 - 0.08
        X = lambda t: lx + t * (rx - lx)
        total_w = sum(w for _, w, _ in tracks)
        gap = 0.036
        avail_h = (y_top - y_bot) - gap * (len(tracks) - 1)
        y = y_top
        boxes = {}
        for label, weight, segs in tracks:
            h = avail_h * (weight / total_w)
            y0 = y - h
            ax.text(lx - 0.04, y0 + h / 2, label, fontsize=5.35, fontweight="bold", color="#1E293B", ha="right", va="center")
            for seg in segs:
                t0, t1, kind, n_sub, txt = seg
                fc = COL[kind]
                dt = (t1 - t0) / max(1, n_sub)
                for k in range(max(1, n_sub)):
                    s0, s1 = t0 + k * dt, t0 + (k + 1) * dt
                    ax.add_patch(Rectangle((X(s0), y0), X(s1) - X(s0), h, fc=fc, ec="white", lw=0.35, zorder=3))
                if txt:
                    tc = "#111111" if kind in ("epi", "scatter", "idle") else "white"
                    ax.text((X(t0) + X(t1)) / 2, y0 + h / 2, txt, fontsize=5.0, fontweight="bold",
                            color=tc, ha="center", va="center", zorder=6)
            boxes[label] = (y0, y0 + h)
            y = y0 - gap
        arrow(ax, lx, y_bot - 0.07, X(time_end), y_bot - 0.07, color="#475569", lw=0.75)
        if ref_end_line:
            ax.plot([rx, rx], [y_bot, y_top], ls=(0, (2.2, 1.4)), lw=0.65, color="#64748B", zorder=5)
        return X, boxes

    # --- Panel (a): Static Megakernel ---
    alt_es = []
    for k in range(4):
        a = 0.22 + k * 0.12
        alt_es.append((a, a + 0.06, "epi", 1, "epi" if k == 0 else ""))
        alt_es.append((a + 0.06, a + 0.12, "scatter", 1, "scat" if k == 2 else ""))
    Xa, ba = draw_tracks(0.08, 2.32, 2.58, 0.64, [
        ("Epi\n×4",  1.15, [(0.00, 0.22, "idle", 1, ""), *alt_es, (0.70, 0.87, "idle", 1, ""), (0.87, 1.00, "combine", 1, "comb")]),
        ("MMA\n×1",  0.85, [(0.00, 0.22, "idle", 1, ""), (0.22, 0.40, "mma", 2, "GEMM1"), (0.40, 0.55, "idle", 1, ""), (0.55, 0.70, "mma", 2, "GEMM2"), (0.70, 1.00, "idle", 1, "")]),
        ("Load\n×2", 0.85, [(0.00, 0.22, "idle", 1, ""), (0.22, 0.39, "load", 2, "load"), (0.39, 0.55, "idle", 1, ""), (0.55, 0.69, "load", 2, "load"), (0.69, 1.00, "idle", 1, "")]),
        ("Comm\n×4", 1.05, [(0.00, 0.34, "pull", 2, "pull"), (0.34, 1.00, "idle", 1, "idle comm bubble")]),
    ], time_end=1.0)
    ax.add_patch(Rectangle((Xa(0.005), ba["Load\n×2"][0] + 0.01), Xa(0.215) - Xa(0.005),
                           ba["Epi\n×4"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.11), (ba["Load\n×2"][0] + ba["Epi\n×4"][1]) / 2, "head\nbubble",
            fontsize=5.0, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=9)
    ax.add_patch(Rectangle((Xa(0.405), ba["Load\n×2"][0] + 0.01), Xa(0.545) - Xa(0.405),
                           ba["MMA\n×1"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.475), (ba["Load\n×2"][0] + ba["MMA\n×1"][1]) / 2, "stage\nsync",
            fontsize=4.7, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=10)
    ax.add_patch(Rectangle((Xa(0.705), ba["Load\n×2"][0] + 0.01), Xa(0.865) - Xa(0.705),
                           ba["Epi\n×4"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.785), (ba["Load\n×2"][0] + ba["Epi\n×4"][1]) / 2, "tail\nsync",
            fontsize=5.0, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=9)
    ax.text((0.08 + 2.32) / 2, 0.45, "Fixed warp split across all SMs stalls compute", fontsize=5.3, color="#991B1B", ha="center", va="center")

    # --- Panel (b): Cross-GPU & Cross-SM/CU Spatial Mixing ---
    # GPU 0 box: Comm-skewed (larger k_0 Comm-rich SMs, smaller Compute-rich SMs)
    rbox(ax, 2.52, 1.66, 4.74, 2.60, ec="#0284C7", fc="white", lw=0.75, r=0.035, z=2)
    ax.text(3.63, 2.48, "GPU 0 (Comm-Heavy): Threshold k_0 = 60% SMs", fontsize=5.45, fontweight="bold", color="#075985", ha="center", va="center")
    # Draw 10 SM tiles on GPU 0 (6 Comm-Rich + 4 Comp-Rich), each having BOTH Tensor Core green AND Comm blue/orange!
    sm_x0 = 2.60
    sm_w = 0.195
    for i in range(10):
        sx = sm_x0 + i * (sm_w + 0.018)
        is_comm_rich = i < 6
        comm_h = 0.30 if is_comm_rich else 0.13
        comp_h = 0.46 - comm_h
        # Bottom: Comm warps; Top: Tensor/Matrix Core + Epi warps (active on EVERY SM!)
        ax.add_patch(Rectangle((sx, 1.88), sm_w, comm_h, fc=COL["pull"] if i < 4 else COL["scatter"], ec="white", lw=0.4, zorder=4))
        ax.add_patch(Rectangle((sx, 1.88 + comm_h), sm_w, comp_h, fc=COL["mma"], ec="white", lw=0.4, zorder=4))
    ax.text(3.22, 1.76, "SM 0..k_0-1: (8c,4e)/(10c,8m)", fontsize=5.0, fontweight="bold", color="#075985", ha="center", va="center")
    ax.text(4.32, 1.76, "SM k_0..N-1: (4c,8e)", fontsize=5.0, fontweight="bold", color="#065F46", ha="center", va="center")

    # GPU 1 box: Compute-heavy (smaller k_1 Comm-rich SMs, larger Compute-rich SMs + D != K)
    rbox(ax, 2.52, 0.62, 4.74, 1.56, ec="#0F766E", fc="white", lw=0.75, r=0.035, z=2)
    ax.text(3.63, 1.44, "GPU 1 (Compute-Heavy): Threshold k_1 = 20% SMs", fontsize=5.45, fontweight="bold", color="#065F46", ha="center", va="center")
    for i in range(10):
        sx = sm_x0 + i * (sm_w + 0.018)
        is_comm_rich = i < 2
        comm_h = 0.28 if is_comm_rich else 0.11
        comp_h = 0.44 - comm_h
        ax.add_patch(Rectangle((sx, 0.86), sm_w, comm_h, fc=COL["pull"] if i < 2 else COL["scatter"], ec="white", lw=0.4, zorder=4))
        ax.add_patch(Rectangle((sx, 0.86 + comm_h), sm_w, comp_h, fc=COL["mma"], ec="white", lw=0.4, zorder=4))
    ax.text(3.63, 0.73, "Active Dispatch SMs (D) != Scatter SMs (K) · All SMs run MMA", fontsize=5.05, fontweight="bold", color="#1E293B", ha="center", va="center")
    ax.text((2.44 + 4.82) / 2, 0.45, "Continuous GPU-wide N_c tuning · 100% SM MMA activity", fontsize=5.3, color="#075985", ha="center", va="center")

    # --- Panel (c): NVIDIA SM103 (16w) & H20 (20w) Dynamic Temporal Roles ---
    draw_tracks(4.94, 7.22, 2.58, 0.64, [
        ("Epi/WG\n8–16w", 1.35, [(0.00, 0.11, "pull", 1, "V_h"), (0.11, 0.63, "epi", 5, "multi-stage epi / reg"), (0.63, 0.78, "combine", 1, "comb")]),
        ("MMA\n2–8w",     0.90, [(0.00, 0.07, "idle", 1, ""), (0.07, 0.61, "mma", 5, "UMMA / WGMMA"), (0.61, 0.78, "combine", 1, "")]),
        ("TMA\n2w",       0.80, [(0.00, 0.04, "idle", 1, ""), (0.04, 0.59, "load", 5, "multi-stage TMA"), (0.59, 0.78, "combine", 1, "")]),
        ("Comm\n2–10w",   1.05, [(0.00, 0.30, "pull", 3, "reg pull"), (0.30, 0.65, "scatter", 4, "smem scatter"), (0.65, 0.78, "combine", 1, "")]),
    ], time_end=0.78, ref_end_line=True)
    ax.text((4.94 + 7.22) / 2, 0.45, "Head V_head pull -> Middle smem/reg -> k/k Combine", fontsize=5.2, color="#065F46", ha="center", va="center")

    # --- Panel (d): AMD MI308X CDNA3 (8 Wave64/CU: (0,8) vs (4,4) CUs) ---
    rbox(ax, 7.42, 1.88, 9.44, 2.58, ec="#EA580C", fc="white", lw=0.65, r=0.03, z=2)
    ax.text(8.43, 2.46, "Compute-Rich CU: Split = (0, 8)", fontsize=5.45, fontweight="bold", color="#9A3412", ha="center", va="center")
    lx_d, rx_d = 7.88, 9.36
    Xd = lambda t: lx_d + t * (rx_d - lx_d)
    ax.text(lx_d - 0.04, 2.14, "W0-7\n×8", fontsize=5.25, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.14, "pull", "pull"),
        (0.14, 0.62, "mma", "8-Wave MFMA"),
        (0.62, 0.75, "scatter", "scat"),
        (0.75, 0.88, "combine", "comb"),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 1.98), Xd(s1) - Xd(s0), 0.31, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        tc = "#111111" if kind == "scatter" else "white"
        ax.text((Xd(s0) + Xd(s1)) / 2, 2.135, lab, fontsize=4.9, fontweight="bold", color=tc, ha="center", va="center", zorder=5)

    rbox(ax, 7.42, 0.64, 9.44, 1.78, ec="#EA580C", fc="white", lw=0.65, r=0.03, z=2)
    ax.text(8.43, 1.66, "Comm-Rich Hybrid CU: Split = (4, 4)", fontsize=5.45, fontweight="bold", color="#9A3412", ha="center", va="center")
    ax.text(lx_d - 0.04, 1.35, "W0-3\nComp", fontsize=5.15, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.10, "load", ""),
        (0.10, 0.73, "mma", "GEMM1/2 MFMA"),
        (0.73, 0.88, "combine", "comb"),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 1.20), Xd(s1) - Xd(s0), 0.30, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        if lab:
            ax.text((Xd(s0) + Xd(s1)) / 2, 1.35, lab, fontsize=4.9, fontweight="bold", color="white", ha="center", va="center", zorder=5)
    ax.text(lx_d - 0.04, 0.90, "W4-7\nComm", fontsize=5.15, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.34, "pull", "8KB pull"),
        (0.34, 0.76, "scatter", "xGMI+RDMA"),
        (0.76, 0.88, "combine", ""),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 0.75), Xd(s1) - Xd(s0), 0.30, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        if lab:
            tc = "#111111" if kind == "scatter" else "white"
            ax.text((Xd(s0) + Xd(s1)) / 2, 0.90, lab, fontsize=4.9, fontweight="bold", color=tc, ha="center", va="center", zorder=5)

    ax.text((7.34 + 9.52) / 2, 0.45, "Co-resident (0,8) & (4,4) CUs overlap xGMI + GDA RDMA", fontsize=5.2, color="#7C2D12", ha="center", va="center")

    leg_handles = [
        Patch(facecolor=COL["pull"], edgecolor="none", label="Remote Pull / Dedup"),
        Patch(facecolor=COL["load"], edgecolor="none", label="Async Weight/Act Load"),
        Patch(facecolor=COL["mma"], edgecolor="none", label="Tensor / Matrix Core GEMM"),
        Patch(facecolor=COL["epi"], edgecolor="none", label="Multi-Stage Epilogue"),
        Patch(facecolor=COL["scatter"], edgecolor="none", label="Cross-GPU Scatter (reg / smem / hbm)"),
        Patch(facecolor=COL["combine"], edgecolor="none", label="Per-Token Ready Combine"),
        Patch(facecolor=COL["idle"], edgecolor="#94A3B8", linewidth=0.5, label="Sync / Idle Bubble"),
    ]
    fig.legend(
        handles=leg_handles, loc="lower center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, 0.018), fontsize=5.5,
        handlelength=1.15, handleheight=0.85, columnspacing=1.0, handletextpad=0.36, borderaxespad=0.0,
    )

    verify_canvas(fig, "weave_design_rolegraph")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "weave_design_rolegraph.png"
    out_pdf = FIG_DIR / "weave_design_rolegraph.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    raw = out_pdf.read_bytes()
    assert b"/Type3" not in raw and (b"/CIDFontType2" in raw or b"/TrueType" in raw)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    render_overview()
    render_rolegraph()


if __name__ == "__main__":
    main()
