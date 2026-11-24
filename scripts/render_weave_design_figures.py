#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the two repository architecture schematics:
  1. docs/figures/weave_design_overview.png
     Two-stage Compile-Time Autotune + Runtime Zero-Env Selection over the
     unified 14-knob (Spatial 5 / Temporal 3 / Implementation 6) surface and
     barrier-free fine-grained atomic megakernel across SM100/SM103, SM90, CDNA3.
  2. docs/figures/weave_design_rolegraph.png
     Cross-hardware intra-SM/CU spatial-temporal warp/wave role schedules
     contrasting (a) Coarse-Barrier Megakernel against Weave on:
       (b) NVIDIA SM100/SM103 (16 warps/CTA: TMEM + UMMA + SMEM->NVLink Scatter),
       (c) NVIDIA H20 SM90 (20 warps/CTA: Lag-W K-block bitmap, zero grid_sync),
       (d) AMD MI308X CDNA3 (8 Wave64/CU, 2-CTA/CU, 0-8 vs 4-4 xGMI+GDA RDMA).

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

FORBIDDEN = [
    "B300", "L20D", "dynamicMK", "v33", "v28", "v70", "v62", "v58",
    "v1864", "D7", "AGY", "Residency", "§",
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
# 1. Weave System Design Overview (Two-Stage + 14 Knobs + Barrier-Free Pipeline)
# ==============================================================================
def render_overview() -> None:
    W, H = 9.0, 3.25
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Top banner strip
    ax.text(
        W / 2, 3.11,
        "Weave Architecture: Compile-Time Calibration + Zero-Env Runtime Scheduling over 14 Public Knobs",
        fontsize=8.6, fontweight="bold", color="#0F172A", ha="center", va="center",
    )

    # ---------------- Column 1: Stage 1 Offline / Compile-Time Calibration ----------------
    c1_x0, c1_x1 = 0.10, 2.52
    rbox(ax, c1_x0, 0.12, c1_x1, 2.94, ec="#B4691E", fc="#FFF9F2", lw=1.0, r=0.06, z=1)
    badge(ax, c1_x0 + 0.15, 2.77, "1", fc="#C96A1B")
    ax.text(c1_x0 + 0.29, 2.81, "Stage 1: Compile-Time Autotune", fontsize=7.4, fontweight="bold", color="#8C4308", va="center")
    ax.text(c1_x0 + 0.29, 2.67, "deep_gemm.mega.autotune (once per SKU)", fontsize=6.0, color="#6B4F35", va="center")

    # Subbox 1A: Dynamic Hardware Fingerprint
    rbox(ax, 0.20, 1.88, 2.42, 2.52, ec="#D99B52", fc="white", lw=0.75, r=0.04, z=2)
    ax.text(1.31, 2.39, "Dynamic Hardware Fingerprint (No SKU Hardcode)", fontsize=6.3, fontweight="bold", color="#1E293B", ha="center", va="center")
    ax.text(1.31, 2.21, "• NVIDIA SM100 / SM103 (TMEM, UMMA, NVLink)", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(1.31, 2.07, "• NVIDIA SM90 H20 (WGMMA, TMA, NVLink)", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(1.31, 1.93, "• AMD CDNA3 gfx942 (8-Wave CU, MFMA, xGMI+RDMA)", fontsize=5.8, color="#334155", ha="center", va="center")

    arrow(ax, 1.31, 1.87, 1.31, 1.73, color="#8C4308")

    # Subbox 1B: 0-Spill Gate + Bucket ABBA
    rbox(ax, 0.20, 0.92, 2.42, 1.74, ec="#D99B52", fc="white", lw=0.75, r=0.04, z=2)
    ax.text(1.31, 1.61, "Two-Pass 0-Spill Gate & Bucket ABBA", fontsize=6.3, fontweight="bold", color="#1E293B", ha="center", va="center")
    ax.text(1.31, 1.44, "1. Prune GEMM1/2 tuples by M_exp & 0-spill check", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(1.31, 1.29, "2. Palindromic ABBA across S in {16 .. 16384}", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(1.31, 1.14, "3. Calibrate 4 online routing break-even thresholds", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(1.31, 0.995, "4. Guard: cos > 0.99 & >= 0.5% vs. default", fontsize=5.6, color="#64748B", ha="center", va="center")

    arrow(ax, 1.31, 0.91, 1.31, 0.79, color="#8C4308")

    # Subbox 1C: JSON Profile Artifact
    rbox(ax, 0.20, 0.20, 2.42, 0.78, ec="#8C4308", fc="#FEF0E1", lw=0.85, r=0.04, z=2)
    ax.text(1.31, 0.62, "Calibrated Hardware Profile JSON", fontsize=6.4, fontweight="bold", color="#7C3A06", ha="center", va="center")
    ax.text(1.31, 0.45, "{hw_id}__E_K_H_I_{mma}.json", fontsize=5.9, color="#334155", ha="center", va="center")
    ax.text(1.31, 0.29, "Per-S base_env + verified_variants + 4 thresholds", fontsize=5.6, color="#57534E", ha="center", va="center")

    # ---------------- Column 2: Stage 2 Runtime Selection + 14 Public Knobs ----------------
    c2_x0, c2_x1 = 2.74, 5.98
    rbox(ax, c2_x0, 0.12, c2_x1, 2.94, ec="#0072B2", fc="#F4F9FD", lw=1.0, r=0.06, z=1)
    badge(ax, c2_x0 + 0.15, 2.77, "2", fc="#0072B2")
    ax.text(c2_x0 + 0.29, 2.81, "Stage 2: Zero-Env Runtime Selection & 14 Knobs", fontsize=7.3, fontweight="bold", color="#075985", va="center")
    ax.text(c2_x0 + 0.29, 2.67, "weave_mega_moe(...) — 0 host overhead in CUDA/HIP Graph", fontsize=5.9, color="#334155", va="center")

    # Selector bar
    rbox(ax, 2.84, 2.02, 5.88, 2.54, ec="#0284C7", fc="white", lw=0.85, r=0.04, z=2)
    ax.text(4.36, 2.40, "Live Post-Dispatch Routing Stats + Profile Lookup", fontsize=6.4, fontweight="bold", color="#0F172A", ha="center", va="center")
    ax.text(4.36, 2.24, "Nearest log2(S) bucket + 4 rules (dedup, scatter, spatial, ready_group)", fontsize=5.8, color="#334155", ha="center", va="center")
    ax.text(4.36, 2.09, "On-device CTA-0 cost model adapts V_HEAD / wave split on live tokens[e]", fontsize=5.7, color="#475569", ha="center", va="center")

    # Arrow from Stage 1 JSON -> Stage 2 Selector
    ax.plot([2.42, 2.62, 2.62, 2.83], [0.51, 0.51, 2.28, 2.28], color="#8C4308", lw=0.95, zorder=7)
    arrow(ax, 2.76, 2.28, 2.84, 2.28, color="#8C4308")
    arrow(ax, 4.36, 2.01, 4.36, 1.88, color="#0072B2")

    # 3 Category Cards for the 14 Public Knobs
    cards = [
        (1.34, 1.87, "#0072B2", "#EBF5FB", "Spatial (5 Knobs) — SM/CU & Warp/Wave Ratios",
         "• DISPATCH_ACTIVE_SMS (D)  • SCATTER_ACTIVE_SMS (K, K != D)",
         "• DISPATCH_WARPS  • SCATTER_WARPS  • CU_WAVE_SPLIT (0-8 / 4-4 / 7-W)"),
        (0.78, 1.29, "#009E73", "#EAFAF4", "Temporal (3 Knobs) — Head / Middle / Tail Overlap",
         "• V_HEAD (cooperative head pull quota)  • READY_GROUP_SIZE (N-tile start)",
         "• SCATTER_OWNER (strict binary: epilogue_only vs. comm_only)"),
        (0.20, 0.73, "#D55E00", "#FEF3EC", "Implementation (6 Knobs) — Independent 4-Op Configs",
         "• PRECISION_MODE  • GEMM1_CONFIG (coupled tuple)  • GEMM2_CONFIG",
         "• REG_PULL_BYTES (1K-4K / 8K)  • DEDUP_MODE  • SCATTER_STAGING"),
    ]
    for y0, y1, ec, fc, title, l1, l2 in cards:
        rbox(ax, 2.84, y0, 5.88, y1, ec=ec, fc=fc, lw=0.8, r=0.038, z=2)
        ax.text(2.92, y1 - 0.125, title, fontsize=6.2, fontweight="bold", color=ec, va="center")
        ax.text(2.92, y1 - 0.280, l1, fontsize=5.65, color="#1E293B", va="center")
        ax.text(2.92, y0 + 0.100, l2, fontsize=5.65, color="#1E293B", va="center")

    # ---------------- Column 3: Barrier-Free Fused Megakernel Execution ----------------
    c3_x0, c3_x1 = 6.20, 8.90
    rbox(ax, c3_x0, 0.12, c3_x1, 2.94, ec="#1E293B", fc="#F8FAFC", lw=1.0, r=0.06, z=1)
    badge(ax, c3_x0 + 0.15, 2.77, "3", fc="#0F766E")
    ax.text(c3_x0 + 0.29, 2.81, "Fused MoE Megakernel", fontsize=7.4, fontweight="bold", color="#0F172A", va="center")
    ax.text(c3_x0 + 0.29, 2.67, "Zero inter-op grid_sync across GPUs", fontsize=6.0, color="#0F766E", fontweight="bold", va="center")

    for yc in (1.60, 1.03, 0.46):
        arrow(ax, 5.89, yc, 6.20, yc, color="#334155")

    # Mini cross-engine overlap timeline inside Column 3
    rbox(ax, 6.30, 1.40, 8.80, 2.54, ec="#CBD5E1", fc="white", lw=0.75, r=0.04, z=2)
    ax.text(7.55, 2.42, "Concurrent Intra-SM / CU Pipeline", fontsize=6.3, fontweight="bold", color="#0F172A", ha="center", va="center")

    tx0, tx1 = 6.82, 8.72
    X = lambda t: tx0 + t * (tx1 - tx0)
    tracks = [
        ("Epi",   2.15, [(0.00, 0.14, "#0072B2", ""), (0.14, 0.80, "#56B4E9", "GEMM1/2 epi"), (0.80, 1.00, "#475569", "comb")]),
        ("Math",  1.94, [(0.00, 0.11, "#0072B2", ""), (0.11, 0.77, "#009E73", "UMMA/WGMMA/MFMA"), (0.77, 1.00, "#475569", "")]),
        ("Load",  1.73, [(0.00, 0.09, "#0072B2", ""), (0.09, 0.74, "#CC79A7", "TMA / Coalesced HBM"), (0.74, 1.00, "#475569", "")]),
        ("Comm",  1.52, [(0.00, 0.38, "#0072B2", "pull"), (0.38, 0.83, "#E69F00", "async scatter"), (0.83, 1.00, "#475569", "")]),
    ]
    rh = 0.165
    for name, yc, segs in tracks:
        ax.text(tx0 - 0.04, yc, name, fontsize=5.8, fontweight="bold", color="#334155", ha="right", va="center")
        for s0, s1, col, lab in segs:
            ax.add_patch(Rectangle((X(s0), yc - rh / 2), X(s1) - X(s0), rh, fc=col, ec="white", lw=0.4, zorder=4))
            if lab:
                tc = "#111111" if col in ("#56B4E9", "#E69F00") else "white"
                ax.text((X(s0) + X(s1)) / 2, yc, lab, fontsize=5.2, fontweight="bold", color=tc, ha="center", va="center", zorder=6)

    # Atomic handoff summary box
    rbox(ax, 6.30, 0.22, 8.80, 1.30, ec="#0F766E", fc="#ECFDF5", lw=0.8, r=0.04, z=2)
    ax.text(7.55, 1.16, "Fine-Grained Atomic Handoffs", fontsize=6.3, fontweight="bold", color="#065F46", ha="center", va="center")
    ax.text(7.55, 0.97, "• Notify -> Pull: per-expert atomic count", fontsize=5.7, color="#1E293B", ha="center", va="center")
    ax.text(7.55, 0.80, "• GEMM1 <-> GEMM2: K-block / ring ready", fontsize=5.7, color="#1E293B", ha="center", va="center")
    ax.text(7.55, 0.63, "• FC2 Scatter: SMEM ring / owner-node RDMA", fontsize=5.7, color="#1E293B", ha="center", va="center")
    ax.text(7.55, 0.46, "• Tail Combine: per-token ld.acquire.sys", fontsize=5.7, color="#1E293B", ha="center", va="center")
    ax.text(7.55, 0.29, "1.30x SM103 · 1.25x H20 · 1.88x MI308X", fontsize=5.8, fontweight="bold", color="#0F766E", ha="center", va="center")

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
# 2. Cross-Hardware Spatial-Temporal Warp/Wave Role Graph (4-Panel)
# ==============================================================================
def render_rolegraph() -> None:
    W, H = 9.4, 3.15
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    COL = {
        "pull":     "#0072B2",
        "scatter":  "#E69F00",
        "load":     "#CC79A7",
        "mma":      "#009E73",
        "epi":      "#56B4E9",
        "combine":  "#475569",
        "idle":     "#E2E8F0",
    }

    ax.text(
        W / 2, 3.01,
        "Spatial Warp/Wave Partitioning & Barrier-Free Temporal Schedules Across NVIDIA SM103, NVIDIA H20, and AMD MI308X",
        fontsize=8.3, fontweight="bold", color="#0F172A", ha="center", va="center",
    )

    panels = [
        (0.08, 2.26, "(a) Coarse-Barrier Megakernel", "Global grid_sync + blocking scatter/combine", "#991B1B", "#FEF2F2"),
        (2.38, 4.64, "(b) NVIDIA SM100 / SM103", "16 Warps/CTA · TMEM + UMMA + SMEM Scatter", "#075985", "#F0F9FF"),
        (4.76, 7.02, "(c) NVIDIA H20 (SM90)", "20 Warps/CTA · Lag-W K-Bitmap (0 grid_sync)", "#065F46", "#ECFDF5"),
        (7.14, 9.32, "(d) AMD MI308X (CDNA3)", "8 Wave64/CU · 2-CTA/CU · 0-8 vs 4-4 Split", "#7C2D12", "#FFF7ED"),
    ]
    for x0, x1, title, sub, ec, fc in panels:
        rbox(ax, x0, 0.36, x1, 2.86, ec=ec, fc=fc, lw=0.9, r=0.05, z=1)
        ax.text((x0 + x1) / 2, 2.74, title, fontsize=6.9, fontweight="bold", color=ec, ha="center", va="center")
        ax.text((x0 + x1) / 2, 2.58, sub, fontsize=5.55, color="#334155", ha="center", va="center")

    def draw_tracks(px0, px1, y_top, y_bot, tracks, time_end=1.0, ref_end_line=False):
        lx = px0 + 0.48
        rx = px1 - 0.08
        X = lambda t: lx + t * (rx - lx)
        total_w = sum(w for _, w, _ in tracks)
        gap = 0.035
        avail_h = (y_top - y_bot) - gap * (len(tracks) - 1)
        y = y_top
        boxes = {}
        for label, weight, segs in tracks:
            h = avail_h * (weight / total_w)
            y0 = y - h
            ax.text(lx - 0.04, y0 + h / 2, label, fontsize=5.5, fontweight="bold", color="#1E293B", ha="right", va="center")
            for seg in segs:
                t0, t1, kind, n_sub, txt = seg
                fc = COL[kind]
                dt = (t1 - t0) / max(1, n_sub)
                for k in range(max(1, n_sub)):
                    s0, s1 = t0 + k * dt, t0 + (k + 1) * dt
                    ax.add_patch(Rectangle((X(s0), y0), X(s1) - X(s0), h, fc=fc, ec="white", lw=0.35, zorder=3))
                if txt:
                    tc = "#111111" if kind in ("epi", "scatter", "idle") else "white"
                    ax.text((X(t0) + X(t1)) / 2, y0 + h / 2, txt, fontsize=5.1, fontweight="bold",
                            color=tc, ha="center", va="center", zorder=6)
            boxes[label] = (y0, y0 + h)
            y = y0 - gap
        arrow(ax, lx, y_bot - 0.07, X(time_end), y_bot - 0.07, color="#475569", lw=0.75)
        if ref_end_line:
            ax.plot([rx, rx], [y_bot, y_top], ls=(0, (2.2, 1.4)), lw=0.65, color="#64748B", zorder=5)
        return X, boxes

    # --- Panel (a): Baseline Coarse-Barrier Megakernel ---
    alt_es = []
    for k in range(4):
        a = 0.22 + k * 0.12
        alt_es.append((a, a + 0.06, "epi", 1, "epi" if k == 0 else ""))
        alt_es.append((a + 0.06, a + 0.12, "scatter", 1, "scat" if k == 2 else ""))
    Xa, ba = draw_tracks(0.08, 2.26, 2.44, 0.62, [
        ("Epi\n×4",  1.15, [(0.00, 0.22, "idle", 1, ""), *alt_es, (0.70, 0.87, "idle", 1, ""), (0.87, 1.00, "combine", 1, "comb")]),
        ("MMA\n×1",  0.85, [(0.00, 0.22, "idle", 1, ""), (0.22, 0.42, "mma", 2, "GEMM1"), (0.42, 0.54, "idle", 1, ""), (0.54, 0.70, "mma", 2, "GEMM2"), (0.70, 1.00, "idle", 1, "")]),
        ("Load\n×2", 0.85, [(0.00, 0.22, "idle", 1, ""), (0.22, 0.41, "load", 2, "load"), (0.41, 0.54, "idle", 1, ""), (0.54, 0.68, "load", 2, "load"), (0.68, 1.00, "idle", 1, "")]),
        ("Comm\n×4", 1.05, [(0.00, 0.34, "pull", 2, "pull"), (0.34, 1.00, "idle", 1, "idle comm bubble")]),
    ], time_end=1.0)
    # Highlight Head bubble, grid_sync gap, and Tail barrier bubble
    ax.add_patch(Rectangle((Xa(0.005), ba["Load\n×2"][0] + 0.01), Xa(0.215) - Xa(0.005),
                           ba["Epi\n×4"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.11), (ba["Load\n×2"][0] + ba["Epi\n×4"][1]) / 2, "head\nbubble",
            fontsize=5.1, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=9)
    ax.add_patch(Rectangle((Xa(0.425), ba["Load\n×2"][0] + 0.01), Xa(0.535) - Xa(0.425),
                           ba["MMA\n×1"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.48), (ba["Load\n×2"][0] + ba["MMA\n×1"][1]) / 2, "grid\nsync",
            fontsize=4.8, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=10)
    ax.add_patch(Rectangle((Xa(0.705), ba["Load\n×2"][0] + 0.01), Xa(0.865) - Xa(0.705),
                           ba["Epi\n×4"][1] - ba["Load\n×2"][0] - 0.02,
                           fc="none", ec="#B91C1C", lw=0.75, ls=(0, (2, 1.2)), zorder=8))
    ax.text(Xa(0.785), (ba["Load\n×2"][0] + ba["Epi\n×4"][1]) / 2, "tail\nsync",
            fontsize=5.0, fontweight="bold", color="#991B1B", ha="center", va="center", zorder=9)
    ax.text((0.08 + 2.26) / 2, 0.44, "3 coarse barriers stall Tensor/Matrix Cores", fontsize=5.4, color="#991B1B", ha="center", va="center")

    # --- Panel (b): NVIDIA SM100 / SM103 (16 Warps / CTA) ---
    draw_tracks(2.38, 4.64, 2.44, 0.62, [
        ("W08-15\nEpi ×8",  1.35, [(0.00, 0.10, "pull", 1, "V_h"), (0.10, 0.63, "epi", 5, "TMEM epi + ring"), (0.63, 0.76, "combine", 2, "comb")]),
        ("W06-07\nUMMA ×2", 0.85, [(0.06, 0.60, "mma", 5, "CLC + UMMA"), (0.00, 0.06, "idle", 1, ""), (0.60, 0.76, "combine", 2, "")]),
        ("W04-05\nTMA ×2",  0.85, [(0.03, 0.57, "load", 5, "9-st TMA A/B"), (0.00, 0.03, "idle", 1, ""), (0.57, 0.76, "combine", 2, "")]),
        ("W00-03\nComm ×4", 1.00, [(0.00, 0.31, "pull", 3, "reg pull"), (0.31, 0.66, "scatter", 4, "SMEM scatter"), (0.66, 0.76, "combine", 2, "")]),
    ], time_end=0.76, ref_end_line=True)
    ax.text((2.38 + 4.64) / 2, 0.44, "All 16 warps active · HBM peak 82% · 0 spill", fontsize=5.4, color="#075985", ha="center", va="center")

    # --- Panel (c): NVIDIA H20 SM90 (20 Warps / CTA, Lag-W Zero-grid_sync) ---
    draw_tracks(4.76, 7.02, 2.44, 0.62, [
        ("W12-19\nWG0/1 ×8", 1.40, [(0.00, 0.09, "pull", 1, "V_h"), (0.09, 0.66, "mma", 6, "Lag-W GEMM1<->2"), (0.66, 0.78, "combine", 2, "comb")]),
        ("W08-09\nTMA ×2",   0.85, [(0.00, 0.07, "pull", 1, ""), (0.07, 0.63, "load", 5, "TMA weight/act"), (0.63, 0.78, "combine", 2, "")]),
        ("W00-07\nComm ×8",  1.20, [(0.00, 0.27, "pull", 3, "dedup pull"), (0.27, 0.69, "scatter", 4, "NVLink scatter"), (0.69, 0.78, "combine", 2, "")]),
    ], time_end=0.78, ref_end_line=True)
    ax.text((4.76 + 7.02) / 2, 0.44, "K-bitmap L1->L2 · Tensor Active x4 = 75-100%", fontsize=5.4, color="#065F46", ha="center", va="center")

    # --- Panel (d): AMD MI308X CDNA3 (8 Wave64/CU: 0-8 Decode vs 4-4 Prefill) ---
    # Upper sub-box: Decode S<=128 (0-8)
    rbox(ax, 7.22, 1.84, 9.24, 2.46, ec="#EA580C", fc="white", lw=0.65, r=0.03, z=2)
    ax.text(8.23, 2.35, "Small S <= 128: CU_WAVE_SPLIT = 0-8", fontsize=5.5, fontweight="bold", color="#9A3412", ha="center", va="center")
    lx_d, rx_d = 7.68, 9.16
    Xd = lambda t: lx_d + t * (rx_d - lx_d)
    ax.text(lx_d - 0.04, 2.07, "W0-7\n×8", fontsize=5.3, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.14, "pull", "pull"),
        (0.14, 0.60, "mma", "8-Wave MFMA"),
        (0.60, 0.74, "scatter", "scat"),
        (0.74, 0.88, "combine", "comb"),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 1.92), Xd(s1) - Xd(s0), 0.29, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        tc = "#111111" if kind == "scatter" else "white"
        ax.text((Xd(s0) + Xd(s1)) / 2, 2.065, lab, fontsize=4.95, fontweight="bold", color=tc, ha="center", va="center", zorder=5)

    # Lower sub-box: Prefill / 2-Node EP16 (4-4)
    rbox(ax, 7.22, 0.64, 9.24, 1.75, ec="#EA580C", fc="white", lw=0.65, r=0.03, z=2)
    ax.text(8.23, 1.63, "Prefill / EP16: CU_WAVE_SPLIT = 4-4", fontsize=5.5, fontweight="bold", color="#9A3412", ha="center", va="center")
    # Compute W0-3
    ax.text(lx_d - 0.04, 1.33, "W0-3\nComp", fontsize=5.2, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.10, "load", ""),
        (0.10, 0.73, "mma", "160x256 MFMA"),
        (0.73, 0.88, "combine", "comb"),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 1.18), Xd(s1) - Xd(s0), 0.29, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        if lab:
            ax.text((Xd(s0) + Xd(s1)) / 2, 1.325, lab, fontsize=4.95, fontweight="bold", color="white", ha="center", va="center", zorder=5)
    # Comm W4-7
    ax.text(lx_d - 0.04, 0.89, "W4-7\nComm", fontsize=5.2, fontweight="bold", color="#1E293B", ha="right", va="center")
    for s0, s1, kind, lab in [
        (0.00, 0.34, "pull", "8KB pull"),
        (0.34, 0.76, "scatter", "xGMI+RDMA"),
        (0.76, 0.88, "combine", ""),
    ]:
        ax.add_patch(Rectangle((Xd(s0), 0.74), Xd(s1) - Xd(s0), 0.29, fc=COL[kind], ec="white", lw=0.35, zorder=3))
        if lab:
            tc = "#111111" if kind == "scatter" else "white"
            ax.text((Xd(s0) + Xd(s1)) / 2, 0.885, lab, fontsize=4.95, fontweight="bold", color=tc, ha="center", va="center", zorder=5)

    ax.text((7.14 + 9.32) / 2, 0.44, "2-CTA/CU · 256 VGPR 0-spill · 1.875x EP16 GM", fontsize=5.4, color="#7C2D12", ha="center", va="center")

    # Shared Bottom Legend (centered using fig.legend so widths are exact)
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor=COL["pull"], edgecolor="none", label="Remote Pull / Dedup"),
        Patch(facecolor=COL["load"], edgecolor="none", label="Async Weight/Act Load"),
        Patch(facecolor=COL["mma"], edgecolor="none", label="Tensor / Matrix Core GEMM"),
        Patch(facecolor=COL["epi"], edgecolor="none", label="Epilogue / Ring Stage"),
        Patch(facecolor=COL["scatter"], edgecolor="none", label="Async NVLink / xGMI / RDMA Scatter"),
        Patch(facecolor=COL["combine"], edgecolor="none", label="Ready-Gated Combine"),
        Patch(facecolor=COL["idle"], edgecolor="#94A3B8", linewidth=0.5, label="Barrier / Idle Bubble"),
    ]
    fig.legend(
        handles=leg_handles, loc="lower center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, 0.018), fontsize=5.55,
        handlelength=1.15, handleheight=0.85, columnspacing=1.05, handletextpad=0.38, borderaxespad=0.0,
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
