#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper Fig. 7: 3-hardware end-to-end MoE-layer latency & speedup across 6 models x 5 S = 90 cells.

Self-contained renderer driven directly from the committed ledger
``results/bound_utilization_90cells_3hw.json`` (no external data paths required):
  - Row 1: 8x SM103 (148-SM), EP=8 (7 systems)
  - Row 2: 8x H20 (SM90), EP=8 (3 systems)
  - Row 3: 16x MI308X (CDNA3), 2-node EP=16 (Weave V70 vs AITER-1stage & vLLM)

Every cell's speedup strip equals ``fastest_baseline_us / weave_us``, matching
``weave_bound_utilization_3hw.png`` to machine precision across all 90 cells.
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "bound_utilization_90cells_3hw.json"
OUT_PNG = ROOT / "docs" / "figures" / "weave_eval_3hw_speedup.png"
OUT_PDF = ROOT / "docs" / "figures" / "weave_eval_3hw_speedup.pdf"

MODELS = [
    "Qwen3.8-2.4T",
    "MiniMax-M3",
    "Qwen3.5-397B",
    "Qwen3.5-122B",
    "DSKv4-Flash",
    "Qwen3.5-35B",
]
S_VALUES = [16, 128, 1024, 8192, 16384]
S_LABELS = ["16", "128", "1K", "8K", "16K"]
HW_ORDER = ["SM103", "H20", "MI308X"]
HW_ROW_LABEL = {
    "SM103": "8× SM103\n(148-SM)",
    "H20": "8× H20\n(SM90)",
    "MI308X": "16× MI308X\n(CDNA3)",
}
DRAW_ORDER = {
    "SM103": ["NanoFlow", "FlashMoE", "mKernel", "SGLang", "FlashInfer", "DeepGEMM-MegaMoE", "Weave"],
    "H20": ["FlashInfer", "DeepGEMM-MegaMoE", "Weave"],
    "MI308X": ["vLLM", "AITER-1stage", "Weave"],
}
LEGEND_SYSTEMS = [
    "Weave", "DeepGEMM-MegaMoE", "FlashInfer", "SGLang",
    "AITER-1stage", "vLLM", "mKernel", "FlashMoE", "NanoFlow",
]
STYLE = {
    "Weave":            ("#0072B2", "o", 2.30, 1.35, 10, "#0072B2"),
    "DeepGEMM-MegaMoE": ("#D55E00", "s", 2.15, 1.00,  8, "white"),
    "FlashInfer":       ("#009E73", "^", 2.25, 0.95,  7, "#009E73"),
    "SGLang":           ("#56B4E9", "P", 2.40, 0.95,  6, "#56B4E9"),
    "AITER-1stage":     ("#C0392B", "d", 2.20, 1.00,  7, "white"),
    "vLLM":             ("#8E5EA2", "h", 2.25, 0.95,  5, "white"),
    "mKernel":          ("#E69F00", "v", 2.15, 0.85,  4, "#E69F00"),
    "FlashMoE":         ("#CC79A7", "D", 1.90, 0.85,  3, "#CC79A7"),
    "NanoFlow":         ("#666666", "X", 2.15, 0.85,  2, "#666666"),
}
FORBIDDEN = [
    "B300", "L20D", "dynamicMK", "v33", "v28", "v70", "v62", "v58",
    "v1864", "D7", "AGY", "ours_tuned", "all16", "Residency",
]


def gm(vals: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def cell_color(sp: float) -> tuple[str, str]:
    if sp >= 1.75:
        return "#1E6B9E", "white"
    if sp >= 1.50:
        return "#6BAED6", "#111111"
    if sp >= 1.35:
        return "#9ECAE1", "#111111"
    if sp >= 1.20:
        return "#C6DBEF", "#111111"
    if sp >= 1.10:
        return "#DEEBF7", "#111111"
    return "#EDF4FA", "#111111"


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    cells = {(c["hardware"], c["model"], int(c["S"])): c for c in ledger["cells"]}
    assert len(cells) == 90

    assert "Arial" in {f.name for f in font_manager.fontManager.ttflist}, "Arial font required"
    plt.rcParams.update({
        "font.family": ["Arial"],
        "font.size": 6.4,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "text.usetex": False,
    })

    fig = plt.figure(figsize=(8.6, 3.60))
    left_x, right_x = 0.104, 0.996
    row_cfg = {
        "SM103":  ((0.875, 0.642), (22.0, 10500.0),  [32, 128, 512, 2048, 8192], ["32", "128", "512", "2K", "8K"]),
        "H20":    ((0.600, 0.367), (36.0, 24000.0),  [64, 256, 1024, 4096, 16384], ["64", "256", "1K", "4K", "16K"]),
        "MI308X": ((0.325, 0.092), (180.0, 36000.0), [256, 1024, 4096, 16384], ["256", "1K", "4K", "16K"]),
    }
    x = list(range(len(S_VALUES)))
    titles, strip_texts = [], []

    for hw in HW_ORDER:
        (top, bot), y_lim, y_ticks, y_labels = row_cfg[hw]
        gs = fig.add_gridspec(
            2, 6, height_ratios=[1.48, 0.44], hspace=0.10, wspace=0.075,
            left=left_x, right=right_x, top=top, bottom=bot,
        )
        sp_all = []
        for ci, m in enumerate(MODELS):
            axt = fig.add_subplot(gs[0, ci])
            axb = fig.add_subplot(gs[1, ci], sharex=axt)
            for sname in DRAW_ORDER[hw]:
                if any(sname not in cells[(hw, m, s)]["latency_us"] for s in S_VALUES):
                    continue
                col, mk, ms, lw, z, mfc = STYLE[sname]
                ys = [float(cells[(hw, m, s)]["latency_us"][sname]) for s in S_VALUES]
                for yv in ys:
                    assert y_lim[0] * 1.25 <= yv <= y_lim[1] / 1.15, f"{hw} {sname} {m} y={yv} out of {y_lim}"
                axt.plot(
                    x, ys, color=col, marker=mk, markersize=ms, linewidth=lw,
                    markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=0.6,
                    zorder=z, clip_on=False,
                )
            axt.set_yscale("log", base=2)
            axt.set_ylim(*y_lim)
            axt.yaxis.set_major_locator(FixedLocator(y_ticks))
            axt.yaxis.set_major_formatter(FixedFormatter(y_labels if ci == 0 else [""] * len(y_ticks)))
            axt.yaxis.set_minor_locator(NullLocator())
            axt.yaxis.set_minor_formatter(NullFormatter())
            axt.yaxis.grid(True, color="#E0E0E0", lw=0.40, zorder=0)
            axt.set_axisbelow(True)
            axt.set_xlim(-0.5, len(S_VALUES) - 0.5)
            axt.spines["left"].set_visible(False)
            axt.spines["bottom"].set_color("#888888")
            axt.tick_params(axis="x", labelbottom=False, length=1.6, width=0.5, color="#666666")
            axt.tick_params(axis="y", length=0.0, pad=1.5)
            if hw == "SM103":
                titles.append(axt.set_title(m, loc="center", pad=3.2, fontsize=7.0, weight="bold", color="#111111"))

            axb.set_xlim(-0.5, len(S_VALUES) - 0.5)
            axb.set_ylim(0.0, 1.0)
            for spn in axb.spines.values():
                spn.set_visible(False)
            axb.yaxis.set_major_locator(NullLocator())
            for xi, s in zip(x, S_VALUES):
                c = cells[(hw, m, s)]
                base = [v for k, v in c["latency_us"].items() if k != "Weave"]
                v = min(base) / float(c["latency_us"]["Weave"])
                assert abs(v - float(c["speedup_vs_fastest"])) < 1e-6
                sp_all.append(v)
                bg, fg = cell_color(v)
                axb.add_patch(Rectangle((xi - 0.46, 0.06), 0.92, 0.88, facecolor=bg, edgecolor="none", zorder=2))
                strip_texts.append(axb.text(
                    xi, 0.50, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color=fg,
                    weight="bold" if v >= 1.75 else "normal", zorder=4,
                ))
            axb.set_xticks(x)
            axb.set_xticklabels(S_LABELS if hw == "MI308X" else [""] * len(S_LABELS))
            axb.tick_params(axis="x", length=0.0, pad=1.8 if hw == "MI308X" else 0.0)

        y_mid = bot + (top - bot) * 0.63
        fig.text(0.004, y_mid + 0.038, HW_ROW_LABEL[hw], ha="left", va="center", fontsize=6.8, weight="bold", color="#111111")
        fig.text(0.004, y_mid - 0.036, "Latency\n(µs)", ha="left", va="center", fontsize=5.9, color="#222222")
        fig.text(0.004, bot + (top - bot) * 0.11, "Speedup (×)", ha="left", va="center", fontsize=5.9, color="#222222")
        print(f"{hw:6s}: GM speedup vs fastest baseline = {gm(sp_all):.4f}x  range=[{min(sp_all):.2f}x, {max(sp_all):.2f}x]")

    fig.text(
        (left_x + right_x) * 0.5, 0.007,
        "Tokens per step (S)    ·    Speedup = Fastest baseline latency / Weave latency",
        ha="center", va="bottom", fontsize=6.2, color="#222222",
    )

    handles = [
        Line2D(
            [0], [0], color=STYLE[s][0], marker=STYLE[s][1], markersize=STYLE[s][2] + 0.5,
            linewidth=STYLE[s][3] + (0.25 if s == "Weave" else 0.15),
            markerfacecolor=STYLE[s][5], markeredgecolor=STYLE[s][0], markeredgewidth=0.65,
        )
        for s in LEGEND_SYSTEMS
    ]
    leg = fig.legend(
        handles, LEGEND_SYSTEMS, loc="upper center", ncol=9, frameon=False,
        bbox_to_anchor=(0.55, 0.996), fontsize=6.3,
        handlelength=1.55, columnspacing=0.95, handletextpad=0.35, borderaxespad=0.0,
    )

    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    vis = []
    for t in fig.findobj(Text):
        s_txt = t.get_text().strip()
        if not t.get_visible() or not s_txt:
            continue
        for f in FORBIDDEN:
            assert f.lower() not in s_txt.lower(), f"forbidden {f!r} in {s_txt!r}"
        bb = t.get_window_extent(rnd)
        assert -0.5 <= bb.x0 and bb.x1 <= fig.bbox.width + 0.5 and -0.5 <= bb.y0 and bb.y1 <= fig.bbox.height + 0.5, (
            f"clipped text {s_txt!r}: {bb}"
        )
        vis.append((s_txt, bb))
    overlaps = [
        (a, b) for (a, ba), (b, bb) in itertools.combinations(vis, 2)
        if min(ba.x1, bb.x1) - max(ba.x0, bb.x0) > 0.4 and min(ba.y1, bb.y1) - max(ba.y0, bb.y0) > 0.4
    ]
    assert not overlaps, f"text overlaps: {overlaps[:5]}"
    assert leg.get_window_extent(rnd).y0 >= max(t.get_window_extent(rnd).y1 for t in titles) + 0.5
    assert len(strip_texts) == 90
    print(f"gates: {len(vis)} visible texts, 0 overlaps, 0 clipped, 0 forbidden strings")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    raw = OUT_PDF.read_bytes()
    assert b"/Type3" not in raw and (b"/CIDFontType2" in raw or b"/TrueType" in raw)
    plt.close(fig)
    print(f"wrote {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
