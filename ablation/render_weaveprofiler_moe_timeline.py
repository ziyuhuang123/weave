#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified 3-panel WeaveProfiler MoE timeline renderer (annotated v3).

Renders ONE identical 3-panel layout for all three hardware platforms on the exact
same matched workload (the platform preset defines the exact EP size, model and S), from real paired SW+HW Perfetto
traces (gzip JSON). All timestamps / durations / counter values are copied verbatim
from the source traces; only display labels and the 7-stage MoE color mapping are
normalized.

Panels (identical axes rectangles across platforms, figsize=(16, 10.2), DPI=220):
  A: aligned hardware utilization curves (Compute / HBM / Interconnect / Active SM-CU)
  B: multi-CTA execution across ALL physical SMs / CUs (148 / 78 / 80)
  C: representative SM 0 / CU 0 complete MoE pipeline by real warp / wave role

Red lines (raised as TraceRedLineError, a ValueError subclass; never `assert`, so
`python -O` cannot strip them). The rule id is the bracketed prefix of the message:
  hw_provenance  Panel A plots real hardware-counter samples only: nsys GPU metrics on
                 NVIDIA (sm103, h20), rocprof PMC counters on AMD (mi308x). A trace
                 whose otherData.source, any other otherData "*source*" field, a counter
                 name, or the counter-track title carries "Model estimate" or another
                 synthetic / analytic marker (estimate, synthetic, analytic, simulated,
                 roofline, per-tile volume reconstruction, ERR_NVGPUCTRPERM, ...) is
                 rejected; so is a trace that records no positive real-counter evidence.
  hw_curves      all 4 curves (compute, hbm, comm = NVLink / xGMI, active SM / CU) are
                 present exactly once, non-empty (>= 3 samples), equal sample counts.
                 Multi-node MI308X EP16 traces may carry an OPTIONAL 5th real curve,
                 rdma_util [Inter-Node RDMA NIC % of <rate>GB/s]: it is accepted only
                 when otherData.meta.ep == 16 and curve_sources records the exact
                 /sys/class/... NIC counter paths; it joins the same sample-count,
                 range and non-zero gates (hw_rdma_zero) and is drawn in magenta
                 #7B1FA2 with the xGMI legend relabeled "Intra-Node xGMI ...".
                 On EP16 the xGMI curve is ALWAYS labeled "Intra-Node xGMI ..." (it
                 only observes intra-node traffic).  When no Inter-Node RDMA NIC
                 counter could be collected, Panel A draws ONE footnote line
                 ("Inter-Node RDMA NIC counters unavailable on this HCA (omitted,
                 not zero-filled)"); the full device evidence stays in the trace
                 metadata and the validator report, never a zero line and never a
                 fabricated series.
  fabric palette MI308X EP16 (2-node) Panels B and C color every slice by its
                 communication FABRIC, not only by its pipeline stage: Inter-Node
                 RDMA (InterNode_* slices) in warm crimson / amber, Intra-Node xGMI
                 (IntraNode_xGMI_* / IntraNode_HomeCombine) in cool teal / royal
                 blue, GEMM1 / GEMM2 in two greens, Route / Sync in slate gray and
                 the on-GPU LocalStage prep in light violet.  Both panels carry
                 separate, side-by-side Inter-Node RDMA vs Intra-Node xGMI legend
                 swatches; the 7-stage semantics (red lines, streaming Reduce) are
                 unchanged and EP8 / NVIDIA keep the 7-stage palette byte-for-byte.
  hw_range       every sample is finite and satisfies 0.0 <= value <= 100.0.
  hw_comm_zero   max(comm curve) > 0.0: a flat-zero NVLink / xGMI curve is rejected.
  hw_rdma_zero   max(rdma curve) > 0.0 when the optional RDMA NIC curve is present.
  hw_alignment   the curves span the software window of the same run.
  sw_stage_map   every physical-unit slice maps to one of the 7 legend stages or to an
                 explicit background-wait drop; recorder stage_id 14 (Combine_Reduce)
                 always maps to Combine / Reduce, even under a new slice name.
  sw_reduce_dropped / sw_track_hidden / sw_reduce_hidden
                 Combine / Reduce slices are never dropped, every SM 0 / CU 0 track gets
                 a Panel C lane, and every Panel C Combine / Reduce slice is verified on
                 the rendered pixels to show its crisp (non-antialiased) legend color
                 (purple; on the EP16 fabric palette its fabric's amber / royal blue).
  forbidden_string
                 no drawn title, banner, legend, axis or tick label contains a forbidden
                 display string (checked before drawing and again on every Text artist).

Dedicated Spatial streaming Reduce warps (WEAVE_REDUCE_WARPS > 0):
  comm track     a warp / wave track that executes Dispatch / Pull work (or, for tid
                 keyed tracks, whose recorder thread name says Dispatch).
  global tail combine start
                 the earliest Combine / Reduce slice start on any NON-comm track of any
                 SM / CU (the epilogue / compute warps drain the tail combine).
  streaming Reduce slice
                 a Combine / Reduce slice on a comm track that starts inside its own
                 unit's Middle GEMM / Scatter window [first middle start, last middle
                 end) and before the global tail combine start (the window rejects both
                 tail-barrier skew and early pre-GEMM combines).
  Panel C keeps every such slice in Combine / Reduce purple on top of all other stages
  (minimum drawn width 2 px, timestamps untouched) and labels the track band, e.g.
  "W02..W03 (Dispatch -> Streaming Reduce)".  On MI308X EP16 (otherData.meta.ep == 16)
  Panel C uses per-wave layout: EVERY wave W00..W07 is labeled on its own row, e.g.
  "W04 (Comm / Scatter -> Streaming Reduce)" — no merged W00..W03 / W04..W07 bands.
  EP8 fixtures and all other platforms keep the merged / fixed layouts unchanged.

Usage:
  python render_weaveprofiler_moe_timeline.py                  # all 3 platforms
  python render_weaveprofiler_moe_timeline.py --platform h20
  python render_weaveprofiler_moe_timeline.py --stats-json s.json
  python render_weaveprofiler_moe_timeline.py --platform sm103 --trace run.sw_hw.perfetto.json.gz
  python render_weaveprofiler_moe_timeline.py --platform h20 --config-banner "..." --repro-script examples/h20_qwen38max_ep4_s1024.sh
  python render_weaveprofiler_moe_timeline.py --platform mi308x --trace run.sw_hw.perfetto.json.gz --no-banner
  python render_weaveprofiler_moe_timeline.py --platform sm103 --trace run.sw_hw.perfetto.json.gz --no-banner
Validate traces without drawing: python tests/validate_weaveprofiler_trace.py TRACE ...
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import re
import struct
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:  # drawing only; every trace red line below is pure Python
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from matplotlib.ticker import FixedLocator
except ImportError:  # pragma: no cover - only on hosts without matplotlib
    plt = None

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
CURVE_COLOR = {"compute": "#1565C0", "hbm": "#E65100", "comm": "#00897B",
               "comm_rx": "#00897B", "comm_tx": "#0288D1",
               "active": "#455A64", "rdma": "#7B1FA2",
               "rdma_rx": "#C2185B", "rdma_tx": "#7B1FA2"}
CURVE_ORDER = ["compute", "hbm", "comm", "active"]
COMM_ROLES = ("comm", "comm_rx", "comm_tx")
RDMA_ROLES = ("rdma", "rdma_rx", "rdma_tx")
# Optional real Inter-Node RDMA NIC hardware counters on multi-node EP16 only.
# V4 full-duplex exports separate TX (Out) and RX (In) curves (never summed);
# legacy single-track rdma_util stays accepted for backward compatibility.
RDMA_CURVE_RX = re.compile(r"rdma_util \[Inter-Node RDMA NIC % of (\d+(?:\.\d+)?)GB/s\]")
RDMA_RX_CURVE_RX = re.compile(r"rdma_rx_util \[Inter-Node RDMA RX \(In\) % of (\d+(?:\.\d+)?)GB/s\]")
RDMA_TX_CURVE_RX = re.compile(r"rdma_tx_util \[Inter-Node RDMA TX \(Out\) % of (\d+(?:\.\d+)?)GB/s\]")

# ---------------------------------------------------------------- EP16 fabric palette
# MI308X 2-node EP16 Panels B / C: a "fabric class" is the display color class of a
# slice, chosen by the communication fabric the slice drives (the exporter's own
# InterNode_ / IntraNode_ slice-name prefixes), so Inter-Node RDMA and Intra-Node
# xGMI can never share a color.  The 7 semantic STAGES above stay authoritative for
# every red line; fabric classes only change what is drawn.  Order = legend order.
# RDMA_TX: InterNode_RDMA_Dispatch.  RDMA_CB: InterNode_RDMA_Return + InterNode_Partial_Reduce.
# XGMI_TX: IntraNode_xGMI_Pull.      XGMI_CB: IntraNode_xGMI_ScatterPush + IntraNode_HomeCombine.
RDMA_TX, RDMA_CB = "Inter-Node RDMA Dispatch", "Inter-Node RDMA Combine"
XGMI_TX, XGMI_CB = "Intra-Node xGMI Dispatch", "Intra-Node xGMI Combine"
LOCAL_PREP = "Local Stage (prep)"
FABRIC_CLASSES = [
    RDMA_TX, RDMA_CB, XGMI_TX, XGMI_CB,
    "GEMM1 (FC1+SwiGLU)", "GEMM2 (FC2)",
    "Route / Sync", LOCAL_PREP,
]
FABRIC_COLOR = {
    RDMA_TX: "#D62728",                      # crimson red
    RDMA_CB: "#F28E2B",                      # warm amber / orange
    XGMI_TX: "#009FB7",                      # vivid teal / cyan
    XGMI_CB: "#2563EB",                      # royal blue
    "GEMM1 (FC1+SwiGLU)": "#16A34A",         # emerald green
    "GEMM2 (FC2)": "#166534",                # deep forest green
    "Route / Sync": "#94A3B8",               # cool slate gray
    LOCAL_PREP: "#8B5CF6",                   # light violet (non-comm on-GPU prep)
}
FABRIC_OF = {"rdma": (RDMA_TX, RDMA_CB), "xgmi": (XGMI_TX, XGMI_CB)}
FABRIC_MAP = {
    # EP16 (2-node) slice names
    "InterNode_RDMA_Dispatch": RDMA_TX,
    "InterNode_RDMA_Return": RDMA_CB,
    "InterNode_Partial_Reduce": RDMA_CB,
    "IntraNode_xGMI_Pull": XGMI_TX,
    "IntraNode_xGMI_ScatterPush": XGMI_CB,
    "IntraNode_HomeCombine": XGMI_CB,
    "IntraNode_LocalStage": LOCAL_PREP,
    "Compute_GEMM1_SwiGLU": "GEMM1 (FC1+SwiGLU)",
    "Compute_GEMM2_MFMA": "GEMM2 (FC2)",
    "Route_Plan_Local": "Route / Sync",
    "Route_Plan_Remote": "Route / Sync",
    "Local_Ballot_Plan": "Route / Sync",
    # EP8 (single-node, all xGMI) names, should one ever appear in an EP16 trace
    "Route_Plan_80CTA": "Route / Sync",
    "Dispatch_Meta_A2A": XGMI_TX,
    "Dispatch_xGMI_A2A": XGMI_TX,
    "GEMM1_Tile_SwiGLU": "GEMM1 (FC1+SwiGLU)",
    "GEMM2_Tile_MFMA": "GEMM2 (FC2)",
    "Scatter_xGMI": XGMI_CB,
    "Home_Combine": XGMI_CB,
}
FABRIC_MIN_RGB_DIST = 60   # min max-channel distance between any two fabric colors (0..255)

REDUCE_STAGE = "Combine / Reduce"
COMM_STAGE = "Dispatch / Pull"
MIDDLE_STAGES = frozenset({"L1 GEMM (FC1+SwiGLU)", "L2 GEMM (FC2)", "Scatter"})
RAW_REDUCE_STAGE_ID = 14          # tile_prof stage id of every Combine_Reduce tile (SM100-family kernels)
STREAMING_REDUCE_ROLE = "Streaming Reduce"
REDUCE_MIN_DRAW_PX = 2.0          # Panel C minimum drawn width of a Combine / Reduce slice (visibility only)
MIN_CURVE_SAMPLES = 3
ALIGN_HEAD_FRAC, ALIGN_TAIL_FRAC = 0.10, 0.90

PANEL_A_TITLE = "(A) Aligned Hardware Utilization Curves (same-run SW+HW capture)"
PANEL_A_YLABEL = "Utilization (%)"
PANEL_A_YTICKS = [0, 25, 50, 75, 100]
X_LABEL = "Time (\u00b5s)"

# Forbidden display strings, base64-encoded so this source never spells them out.
# Matching is case-insensitive substring matching over every drawn string.
_FORBIDDEN_B64 = [
    # mission list (15)
    "QjMwMA==", "TDIwQQ==", "TDIwRA==", "ZHluYW1pY01L", "djMz", "djI4", "djIw", "djQ2", "cjMx",
    "VjE4NjQ=", "QUdZXw==", "REdfVFVORV9E", "UGhhc2Ut", "UmVzaWRlbmN5", "V0VBVkVfSE9UX0NPTERfTU9ERQ==",
    # carried over unchanged from the previous gate (6)
    "RDc=", "b3Vyc190dW5lZA==", "YWxsMTY=", "bWVnYW1vZTRwNA==", "UlRY", "MzA1MA==",
]
FORBIDDEN = [base64.b64decode(x).decode("utf-8") for x in _FORBIDDEN_B64]

# Provenance markers of synthetic / analytic "hardware" curves (case-insensitive).
SYNTHETIC_HW_MARKERS = [
    r"model[\s_\-:]*estimat",            # "Model estimate", "Model estimates", "model_estimate"
    r"\bestimat",                        # estimate(s) / estimated / estimation
    r"\bsynthetic\b",
    r"\banalytic",                       # analytic / analytical
    r"\bsimulat",                        # simulated / simulation / simulator
    r"\bemulat",
    r"\bnot\s+measured\b",
    r"\broofline\b",
    r"\bper[\s/_\-]+tile(?=[\s/_\-]|$)",             # per-tile / per_tile FLOP or byte volume
    r"\btile[\s/_\-]*(?:route[\s/_\-]*)?volumes?(?=[\s/_\-]|$)",  # tile volume / tile-volume / tile_volume
    r"\bmodel(?:l)?ed\b",                # modelled / modeled curves
    r"\bextrapolat",                     # extrapolated samples
    r"\bERR_NVGPUCTRPERM\b",             # GPU PMU access denied: curves cannot be PMU samples
]
_SYNTHETIC_RX = [re.compile(p, re.IGNORECASE) for p in SYNTHETIC_HW_MARKERS]
_NSYS_TITLE_RX = re.compile(r"\bnsys\b", re.IGNORECASE)
_COUNTER_WORD_RX = re.compile(r"counter|metric", re.IGNORECASE)
_NSYS_METRICS_RX = re.compile(r"gpu[\s_\-]?metrics|\[Throughput %\]", re.IGNORECASE)
_NSYS_METRIC_NAME_RX = re.compile(r"\[Throughput %\]")
_ROCPROF_RX = re.compile(r"\brocprof", re.IGNORECASE)
_AMD_PMC_RX = re.compile(r"\bpmc\b|\bSQ_[A-Z]|\bTCC_[A-Z]|\bGRBM_[A-Z]|\bTA_[A-Z]|"
                         r"hardware[\s\-]counters?|performance[\s\-]counters?|counter[\s\-]collection",
                         re.IGNORECASE)
HW_TOOL_TEXT = {"nsys": "nsys GPU-metrics", "rocprof": "rocprof PMC"}


class TraceRedLineError(ValueError):
    """A trace, or a string that would be drawn from it, violates a WeaveProfiler red line."""

    def __init__(self, rule: str, message: str):
        super().__init__(f"[{rule}] {message}")
        self.rule = rule


# ---------------------------------------------------------------- platform configs
SHARED_BANNERS = {
    "sm103": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=40 (measured winner at S=1024); all other spatial knobs at their defaults.\n"
        "Temporal: all knobs at their defaults.\n"
        "Implementation: WEAVE_PRECISION_MODE=0 (fp8xfp4); all other implementation knobs at their defaults."
    ),
    "h20": (
        "Spatial: all knobs at their defaults; 78 SMs.\n"
        "Temporal: WEAVE_SCATTER_OWNER=comm_only.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_BLOCK_M_L1=32, WEAVE_BLOCK_M_L2=64, "
        "WEAVE_SCATTER_STAGING=smem; all other implementation knobs at their defaults."
    ),
    "mi308x": (
        "Spatial: WEAVE_DISPATCH_ACTIVE_SMS=0, WEAVE_SCATTER_ACTIVE_SMS=0, WEAVE_SM0_STREAM_SCHED=0, "
        "WEAVE_CU_WAVE_SPLIT=4-4 (80 CUs; 4/4 hybrid CUs run 4 compute + 4 comm waves).\n"
        "Temporal: WEAVE_V_HEAD=0 (auto), WEAVE_SCATTER_OWNER=comm_only.\n"
        "Implementation: WEAVE_PRECISION_MODE=2 (fp8xfp8), WEAVE_ADAPT_BM=1, WEAVE_BLOCK_M_L1=0 (auto), "
        "WEAVE_BLOCK_M_L2=0 (auto), WEAVE_DEDUP_MODE=1, WEAVE_LOCAL_SCATTER_DEDUP=1."
    ),
}

REPRO_SCRIPTS = {
    "sm103": "examples/sm103_qwen38max_ep4_s1024.sh",
    "h20": "examples/h20_qwen38max_ep4_s1024.sh",
    "mi308x": "examples/mi308x_qwen38max_ep16_s16384.sh",
}

# SM103 recorder slice names start with an ordinal family token ("5a.", "8b.", ...);
# the bracketed suffix varies across recorder versions, the family does not.
SM103_STAGE_FAMILIES = {
    "1.": "Route / Sync", "2.": "Route / Sync", "3.": "Dispatch / Pull",
    "4.": "TMA / Tile Load", "5a.": "TMA / Tile Load", "5b.": "TMA / Tile Load",
    "5c.": "L1 GEMM (FC1+SwiGLU)", "5d.": "L1 GEMM (FC1+SwiGLU)",
    "6a.": "TMA / Tile Load", "6b.": "TMA / Tile Load", "6c.": "L2 GEMM (FC2)",
    "7a.": "Scatter", "7b.": "Scatter",
    "8a.": "Route / Sync", "8b.": "Combine / Reduce",
}
# Panel C role token per family (None = shared by every warp, not a role).
SM103_ROLE_FAMILIES = {
    "1.": "Dispatch", "2.": "Dispatch", "3.": "Dispatch", "7b.": "Scatter",
    "4.": "CLC Scheduler", "5a.": "TMA-A Load", "6a.": "TMA-A Load",
    "5b.": "TMA-B Load", "6b.": "TMA-B Load", "5c.": "UMMA Issue", "6c.": "UMMA Issue",
    "5d.": "Epilogue / Scatter", "7a.": "Epilogue / Scatter",
    "8a.": None, "8b.": None,
}

PLATFORMS = {
    "sm103": dict(
        trace=None,
        trace_env="WEAVE_SM103_TRACE",
        trace_glob="*qwen38max_ep8_bs1_seqlen1024_local128_ours_*.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_SM103_RESULT_ROOT",
        out="weaveprofiler_sm103_timeline.png",
        title="Weave on NVIDIA SM103 (148-SM, 4x EP4, E=256, S=1024)",
        unit="SM", n_units=148, pid_base=100000, counter_pid=90000, rank_label="rank 0",
        hw_tool="nsys", comm_name="NVLink", arch_prefixes=("sm_100", "sm_103"),
        curves={
            "Tensor Active / MMA Pipe (%)": ("compute", "Compute \u2014 Tensor Active / MMA Pipe %"),
            "HBM / TMA Bandwidth (%)": ("hbm", "HBM \u2014 HBM / TMA Bandwidth %"),
            "NVLink RX/TX User Data (%)": ("comm", "NVLink \u2014 RX/TX User Data %"),
            "SM Warp Issue Activity (%)": ("active", "Active \u2014 SM Warp Issue Activity %"),
            "Active SMs (%)": ("active", "Active \u2014 SMs Active %"),
        },
        stage_map={"Combine_Reduce": "Combine / Reduce"},
        stage_families=SM103_STAGE_FAMILIES,
        raw_stage_ids=True,
        drop=set(),
        c_key="warp", c_layout="families", c_prefix="W",
        c_title="(C) Representative SM 0 \u2014 Complete MoE Pipeline by Real Warp Role",
        c_ylabel="SM 0 warp",
        launch_skew_check=True,
    ),
    "h20": dict(
        trace=None,
        trace_env="WEAVE_H20_TRACE",
        trace_glob="h20_*_tuned__qwen38max_S1024.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_H20_RESULT_ROOT",
        out="weaveprofiler_h20_timeline.png",
        title="Weave on NVIDIA H20 (4x EP4, Qwen3.8-Max, S=1024)",
        unit="SM", n_units=78, pid_base=100, counter_pid=1, rank_label="rank 3",
        hw_tool="nsys", comm_name="NVLink", arch_prefixes=("sm_90",),
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
            "Tile_Sync": "Route / Sync",          # in-SM tile barrier (canonical_semantic=barrier)
            "Workspace_Cleanup": "Route / Sync",
            "Dispatch_Pull": "Dispatch / Pull",
            "L1_TMA_Load": "TMA / Tile Load",
            "L2_TMA_Load": "TMA / Tile Load",
            "L1_GEMM_SwiGLU": "L1 GEMM (FC1+SwiGLU)",
            "L2_GEMM": "L2 GEMM (FC2)",
            "L2_Scatter": "Scatter",
            "Combine_Reduce": "Combine / Reduce",
        },
        drop={"wait", "Disp_Idle_Wait"},           # background waits (dispatch idle until cleanup)
        c_key="tid", c_layout="fixed",
        c_rows=[
            (1, "Producer: Dispatch W0"), (8, "Producer: Dispatch W1"),
            (2, "Producer: TMA-A (Acts+SF)"), (3, "Producer: TMA-B (Weights)"),
            (4, "Compute: Math+Epi WG0"), (5, "Compute: Math+Epi WG1"),
            (6, "Compute: Math+Epi WG2"), (7, "Compute: Math+Epi WG3"),
        ],
        c_thread_expect={1: "Dispatch Warp 0", 8: "Dispatch Warp Last", 2: "TMA Warp A", 3: "TMA Warp B",
                         4: "Math+Epi WG0", 5: "Math+Epi WG1", 6: "Math+Epi WG2", 7: "Math+Epi WG3"},
        c_title="(C) Representative SM 0 \u2014 Complete MoE Pipeline by Real Warp Role",
        c_ylabel="SM 0 warp",
    ),
    "mi308x": dict(
        trace=None,
        trace_env="WEAVE_MI308X_TRACE",
        trace_glob="mi308x_*_tuned__qwen38max_S1024.sw_hw.perfetto.json.gz",
        trace_root_env="WEAVE_MI308X_TRACE_ROOT",
        out="weaveprofiler_mi308x_timeline.png",
        title="Weave on AMD MI308X (16x EP16, Qwen3.8-Max, S=16384) \u2014 WeaveProfiler SW+HW Timeline",
        unit="CU", n_units=80, pid_base=100, counter_pid=1, rank_label="rank 0",
        hw_tool="rocprof", comm_name="xGMI", arch_prefixes=("gfx942",),
        curves={
            "compute_util [FP8 Matrix Core Pipe Active %]": ("compute", "Compute \u2014 FP8 Matrix Core Pipe Active %"),
            "compute_util [FP8 Matrix Core % of 606.2T]": ("compute", "Compute \u2014 FP8 Matrix Core % of 606.2T"),
            "memory_util [HBM3 Bandwidth % of 3684GB/s]": ("hbm", "HBM \u2014 HBM3 Bandwidth % of 3684GB/s"),
            "communication_util [xGMI P2P % of 287.2GB/s]": ("comm", "xGMI \u2014 P2P % of 287.2GB/s"),
            "communication_rx_util [xGMI P2P RX (In) % of 287.2GB/s]": ("comm_rx", "xGMI RX (In) \u2014 % of 287.2GB/s"),
            "communication_rx_util [xGMI RX (In) % of 287.2GB/s]": ("comm_rx", "xGMI RX (In) \u2014 % of 287.2GB/s"),
            "communication_tx_util [xGMI TX (Out) % of 287.2GB/s]": ("comm_tx", "xGMI TX (Out) \u2014 % of 287.2GB/s"),
            "sm_active [Active MI308X CUs % of 80]": ("active", "Active \u2014 CUs % of 80"),
        },
        stage_map={
            # EP8 (4/4 uniform) names -- kept so the EP8 self-test / regression suite stays green.
            "Route_Plan_80CTA": "Route / Sync",
            "Local_Ballot_Plan": "Route / Sync",
            "Dispatch_Meta_A2A": "Dispatch / Pull",
            "Dispatch_xGMI_A2A": "Dispatch / Pull",
            "GEMM1_Tile_SwiGLU": "L1 GEMM (FC1+SwiGLU)",
            "GEMM2_Tile_MFMA": "L2 GEMM (FC2)",
            "Scatter_xGMI": "Scatter",
            "Home_Combine": "Combine / Reduce",
            # EP16 (2-node, 0/8 + 4/4 + 8/0 CTA regimes) names.
            "Route_Plan_Local": "Route / Sync",
            "Route_Plan_Remote": "Route / Sync",
            "IntraNode_LocalStage": "Dispatch / Pull",
            "IntraNode_xGMI_Pull": "Dispatch / Pull",
            "InterNode_RDMA_Dispatch": "Dispatch / Pull",
            "Compute_GEMM1_SwiGLU": "L1 GEMM (FC1+SwiGLU)",
            "Compute_GEMM2_MFMA": "L2 GEMM (FC2)",
            "IntraNode_xGMI_ScatterPush": "Scatter",
            "InterNode_RDMA_Return": "Scatter",
            "InterNode_Partial_Reduce": "Combine / Reduce",
            "IntraNode_HomeCombine": "Combine / Reduce",
        },
        drop={"GEMM1_Wait_Payload", "GEMM2_Wait_Stage1", "Post_Plan_Wait",
              "Scatter_Ready_Wait", "Dispatch_Wait_Plan", "Home_Ready_Wait",
              "GEMM1_Wait_Route", "Route_Wait_Meta",
              "InterNode_RDMA_RxWatch", "RDMA_WQE_CQE_Service", "Legacy_GDA_Service"},
        c_key="wave", c_layout="groups", c_prefix="W",
        # EP16 hybrid compute waves also run a short front-end LocalStage/Pull, and the
        # pure-comm CTAs' reduce-specialist waves stream Reduce early, so the EP8 stage
        # heuristic alone cannot separate roles.  The exporter therefore stamps each wave's
        # thread name with its kernel-geometry role ([Compute]/[Comm]); when present it is
        # authoritative.  EP8 traces carry no marker and fall back to the stage heuristic.
        wave_role_markers=True, c_tid_base=1,
        c_group_roles={"comm": "Comm / Scatter", "compute": "Compute / Combine"},
        # MI308X EP16 (2-node) resolves to c_layout="per_wave" via trace_cfg(): every
        # wave W00..W07 gets its own labeled Panel C row (no merged W00..W03 bands).
        rdma_curve_rx=RDMA_CURVE_RX,
        c_title="(C) Representative CU 0 \u2014 Complete MoE Pipeline by Real Wave Role",
        c_ylabel="CU 0 wave",
    ),
}


def _check_platform_configs():
    """Config-level red line: no Combine / Reduce slice name may ever sit in a drop set."""
    for key, cfg in PLATFORMS.items():
        bad = sorted(nm for nm in cfg["drop"]
                     if nm == "Combine_Reduce" or cfg["stage_map"].get(nm) == REDUCE_STAGE)
        if bad:
            raise TraceRedLineError("sw_reduce_dropped", f"{key}: drop set contains Combine / Reduce names {bad}")
        for st in list(cfg["stage_map"].values()) + list((cfg.get("stage_families") or {}).values()):
            if st not in STAGE_COLOR:
                raise TraceRedLineError("sw_stage_map", f"{key}: stage {st!r} is not a legend stage")


_check_platform_configs()

# identical axes rectangles for every platform (figure fractions)
LEFT, WIDTH = 0.160, 0.825
RECT_A = [LEFT, 0.6225, WIDTH, 0.200]
RECT_B = [LEFT, 0.3095, WIDTH, 0.285]
RECT_C = [LEFT, 0.056, WIDTH, 0.183]
FIG_SIZE_IN, FIG_DPI, FIG_PX = (16, 10.2), 220, (3520, 2244)


def platform_cfg(key: str, trace=None) -> dict:
    cfg = dict(PLATFORMS[key])
    cfg["platform"] = key
    if trace is not None:
        cfg["trace"] = Path(trace)
    return cfg


def resolve_trace(cfg):
    """Resolve an explicit trace path, then an env override, then a glob."""
    if cfg.get("trace"):
        return Path(cfg["trace"])
    env_name = cfg.get("trace_env")
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name])
    pattern = cfg.get("trace_glob")
    if pattern:
        default_root = cfg.get("trace_root") or Path.cwd()
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


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _family(name: str) -> str:
    return str(name).split(" ", 1)[0]


def _other_data(doc) -> dict:
    other = doc.get("otherData") if isinstance(doc, dict) else None
    return other if isinstance(other, dict) else {}


def detect_platform(doc) -> str:
    """Platform key from otherData card / arch, else from the counter-track names."""
    other = _other_data(doc)
    meta = other.get("meta") if isinstance(other.get("meta"), dict) else {}
    for card in (meta.get("card"), other.get("card")):
        if isinstance(card, str) and card.lower() in PLATFORMS:
            return card.lower()
    arch = str(meta.get("arch") or other.get("arch") or "").lower()
    for key, cfg in PLATFORMS.items():
        if arch and arch.startswith(cfg["arch_prefixes"]):
            return key
    names = {e.get("name") for e in doc.get("traceEvents", []) if isinstance(e, dict) and e.get("ph") == "C"}
    for key, cfg in PLATFORMS.items():
        if names and names <= set(cfg["curves"]):
            return key
    raise TraceRedLineError("platform", "cannot detect platform from otherData.meta.card / arch or "
                                        "counter names; pass --platform sm103|h20|mi308x")


def trace_names(evs):
    pnames, tnames = {}, defaultdict(dict)
    for e in evs:
        if e.get("ph") != "M":
            continue
        pid, args = _as_int(e.get("pid")), e.get("args") or {}
        if e.get("name") == "process_name" and pid is not None:
            pnames[pid] = str(args.get("name", ""))
        elif e.get("name") == "thread_name" and pid is not None and _as_int(e.get("tid")) is not None:
            tnames[pid][int(e["tid"])] = str(args.get("name", ""))
    return pnames, tnames


# ================================================================= hardware red lines
def collect_counters(evs, cfg):
    curves = defaultdict(list)
    for e in evs:
        if e.get("ph") != "C" or _as_int(e.get("pid")) != cfg["counter_pid"]:
            continue
        args = e.get("args") or {}
        # float(True) == 1.0 would silently turn a boolean into a 1 % sample.
        if isinstance(e.get("ts"), bool) or (isinstance(args, dict) and isinstance(args.get("value"), bool)):
            raise TraceRedLineError("hw_curves", f"{cfg['platform']}: boolean counter sample {e!r} (expected a number)")
        try:
            curves[str(e["name"])].append((float(e["ts"]), float(args["value"])))
        except (KeyError, TypeError, ValueError):
            raise TraceRedLineError("hw_curves", f"{cfg['platform']}: malformed counter sample {e!r}") from None
    for pts in curves.values():
        pts.sort()
    return dict(curves)


def _string_leaves(obj, path):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_leaves(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _string_leaves(v, f"{path}[{i}]")


def _source_fields(obj, path="otherData"):
    """String leaves under every otherData key that mentions 'source' (source, hw_counter_source, curve_sources, ...)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if "source" in str(k).lower():
                yield from _string_leaves(v, p)
            elif isinstance(v, (dict, list)):
                yield from _source_fields(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _source_fields(v, f"{path}[{i}]")


def _curve_sources(other):
    for holder in (other, other.get("meta") if isinstance(other.get("meta"), dict) else {}):
        cs = holder.get("curve_sources")
        if isinstance(cs, dict) and cs:
            return cs
    return None


def hw_provenance_strings(doc, cfg, curves, pnames, tnames):
    cpid = cfg["counter_pid"]
    out = list(_source_fields(_other_data(doc)))
    out += [(f"counter name {nm!r}", nm) for nm in sorted(curves)]
    if cpid in pnames:
        out.append((f"counter-track title (pid {cpid})", pnames[cpid]))
    out += [(f"counter-track thread (pid {cpid}, tid {t})", n) for t, n in sorted(tnames.get(cpid, {}).items())]
    return out


def _real_counter_evidence(cfg, doc, curves, pnames, prov):
    other, title = _other_data(doc), pnames.get(cfg["counter_pid"], "")
    other_texts = [(w, t) for w, t in prov if w.startswith("otherData")]
    ev = []
    if cfg["hw_tool"] == "nsys":
        if _NSYS_TITLE_RX.search(title) and _COUNTER_WORD_RX.search(title):
            ev.append(f"counter-track title {title!r}")
        cs = _curve_sources(other)
        if cs and len(cs) >= 4 and all(isinstance(v, str) and _NSYS_METRIC_NAME_RX.search(v) for v in cs.values()):
            ev.append("curve_sources are nsys GPU-metrics names (" + ", ".join(sorted(map(str, cs))) + ")")
        ev += [f"{w} = {t!r}" for w, t in other_texts if _NSYS_TITLE_RX.search(t) and _NSYS_METRICS_RX.search(t)]
        if curves and all(_NSYS_METRIC_NAME_RX.search(nm) for nm in curves):
            ev.append("counter names are raw nsys GPU-metrics names")
    else:
        texts = [t for _, t in other_texts] + [title]
        if any(_ROCPROF_RX.search(t) for t in texts) and any(_AMD_PMC_RX.search(t) for t in texts):
            ev.append("rocprof PMC counter provenance recorded in otherData / counter-track title")
    return ev


def check_hw_red_lines(doc, cfg, curves, pnames, tnames) -> dict:
    """Strict real-hardware-counter red lines; raises TraceRedLineError (a ValueError)."""
    plat, tool = cfg["platform"], cfg["hw_tool"]
    tool_text = HW_TOOL_TEXT[tool]
    prov = hw_provenance_strings(doc, cfg, curves, pnames, tnames)
    for where, text in prov:                      # 1. reject synthetic / analytic provenance
        for rx in _SYNTHETIC_RX:
            m = rx.search(text)
            if m:
                raise TraceRedLineError(
                    "hw_provenance",
                    f"{plat}: synthetic / analytic hardware curves rejected: {where} = {text!r} "
                    f"matches {m.group(0)!r}; Panel A only plots real {tool_text} samples")
    evidence = _real_counter_evidence(cfg, doc, curves, pnames, prov)
    if not evidence:                              # 2. require positive real-counter evidence
        raise TraceRedLineError(
            "hw_provenance",
            f"{plat}: cannot certify real {tool_text} hardware-counter provenance: no evidence in "
            f"otherData.source / otherData '*source*' fields / counter names / counter-track title "
            f"(pid {cfg['counter_pid']}); record the capture tool and metric names at capture time")
    known = dict(cfg["curves"])                   # 3. compute, hbm, comm (or full-duplex comm_rx/tx), active (+ optional rdma / rdma_rx+tx)
    rdma_tracks = {}
    if cfg.get("rdma_curve_rx") is not None:
        rdma_specs = (
            ("rdma", cfg["rdma_curve_rx"], lambda r: f"Inter-Node RDMA (NIC) \u2014 % of {r}GB/s"),
            ("rdma_rx", RDMA_RX_CURVE_RX, lambda r: f"Inter-Node RDMA RX (In) \u2014 % of {r}GB/s"),
            ("rdma_tx", RDMA_TX_CURVE_RX, lambda r: f"Inter-Node RDMA TX (Out) \u2014 % of {r}GB/s"),
        )
        for nm in sorted(curves):
            for role, rx_pat, lab_fn in rdma_specs:
                m = rx_pat.fullmatch(nm)
                if m:
                    if role in rdma_tracks:
                        raise TraceRedLineError("hw_curves", f"{plat}: two Inter-Node RDMA NIC counter tracks for {role!r}: "
                                                             f"{rdma_tracks[role][0]!r} and {nm!r}")
                    rdma_tracks[role] = (nm, m.group(1), lab_fn(m.group(1)))
    if "rdma" in rdma_tracks and ("rdma_rx" in rdma_tracks or "rdma_tx" in rdma_tracks):
        raise TraceRedLineError("hw_curves", f"{plat}: cannot mix combined rdma_util with full-duplex rdma_rx_util/rdma_tx_util")
    if rdma_tracks:
        # RDMA NIC curves are legal ONLY on a multi-node EP16 trace and must carry
        # real NIC sysfs provenance (exact /sys/class/... counter paths + rate).
        meta = _other_data(doc).get("meta")
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("ep") != 16:
            first_nm = next(iter(rdma_tracks.values()))[0]
            raise TraceRedLineError("hw_curves", f"{plat}: Inter-Node RDMA NIC curve {first_nm!r} requires a "
                                                 f"multi-node EP16 trace (otherData.meta.ep == 16), got "
                                                 f"{meta.get('ep')!r}; single-node runs have no RDMA fabric")
        cs_map = _curve_sources(_other_data(doc)) or {}
        for role, (nm, _rate, lab) in rdma_tracks.items():
            nic_src = cs_map.get(nm)
            if not (isinstance(nic_src, str) and "/sys/class/" in nic_src):
                raise TraceRedLineError("hw_provenance", f"{plat}: Inter-Node RDMA NIC curve {nm!r} has no "
                                                         f"real NIC hardware-counter provenance: curve_sources must "
                                                         f"record the exact /sys/class/... counter paths and link rate")
            known[nm] = (role, lab)
    unknown = sorted(set(curves) - set(known))
    if unknown:
        raise TraceRedLineError("hw_curves", f"{plat}: unexpected counter track(s) {unknown} on pid "
                                             f"{cfg['counter_pid']}; expected names {sorted(known)}")
    role_src = {}
    for nm in sorted(curves):
        role = known[nm][0]
        if role in role_src:
            raise TraceRedLineError("hw_curves", f"{plat}: two counter tracks for role {role!r}: "
                                                 f"{role_src[role]!r} and {nm!r}")
        role_src[role] = nm
    if "comm" in role_src and ("comm_rx" in role_src or "comm_tx" in role_src):
        raise TraceRedLineError("hw_curves", f"{plat}: cannot mix combined comm curve with directional comm_rx/comm_tx")
    has_comm = any(r in role_src for r in COMM_ROLES)
    missing = [r for r in ("compute", "hbm", "active") if r not in role_src]
    if not has_comm:
        missing.append("comm")
    if missing:
        raise TraceRedLineError("hw_curves", f"{plat}: missing real {tool_text} curve(s) for role(s) {missing}; "
                                             f"all 4 of compute / hbm / comm ({cfg['comm_name']}) / active are required")
    roles = (["compute", "hbm"]
             + [r for r in COMM_ROLES if r in role_src]
             + [r for r in RDMA_ROLES if r in role_src]
             + ["active"])
    lens, stats = {}, {}
    for role in roles:                            # 4. every sample finite and inside [0, 100]
        nm = role_src[role]
        pts = curves[nm]
        if len(pts) < MIN_CURVE_SAMPLES:
            raise TraceRedLineError("hw_curves", f"{plat}: curve {nm!r} has {len(pts)} sample(s); "
                                                 f"need >= {MIN_CURVE_SAMPLES}")
        for i, (ts, v) in enumerate(pts):
            if not (math.isfinite(ts) and math.isfinite(v) and 0.0 <= v <= 100.0):
                raise TraceRedLineError("hw_range", f"{plat}: curve {nm!r} sample #{i} at ts={ts} us is {v!r}, "
                                                    f"outside 0.0 <= value <= 100.0")
        vs = [v for _, v in pts]
        lens[role] = len(pts)
        stats[role] = dict(name=nm, label=known[nm][1], points=len(pts), ts_min=pts[0][0], ts_max=pts[-1][0],
                           min=round(min(vs), 3), max=round(max(vs), 3))
    if len(set(lens.values())) != 1:
        raise TraceRedLineError("hw_curves", f"{plat}: the {len(roles)} curves have unequal sample counts {lens}")
    comm_roles_present = [r for r in COMM_ROLES if r in role_src]
    comm_max = max(max(v for _, v in curves[role_src[r]]) for r in comm_roles_present)
    if not comm_max > 0.0:
        raise TraceRedLineError("hw_comm_zero", f"{plat}: {cfg['comm_name']} curve(s) "
                                                f"{[role_src[r] for r in comm_roles_present]!r} flat zero "
                                                f"(max = {comm_max}); a missing or all-zero interconnect counter "
                                                f"cannot be plotted as a real {tool_text} curve")
    rdma_max = None
    rdma_roles_present = [r for r in RDMA_ROLES if r in role_src]
    if rdma_roles_present:                        # 6. present RDMA NIC curve(s) must be non-zero too
        rdma_max = max(max(v for _, v in curves[role_src[r]]) for r in rdma_roles_present)
        if not rdma_max > 0.0:
            raise TraceRedLineError("hw_rdma_zero", f"{plat}: Inter-Node RDMA NIC curve(s) "
                                                    f"{[role_src[r] for r in rdma_roles_present]!r} flat zero "
                                                    f"(max = {rdma_max}); an all-zero NIC counter cannot be "
                                                    f"plotted as a real hardware curve")
    return dict(tool=tool, tool_text=tool_text, evidence=evidence, role_src=role_src, samples=lens["compute"],
                comm_max=comm_max, rdma_max=rdma_max, curves=stats,
                full_duplex_comm=("comm_rx" in role_src or "comm_tx" in role_src),
                full_duplex_rdma=("rdma_rx" in role_src or "rdma_tx" in role_src),
                curve_ts_min=min(s["ts_min"] for s in stats.values()),
                curve_ts_max=max(s["ts_max"] for s in stats.values()))


def panel_a_legend(cfg, hw) -> list:
    """Ordered (role, counter-track name, legend label) for Panel A.  On the
    2-node EP16 fabric the xGMI curve(s) are always labeled 'Intra-Node xGMI ...'
    (they only ever observe intra-node traffic); full-duplex RX (In) and TX (Out)
    curves stay strictly separate and are never summed."""
    has_rdma = any(r in hw["role_src"] for r in RDMA_ROLES)
    order = (["compute", "hbm"]
             + [r for r in COMM_ROLES if r in hw["role_src"]]
             + [r for r in RDMA_ROLES if r in hw["role_src"]]
             + ["active"])
    out = []
    for role in order:
        src = hw["role_src"][role]
        lab = cfg["curves"][src][1] if src in cfg["curves"] else hw["curves"][role]["label"]
        if role in COMM_ROLES and (has_rdma or cfg.get("ep") == 16):
            lab = "Intra-Node " + lab
        out.append((role, src, lab))
    return out


def trace_cfg(cfg, doc) -> dict:
    """Per-trace Panel C layout: MI308X EP16 (2-node, otherData.meta.ep == 16)
    labels EVERY wave W00..W07 on its own row (c_layout='per_wave'); EP8 and all
    other platforms keep their preset layout unchanged."""
    meta = _other_data(doc).get("meta")
    meta = meta if isinstance(meta, dict) else {}
    if cfg.get("platform") == "mi308x" and meta.get("ep") == 16:
        if cfg.get("c_layout") == "groups":
            cfg = dict(cfg, c_layout="per_wave")
        cap = meta.get("capture_meta") if isinstance(meta.get("capture_meta"), dict) else {}
        ecfg = (meta.get("env_cfg") if isinstance(meta.get("env_cfg"), dict)
                else (cap.get("env_cfg") if isinstance(cap.get("env_cfg"), dict) else {}))
        no_sr = str(ecfg.get("WEAVE_REDUCE_WARPS", "")) == "0"
        # 2-node EP16 has two fabrics: Panels B / C color by fabric (see FABRIC_MAP)
        cfg = dict(cfg, ep=16, fabric_palette=True, ep16_tail_combine_label=no_sr)
    return cfg


def fabric_class(cfg, nm: str, st: str) -> str:
    """Fabric display class of one mapped slice (EP16 fabric palette only).  The
    slice name decides the fabric: InterNode_* -> Inter-Node RDMA, IntraNode_xGMI_* /
    IntraNode_HomeCombine -> Intra-Node xGMI.  An unknown name is a red line, never
    silently recolored into either fabric."""
    cls = FABRIC_MAP.get(nm)
    if cls is None and nm.startswith("InterNode_"):
        cls = RDMA_CB if st in (REDUCE_STAGE, "Scatter") else RDMA_TX
    elif cls is None and nm.startswith("IntraNode_xGMI_"):
        cls = XGMI_CB if st in (REDUCE_STAGE, "Scatter") else XGMI_TX
    if cls is None:
        raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: slice {nm!r} (stage {st!r}) has no EP16 "
                                                f"fabric class; refusing to guess Inter-Node RDMA vs Intra-Node xGMI")
    return cls


def stage_palette(cfg):
    """(ordered legend classes, class -> hex color) drawn on Panels B / C."""
    if cfg.get("fabric_palette"):
        return list(FABRIC_CLASSES), dict(FABRIC_COLOR)
    return list(STAGES), dict(STAGE_COLOR)


def span_class(cfg, st: str, nm: str) -> str:
    return fabric_class(cfg, nm, st) if cfg.get("fabric_palette") else st


def panel_b_legend_classes(cfg) -> list:
    """Panel B's own legend: the four side-by-side fabric swatches (EP16 only)."""
    return [RDMA_TX, RDMA_CB, XGMI_TX, XGMI_CB] if cfg.get("fabric_palette") else []


def check_fabric_palette():
    """Palette red line: Inter-Node RDMA and Intra-Node xGMI never share a color, and
    every fabric class is clearly distinct from every other one."""
    rgb = {c: tuple(int(FABRIC_COLOR[c][i:i + 2], 16) for i in (1, 3, 5)) for c in FABRIC_CLASSES}
    if len({FABRIC_COLOR[c].upper() for c in FABRIC_CLASSES}) != len(FABRIC_CLASSES):
        raise TraceRedLineError("fabric_palette", "two EP16 fabric classes share one hex color")
    worst = min((max(abs(a - b) for a, b in zip(rgb[x], rgb[y])), x, y)
                for x, y in itertools.combinations(FABRIC_CLASSES, 2))
    if worst[0] < FABRIC_MIN_RGB_DIST:
        raise TraceRedLineError("fabric_palette", f"fabric colors {worst[1]!r} / {worst[2]!r} differ by only "
                                                  f"{worst[0]} (< {FABRIC_MIN_RGB_DIST}) in some RGB channel")
    rdma = {FABRIC_COLOR[c].upper() for c in FABRIC_OF["rdma"]}
    xgmi = {FABRIC_COLOR[c].upper() for c in FABRIC_OF["xgmi"]}
    if rdma & xgmi:
        raise TraceRedLineError("fabric_palette", f"Inter-Node RDMA and Intra-Node xGMI share {sorted(rdma & xgmi)}")
    return worst[0]


# Absent same-run interconnect evidence is recorded as unavailable with a reason; it
# is never drawn as a zero line and never replaced by a fabricated series.  Panel A
# shows ONE publication-grade footnote line; the full device evidence (HCA names,
# port state, probe output, recorded reason) stays in otherData.meta and in the
# validator report (panel_a_rdma_evidence), never on the figure.
RDMA_NOTE_HCA = "Inter-Node RDMA NIC counters unavailable on this HCA (omitted, not zero-filled)"
RDMA_NOTE_NONE = "Inter-Node RDMA NIC counters not recorded in this capture (omitted, not zero-filled)"


def _device_range(devs) -> str:
    """'fic2_soe_bond0..3' for one contiguous device family, else a comma list."""
    devs = [str(d) for d in devs]
    if not devs:
        return ""
    m = [re.fullmatch(r"(.*?)(\d+)", d) for d in devs]
    if all(m) and len({x.group(1) for x in m}) == 1:
        nums = sorted(int(x.group(2)) for x in m)
        if nums == list(range(nums[0], nums[-1] + 1)):
            stem = m[0].group(1)
            return f"{stem}{nums[0]}" if len(nums) == 1 else f"{stem}{nums[0]}..{nums[-1]}"
    return ", ".join(devs)


def _rdma_meta(doc):
    meta = _other_data(doc).get("meta")
    meta = meta if isinstance(meta, dict) else {}
    nic = meta.get("nic_capture") if isinstance(meta.get("nic_capture"), dict) else {}
    hca = meta.get("nic_rdma_hca") if isinstance(meta.get("nic_rdma_hca"), dict) else {}
    return nic, hca


def panel_a_rdma_note(cfg, hw, doc):
    """Panel A's one-line Inter-Node RDMA footnote for a 2-node EP16 MI308X trace
    carrying NO rdma curve.  None when the RDMA curve is present (both fabrics then
    appear in the legend) or on single-node EP8 / other platforms.  'on this HCA' is
    only claimed when the trace records HCA / NIC evidence; otherwise the line says
    the counter was not recorded.  Never a zero curve in place of the counter."""
    if cfg.get("platform") != "mi308x" or cfg.get("ep") != 16 or any(r in hw["role_src"] for r in RDMA_ROLES):
        return None
    nic, hca = _rdma_meta(doc)
    has_evidence = bool(hca.get("devices") or hca.get("port") or nic.get("rdma_skipped_reason")
                        or hca.get("reason") or nic.get("ports"))
    return RDMA_NOTE_HCA if has_evidence else RDMA_NOTE_NONE


def panel_a_rdma_evidence(cfg, hw, doc):
    """Full recorded Inter-Node RDMA evidence (validator report only, never drawn)."""
    if panel_a_rdma_note(cfg, hw, doc) is None:
        return None
    nic, hca = _rdma_meta(doc)
    head = "Inter-Node RDMA (NIC) - hardware counter UNAVAILABLE"
    if hca.get("devices"):
        head += " on " + _device_range(hca["devices"])
    if hca.get("node_desc"):
        head += f" ({hca['node_desc']})"
    parts = [head]
    port = hca.get("port") if isinstance(hca.get("port"), dict) else {}
    if port:
        line = (f"port {port.get('port')}: {port.get('state')}, {port.get('rate')}, "
                f"counters={port.get('counters')} and hw_counters={port.get('hw_counters')}")
        stat = hca.get("rdma_statistic") if isinstance(hca.get("rdma_statistic"), dict) else {}
        if stat.get("cmd"):
            line += f"; {stat['cmd']} -> {str(stat.get('stderr', '')).strip()}"
        parts.append(line)
    reason = nic.get("rdma_skipped_reason") or hca.get("reason")
    if reason:
        parts.append(str(reason))
    elif nic.get("ports"):
        parts.append("The readable NIC sysfs counters recorded in nic_capture did not move inside the "
                     "FI window, so this capture has no real Inter-Node RDMA hardware curve.")
    else:
        parts.append("No Inter-Node RDMA NIC hardware counter was recorded for this capture.")
    return "\n".join(parts)


def _panel_a_free_bands(data, top: float = 105.0):
    """Curve-free horizontal bands of Panel A (data units), widest first.  The
    fabric note is placed in one of these so it never covers a real counter sample."""
    vals = sorted({float(v) for pts in data["curves"].values() for _, v in pts})
    edges = [0.0] + vals + [top]
    return sorted([(a, b) for a, b in zip(edges, edges[1:]) if b > a],
                  key=lambda ab: ab[1] - ab[0], reverse=True)


def _fit_panel_a_note(axA, artist, data, fig, renderer, base_font: float = 8.4):
    """Place the fabric note in the widest curve-free band, shrinking the font until
    the whole box fits that band and the axes width and no real sample lies under it."""
    pts = [(t, v) for p in data["curves"].values() for t, v in p]
    ab = axA.get_window_extent(renderer)
    for lo, hi in _panel_a_free_bands(data):
        band_px = axA.transData.transform((0, hi))[1] - axA.transData.transform((0, lo))[1]
        cy_frac = (lo + hi) / 2.0 / 105.0
        for i in range(10):
            font = round(base_font - 0.3 * i, 2)
            artist.set_fontsize(font)
            artist.set_position((0.012, cy_frac))
            fig.canvas.draw()
            bb = artist.get_window_extent(renderer)
            pad = 0.30 * font * fig.dpi / 72.0
            if bb.height + 2 * pad > band_px - 6.0:
                continue
            if bb.x1 + pad > ab.x1 - 4.0 or bb.x0 - pad < ab.x0 + 2.0:
                continue
            x0, x1, y0, y1 = bb.x0 - pad, bb.x1 + pad, bb.y0 - pad, bb.y1 + pad
            if any(x0 <= axA.transData.transform((t, v))[0] <= x1
                   and y0 <= axA.transData.transform((t, v))[1] <= y1 for t, v in pts):
                continue
            return font
    raise RuntimeError("Panel A Inter-Node RDMA unavailable note fits no curve-free band; "
                       "refusing to draw it over real counter samples")


# ================================================================= software red lines
def _is_raw_reduce_id(v) -> bool:
    try:
        return int(float(v)) == RAW_REDUCE_STAGE_ID and float(v) == float(RAW_REDUCE_STAGE_ID)
    except (TypeError, ValueError):
        return False


def classify_stage(cfg, e):
    """Legend stage of one physical-unit slice, or None for an explicit background-wait drop."""
    nm = str(e.get("name", ""))
    args = e.get("args") or {}
    reduce_marked = (nm == "Combine_Reduce" or args.get("stage") == "Combine_Reduce"
                     or (cfg.get("raw_stage_ids") and _is_raw_reduce_id(args.get("stage_id"))))
    if nm in cfg["drop"]:
        if reduce_marked:
            raise TraceRedLineError("sw_reduce_dropped", f"{cfg['platform']}: Combine_Reduce slice {nm!r} "
                                                         f"would be dropped as a wait span")
        return None
    st = cfg["stage_map"].get(nm)
    if st is None and cfg.get("stage_families"):
        st = cfg["stage_families"].get(_family(nm))
    if reduce_marked:
        if st is not None and st != REDUCE_STAGE:
            raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: slice {nm!r} carries the Combine_Reduce "
                                                    f"marker (stage_id {RAW_REDUCE_STAGE_ID}) but its name maps to {st!r}")
        return REDUCE_STAGE
    if st is None:
        raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: unmapped event name {nm!r}")
    return st


def track_key(e, cfg):
    if cfg["c_key"] == "tid":
        k = _as_int(e.get("tid"))
    else:
        k = _as_int((e.get("args") or {}).get(cfg["c_key"]))
    if k is None:
        raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: slice {e.get('name')!r} has no integer "
                                                f"{cfg['c_key']} track key")
    return k


def parse_sw(doc, cfg) -> dict:
    """Per-unit software spans (physical SMs / CUs only; summary / macro tracks excluded)."""
    base, n = cfg["pid_base"], cfg["n_units"]
    unit_spans = [[] for _ in range(n)]
    unit_tracks = [defaultdict(list) for _ in range(n)]
    stage_counts, drop_counts, name_counts = Counter(), Counter(), Counter()
    sw_end = 0.0
    for e in doc["traceEvents"]:
        if e.get("ph") != "X":
            continue
        pid = _as_int(e.get("pid"))
        if pid is None or not (base <= pid < base + n):
            continue
        idx = pid - base
        nm = str(e.get("name", ""))
        name_counts[nm] += 1
        try:
            ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
        except (KeyError, TypeError, ValueError):
            raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: malformed software slice {nm!r}: "
                                                     f"ts={e.get('ts')!r} dur={e.get('dur')!r}") from None
        if not (math.isfinite(ts) and math.isfinite(dur) and dur >= 0.0):
            raise TraceRedLineError("sw_window", f"{cfg['platform']}: software slice {nm!r} has a non-finite or "
                                                 f"negative interval ts={ts!r} dur={dur!r}")
        sw_end = max(sw_end, ts + dur)
        st = classify_stage(cfg, e)
        if st is None:
            drop_counts[nm] += 1
            continue
        stage_counts[st] += 1
        unit_spans[idx].append((ts, dur, st, e))
        unit_tracks[idx][track_key(e, cfg)].append((ts, dur, st, nm))
    if not sum(stage_counts.values()):
        raise TraceRedLineError("sw_stage_map", f"{cfg['platform']}: no software slices on pids "
                                                f"{base}..{base + n - 1}")
    for tracks in unit_tracks:
        for spans in tracks.values():
            spans.sort()
    return dict(unit_spans=unit_spans, unit_tracks=[dict(t) for t in unit_tracks], stage_counts=stage_counts,
                drop_counts=drop_counts, name_counts=name_counts, sw_end=sw_end)


def check_alignment(cfg, hw, sw):
    end = sw["sw_end"]
    if not (hw["curve_ts_min"] <= ALIGN_HEAD_FRAC * end and hw["curve_ts_max"] >= ALIGN_TAIL_FRAC * end):
        raise TraceRedLineError("hw_alignment", f"{cfg['platform']}: hardware curves span "
                                                f"[{hw['curve_ts_min']}, {hw['curve_ts_max']}] us but the software "
                                                f"window ends at {end:.3f} us; not a same-run aligned capture")


def verify_tracks(cfg, sw, tnames):
    """Truthfulness checks for Panel C (labels themselves are derived from the slices)."""
    spans0 = sw["unit_spans"][0]
    if not spans0:
        raise TraceRedLineError("sw_roles", f"{cfg['platform']}: representative {cfg['unit']} 0 has no slices")
    if cfg["c_layout"] == "fixed":
        tn = tnames.get(cfg["pid_base"], {})
        for tid, frag in cfg["c_thread_expect"].items():
            if frag not in tn.get(tid, ""):
                raise TraceRedLineError("sw_roles", f"{cfg['platform']}: Panel C row tid {tid} needs a recorder "
                                                    f"thread name containing {frag!r}; trace has {tn.get(tid)!r}")
    if cfg.get("launch_skew_check"):  # clean window: no giant cross-rank launch-skew wait at start
        first_ts = min(ts for ts, _, _, _ in spans0)
        if not first_ts < 5.0:
            raise TraceRedLineError("sw_window", f"{cfg['platform']}: first SM 0 slice starts at {first_ts} us (>= 5 us)")
        sync = [float(e.get("dur", 0.0)) for _, _, _, e in spans0 if _family(e.get("name", "")) == "2."]
        if sync and not max(sync) < 15.0:
            raise TraceRedLineError("sw_window", f"{cfg['platform']}: SM 0 cross-rank rendezvous lasts "
                                                 f"{max(sync)} us (>= 15 us launch skew)")


def _is_comm_track(cfg, spans, tname) -> bool:
    # An explicit kernel-geometry role marker on the wave's thread name is authoritative
    # (EP16 hybrid compute waves also run a front-end Pull, and pure-comm reduce-specialist
    # waves stream Reduce early, so the stage heuristic alone mislabels them).  Traces
    # without a marker (all EP8 traces / fixtures) fall through to the stage heuristic.
    if cfg.get("wave_role_markers") and tname:
        if "[Comm]" in tname:
            return True
        if "[Compute]" in tname:
            return False
    if any(st == COMM_STAGE for _, _, st, _ in spans):
        return True
    return cfg["c_key"] == "tid" and "dispatch" in (tname or "").lower()


def analyze_reduce(cfg, sw, tnames) -> dict:
    """Locate streaming (Middle) Reduce slices on communication tracks; see module docstring."""
    base = cfg["pid_base"]
    comm, tail_starts, unit_mid = {}, [], []
    for u, tracks in enumerate(sw["unit_tracks"]):
        mids = [(ts, ts + dur) for spans in tracks.values() for ts, dur, st, _ in spans if st in MIDDLE_STAGES]
        unit_mid.append((min(a for a, _ in mids), max(b for _, b in mids)) if mids else None)
        for k, spans in tracks.items():
            tid_key = k if cfg["c_key"] == "tid" else k + int(cfg.get("c_tid_base", 1))
            tname = tnames.get(base + u, {}).get(tid_key)
            comm[(u, k)] = _is_comm_track(cfg, spans, tname)
            if not comm[(u, k)]:
                tail_starts.extend(ts for ts, _, st, _ in spans if st == REDUCE_STAGE)
    tail_start = min(tail_starts) if tail_starts else math.inf
    streaming, comm_reduce = {}, 0
    for (u, k), is_comm in comm.items():
        if not is_comm:
            continue
        reds = [(ts, dur) for ts, dur, st, _ in sw["unit_tracks"][u][k] if st == REDUCE_STAGE]
        comm_reduce += len(reds)
        if unit_mid[u] is None:
            continue
        m_start, m_end = unit_mid[u]
        hits = [(ts, dur) for ts, dur in reds if m_start <= ts < m_end and ts < tail_start]
        if hits:
            streaming[(u, k)] = hits
    mids = [m for m in unit_mid if m is not None]
    return dict(
        tail_combine_start_us=None if math.isinf(tail_start) else tail_start,
        middle_window_us=[min(a for a, _ in mids), max(b for _, b in mids)] if mids else None,
        comm_tracks_unit0=sorted(k for (u, k), c in comm.items() if u == 0 and c),
        comm_reduce_slices=comm_reduce,
        streaming_slices=sum(len(v) for v in streaming.values()),
        streaming_units=len({u for u, _ in streaming}),
        streaming_tracks=len(streaming),
        first_streaming_ts_us=min((ts for v in streaming.values() for ts, _ in v), default=None),
        unit0_streaming_tracks=sorted(k for (u, k) in streaming if u == 0),
        unit0_streaming_slices=sum(len(v) for (u, _), v in streaming.items() if u == 0),
    )


def streaming_label(label: str) -> str:
    """'W02..W03 (Dispatch)' -> 'W02..W03 (Dispatch -> Streaming Reduce)'."""
    if label.endswith(")"):
        return f"{label[:-1]} -> {STREAMING_REDUCE_ROLE})"
    return f"{label} -> {STREAMING_REDUCE_ROLE}"


def _track_role(cfg, spans, is_comm) -> str:
    if cfg["c_layout"] in ("groups", "per_wave"):
        return cfg["c_group_roles"]["comm" if is_comm else "compute"]
    tokens = []
    for _, _, _, nm in spans:
        tok = SM103_ROLE_FAMILIES.get(_family(nm))
        if tok and tok not in tokens:
            tokens.append(tok)
    if tokens:
        return " / ".join(tokens)
    return REDUCE_STAGE if any(st == REDUCE_STAGE for _, _, st, _ in spans) else spans[0][2]


def panel_c_layout(cfg, sw, red) -> dict:
    """Panel C lanes (one per real SM 0 / CU 0 track) grouped into labeled role bands."""
    tracks0 = sw["unit_tracks"][0]
    stream0 = set(red["unit0_streaming_tracks"])
    comm0 = set(red["comm_tracks_unit0"])
    if cfg["c_layout"] == "fixed":
        lanes = [k for k, _ in cfg["c_rows"]]
        hidden = sorted(set(tracks0) - set(lanes))
        if hidden:
            raise TraceRedLineError("sw_track_hidden", f"{cfg['platform']}: Panel C has no lane for SM 0 "
                                                       f"track(s) {hidden}; their slices would be hidden")
        bands = [dict(first=i, last=i, keys=[k], streaming=k in stream0,
                      label=streaming_label(lab) if k in stream0 else lab)
                 for i, (k, lab) in enumerate(cfg["c_rows"])]
        return dict(lanes=lanes, bands=bands)
    lanes = sorted(tracks0)
    bands = []
    per_wave = cfg["c_layout"] == "per_wave"   # EP16: one labeled row per wave, never merged
    for i, k in enumerate(lanes):
        role = _track_role(cfg, tracks0[k], k in comm0)
        if cfg.get("ep16_tail_combine_label") and (k in comm0):
            role = "Comm / Scatter & Combine"
        elif k in stream0:
            role = f"{role} -> {STREAMING_REDUCE_ROLE}"
        if not per_wave and bands and bands[-1]["role"] == role and bands[-1]["keys"][-1] == k - 1:
            bands[-1]["keys"].append(k)
            bands[-1]["last"] = i
        else:
            bands.append(dict(first=i, last=i, keys=[k], role=role, streaming=k in stream0))
    p = cfg["c_prefix"]
    for b in bands:
        a, z = b["keys"][0], b["keys"][-1]
        rng = f"{p}{a:02d}" if a == z else f"{p}{a:02d}..{p}{z:02d}"
        b["label"] = f"{rng} ({b['role']})"
    return dict(lanes=lanes, bands=bands)


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw - 1e-12:
            return m * mag
    return 10.0 * mag


def _fmt_tick(t: float) -> str:
    return "%d" % round(t) if abs(t - round(t)) < 1e-9 else "%g" % t


def x_axis(cfg, sw_end: float, curve_ts_max: float) -> dict:
    x_end = max(sw_end, curve_ts_max)
    x_max = float(math.ceil(x_end)) if x_end > 0 else 1.0
    step = float(cfg.get("x_step") or _nice_step(x_max / 8.0))
    ticks = [round(i * step, 6) for i in range(int(math.floor(x_max / step + 1e-9)) + 1)]
    return dict(x_max=x_max, step=step, ticks=ticks, labels=[_fmt_tick(t) for t in ticks])


def panel_b_ticks(cfg):
    n = cfg["n_units"]
    ticks = sorted(set([round(i * (n - 1) / 7) for i in range(8)]))
    return ticks, ["%s%d" % (cfg["unit"], t) for t in ticks]


def panel_b_title(cfg):
    return "(B) Multi-CTA Execution Across All %d Physical %ss" % (cfg["n_units"], cfg["unit"])


def panel_b_ylabel(cfg):
    return "Physical %s (%s)" % (cfg["unit"], cfg["rank_label"])


def compose_banner(config_banner: str = "", repro_script: str = "") -> str:
    banner_text = (config_banner or "").strip()
    if (repro_script or "").strip():
        sep = "\n" if banner_text else ""
        banner_text = banner_text + sep + f"Reproduce: {repro_script.strip()}"
    return banner_text


def display_strings(cfg, data=None, banner_text: str = ""):
    """Every string the renderer draws for this trace, as (where, text) pairs."""
    data = data or {}
    hw, layout, xa = data.get("hw"), data.get("layout"), data.get("xaxis")
    out = [("suptitle", cfg["title"])]
    if banner_text:
        out.append(("banner", banner_text))
    out += [("panel A title", PANEL_A_TITLE), ("panel A y label", PANEL_A_YLABEL)]
    if hw:
        out += [("panel A legend", lab) for _, _, lab in panel_a_legend(cfg, hw)]
        note = panel_a_rdma_note(cfg, hw, data.get("doc"))
        if note:
            out.append(("panel A RDMA fabric note", note))
    else:
        out += [("panel A legend", lab) for _, lab in cfg["curves"].values()]
    out += [("panel A y tick", str(v)) for v in PANEL_A_YTICKS]
    out += [("panel B title", panel_b_title(cfg)), ("panel B y label", panel_b_ylabel(cfg))]
    out += [("panel B legend", c) for c in panel_b_legend_classes(cfg)]
    out += [("panel B y tick", lab) for lab in panel_b_ticks(cfg)[1]]
    out += [("panel C title", cfg["c_title"]), ("panel C y label", cfg["c_ylabel"]), ("x label", X_LABEL)]
    if layout:
        out += [("panel C track label", b["label"]) for b in layout["bands"]]
    out += [("panel C legend", st) for st in stage_palette(cfg)[0]]
    if xa:
        out += [("x tick", lab) for lab in xa["labels"]]
    return out


def forbidden_hits(text: str):
    s = str(text)
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        return []
    low = s.lower()
    return [f for f in FORBIDDEN if f.lower() in low]


def audit_forbidden(strings):
    for where, text in strings:
        hits = forbidden_hits(text)
        if hits:
            raise TraceRedLineError("forbidden_string", f"{where} {text!r} contains forbidden display string(s) {hits}")
    return len(strings)


def banner_env_mismatches(banner_text: str, doc):
    """WEAVE_* values stated in the banner that contradict the trace's recorded capture env (warning only)."""
    meta = _other_data(doc).get("meta")
    meta = meta if isinstance(meta, dict) else {}
    cap = meta.get("capture_meta") if isinstance(meta.get("capture_meta"), dict) else {}
    env = cap.get("env_cfg") if isinstance(cap.get("env_cfg"), dict) else meta.get("env_cfg")
    if not isinstance(env, dict) or not banner_text:
        return []
    stated = dict(re.findall(r"\b(WEAVE_[A-Z0-9_]+)=([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*)", banner_text))
    return [dict(knob=k, banner=v, trace=str(env[k])) for k, v in sorted(stated.items())
            if k in env and str(env[k]) != v]


def parse(cfg, doc=None) -> dict:
    """Load one trace and run every red line; returns the data dict consumed by render()."""
    d = doc if doc is not None else load_trace(cfg["trace"])
    if not isinstance(d, dict) or not isinstance(d.get("traceEvents"), list):
        raise TraceRedLineError("sw_stage_map", "not a Chrome-trace JSON object with a traceEvents list")
    evs = d["traceEvents"]
    bad = next((i for i, e in enumerate(evs) if not isinstance(e, dict)), None)
    if bad is not None:
        raise TraceRedLineError("sw_stage_map", f"traceEvents[{bad}] is not a JSON object")
    cfg = trace_cfg(cfg, d)
    pnames, tnames = trace_names(evs)
    # Panel B must name the rank the trace actually captured, not the preset's
    # reference rank.  An explicit --rank-label still wins.
    _meta = _other_data(d).get("meta")
    _meta = _meta if isinstance(_meta, dict) else {}
    if not cfg.get("rank_label_explicit") and isinstance(_meta.get("rank"), int):
        cfg["rank_label"] = "rank %d" % _meta["rank"]
    curves = collect_counters(evs, cfg)
    hw = check_hw_red_lines(d, cfg, curves, pnames, tnames)
    sw = parse_sw(d, cfg)
    check_alignment(cfg, hw, sw)
    verify_tracks(cfg, sw, tnames)
    red = analyze_reduce(cfg, sw, tnames)
    layout = panel_c_layout(cfg, sw, red)
    xa = x_axis(cfg, sw["sw_end"], hw["curve_ts_max"])
    return dict(curves=curves, hw=hw, reduce=red, layout=layout, xaxis=xa, tnames=tnames, pnames=pnames,
                doc=d, rank_label=cfg["rank_label"], **sw)


# ================================================================= drawing
def rects_by_stage(spans, y, h, classes=None):
    out = {st: [] for st in (STAGES if classes is None else classes)}
    for ts, dur, st in spans:
        out[st].append([(ts, y - h / 2), (ts + dur, y - h / 2), (ts + dur, y + h / 2), (ts, y + h / 2)])
    return out


def add_stage_collections(ax, rectmap, zbase=2, classes=None, colors=None):
    classes = STAGES if classes is None else classes
    colors = STAGE_COLOR if colors is None else colors
    for i, st in enumerate(classes):
        if not rectmap.get(st):
            continue
        pc = PolyCollection(rectmap[st], facecolors=colors[st], edgecolors="none",
                            linewidths=0, antialiased=False, rasterized=True, zorder=zbase + i)
        ax.add_collection(pc)


def _legend_order(handles, ncol):
    """Matplotlib fills legend columns first; reorder so the rows read left-to-right."""
    nrow = -(-len(handles) // ncol)
    grid = [handles[r * ncol:(r + 1) * ncol] for r in range(nrow)]
    return [grid[r][c] for c in range(ncol) for r in range(nrow) if c < len(grid[r])]


def _bbox_overlap(b1, b2, tol=0.4):
    return min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > tol and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > tol


def _panel_c_labels_fit(axC, renderer) -> bool:
    labs = [t for t in axC.get_yticklabels() if t.get_visible() and t.get_text().strip()]
    boxes = [t.get_window_extent(renderer) for t in labs]
    ylab = axC.yaxis.label.get_window_extent(renderer)
    if ylab.x0 < 1.0 or any(bb.x0 < 1.0 for bb in boxes):
        return False
    return not any(_bbox_overlap(b1, b2) for b1, b2 in itertools.combinations(boxes + [ylab], 2))


def render(cfg, data, out_path=None, config_banner: str = "", repro_script: str = ""):
    """Draw the 3-panel figure; out_path=None renders into memory (PNG bytes in the result)."""
    if plt is None:
        raise RuntimeError("matplotlib is required to draw; trace red lines do not need it")
    # parse() may run on a copy of cfg (the validator does this); honour the rank
    # label it resolved from the trace so Panel B names the captured rank.
    if data.get("rank_label"):
        cfg = dict(cfg, rank_label=data["rank_label"])
    if data.get("doc") is not None:          # same per-trace layout / palette parse() resolved
        cfg = trace_cfg(cfg, data["doc"])
    fabric = bool(cfg.get("fabric_palette"))
    classes, colors = stage_palette(cfg)
    if fabric:
        check_fabric_palette()
    banner_text = compose_banner(config_banner, repro_script)
    n_strings = audit_forbidden(display_strings(cfg, data, banner_text))   # pre-draw gate
    hw, layout, xa = data["hw"], data["layout"], data["xaxis"]
    x_max = xa["x_max"]
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
        "axes.titlesize": 12.5, "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9, "legend.fontsize": 9.5, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=FIG_SIZE_IN, dpi=FIG_DPI)
    fig.suptitle(cfg["title"], x=0.5, y=0.988, ha="center", va="top", fontsize=15, weight="bold")
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

    # ---- Panel A: aligned HW utilization curves (real counter samples only, see hw red lines)
    handles = []
    for role, src, lab in panel_a_legend(cfg, hw):
        pts = data["curves"][src]
        ts = [p[0] for p in pts]; vs = [p[1] for p in pts]
        ls = "--" if role == "active" else ("-." if role in ("comm_tx", "rdma_tx") else "-")
        axA.plot(ts, vs, color=CURVE_COLOR[role], lw=1.8, ls=ls, zorder=3)
        handles.append(Line2D([], [], color=CURVE_COLOR[role], lw=2.0, ls=ls, label=lab))
    axA.axhline(100.0, color="#9E9E9E", lw=0.8, ls=(0, (4, 3)), zorder=2)
    axA.set_ylim(0, 105)
    axA.set_yticks(PANEL_A_YTICKS)
    axA.set_ylabel(PANEL_A_YLABEL)
    axA.yaxis.grid(True, color="#ECEFF1", lw=0.6, zorder=0)
    axA.set_axisbelow(True)
    axA.set_xlim(0, x_max)
    leg_ncol = 3 if len(handles) > 4 else 2
    leg_fs = 8.4 if len(handles) >= 6 else 9.5
    title_pad = 50 if len(handles) >= 7 else 44
    axA.set_title(PANEL_A_TITLE, pad=title_pad)
    legA = axA.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="lower center",
                      ncol=leg_ncol, frameon=False, handlelength=2.2, columnspacing=1.5,
                      handletextpad=0.55, borderaxespad=0.0, fontsize=leg_fs)
    # EP16 without a collectable Inter-Node RDMA NIC counter: say so on the panel
    # instead of leaving the second fabric unmentioned (never a zero curve).
    note_artist = None
    rdma_note = panel_a_rdma_note(cfg, hw, data.get("doc"))
    if rdma_note:
        if "\n" in rdma_note:
            raise RuntimeError("Panel A Inter-Node RDMA footnote must be exactly one line")
        note_artist = axA.text(0.012, 0.5, rdma_note, transform=axA.transAxes, ha="left", va="center",
                               fontsize=8.4, style="italic", color="#7F1D1D", zorder=6,
                               bbox=dict(boxstyle="square,pad=0.30", facecolor="white",
                                         edgecolor="none", alpha=0.9))

    # ---- Panel B: all physical SMs / CUs (EP16: colored by fabric class)
    n = cfg["n_units"]
    bm = {st: [] for st in classes}
    for i, spans in enumerate(data["unit_spans"]):
        y = n - 1 - i
        rm = rects_by_stage([(t, d, span_class(cfg, s, str(e.get("name", "")))) for t, d, s, e in spans],
                            y, 0.86, classes)
        for st in classes:
            bm[st].extend(rm[st])
    add_stage_collections(axB, bm, classes=classes, colors=colors)
    used_b = {st for st in classes if bm[st]}
    axB.set_ylim(-0.6, n - 0.4)
    b_ticks, b_labels = panel_b_ticks(cfg)
    axB.set_yticks([n - 1 - t for t in b_ticks])
    axB.set_yticklabels(b_labels, fontsize=8.5)
    axB.set_ylabel(panel_b_ylabel(cfg))
    # EP16 shares the Panel B title row with the fabric legend: sit both lower so
    # neither touches Panel A's x axis.
    if fabric:
        axB.set_title(panel_b_title(cfg), pad=4, loc="left", fontsize=11.5)
    else:
        axB.set_title(panel_b_title(cfg), pad=10)
    axB.tick_params(axis="y", length=2, width=0.6, pad=2)

    # ---- Panel C: representative unit 0, one lane per real track, labeled role bands
    lanes, bands = layout["lanes"], layout["bands"]
    tracks0 = data["unit_tracks"][0]
    nl = len(lanes)
    h = 0.72 if nl > 10 else 0.62
    min_w = REDUCE_MIN_DRAW_PX * x_max / (WIDTH * FIG_SIZE_IN[0] * FIG_DPI)
    cm = {st: [] for st in STAGES}
    cm_f, cm_fr = {c: [] for c in classes}, {c: [] for c in classes}   # fabric: body / reduce rects
    reduce_probe = []   # (x0, x1, y, legend hex) of every drawn Panel C Combine / Reduce rectangle
    n_reduce0 = 0
    for li, k in enumerate(lanes):
        y = nl - 1 - li
        for ts, dur, st, nm in tracks0.get(k, []):
            cls = span_class(cfg, st, nm)
            if st == REDUCE_STAGE:
                n_reduce0 += 1
                w = max(dur, min_w)
                x0 = min(ts, x_max - w)
                rect = [(x0, y - h / 2), (x0 + w, y - h / 2), (x0 + w, y + h / 2), (x0, y + h / 2)]
                (cm_fr[cls] if fabric else cm[st]).append(rect)
                reduce_probe.append((x0, x0 + w, y, colors[cls]))
            else:
                rect = [(ts, y - h / 2), (ts + dur, y - h / 2), (ts + dur, y + h / 2), (ts, y + h / 2)]
                (cm_f[cls] if fabric else cm[st]).append(rect)
    if fabric:   # every Combine / Reduce rect (either fabric) above every non-reduce rect
        add_stage_collections(axC, cm_f, classes=classes, colors=colors)
        add_stage_collections(axC, cm_fr, zbase=2 + len(classes), classes=classes, colors=colors)
        used_c = {c for c in classes if cm_f[c] or cm_fr[c]}
    else:
        add_stage_collections(axC, cm)   # Combine / Reduce is the last legend stage -> topmost zorder
        used_c = {st for st in STAGES if cm[st]}
    if any(b["last"] > b["first"] for b in bands):   # thin separators only when lanes are grouped
        for b in bands[1:]:
            axC.axhline(nl - 1 - b["first"] + 0.5, color="#CFD8DC", lw=0.6, zorder=1)
    axC.set_ylim(-0.6, nl - 0.4)
    axC.set_yticks([nl - 1 - (b["first"] + b["last"]) / 2.0 for b in bands])
    c_font = 8.2 if nl > 10 else 9.0
    axC.set_yticklabels([b["label"] for b in bands], fontsize=c_font)
    axC.set_ylabel(cfg["c_ylabel"])
    axC.set_title(cfg["c_title"], pad=30)
    axC.tick_params(axis="y", length=2, width=0.6, pad=3)
    axC.set_xlabel(X_LABEL)
    for ax in (axA, axB, axC):
        ax.xaxis.set_major_locator(FixedLocator(xa["ticks"]))
    axC.set_xticks(xa["ticks"])
    axC.set_xticklabels(xa["labels"])
    axC.xaxis.grid(True, color="#F5F5F5", lw=0.5, zorder=0)
    axC.set_axisbelow(True)
    stage_handles = [Patch(facecolor=colors[st], edgecolor="none", label=st) for st in classes]
    if fabric:   # one row: Inter-Node RDMA | Intra-Node xGMI | GEMM | Route | prep, side by side
        legC = axC.legend(handles=stage_handles, bbox_to_anchor=(0.5, 1.02), loc="lower center",
                          ncol=len(classes), frameon=False, handlelength=1.4, columnspacing=1.0,
                          handletextpad=0.45, borderaxespad=0.0, fontsize=9.0)
    else:
        legC = axC.legend(handles=stage_handles, bbox_to_anchor=(0.5, 1.02), loc="lower center",
                          ncol=7, frameon=False, handlelength=1.4, columnspacing=1.2,
                          handletextpad=0.5, borderaxespad=0.0)
    legB = None
    if fabric:   # Panel B's own fabric legend: title left, four fabric swatches right
        b_handles = [Patch(facecolor=colors[c], edgecolor="none", label=c) for c in panel_b_legend_classes(cfg)]
        legB = axB.legend(handles=b_handles, bbox_to_anchor=(1.0, 1.0), loc="lower right",
                          ncol=len(b_handles), frameon=False, handlelength=1.4, columnspacing=1.0,
                          handletextpad=0.45, borderaxespad=0.15, fontsize=9.0)

    for ax in (axA, axB):
        ax.tick_params(axis="x", labelbottom=False)

    # Panel C role labels must fit left of the axes: first wrap long multi-lane band labels
    # before their "->", then shrink the tick font if they still do not fit
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plain = [b["label"] for b in bands]
    if not _panel_c_labels_fit(axC, renderer):
        wrapped = [lab.replace(" -> ", "\n-> ", 1) if (b["last"] > b["first"] and " -> " in lab) else lab
                   for b, lab in zip(bands, plain)]
        if wrapped != plain:
            axC.set_yticklabels(wrapped, fontsize=c_font)
            fig.canvas.draw()
            if not _panel_c_labels_fit(axC, renderer):   # wrapping did not help: back to one line
                axC.set_yticklabels(plain, fontsize=c_font)
                fig.canvas.draw()
    while not _panel_c_labels_fit(axC, renderer) and c_font > 6.0 + 1e-9:
        c_font = round(c_font - 0.3, 2)
        for t in axC.get_yticklabels():
            t.set_fontsize(c_font)
        fig.canvas.draw()

    note_font = None
    if note_artist is not None:
        note_font = _fit_panel_a_note(axA, note_artist, data, fig, renderer)

    # ================= strict renderer gates =================
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        s = t.get_text()
        audit_forbidden([("drawn text", s)])
        bb = t.get_window_extent(renderer)
        if not (bb.x0 >= -0.5 and bb.y0 >= -0.5 and bb.x1 <= W + 0.5 and bb.y1 <= H + 0.5):
            raise RuntimeError(f"Clipped: {s!r}")
        boxes.append((s, bb))
    for (s1, b1), (s2, b2) in itertools.combinations(boxes, 2):
        if _bbox_overlap(b1, b2):
            raise RuntimeError(f"Text overlap: {s1!r} vs {s2!r}")
    bA, bB, bC = (ax.get_window_extent(renderer) for ax in (axA, axB, axC))
    lA, lC = legA.get_window_extent(renderer), legC.get_window_extent(renderer)
    tA, tC = axA.title.get_window_extent(renderer), axC.title.get_window_extent(renderer)
    if not lA.y0 >= bA.y1 - 0.5:
        raise RuntimeError("Panel A legend must sit strictly above Panel A axes")
    if not lA.y1 <= tA.y0 - 0.5:
        raise RuntimeError("Panel A legend overlaps its title")
    if not lC.y0 >= bC.y1 - 0.5:
        raise RuntimeError("Panel C legend must sit strictly above Panel C axes")
    if not lC.y1 <= tC.y0 - 0.5:
        raise RuntimeError("Panel C legend overlaps its title")
    if banner_artist is not None:
        bBan = banner_artist.get_window_extent(renderer)
        bSup = fig._suptitle.get_window_extent(renderer)
        if not bBan.y1 <= bSup.y0 - 0.5:
            raise RuntimeError("Config banner overlaps the main title")
        if not bBan.y0 >= tA.y1 + 0.5:
            raise RuntimeError("Config banner overlaps Panel A title")
    if not tC.y1 <= bB.y0 - 0.5:
        raise RuntimeError("Panel C title/legend intrudes into Panel B")
    # 100% of plotted categories are in the 7-item legend
    used = set(data["stage_counts"])
    if not used <= set(STAGES):
        raise RuntimeError(f"unlabeled plotted stages {used - set(STAGES)}")
    if {hd.get_label() for hd in stage_handles} != set(classes):
        raise RuntimeError(f"stage legend does not list exactly the {len(classes)} palette classes")
    if not (used_b | used_c) <= set(classes):
        raise RuntimeError(f"unlabeled plotted classes {(used_b | used_c) - set(classes)}")
    fabric_report = None
    if fabric:   # EP16 fabric gates: RDMA vs xGMI separate swatches on BOTH Panel B and Panel C
        lB = legB.get_window_extent(renderer)
        tB = axB._left_title.get_window_extent(renderer)
        if not lB.y0 >= bB.y1 - 0.5:
            raise RuntimeError("Panel B fabric legend must sit strictly above Panel B axes")
        if _bbox_overlap(lB, tB):
            raise RuntimeError("Panel B fabric legend overlaps the Panel B title")
        if not lB.y1 <= bA.y0 + 0.5:
            raise RuntimeError("Panel B fabric legend intrudes into Panel A")
        if not lB.x1 <= W + 0.5:
            raise RuntimeError("Panel B fabric legend is clipped")
        legend_hex = {}
        for where, leg in (("panel B", legB), ("panel C", legC)):
            got = {t.get_text(): h.get_facecolor() for t, h in zip(leg.get_texts(), leg.legend_handles)}
            for fab, pair in FABRIC_OF.items():
                for c in pair:
                    if c not in got:
                        raise RuntimeError(f"{where} legend has no {c!r} swatch")
                    if to_rgb(got[c]) != to_rgb(colors[c]):
                        raise RuntimeError(f"{where} legend swatch {c!r} is not {colors[c]}")
            legend_hex[where] = {c: colors[c] for c in got}
        rdma_hex = {colors[c].upper() for c in FABRIC_OF["rdma"]}
        xgmi_hex = {colors[c].upper() for c in FABRIC_OF["xgmi"]}
        fabric_report = dict(palette=dict(colors), min_rgb_channel_dist=check_fabric_palette(),
                             rdma_hex=sorted(rdma_hex), xgmi_hex=sorted(xgmi_hex),
                             rdma_xgmi_disjoint=not (rdma_hex & xgmi_hex),
                             panel_b_legend=[t.get_text() for t in legB.get_texts()],
                             panel_c_legend=[t.get_text() for t in legC.get_texts()],
                             panel_b_classes_drawn=sorted(used_b), panel_c_classes_drawn=sorted(used_c),
                             legend_hex=legend_hex)
    # Panel C Combine / Reduce: none dropped, none hidden, crisp purple on the rendered pixels
    if len(reduce_probe) != n_reduce0 or n_reduce0 != sum(
            1 for spans in tracks0.values() for _, _, st, _ in spans if st == REDUCE_STAGE):
        raise TraceRedLineError("sw_reduce_dropped", f"{cfg['platform']}: Panel C drew {len(reduce_probe)} of "
                                                     f"{n_reduce0} SM 0 Combine / Reduce slices")
    buf = np.asarray(fig.canvas.buffer_rgba())
    Hp, Wp = buf.shape[0], buf.shape[1]
    hidden = []
    for x0, x1, y, hexc in reduce_probe:
        target = np.array([int(round(c * 255)) for c in to_rgb(hexc)])
        px, py = axC.transData.transform(((x0 + x1) / 2.0, y))
        col = min(max(int(math.floor(px)), 0), Wp - 1)
        row = min(max(Hp - 1 - int(math.floor(py)), 0), Hp - 1)
        if int(np.abs(buf[row, col, :3].astype(int) - target).max()) > 2:
            hidden.append(((x0 + x1) / 2.0, y, tuple(int(v) for v in buf[row, col, :3])))
    if hidden:
        raise TraceRedLineError("sw_reduce_hidden", f"{cfg['platform']}: {len(hidden)} of {len(reduce_probe)} "
                                                    f"Panel C Combine / Reduce slices are not crisp "
                                                    f"{'fabric-colored' if fabric else 'purple'} on the "
                                                    f"rendered pixels, first {hidden[:3]}")

    target_io = io.BytesIO() if out_path is None else Path(out_path)
    if out_path is not None:
        target_io.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_io, dpi=FIG_DPI, format="png")
    plt.close(fig)
    if out_path is None:
        png = target_io.getvalue()
    else:
        with open(target_io, "rb") as fh:
            png = fh.read()
    w_px, h_px = struct.unpack(">II", png[16:24])
    if (w_px, h_px) != FIG_PX:
        raise RuntimeError(f"unexpected PNG size {(w_px, h_px)} (want {FIG_PX})")
    res = dict(png=None if out_path is None else str(out_path), png_bytes=len(png), px=[w_px, h_px],
               texts_audited=len(boxes), strings_preaudited=n_strings, panel_c_font=c_font,
               panel_a_rdma_note_font=note_font, panel_a_rdma_note=rdma_note, fabric=fabric_report,
               panel_c_reduce_rects=len(reduce_probe), panel_c_reduce_pixel_probes_ok=len(reduce_probe))
    if out_path is None:
        res["png_data"] = png
    return res


def curve_stats(cfg, data):
    return {role: dict(s) for role, s in data["hw"]["curves"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["sm103", "h20", "mi308x", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--stats-json", type=Path, default=None)
    ap.add_argument("--config-banner", default=None,
                    help="plain-language Spatial / Temporal / Implementation summary; "
                         "pass an empty string to suppress the banner box entirely")
    ap.add_argument("--repro-script", default=None,
                    help="path to the preset that produced the trace; an empty string suppresses the "
                         "Reproduce line (and therefore the whole banner box)")
    ap.add_argument("--no-banner", action="store_true",
                    help="draw no parameter banner box at all (clean title + Panels A/B/C only)")
    ap.add_argument("--trace", type=Path, default=None, help="override the trace for a single platform")
    # Per-capture overrides (single platform): a new capture can have its own title
    # (e.g. a different EP size or sequence length than the preset title), representative rank and tick step.
    # The time-axis end and the HW curve sample count are NOT overridable: this
    # renderer derives both from the trace and gates them with red lines.
    ap.add_argument("--title", default="", help="override the figure title")
    ap.add_argument("--rank-label", default="", help="override the Panel B rank label, e.g. 'rank 2'")
    ap.add_argument("--x-step", type=float, default=0.0, help="override the time-axis tick step (us)")
    args = ap.parse_args(argv)
    keys = ["sm103", "h20", "mi308x"] if args.platform == "all" else [args.platform]
    if args.trace is not None and len(keys) != 1:
        raise SystemExit("--trace requires one concrete --platform")
    if (args.title or args.rank_label or args.x_step > 0) and len(keys) != 1:
        raise SystemExit("--title/--rank-label/--x-step require one concrete --platform")
    all_stats, failures = {}, {}
    for k in keys:
        cfg = platform_cfg(k, trace=args.trace)
        if args.title:
            cfg["title"] = args.title
        if args.rank_label:
            cfg["rank_label"] = args.rank_label
            cfg["rank_label_explicit"] = True
        if args.x_step > 0:
            cfg["x_step"] = args.x_step
        cfg["trace"] = resolve_trace(cfg)
        # None = preset banner; "" = deliberately suppress the banner box.
        if args.no_banner:
            banner, repro = "", ""
        else:
            banner = SHARED_BANNERS[k] if args.config_banner is None else args.config_banner
            repro = REPRO_SCRIPTS[k] if args.repro_script is None else args.repro_script
        try:
            data = parse(cfg)
            out = args.out_dir / cfg["out"]
            r = render(cfg, data, out, banner, repro)
        except TraceRedLineError as exc:
            failures[k] = dict(rule=exc.rule, message=str(exc))
            print(f"[{k}] RED LINE REJECTED: {exc}", file=sys.stderr)
            continue
        warns = banner_env_mismatches(compose_banner(banner, repro), data["doc"])
        for w in warns:
            print(f"[{k}] WARNING banner states {w['knob']}={w['banner']} but the trace capture env records "
                  f"{w['knob']}={w['trace']}", file=sys.stderr)
        sha = hashlib.sha256(cfg["trace"].read_bytes()).hexdigest()
        red = data["reduce"]
        st = dict(platform=k,
                  source_path_b64=base64.b64encode(str(cfg["trace"]).encode("utf-8")).decode("ascii"),
                  source_sha256=sha,
                  source_bytes=cfg["trace"].stat().st_size,
                  sw_window_us=round(data["sw_end"], 3), x_max=data["xaxis"]["x_max"],
                  units=cfg["n_units"], panel_c_lanes=len(data["layout"]["lanes"]),
                  panel_c_tracks=[b["label"] for b in data["layout"]["bands"]],
                  hw_tool=data["hw"]["tool_text"], hw_evidence=data["hw"]["evidence"],
                  curves=curve_stats(cfg, data),
                  reduce=red,
                  banner_env_mismatches=warns,
                  plotted_stage_counts=dict(data["stage_counts"]),
                  plotted_total=sum(data["stage_counts"].values()),
                  dropped_wait_counts=dict(data["drop_counts"]),
                  dropped_total=sum(data["drop_counts"].values()), **r)
        all_stats[k] = st
        print(f"[{k}] {out}  {r['png_bytes']} bytes  {r['px']}  plotted={st['plotted_total']} "
              f"dropped={st['dropped_total']}  streaming_reduce_slices={red['streaming_slices']}")
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(all_stats)
        if failures:
            payload["red_line_failures"] = failures
        args.stats_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"stats -> {args.stats_json}")
    if failures:
        raise SystemExit(f"red-line rejected platform(s): {', '.join(sorted(failures))}")


if __name__ == "__main__":
    main()
