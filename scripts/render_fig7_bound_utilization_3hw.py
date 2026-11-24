#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper Fig. 7 style: bound-resource theoretical-peak utilization (%) of every system.

Layout (identical grid to the Fig. 7 speedup figure ``weave_eval_3hw_speedup.png``):
  3 rows (8x SM103 EP8 / 8x H20 EP8 / 16x MI308X 2-node EP16) x 6 models
  (paper Table III) x 5 token counts S = 16, 128, 1K, 8K, 16K; every subplot is
  a curve panel (one curve per measured system) plus a bottom heat-strip with
  Weave's own utilization.

Metric (per cell = (hardware, model, S)):
    T_bound_us   = max over ranks r of max(T_compute_us[r], T_memory_us[r], T_interconnect_us[r])
    util_pct[s]  = 100 * T_bound_us / T_s_us          (T_s_us = measured latency of system s)
so util_pct["Weave"] / util_pct[fastest baseline] == fastest_baseline_us / weave_us,
i.e. exactly the Fig. 7 speedup.  0 < util_pct <= 100 is asserted for every
system in every one of the 90 cells.

Two modes:
  python scripts/render_fig7_bound_utilization_3hw.py            # render from the committed ledger only
  python scripts/render_fig7_bound_utilization_3hw.py --rebuild  # rebuild the ledger from the read-only
                                                                 # paper data tree, then render
The ledger ``results/bound_utilization_90cells_3hw.json`` is self-contained:
re-rendering needs no external path.  ``--copy-to PATH`` additionally copies the PNG.

Data provenance (all measured; nothing is estimated or simulated):
  * latencies: the exact same files and selection rules as the paper Fig. 7
    renderer (``render_fig1_speedup.py``: load_b300 / load_h20 / load_mi308x_ep16);
  * rows 1-2 bounds: per-cell T_compute / T_memory / T_interconnect / T_bound /
    bound_resource from ``weave_roofline_90cells.csv`` (hardware keys of that CSV:
    the SM103 rows are stored there under the legacy key of the 148-SM part);
  * row 3 bound: computed here on the same ShareGPT ``*_S262144.pt`` routing,
    sliced to [:S], over WORLD = 16 ranks (2 nodes x 8 GPUs).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import os
import pathlib
import pickle
import shutil
import statistics
import sys
import zipfile

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "bound_utilization_90cells_3hw.json"
OUT_PNG = ROOT / "docs" / "figures" / "weave_bound_utilization_3hw.png"
OUT_PDF = ROOT / "docs" / "figures" / "weave_bound_utilization_3hw.pdf"

# ----------------------------------------------------------------- read-only sources (--rebuild only)
PAPER_DATA = pathlib.Path(os.environ.get("WEAVE_PAPER_DATA_DIR", "paper_data"))
B300_DIR = PAPER_DATA / "b300_ours_vs_dg_20261005"
H20_DIR = PAPER_DATA / "h20_moe_b641_20261117"
MI308X_DIR = PAPER_DATA / "mi308x_ep16_v33_1stage_20261119"
ROOFLINE_CSV = PAPER_DATA / "baseline_roofline_3hw" / "weave_roofline_90cells.csv"
ROUTING_DIR = PAPER_DATA / "routing_ep8_s262144"

# ----------------------------------------------------------------- figure constants (from Fig. 7)
NAME = {"qwen38max": "Qwen3.8-2.4T", "minimax_m3": "MiniMax-M3", "qwen35_397b": "Qwen3.5-397B",
        "qwen35_122b": "Qwen3.5-122B", "dskv4_flash": "DSKv4-Flash", "qwen36_35b": "Qwen3.5-35B"}
# (paper name, routing stem, E, K, H, I)  -- paper Table III
MODEL_SPECS = [
    ("Qwen3.8-2.4T", "qwen38max", 512, 10, 8192, 2048),
    ("MiniMax-M3", "minimax_m3", 128, 4, 6144, 3072),
    ("Qwen3.5-397B", "Qwen3.5-397B-A17B", 512, 10, 4096, 1024),
    ("Qwen3.5-122B", "Qwen3.5-122B-A10B", 256, 8, 3072, 1024),
    ("DSKv4-Flash", "DeepSeek-V4-Flash", 256, 6, 4096, 2048),
    ("Qwen3.5-35B", "Qwen3.5-35B-A3B", 256, 8, 2048, 512),
]
MODELS = [m[0] for m in MODEL_SPECS]
S_VALUES = [16, 128, 1024, 8192, 16384]
S_LABELS = ["16", "128", "1K", "8K", "16K"]
HW_ORDER = ["SM103", "H20", "MI308X"]
HW_ROW_LABEL = {"SM103": "8× SM103\n(148-SM)", "H20": "8× H20\n(SM90)", "MI308X": "16× MI308X\n(CDNA3)"}
DRAW_ORDER = {
    "SM103": ["NanoFlow", "FlashMoE", "mKernel", "SGLang", "FlashInfer", "DeepGEMM-MegaMoE", "Weave"],
    "H20": ["FlashInfer", "DeepGEMM-MegaMoE", "Weave"],
    "MI308X": ["vLLM", "AITER-1stage", "Weave"],
}
LEGEND_SYSTEMS = ["Weave", "DeepGEMM-MegaMoE", "FlashInfer", "SGLang", "AITER-1stage", "vLLM",
                  "mKernel", "FlashMoE", "NanoFlow"]
# (color, marker, markersize, linewidth, zorder, markerfacecolor) -- Fig. 7 STYLE
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
WEAVE_COLOR = "#0072B2"
# Weave point marker = the cell's bound resource
BOUND_MARKER = {
    "memory":       dict(marker="o", markersize=3.8, markerfacecolor="white", markeredgecolor=WEAVE_COLOR,
                         markeredgewidth=1.15),
    "interconnect": dict(marker="^", markersize=4.3, markerfacecolor=WEAVE_COLOR, markeredgecolor=WEAVE_COLOR,
                         markeredgewidth=0.6),
    "compute":      dict(marker="s", markersize=3.6, markerfacecolor=WEAVE_COLOR, markeredgecolor=WEAVE_COLOR,
                         markeredgewidth=0.6),
}
BOUND_LABEL = {"memory": "HBM-bound", "interconnect": "Comm-bound", "compute": "Compute-bound"}
FORBIDDEN = ["B300", "L20D", "dynamicMK", "v33", "v28", "v70", "v58", "v1864", "D7", "AGY", "ours_tuned",
             "all16", "megamoe4p4", "Residency"]

# ----------------------------------------------------------------- hardware peaks
HW_SPEC = {
    "SM103": dict(ep=8, world=8, nodes=1, label="8x SM103 (148-SM), EP=8",
                  compute_tflops=9852.0, compute_dtype="FP4 (FP8xNVFP4)", hbm_gbps=7680.0, hbm_kind="HBM3e",
                  interconnect="NVLink", interconnect_gbps=780.0,
                  bound_source="weave_roofline_90cells.csv (rows with the legacy SM103 hardware key)"),
    "H20": dict(ep=8, world=8, nodes=1, label="8x H20 (SM90), EP=8",
                compute_tflops=296.0, compute_dtype="FP8", hbm_gbps=4000.0, hbm_kind="HBM3",
                interconnect="NVLink", interconnect_gbps=321.4537855645814,
                bound_source="weave_roofline_90cells.csv (H20 rows)"),
    "MI308X": dict(ep=16, world=16, nodes=2, gpus_per_node=8, label="16x MI308X (CDNA3), 2-node EP=16",
                   compute_tflops=398.7, compute_dtype="FP8 Matrix Core (MI308X SKU)", hbm_gbps=3684.0, hbm_kind="HBM3",
                   interconnect="inter-node GDA RDMA (4x 400GbE per 8-GPU node) + intra-node xGMI",
                   rdma_gbps_per_gpu=25.0, xgmi_gbps=287.194329,
                   bound_source="computed in this script on ShareGPT S262144 routing sliced to [:S]"),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(vals):
    return math.exp(statistics.fmean(math.log(v) for v in vals))


# ================================================================= ledger rebuild (read-only sources)
_PT_TYPES = {"LongStorage": "int64", "IntStorage": "int32", "ShortStorage": "int16", "ByteStorage": "uint8",
             "CharStorage": "int8", "FloatStorage": "float32", "DoubleStorage": "float64",
             "HalfStorage": "float16", "BoolStorage": "bool"}


def load_pt(path: pathlib.Path) -> dict:
    """Torch-free .pt reader (zip + whitelisted unpickler; builds numpy arrays only, executes no code)."""
    import numpy as np

    def _rebuild(storage, offset, size, stride, *a):
        n = int(np.prod(size)) if len(size) else 1
        return storage[offset:offset + n].reshape(size)

    class _U(pickle.Unpickler):
        def __init__(self, f, z, root):
            super().__init__(f)
            self.z, self.root = z, root

        def find_class(self, mod, name):
            if name in _PT_TYPES:
                return name
            if name.startswith("_rebuild_tensor"):
                return _rebuild
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")

        def persistent_load(self, pid):
            _, typ, key, _, n = pid
            return np.frombuffer(self.z.read(f"{self.root}/data/{key}"), dtype=_PT_TYPES[typ], count=n)

    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        return _U(io.BytesIO(z.read(root + "/data.pkl")), z, root).load()


def _load_fi_json_records(path: pathlib.Path) -> dict:
    out = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        if r.get("error") or not r.get("y_finite", True):
            continue
        t_mega = r.get("t_mega_us_slow")
        if t_mega is None:
            continue
        t_red = r.get("t_reduce_us")
        tot = float(t_mega) + (float(t_red) if (t_red is not None and math.isfinite(float(t_red))) else 0.0)
        key = (NAME[r["model"]], int(r["S"]))
        if key not in out or tot < out[key]:
            out[key] = tot
    return out


def load_sm103_latencies() -> dict:
    """Identical to Fig. 7 ``load_b300()``."""
    c30 = {(NAME[r["model"]], int(r["S"])): r
           for r in csv.DictReader((B300_DIR / "all_systems_30cells.csv").open(encoding="utf-8", newline=""))}
    b30 = {(NAME[r["model"]], int(r["S"])): r
           for r in csv.DictReader((B300_DIR / "b300_30cell_roofline_and_baselines.csv").open(encoding="utf-8",
                                                                                                newline=""))}
    fi_pts = _load_fi_json_records(B300_DIR / "final_grid_fi_30pt.json")
    arch = json.loads((B300_DIR / "fi30_summary.json").read_text(encoding="utf-8"))["fi_official"]
    for k_str, rec in arch.items():
        m_raw, s_str = k_str.split(":")
        fi_pts.setdefault((NAME[m_raw], int(s_str)), float(rec["tot_slow_us"]))
    t = {}
    for m in MODELS:
        for s in S_VALUES:
            rc, rb = c30[(m, s)], b30[(m, s)]
            t[("Weave", m, s)] = float(rb["ours_nvfp4_geo_table_us"])
            t[("DeepGEMM-MegaMoE", m, s)] = float(rb["deepgemm_official_us"])
            t[("FlashInfer", m, s)] = float(fi_pts[(m, s)])
            t[("SGLang", m, s)] = float(rb["sglang_us"])
            t[("mKernel", m, s)] = float(rc["mkernel"])
            t[("FlashMoE", m, s)] = float(rb["flashmoe_us"])
            t[("NanoFlow", m, s)] = float(rb["nanoflow_us"])
    assert len(t) == 7 * 6 * 5
    return t


def load_h20_latencies() -> dict:
    """Identical to Fig. 7 ``load_h20()``."""
    col_map = {"Weave": "weave_us", "DeepGEMM-MegaMoE": "megamoe_us", "FlashInfer": "flashinfer_us"}
    t = {}
    with (H20_DIR / "h20_moe_layer_30cells.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            for sname, col in col_map.items():
                v = float(r[col])
                assert v > 0
                t[(sname, r["model"], int(r["S"]))] = v
    assert len(t) == 3 * 6 * 5
    return t


def load_mi308x_latencies() -> dict:
    """Identical to Fig. 7 ``load_mi308x_ep16()`` on the latest 16-rank FP64 verified
    2-node EP16 CSV (``results/mi308x_ep16_weave_best_30cells.csv``): AITER-1stage is
    excluded on MiniMax-M3 (AITER 0.1.11 1-stage ASM fails FP64 verification there),
    giving 25 + 30 + 30 cells."""
    col_map = {"Weave": "weave_us", "AITER-1stage": "base_1stage_us", "vLLM": "vllm_2stage_us"}
    csv_path = ROOT / "results" / "mi308x_ep16_weave_best_30cells.csv"
    t = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            for sname, col in col_map.items():
                if sname == "AITER-1stage" and r["model"] == "MiniMax-M3":
                    continue
                v = float(r[col])
                assert v > 0
                t[(sname, r["model"], int(r["S"]))] = v
    assert len(t) == 2 * 6 * 5 + 5 * 5
    return t


def mi308x_ep16_bound(idx_all, E: int, K: int, H: int, I: int, S: int) -> dict:
    """Per-rank roofline of one EP16 cell (2 nodes x 8 GPUs); returns the critical-rank record."""
    import numpy as np

    spec = HW_SPEC["MI308X"]
    world, gpn = spec["world"], spec["gpus_per_node"]
    epr = E // world
    idx = idx_all[:S].astype(np.int64)
    assert idx.shape == (S, K) and S % world == 0
    tpr = S // world
    owner = np.arange(S) // tpr                      # token -> owning (source) rank
    owner_node = owner // gpn
    dst = idx // epr                                  # (token, k) -> destination rank
    counts = np.bincount(idx.ravel(), minlength=E).reshape(world, epr)
    p = counts.sum(axis=1)                            # routed (token, expert) pairs per rank
    t = (counts > 0).sum(axis=1)                      # touched local experts per rank
    hit = np.zeros((S, world), dtype=bool)            # unique (token, destination rank)
    hit[np.arange(S)[:, None], dst] = True
    pairs = np.zeros((S, world), dtype=np.int64)      # (token, destination rank) pair multiplicity
    np.add.at(pairs, (np.repeat(np.arange(S), K), dst.ravel()), 1)

    disp_b = H + math.ceil(H / 128) * 4               # FP8 token + fp32 per-128 scale
    comb_b = 2 * H                                    # BF16 expert output
    rdma_bps = spec["rdma_gbps_per_gpu"] * 1e9
    xgmi_bps = spec["xgmi_gbps"] * 1e9
    peak_flops = spec["compute_tflops"] * 1e12
    hbm_bps = spec["hbm_gbps"] * 1e9
    ranks = []
    for r in range(world):
        node = r // gpn
        own = owner == r
        from_other_node = owner_node != node
        from_same_node_peer = (owner_node == node) & (owner != r)
        other_node_ranks = [q for q in range(world) if q // gpn != node]
        same_node_peers = [q for q in range(world) if q // gpn == node and q != r]
        # inter-node GDA RDMA (per-direction NIC wire peak, rank-pair deduplicated on Dispatch & Combine)
        rem_uniq_to_r = int(hit[from_other_node, r].sum())
        own_disp_to_other = int(hit[own][:, other_node_ranks].sum())
        rdma_rx = rem_uniq_to_r * disp_b + own_disp_to_other * comb_b
        rdma_tx = own_disp_to_other * disp_b + rem_uniq_to_r * comb_b
        # intra-node xGMI (same symmetric rank-pair accounting against same-node peers)
        loc_uniq_to_r = int(hit[from_same_node_peer, r].sum())
        own_disp_to_peer = int(hit[own][:, same_node_peers].sum())
        xgmi_rx = loc_uniq_to_r * disp_b + own_disp_to_peer * comb_b
        xgmi_tx = own_disp_to_peer * disp_b + loc_uniq_to_r * comb_b
        tc = float(p[r]) * 6.0 * H * I / peak_flops * 1e6
        tm = float(t[r]) * 3.0 * H * I / hbm_bps * 1e6
        trd = max(rdma_rx, rdma_tx) / rdma_bps * 1e6
        txg = max(xgmi_rx, xgmi_tx) / xgmi_bps * 1e6
        ranks.append(dict(rank=r, recv_pairs=int(p[r]), touched_experts=int(t[r]),
                          rdma_rx_bytes=rdma_rx, rdma_tx_bytes=rdma_tx, xgmi_rx_bytes=xgmi_rx, xgmi_tx_bytes=xgmi_tx,
                          T_compute_us=tc, T_memory_us=tm, T_rdma_us=trd, T_xgmi_us=txg,
                          T_interconnect_us=max(trd, txg), T_rank_us=max(tc, tm, trd, txg)))
    rc = max(range(world), key=lambda r: ranks[r]["T_rank_us"])
    c = ranks[rc]
    trio = [c["T_compute_us"], c["T_memory_us"], c["T_interconnect_us"]]
    bound = ["compute", "memory", "interconnect"][max(range(3), key=lambda i: trio[i])]
    return dict(critical_rank=rc, crit_recv_pairs=c["recv_pairs"], max_recv_pairs=int(p.max()),
                crit_touched_experts=c["touched_experts"],
                T_compute_us=c["T_compute_us"], T_memory_us=c["T_memory_us"],
                T_interconnect_us=c["T_interconnect_us"], T_rdma_us=c["T_rdma_us"], T_xgmi_us=c["T_xgmi_us"],
                interconnect_fabric="rdma" if c["T_rdma_us"] >= c["T_xgmi_us"] else "xgmi",
                crit_rdma_rx_bytes=c["rdma_rx_bytes"], crit_rdma_tx_bytes=c["rdma_tx_bytes"],
                crit_xgmi_rx_bytes=c["xgmi_rx_bytes"], crit_xgmi_tx_bytes=c["xgmi_tx_bytes"],
                T_bound_us=c["T_rank_us"], bound_resource=bound)


def rebuild_ledger() -> dict:
    sources = [B300_DIR / "all_systems_30cells.csv", B300_DIR / "b300_30cell_roofline_and_baselines.csv",
               B300_DIR / "final_grid_fi_30pt.json", B300_DIR / "fi30_summary.json",
               H20_DIR / "h20_moe_layer_30cells.csv",
               ROOT / "results" / "mi308x_ep16_weave_best_30cells.csv", ROOFLINE_CSV]
    sources += [ROUTING_DIR / f"{stem}_S262144.pt" for _, stem, *_ in MODEL_SPECS]
    lat = {"SM103": load_sm103_latencies(), "H20": load_h20_latencies(), "MI308X": load_mi308x_latencies()}
    roof = {}
    with ROOFLINE_CSV.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            hw = {"B300": "SM103", "H20": "H20"}.get(r["hardware"])
            if hw is not None:
                roof[(hw, r["model"], int(r["S"]))] = r
    cells = []
    for hw in HW_ORDER:
        for m, stem, E, K, H, I in MODEL_SPECS:
            idx_all = load_pt(ROUTING_DIR / f"{stem}_S262144.pt")["topk_idx"] if hw == "MI308X" else None
            for S in S_VALUES:
                cell = dict(hardware=hw, model=m, S=S, E=E, K=K, H=H, I=I)
                if hw == "MI308X":
                    cell.update(mi308x_ep16_bound(idx_all, E, K, H, I, S))
                else:
                    r = roof[(hw, m, S)]
                    assert (int(r["E"]), int(r["K"]), int(r["H"]), int(r["I"])) == (E, K, H, I), (hw, m, S)
                    cell.update(critical_rank=int(r["critical_rank"]), crit_recv_pairs=int(r["crit_recv_pairs"]),
                                max_recv_pairs=int(r["max_recv_pairs"]),
                                crit_touched_experts=int(r["crit_touched_experts"]),
                                T_compute_us=float(r["T_compute_us"]), T_memory_us=float(r["T_memory_us"]),
                                T_interconnect_us=float(r["T_interconnect_us"]), T_bound_us=float(r["T_bound_us"]),
                                bound_resource=r["bound_resource"])
                    # the roofline CSV carries its own copy of the Weave latency: it must be the Fig. 7 one
                    assert abs(float(r["weave_latency_us"]) - lat[hw][("Weave", m, S)]) < 5e-3, (hw, m, S)
                cell["latency_us"] = {s: lat[hw][(s, m, S)] for s in DRAW_ORDER[hw] if (s, m, S) in lat[hw]}
                cells.append(cell)
    ledger = dict(
        schema="weave.bound_utilization_90cells_3hw.v1",
        figure="docs/figures/weave_bound_utilization_3hw.png",
        metric="util_pct[system] = 100 * T_bound_us / measured_latency_us[system]; "
               "T_bound_us = max over ranks of max(T_compute_us, T_memory_us, T_interconnect_us)",
        identity="util_pct[Weave] / util_pct[fastest baseline] == fastest_baseline_us / weave_us == Fig. 7 speedup",
        mi308x_bound_model=dict(
            world=16, nodes=2, gpus_per_node=8, experts_per_rank="E // 16",
            token_ownership="contiguous S // 16 tokens per rank (rank r owns tokens [r*S/16, (r+1)*S/16))",
            T_compute_us="p[r] * 6 * H * I / 398.7e12 * 1e6  (p = routed (token, expert) pairs on rank r; MI308X SKU peak)",
            T_memory_us="t[r] * 3 * H * I / 3684.0e9 * 1e6  (t = touched local experts, FP8 weights)",
            dispatch_bytes="disp_b = H + ceil(H/128) * 4", combine_bytes="comb_b = 2 * H",
            rdma_rx_bytes="rem_uniq_to_r * disp_b + own_disp_to_other * comb_b",
            rdma_tx_bytes="own_disp_to_other * disp_b + rem_uniq_to_r * comb_b",
            T_rdma_us="max(rdma_rx_bytes, rdma_tx_bytes) / 25.0e9 * 1e6  (4x 400GbE per 8-GPU node)",
            T_xgmi_us="max(xgmi_rx_bytes, xgmi_tx_bytes) / 287.194329e9 * 1e6  (same accounting vs same-node peers)",
            T_interconnect_us="max(T_rdma_us, T_xgmi_us)",
            definitions="'other' = ranks on the other node; rem_uniq_to_r = unique (token, r) hits from other-node "
                        "tokens; own_disp_to_other = unique (own token, other-node rank) hits",
        ),
        hardware=HW_SPEC,
        sources=[dict(path=str(p), sha256=sha256(p)) for p in sources],
        latency_selection=dict(
            SM103="Fig. 7 load_b300(): Weave = ours_nvfp4_geo_table_us, DeepGEMM-MegaMoE = deepgemm_official_us, "
                  "FlashInfer = min over final_grid_fi_30pt.json (t_mega_us_slow + t_reduce_us) else "
                  "fi30_summary.json tot_slow_us, SGLang/FlashMoE/NanoFlow from b300_30cell_roofline_and_baselines.csv, "
                  "mKernel from all_systems_30cells.csv",
            H20="Fig. 7 load_h20(): h20_moe_layer_30cells.csv weave_us / megamoe_us / flashinfer_us",
            MI308X="Fig. 7 load_mi308x_ep16(): mi308x_ep16_30cells.csv weave_us / base_1stage_us (AITER-1stage, "
                   "excluded on MiniMax-M3: AITER 0.1.11 1-stage ASM fails FP64 verification) / vllm_2stage_us",
        ),
        cells=cells,
    )
    finalize(ledger)
    return ledger


# ================================================================= derived fields + gates (both modes)
def finalize(ledger: dict) -> None:
    """(Re)compute util_pct / speedup from T_bound and latencies and assert every invariant."""
    cells = ledger["cells"]
    assert len(cells) == 90, len(cells)
    seen = set()
    for c in cells:
        key = (c["hardware"], c["model"], c["S"])
        assert key not in seen, key
        seen.add(key)
        assert c["bound_resource"] in BOUND_MARKER, key
        trio = {"compute": c["T_compute_us"], "memory": c["T_memory_us"], "interconnect": c["T_interconnect_us"]}
        assert abs(max(trio.values()) - c["T_bound_us"]) <= 1e-3 * max(1.0, c["T_bound_us"]), key
        assert max(trio, key=trio.get) == c["bound_resource"], key
        lat = c["latency_us"]
        assert "Weave" in lat and len(lat) >= 2, key
        util = {}
        for s, us in lat.items():
            u = 100.0 * c["T_bound_us"] / us
            assert 0.0 < u <= 100.0, f"util out of (0, 100]: {key} {s} {u:.3f}%"
            util[s] = u
        if "util_pct" in c:      # ledger copy must equal the recomputation
            for s, u in util.items():
                assert abs(c["util_pct"][s] - u) < 1e-6, (key, s)
        c["util_pct"] = util
        base = {s: us for s, us in lat.items() if s != "Weave"}
        fastest = min(base, key=base.get)
        sp = base[fastest] / lat["Weave"]
        assert abs(util["Weave"] / util[fastest] - sp) < 1e-9, key
        c["fastest_baseline"] = fastest
        c["speedup_vs_fastest"] = sp
    assert {(h, m, s) for h in HW_ORDER for m in MODELS for s in S_VALUES} == seen
    n_sys = collections.Counter(c["hardware"] for c in cells for _ in c["latency_us"])
    assert n_sys == {"SM103": 7 * 30, "H20": 3 * 30, "MI308X": 2 * 30 + 25}, n_sys
    summary = {}
    for hw in HW_ORDER:
        hc = [c for c in cells if c["hardware"] == hw]
        uw = [c["util_pct"]["Weave"] for c in hc]
        per_sys = {}
        for s in DRAW_ORDER[hw]:
            us = [c["util_pct"][s] for c in hc if s in c["util_pct"]]
            per_sys[s] = dict(cells=len(us), geomean_pct=round(gm(us), 2), mean_pct=round(statistics.fmean(us), 2),
                              min_pct=round(min(us), 2), max_pct=round(max(us), 2))
        by_s = {}
        for S in S_VALUES:
            v = [c["util_pct"]["Weave"] for c in hc if c["S"] == S]
            by_s[str(S)] = dict(geomean_pct=round(gm(v), 2), min_pct=round(min(v), 2), max_pct=round(max(v), 2))
        best = max(hc, key=lambda c: c["util_pct"]["Weave"])
        summary[hw] = dict(
            weave=dict(geomean_pct=round(gm(uw), 2), mean_pct=round(statistics.fmean(uw), 2),
                       min_pct=round(min(uw), 2), max_pct=round(max(uw), 2),
                       max_cell=f'{best["model"]} S={best["S"]}', cells_ge_75pct=sum(u >= 75.0 for u in uw),
                       cells_ge_50pct=sum(u >= 50.0 for u in uw), by_S=by_s),
            per_system=per_sys,
            bound_resource_counts=dict(collections.Counter(c["bound_resource"] for c in hc)),
            weave_speedup_vs_fastest_geomean=round(gm([c["speedup_vs_fastest"] for c in hc]), 4),
            weave_highest_util_cells=sum(c["util_pct"]["Weave"] >= max(c["util_pct"].values()) for c in hc),
        )
    ledger["summary"] = summary


# ================================================================= rendering
def render(ledger: dict, out_png: pathlib.Path, out_pdf: pathlib.Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    cell = {(c["hardware"], c["model"], c["S"]): c for c in ledger["cells"]}
    assert "Arial" in {f.name for f in font_manager.fontManager.ttflist}, "Arial font required"
    plt.rcParams.update({
        "font.family": ["Arial"], "font.size": 6.6, "axes.titlesize": 7.2, "axes.labelsize": 6.4,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.0, "legend.fontsize": 6.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.55, "text.usetex": False,
    })
    fig = plt.figure(figsize=(8.6, 3.85))
    left_x, right_x = 0.104, 0.996
    rows = {"SM103": (0.830, 0.615), "H20": (0.566, 0.351), "MI308X": (0.302, 0.087)}
    x = list(range(len(S_VALUES)))
    titles, strip_texts = [], []

    for hw in HW_ORDER:
        top, bot = rows[hw]
        gs = fig.add_gridspec(2, 6, height_ratios=[1.50, 0.42], hspace=0.10, wspace=0.075,
                              left=left_x, right=right_x, top=top, bottom=bot)
        for ci, m in enumerate(MODELS):
            axt = fig.add_subplot(gs[0, ci])
            axb = fig.add_subplot(gs[1, ci], sharex=axt)
            axt.axhline(100.0, color="#666666", lw=0.6, ls="--", zorder=1)
            for sname in DRAW_ORDER[hw]:
                if any(sname not in cell[(hw, m, s)]["util_pct"] for s in S_VALUES):
                    continue          # e.g. AITER-1stage on MiniMax-M3 (fails FP64 verification)
                ys = [cell[(hw, m, s)]["util_pct"][sname] for s in S_VALUES]
                col, mk, ms, lw, z, mfc = STYLE[sname]
                if sname == "Weave":
                    axt.plot(x, ys, color=WEAVE_COLOR, lw=1.55, zorder=12, clip_on=False)
                    for xi, yi, s in zip(x, ys, S_VALUES):
                        axt.plot([xi], [yi], linestyle="None", zorder=13, clip_on=False,
                                 **BOUND_MARKER[cell[(hw, m, s)]["bound_resource"]])
                else:
                    axt.plot(x, ys, color=col, marker=mk, markersize=ms, linewidth=lw, markerfacecolor=mfc,
                             markeredgecolor=col, markeredgewidth=0.6, zorder=z, clip_on=False)
            axt.set_ylim(0.0, 105.0)
            axt.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
            axt.yaxis.set_major_formatter(FixedFormatter(["0%", "25%", "50%", "75%", "100%"] if ci == 0
                                                         else [""] * 5))
            axt.yaxis.set_minor_locator(NullLocator())
            axt.yaxis.grid(True, color="#E0E0E0", lw=0.40, zorder=0)
            axt.set_axisbelow(True)
            axt.set_xlim(-0.5, len(S_VALUES) - 0.5)
            axt.spines["left"].set_visible(False)
            axt.spines["bottom"].set_color("#888888")
            axt.tick_params(axis="x", labelbottom=False, length=1.6, width=0.5, color="#666666")
            axt.tick_params(axis="y", length=0.0, pad=1.5)
            if hw == "SM103":
                titles.append(axt.set_title(m, loc="center", pad=3.0, fontsize=7.2, weight="bold", color="#111111"))

            axb.set_xlim(-0.5, len(S_VALUES) - 0.5)
            axb.set_ylim(0.0, 1.0)
            for sp in axb.spines.values():
                sp.set_visible(False)
            axb.yaxis.set_major_locator(NullLocator())
            for xi, s in zip(x, S_VALUES):
                v = cell[(hw, m, s)]["util_pct"]["Weave"]
                bg, fg = cell_color(v)
                axb.add_patch(Rectangle((xi - 0.46, 0.06), 0.92, 0.88, facecolor=bg, edgecolor="none", zorder=2))
                strip_texts.append(axb.text(xi, 0.50, f"{v:.1f}", ha="center", va="center", fontsize=5.9, color=fg,
                                            weight="bold" if v >= 75.0 else "normal", zorder=4))
            axb.set_xticks(x)
            axb.set_xticklabels(S_LABELS if hw == "MI308X" else [""] * len(S_LABELS))
            axb.tick_params(axis="x", length=0.0, pad=1.8 if hw == "MI308X" else 0.0)

        y_curve_mid = bot + (top - bot) * 0.62
        fig.text(0.004, y_curve_mid + 0.040, HW_ROW_LABEL[hw], ha="left", va="center", fontsize=7.0,
                 weight="bold", color="#111111")
        fig.text(0.004, y_curve_mid - 0.042, "Bound Peak\nUtil. (%)", ha="left", va="center", fontsize=6.1,
                 color="#222222")
        fig.text(0.004, bot + (top - bot) * 0.11, "Weave (%)", ha="left", va="center", fontsize=6.1,
                 color="#222222")

    fig.text((left_x + right_x) * 0.5, 0.006,
             "Tokens per step (S)   ·   Bound Peak Util. (%) = min-time of bottleneck resource "
             "(HBM / Comm / Compute) / Measured latency × 100%",
             ha="center", va="bottom", fontsize=6.3, color="#222222")

    sys_handles = []
    for s in LEGEND_SYSTEMS:
        col, mk, ms, lw, _, mfc = STYLE[s]
        if s == "Weave":
            sys_handles.append(Line2D([0], [0], color=WEAVE_COLOR, lw=1.75))
        else:
            sys_handles.append(Line2D([0], [0], color=col, marker=mk, markersize=ms + 0.6, lw=lw + 0.15,
                                      markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=0.65))
    leg_sys = fig.legend(sys_handles, LEGEND_SYSTEMS, loc="upper center", ncol=9, frameon=False,
                         bbox_to_anchor=(0.55, 0.998), fontsize=6.6, handlelength=1.6, columnspacing=1.05,
                         handletextpad=0.35, borderaxespad=0.0)
    b_handles = [Line2D([0], [0], linestyle="None", marker="None")]
    b_handles += [Line2D([0], [0], linestyle="None",
                         **dict(BOUND_MARKER[b], markersize=BOUND_MARKER[b]["markersize"] + 0.8))
                  for b in ("memory", "interconnect", "compute")]
    b_labels = ["Weave point marker = bottleneck resource of the cell:", "HBM-bound", "Comm-bound", "Compute-bound"]
    leg_bnd = fig.legend(b_handles, b_labels, loc="upper center", ncol=4, frameon=False,
                         bbox_to_anchor=(0.55, 0.944), fontsize=6.6, handlelength=1.0, columnspacing=1.6,
                         handletextpad=0.35, borderaxespad=0.0)
    leg_bnd.get_texts()[0].set_color("#333333")
    leg_bnd.legend_handles[0].set_visible(False)

    # ------------------------------------------------------------- programmatic gates
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    vis = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip():
            continue
        for f in FORBIDDEN:
            assert f.lower() not in s.lower(), f"forbidden {f!r} in drawn text {s!r}"
        bb = t.get_window_extent(rnd)
        assert (bb.x0 >= -0.5 and bb.y0 >= -0.5 and bb.x1 <= fig.bbox.width + 0.5
                and bb.y1 <= fig.bbox.height + 0.5), f"text outside canvas: {s!r} {bb}"
        vis.append((s, bb))
    overlaps = [(a, b) for (a, ba), (b, bb) in itertools.combinations(vis, 2)
                if min(ba.x1, bb.x1) - max(ba.x0, bb.x0) > 0.4 and min(ba.y1, bb.y1) - max(ba.y0, bb.y0) > 0.4]
    assert not overlaps, f"text overlaps: {overlaps[:5]}"
    b_sys, b_bnd = leg_sys.get_window_extent(rnd), leg_bnd.get_window_extent(rnd)
    t_top = max(t.get_window_extent(rnd).y1 for t in titles)
    ax_top = max(ax.get_window_extent(rnd).y1 for ax in fig.axes)
    assert b_sys.y0 >= b_bnd.y1 + 0.5, f"system legend overlaps bound legend: {b_sys} vs {b_bnd}"
    assert b_bnd.y0 >= t_top + 0.5, f"bound legend overlaps column titles: {b_bnd.y0} vs {t_top}"
    assert b_bnd.y0 >= ax_top + 0.5, "legends must sit strictly above every subplot"
    assert len(strip_texts) == 90
    print(f"gates: {len(vis)} visible texts, 0 overlaps, 0 clipped, 0 forbidden strings; legends above titles")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
        raw = out_pdf.read_bytes()
        assert b"/Type3" not in raw and (b"/CIDFontType2" in raw or b"/TrueType" in raw), "PDF must embed Type 42 fonts"
    plt.close(fig)


def cell_color(v: float) -> tuple[str, str]:
    if v >= 75.0:
        return "#1E6B9E", "white"
    if v >= 55.0:
        return "#6BAED6", "#111111"
    if v >= 40.0:
        return "#9ECAE1", "#111111"
    if v >= 25.0:
        return "#C6DBEF", "#111111"
    if v >= 12.0:
        return "#DEEBF7", "#111111"
    return "#EDF4FA", "#111111"


def print_summary(ledger: dict) -> None:
    for hw in HW_ORDER:
        s = ledger["summary"][hw]
        w = s["weave"]
        print(f"{hw:6s}: Weave bound-peak util GM={w['geomean_pct']:.2f}% mean={w['mean_pct']:.2f}% "
              f"range=[{w['min_pct']:.2f}, {w['max_pct']:.2f}]% (max at {w['max_cell']}); "
              f"bound={s['bound_resource_counts']}; GM speedup vs fastest={s['weave_speedup_vs_fastest_geomean']:.4f}x; "
              f"Weave highest util in {s['weave_highest_util_cells']}/30 cells")
        for name, ps in s["per_system"].items():
            if name != "Weave":
                print(f"         {name:17s} GM={ps['geomean_pct']:6.2f}%  range=[{ps['min_pct']:.2f}, {ps['max_pct']:.2f}]% "
                      f"({ps['cells']} cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true", help="rebuild the ledger from the read-only paper data tree")
    ap.add_argument("--copy-to", type=pathlib.Path, default=None, help="also copy the PNG to this path")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    if args.rebuild:
        ledger = rebuild_ledger()
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    finalize(ledger)            # recompute + re-assert every invariant from the committed ledger
    print_summary(ledger)
    render(ledger, OUT_PNG, None if args.no_pdf else OUT_PDF)
    print(f"wrote {OUT_PNG} (300 DPI)" + ("" if args.no_pdf else f" and {OUT_PDF} (Type 42)"))
    if args.copy_to is not None:
        shutil.copyfile(OUT_PNG, args.copy_to)
        print(f"copied to {args.copy_to}")


if __name__ == "__main__":
    main()
