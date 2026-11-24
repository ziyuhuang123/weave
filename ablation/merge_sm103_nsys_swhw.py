#!/usr/bin/env python3
"""Merge rank-0 SM103 tile-profiler SW records with same-run nsys HW counters.

SW: the DG_MEGA_TILE_PROF uint4 records (stage_id, t_start_ns, t_end_ns, tile_id)
    laid out at word 1024 + (sm * 20 + warp) * 2048 + rec_idx.
HW: nsys --gpu-metrics-set=gb11x GPU_METRICS samples inside the same-run
    NVTX-bracketed mega_moe window.

Curve rules (Panel A):
  hbm      = DRAM Read Bandwidth % + DRAM Write Bandwidth % per timestamp.  HBM
             is one shared half-duplex DRAM interface, so the two directions
             compete for the same bandwidth and are summed (clamped to 100).
  comm RX  = NVLink RX (inbound) user data only: RX Requests User Data % (peer
             writes landing here) + RX Responses User Data % (data returned for
             this GPU's remote reads), both on the RX direction of the link.
  comm TX  = NVLink TX (outbound) user data only, same construction.
  NVLink is full duplex: RX and TX are independent channels with their own
  bandwidth, so they are written as TWO separate counter tracks and are never
  added together.

Window selection: CUPTI_ACTIVITY_KIND_KERNEL is preferred.  This host's nsys
build exports no kernel activity table, so the fallback bounds GPU_METRICS with
the NVTX_EVENTS 'WEAVE_CAPTURE_WINDOW' start/end from the SAME sqlite.  The
provenance string records which of the two was used; the NVTX window brackets
launch+sync and is therefore slightly wider than the kernel, which makes the
reported utilization conservative.

Writes a *.sw_hw.perfetto.json.gz consumable by
ablation/render_weaveprofiler_moe_timeline.py --platform sm103.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HEADER = 1024
WARPS = 20
RECS = 2048

STAGE_NAMES = {
    1: "1. Opening: Top-K Histogram & Route Publish",
    2: "2. Dispatch: Cross-Rank Sync & Rendezvous",
    3: "3. Dispatch Tile: Remote Token & SF Pull",
    4: "4. CLC Scheduler: Dynamic Tile Dispatch",
    5: "5a. FC1 Tile: TMA-A & SFA Load [L1 GEMM]",
    6: "5b. FC1 Tile: TMA-B & SFB Load [L1 GEMM]",
    7: "5c. FC1 Tile: UMMA Issue & Commit [L1 GEMM]",
    8: "6a. FC2 Tile: TMA-A & SFA Load [L2 GEMM]",
    9: "6b. FC2 Tile: TMA-B & SFB Load [L2 GEMM]",
    10: "6c. FC2 Tile: UMMA Issue & Commit [L2 GEMM]",
    11: "5d. FC1 Epilogue Tile: SwiGLU Quant & TMA Store",
    12: "7a. Scatter Tile [FC2 Epilogue]: BF16 Convert & NVLink Scatter",
    13: "8a. Pre-Combine: TMEM Release / Cleanup & Sync",
    14: "8b. Reduce Tile: Top-K FP32 Reduce & BF16 TMA Store",
    23: "5d. FC1 Epilogue Tile: SwiGLU Quant & TMA Store",
    24: "6d. FC2 Epilogue Tile: BF16 Convert & Local Handoff (no NVLink)",
    29: "7b. Scatter Tile [Comm-Warp Executor]: NVLink Scatter",
    30: "3. Head Dispatch Tile [All Warps]: Remote Token & SF Pull",
}

CURVE_NAMES = {
    "compute": "Tensor Active / MMA Pipe (%)",
    "hbm": "HBM / TMA Bandwidth (%)",
    "comm_rx": "NVLink RX User Data (%)",
    "comm_tx": "NVLink TX User Data (%)",
    "active": "Active SMs (%)",
}


def load_renderer(repo: Path):
    p = repo / "ablation" / "render_weaveprofiler_moe_timeline.py"
    spec = importlib.util.spec_from_file_location("w8f_renderer", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_sw(buf: np.ndarray, num_sms: int):
    body = buf[HEADER:].reshape(num_sms, WARPS, RECS, 4)
    recs, t0, t1 = [], None, 0
    for sm in range(num_sms):
        for w in range(WARPS):
            chunk = body[sm, w]
            nz = np.nonzero(chunk[:, 1])[0]
            if nz.size == 0:
                continue
            c = chunk[nz].astype(np.uint32)
            sid, ts, te, tile = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
            dur = (te - ts).astype(np.uint32)          # 32-bit globaltimer wrap-safe
            keep = dur < np.uint32(2_000_000_000)      # drop any wrapped/corrupt record
            sid, ts, dur, tile = sid[keep], ts[keep], dur[keep], tile[keep]
            if ts.size == 0:
                continue
            mn, mx = int(ts.min()), int((ts + dur).max())
            t0 = mn if t0 is None else min(t0, mn)
            t1 = max(t1, mx)
            recs.append((sm, w, sid, ts, dur, tile))
    if t0 is None:
        raise SystemExit("no SW records decoded")
    return recs, t0, t1


def pick_series(by_metric, candidates, combine="first"):
    # SKILL.md: "When only directional read/write counters exist, sum them per
    # timestamp; the per-sample maximum understates the shared DRAM interface
    # and is invalid."  sum_all therefore unions EVERY candidate before summing,
    # instead of stopping at the first candidate that matched anything.
    if combine == "sum_all":
        matched, seen = [], set()
        for cand in candidates:
            for k, pts in by_metric.items():
                if cand.lower() in k.lower() and k not in seen:
                    seen.add(k)
                    matched.append((k, pts))
        if not matched:
            return None, []
        agg = {}
        for _, pts in matched:
            for t, v in pts:
                agg[t] = agg.get(t, 0.0) + v
        return "+".join(k for k, _ in matched), sorted(agg.items())
    for cand in candidates:
        matched = [(k, pts) for k, pts in by_metric.items() if cand.lower() in k.lower()]
        if not matched:
            continue
        if combine == "first":
            return matched[0]
        agg = {}
        for _, pts in matched:
            for t, v in pts:
                agg[t] = agg.get(t, 0.0) + v
        return "+".join(k for k, _ in matched), sorted(agg.items())
    return None, []


def has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone() is not None


def select_window(con):
    """Return (start_ns, end_ns, device_id, label, provenance_kind)."""
    nvtx = con.execute("SELECT start, end FROM NVTX_EVENTS WHERE text = 'WEAVE_CAPTURE_WINDOW' "
                       "ORDER BY start DESC").fetchall()
    if not nvtx:
        raise SystemExit("WEAVE_CAPTURE_WINDOW NVTX range missing")
    win_start, win_end = nvtx[0]
    if has_table(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        kernels = con.execute(
            "SELECT k.start, k.end, k.deviceId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id WHERE s.value LIKE '%mega_moe%' "
            "ORDER BY k.start ASC").fetchall()
        if kernels:
            dev0 = min(r[2] for r in kernels)
            k_dev0 = [(a, b, n) for a, b, did, n in kernels if did == dev0]
            chosen = None
            for a, b, n in k_dev0:
                if a >= win_start - 500000 and b <= win_end + 500000:
                    chosen = (a, b, n)
            if chosen is None:
                chosen = k_dev0[-1]
            a, b, n = chosen
            return a, b, dev0, n, "kernel"
    print("[merge] CUPTI_ACTIVITY_KIND_KERNEL absent/empty: bounding GPU_METRICS with the "
          "same-run NVTX WEAVE_CAPTURE_WINDOW timestamps", flush=True)
    return win_start, win_end, 0, "mega_moe (NVTX capture window)", "nvtx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sw-pt", required=True)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--nsys-cmd", default="")
    args = ap.parse_args()

    d = torch.load(args.sw_pt, map_location="cpu", weights_only=False)
    buf = d["buf"].numpy().astype(np.int32)
    meta = d["meta"]
    num_sms = int(meta["num_sms"])
    recs, t0, t1 = decode_sw(buf, num_sms)
    sw_dur_us = (t1 - t0) / 1000.0
    print(f"[merge] SW records: {sum(len(r[2]) for r in recs)} spans, "
          f"{len(recs)} (sm,warp) tracks, sw_dur={sw_dur_us:.3f} us", flush=True)

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    k_start, k_end, dev0, k_name, win_kind = select_window(con)
    hw_dur_us = (k_end - k_start) / 1000.0

    # w8h_d: key metric names by (typeId, metricId).  This host's export carries
    # two GPU_METRICS typeIds; a bare metricId dict collapses them and can rename
    # every gb11x series, which empties the compute/hbm/comm/active roles.
    m_info = cur.execute("SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS").fetchall()
    id_to_name = {(tid, mid): nm for tid, mid, nm in m_info}
    rows = cur.execute("SELECT timestamp, typeId, metricId, value FROM GPU_METRICS WHERE timestamp "
                       "BETWEEN ? AND ? ORDER BY timestamp ASC",
                       (k_start - 20000, k_end + 20000)).fetchall()
    con.close()
    tid_counts = {}
    for ts, tid, mid, val in rows:
        tid_counts[tid] = tid_counts.get(tid, 0) + 1
    keep_tid = max(tid_counts, key=tid_counts.get) if tid_counts else None
    print(f"[merge] GPU_METRICS typeIds in window: {tid_counts}; keeping {keep_tid}", flush=True)
    by_metric = {}
    for ts, tid, mid, val in rows:
        if tid != keep_tid:
            continue
        nm = id_to_name.get((tid, mid), str(mid))
        by_metric.setdefault(nm, []).append(((ts - k_start) / 1000.0, float(val)))
    print(f"[merge] window={win_kind} label={k_name} hw_dur={hw_dur_us:.3f} us, "
          f"{len(by_metric)} metric series", flush=True)

    comp_src, comp_pts = pick_series(by_metric, ["Tensor Active", "Tensor Pipe Active", "HMMA"])
    hbm_src, hbm_pts = pick_series(by_metric, ["DRAM Read Bandwidth", "DRAM Write Bandwidth"], "sum_all")
    if not hbm_pts:
        hbm_src, hbm_pts = pick_series(by_metric, ["DRAM Bandwidth", "HBM"])
    nvl_names = sorted(k for k in by_metric if "nvlink" in k.lower())
    print(f"[merge] NVLink metric series available: {nvl_names}", flush=True)

    def nvlink_direction(direction):
        # One direction only (RX or TX).  Requests + Responses user data of the
        # SAME direction share that direction's link bandwidth and are summed;
        # the opposite direction is never mixed in.
        src, pts = pick_series(by_metric, [f"NVLink {direction} Requests User Data",
                                           f"NVLink {direction} Responses User Data"], "sum_all")
        if not pts:
            src, pts = pick_series(by_metric, [f"NVLink {direction} User Data", f"NVLink {direction}"])
        other = "TX" if direction == "RX" else "RX"
        if src and other.lower() in src.lower().replace(direction.lower(), ""):
            raise SystemExit(f"NVLink {direction} curve source {src!r} mixes in the {other} direction")
        return src, pts

    rx_src, rx_pts = nvlink_direction("RX")
    tx_src, tx_pts = nvlink_direction("TX")
    act_src, act_pts = pick_series(by_metric, ["SMs Active", "SM Active", "SM Issue"])
    for role, (src, pts) in dict(compute=(comp_src, comp_pts), hbm=(hbm_src, hbm_pts),
                                 comm_rx=(rx_src, rx_pts), comm_tx=(tx_src, tx_pts),
                                 active=(act_src, act_pts)).items():
        if not pts:
            raise SystemExit(f"HW curve role {role!r} is empty; available: {sorted(by_metric)[:40]}")
        print(f"[merge] {role}: src={src} n={len(pts)} max={max(v for _, v in pts):.3f}", flush=True)

    grid = sorted({t for t, _ in comp_pts})
    x_end = max(hw_dur_us, sw_dur_us)
    grid = [t for t in grid if -2.0 <= t <= x_end + 2.0]
    if len(grid) < 3:
        raise SystemExit(f"only {len(grid)} common grid samples")

    def resample(pts):
        xs = np.array([t for t, _ in pts], dtype=float)
        ys = np.array([v for _, v in pts], dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        g = np.array(grid, dtype=float)
        v = np.interp(g, xs, ys)
        return [(round(max(0.0, min(x_end, t)), 3), round(float(max(0.0, min(100.0, x))), 3))
                for t, x in zip(g, v)]

    curves = {CURVE_NAMES["compute"]: resample(comp_pts), CURVE_NAMES["hbm"]: resample(hbm_pts),
              CURVE_NAMES["comm_rx"]: resample(rx_pts), CURVE_NAMES["comm_tx"]: resample(tx_pts),
              CURVE_NAMES["active"]: resample(act_pts)}

    evs = [{"ph": "M", "pid": 90000, "tid": 0, "name": "process_name",
            "args": {"name": "Hardware Counters (nsys gb11x GPU Metrics)"}}]
    for sm in range(num_sms):
        evs.append({"ph": "M", "pid": 100000 + sm, "tid": 0, "name": "process_name",
                    "args": {"name": f"SM {sm:03d}"}})
    for nm, pts in curves.items():
        for t, v in pts:
            evs.append({"ph": "C", "pid": 90000, "ts": t, "name": nm, "args": {"value": v}})

    stage_counts = {}
    warp_reduce = {}
    for sm, w, sid, ts, dur, tile in recs:
        pid = 100000 + sm
        for i in range(len(sid)):
            s = int(sid[i])
            nm = STAGE_NAMES.get(s)
            if nm is None:
                continue
            stage_counts[s] = stage_counts.get(s, 0) + 1
            if s == 14:
                warp_reduce[w] = warp_reduce.get(w, 0) + 1
            evs.append({"ph": "X", "pid": pid, "tid": w, "name": nm,
                        "ts": round((int(ts[i]) - t0) / 1000.0, 3),
                        "dur": round(int(dur[i]) / 1000.0, 3),
                        "args": {"warp": w, "stage_id": s, "tile_id": int(tile[i])}})
    print("[merge] stage_id counts:", dict(sorted(stage_counts.items())), flush=True)
    print("[merge] stage 14 per warp:", dict(sorted(warp_reduce.items())), flush=True)

    nsys_cmd = args.nsys_cmd or ("nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 "
                                 "--gpu-metrics-set=gb11x --gpu-metrics-frequency=200000")
    if win_kind == "kernel":
        win_desc = (f"same-run NVTX-bracketed mega_moe kernel window "
                    f"[{k_start}, {k_end}] ns on device {dev0}")
    else:
        win_desc = (f"same-run NVTX_EVENTS 'WEAVE_CAPTURE_WINDOW' window "
                    f"[{k_start}, {k_end}] ns on device {dev0} "
                    f"(this nsys build exports no CUPTI_ACTIVITY_KIND_KERNEL table; the NVTX "
                    f"range brackets launch+sync so it is wider than the kernel and the "
                    f"resulting utilization is conservative)")
    # The host masks the marketing name; record the measured identity instead of
    # the masked string, and keep every identifying property.
    gpu_raw = d.get("gpu") or {}
    gpu = dict(platform="sm103", sm_count=gpu_raw.get("sm_count"),
               compute_capability=gpu_raw.get("compute_capability"),
               marketing_name_masked_by_host=True)
    other = {"meta": dict(
        card="sm103", arch="sm_103a", num_sms=num_sms,
        hw_dur_us=round(hw_dur_us, 3), sw_dur_us=round(sw_dur_us, 3),
        hw_window_kind=win_kind,
        hw_counter_source=(f"{nsys_cmd}; real GPU_METRICS samples from the nsys gb11x set, "
                           f"{win_desc}; five [Throughput %] series resampled by linear "
                           f"interpolation onto one common {len(grid)}-sample grid"),
        curve_sources=dict(compute=comp_src, hbm=hbm_src, nvlink_rx=rx_src, nvlink_tx=tx_src,
                           active=act_src),
        curve_sum_rule=("hbm = DRAM Read Bandwidth + DRAM Write Bandwidth summed per timestamp "
                        "(shared half-duplex DRAM bus); NVLink is full duplex, so RX (In) and "
                        "TX (Out) are two separate curves and are never summed with each other "
                        "(each direction = its Requests + Responses User Data); all clamped to "
                        "[0, 100] after resampling"),
        curve_grid_samples=len(grid),
        sw_record_source="DG_MEGA_TILE_PROF %globaltimer uint4 records (stage_id, t_start_ns, t_end_ns, tile_id)",
        capture_meta=dict(model=str(d.get("model") or "qwen38max"), ep=4,
                          shape=d.get("shape"),
                          tokens_global=4 * int((d.get("shape") or {}).get("S", 256)),
                          tokens_per_rank=int((d.get("shape") or {}).get("S", 256)),
                          num_shared_experts=0, mma_type="fp8xfp4",
                          arm=d.get("arm"), env_cfg={k: str(v) for k, v in (d.get("env") or {}).items()},
                          gpu=gpu, total_events=sum(stage_counts.values()),
                          stage14_warps=sorted(warp_reduce), overflow=0),
    )}

    R = load_renderer(Path(args.repo))
    bad = []
    for e in evs:
        texts = []
        if e.get("ph") == "M":
            texts.append(str((e.get("args") or {}).get("name", "")))
        elif e.get("ph") in ("C", "X"):
            texts.append(str(e.get("name", "")))
        for t in texts:
            h = R.forbidden_hits(t)
            if h:
                bad.append((t, h))
    if bad:
        raise SystemExit(f"forbidden display strings in trace: {bad[:5]}")
    for nm, pts in curves.items():
        vals = [v for _, v in pts]
        print(f"[merge] curve {nm}: n={len(pts)} min={min(vals):.2f} "
              f"med={statistics.median(vals):.2f} max={max(vals):.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump({"traceEvents": evs, "displayTimeUnit": "ns", "otherData": other}, fh)
    print(f"[merge] wrote {out} ({out.stat().st_size/1e6:.2f} MB), {len(evs)} events", flush=True)


if __name__ == "__main__":
    main()
