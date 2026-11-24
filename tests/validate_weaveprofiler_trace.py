#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate WeaveProfiler paired SW+HW Perfetto traces against the timeline red lines.

Every rule lives in ablation/render_weaveprofiler_moe_timeline.py (single source of
truth). This tool runs them on any *.sw_hw.perfetto.json.gz WITHOUT drawing (pure
Python; matplotlib only for --render-check) and also scans the trace's own
Perfetto-visible strings. Each finding carries the renderer's rule id:

  hw_provenance / hw_curves / hw_range / hw_comm_zero / hw_rdma_zero / hw_alignment /
  hw_nvlink_merged (errors)
      sm103 NVLink is full duplex: Panel A carries "NVLink RX User Data (%)" and
      "NVLink TX User Data (%)" as two separate curves; a merged RX+TX track is an
      error (only HBM DRAM Read % + Write % is summed).
      strict real-hardware-counter red lines: NVIDIA sm103 + h20 need nsys GPU-metrics
      curves, AMD mi308x needs rocprof PMC curves; all 4 curves (compute, hbm,
      comm = NVLink / xGMI, active SM / CU) present and non-empty, every sample
      0.0 <= v <= 100.0, max(comm) > 0.0, and no "Model estimate" / synthetic /
      analytic provenance in otherData.source, otherData "*source*" fields, counter
      names or the counter-track title.  Multi-node MI308X EP16 traces may carry an
      OPTIONAL 5th real curve rdma_util [Inter-Node RDMA NIC % of <rate>GB/s]:
      accepted only with otherData.meta.ep == 16 plus /sys/class/... NIC counter
      provenance in curve_sources, and gated by the same range / sample-count /
      max(rdma) > 0.0 red lines.  EP16 Panel C labels every wave W00..W07 on its
      own row (per-wave layout); EP8 keeps the merged role bands.
  sw_stage_map / sw_reduce_dropped / sw_track_hidden / sw_roles / sw_window  (errors)
      every slice maps to a legend stage, Combine_Reduce (recorder stage_id 14) is never
      dropped, every SM 0 / CU 0 track has a Panel C lane, Panel C labels are truthful,
      and malformed / non-finite / negative slice timings are reported rather than crashed.
  streaming Reduce                                                         (reported)
      comm tracks executing Combine_Reduce that start inside their unit's Middle
      GEMM / Scatter window and before the global tail combine start; their Panel C
      label gains "-> Streaming Reduce", e.g. "W02..W03 (Dispatch -> Streaming Reduce)".
  forbidden_string                                                         (errors)
      every string the renderer would draw (title, banner, legends, axis / tick labels,
      Panel C track labels) and every Perfetto-visible title / label of the trace
      (process, thread, counter and slice names, including numeric names rendered as text).
  forbidden_string_metadata / banner_env_mismatch                          (warnings)
      forbidden strings inside slice args / otherData (never drawn); banner WEAVE_*
      values that contradict otherData.meta.capture_meta.env_cfg.

--render-check draws the figure in memory and re-runs the drawn-text and rendered-pixel
gates (crisp purple Combine / Reduce probes); --render-out PATH also saves that PNG.
--self-test pushes synthetic in-memory fixtures (never written to disk, never presented
as measurements) through every rule.

Usage:
  python tests/validate_weaveprofiler_trace.py TRACE [TRACE ...] [--platform auto]
         [--config-banner TEXT] [--repro-script PATH] [--no-banner] [--json-out REPORT.json]
         [--require-streaming-reduce] [--render-check] [--render-out PNG]
  python tests/validate_weaveprofiler_trace.py --self-test
  python tests/validate_weaveprofiler_trace.py          (no trace: same as --self-test)
Exit status: 0 = every trace passed, 1 = at least one trace violated a red line,
2 = usage error.  Edge-case regression tests: tests/test_validate_weaveprofiler_trace.py.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "ablation" / "render_weaveprofiler_moe_timeline.py"
_RENDERER_CACHE = {}


def load_renderer(path: Path = RENDERER_PATH):
    path = Path(path).resolve()
    if path not in _RENDERER_CACHE:
        spec = importlib.util.spec_from_file_location("weave_timeline_renderer_v3", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"ERROR: cannot import renderer: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RENDERER_CACHE[path] = module
    return _RENDERER_CACHE[path]


def load_doc(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- trace string scans
_SLICE_PHASES = {"X", "B", "E", "i", "I", "b", "e", "n", "s", "t", "f"}


def _display_text(v):
    """Text Perfetto renders for a name; keep numeric names and label lists."""
    if isinstance(v, str):
        return v
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        parts = [str(x) for x in v if isinstance(x, (str, int, float)) and not isinstance(x, bool)]
        return " ".join(parts) if parts else None
    return None


def trace_visible_strings(doc):
    """Unique strings Perfetto shows without interaction: track titles and slice labels."""
    seen = {}
    for e in doc.get("traceEvents", []):
        if not isinstance(e, dict):
            continue
        ph, nm = e.get("ph"), e.get("name")
        if ph == "M" and nm in ("process_name", "thread_name", "process_labels"):
            args = e.get("args") if isinstance(e.get("args"), dict) else {}
            val = _display_text(args.get("name", args.get("labels")))
            if val is not None:
                seen.setdefault(val, f"{nm.replace('_', ' ')}")
        elif ph == "C":
            val = _display_text(nm)
            if val is not None:
                seen.setdefault(val, "counter name")
        elif ph in _SLICE_PHASES:
            val = _display_text(nm)
            if val is not None:
                seen.setdefault(val, "slice name")
    return [(where, s) for s, where in seen.items()]


def _leaves(obj, path, out):
    if isinstance(obj, str):
        out.setdefault(obj, path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _leaves(v, f"{path}.{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _leaves(v, f"{path}[{i}]", out)


def trace_metadata_strings(doc):
    """Unique string leaves of otherData and of event args (shown only on click)."""
    out = {}
    _leaves(doc.get("otherData"), "otherData", out)
    for e in doc.get("traceEvents", []):
        if isinstance(e, dict) and e.get("ph") != "M" and isinstance(e.get("args"), dict):
            _leaves(e["args"], f"args[{e.get('name')!r}]", out)
    return [(where, s) for s, where in out.items()]


# ---------------------------------------------------------------- validation
def _add(rep, kind, rule, message):
    if not message.startswith(f"[{rule}]"):
        message = f"[{rule}] {message}"
    rep[kind].append(dict(rule=rule, message=message))


def _add_exc(rep, exc):
    _add(rep, "errors", getattr(exc, "rule", type(exc).__name__), str(exc))


def validate_doc(doc, *, name="<memory>", platform="auto", config_banner=None, repro_script=None,
                 render_check=False, render_out=None, renderer=None, require_streaming_reduce=False):
    """Validate one decoded trace; never raises on rule violations (they land in the report)."""
    R = renderer or load_renderer()
    rep = dict(trace=name, ok=False, platform=None, errors=[], warnings=[], checks={})
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        _add(rep, "errors", "sw_stage_map", "not a Chrome-trace JSON object with a traceEvents list")
        return rep
    bad = next((i for i, e in enumerate(doc["traceEvents"]) if not isinstance(e, dict)), None)
    if bad is not None:
        _add(rep, "errors", "sw_stage_map", f"traceEvents[{bad}] is not a JSON object")
        return rep
    try:
        plat = R.detect_platform(doc) if platform in (None, "auto") else platform
    except R.TraceRedLineError as exc:
        _add_exc(rep, exc)
        return rep
    rep["platform"] = plat
    cfg = R.trace_cfg(R.platform_cfg(plat), doc)
    banner = R.SHARED_BANNERS[plat] if config_banner is None else config_banner
    repro = R.REPRO_SCRIPTS[plat] if repro_script is None else repro_script
    banner_text = R.compose_banner(banner, repro)
    evs = doc["traceEvents"]
    pnames, tnames = R.trace_names(evs)

    hw = sw = layout = None
    try:   # strict real-hardware-counter red lines
        curves = R.collect_counters(evs, cfg)
        hw = R.check_hw_red_lines(doc, cfg, curves, pnames, tnames)
        rep["checks"]["hw"] = dict(tool=hw["tool_text"], evidence=hw["evidence"],
                                   samples_per_curve=hw["samples"], comm_max=hw["comm_max"],
                                   comm_tx_max=hw.get("comm_tx_max"),
                                   rdma_max=hw.get("rdma_max"), curves=hw["curves"])
    except R.TraceRedLineError as exc:
        _add_exc(rep, exc)
    try:   # software stage map, Panel C tracks, streaming Reduce detection
        sw = R.parse_sw(doc, cfg)
        R.verify_tracks(cfg, sw, tnames)
        red = R.analyze_reduce(cfg, sw, tnames)
        layout = R.panel_c_layout(cfg, sw, red)
        rep["checks"]["sw"] = dict(sw_window_us=round(sw["sw_end"], 3),
                                   plotted=sum(sw["stage_counts"].values()),
                                   dropped_wait=sum(sw["drop_counts"].values()),
                                   stage_counts=dict(sw["stage_counts"]))
        rep["checks"]["reduce"] = red
        if require_streaming_reduce and not red["streaming_slices"]:
            _add(rep, "errors", "sw_streaming_reduce_missing",
                 f"{plat}: --require-streaming-reduce is set but no comm track streams Combine_Reduce "
                 f"inside its unit's Middle window (streaming_slices=0); the dedicated WEAVE_REDUCE_WARPS "
                 f"Middle overlap is missing from this trace")
        rep["checks"]["panel_c_tracks"] = [b["label"] for b in layout["bands"]]
    except R.TraceRedLineError as exc:
        _add_exc(rep, exc)
    xa = None
    if hw and sw:
        try:
            R.check_alignment(cfg, hw, sw)
            xa = R.x_axis(cfg, sw["sw_end"], hw["curve_ts_max"])
        except R.TraceRedLineError as exc:
            _add_exc(rep, exc)

    strings = R.display_strings(cfg, dict(hw=hw, layout=layout, xaxis=xa, doc=doc), banner_text)
    try:   # zero forbidden strings in anything the renderer draws
        rep["checks"]["display_strings_audited"] = R.audit_forbidden(strings)
    except R.TraceRedLineError as exc:
        _add_exc(rep, exc)
    if hw:
        note = R.panel_a_rdma_note(cfg, hw, doc)
        if note:
            rep["checks"]["panel_a_rdma_unavailable_note"] = note
            # full recorded device evidence: report only, never drawn on the figure
            rep["checks"]["panel_a_rdma_evidence"] = R.panel_a_rdma_evidence(cfg, hw, doc)
    if cfg.get("fabric_palette"):
        try:   # EP16: Inter-Node RDMA and Intra-Node xGMI never share a Panel B / C color
            rep["checks"]["fabric_palette_min_rgb_dist"] = R.check_fabric_palette()
        except R.TraceRedLineError as exc:
            _add_exc(rep, exc)
    vis = trace_visible_strings(doc)
    vis_hits = [(w, s, R.forbidden_hits(s)) for w, s in vis if R.forbidden_hits(s)]
    rep["checks"]["trace_visible_strings_audited"] = len(vis)
    if vis_hits:
        _add(rep, "errors", "forbidden_string",
             f"{len(vis_hits)} Perfetto-visible trace string(s) contain forbidden display strings; first: "
             + "; ".join(f"{w} {s!r} -> {h}" for w, s, h in vis_hits[:3]))
    meta = trace_metadata_strings(doc)
    meta_hits = [(w, s, R.forbidden_hits(s)) for w, s in meta if R.forbidden_hits(s)]
    rep["checks"]["metadata_strings_audited"] = len(meta)
    if meta_hits:
        _add(rep, "warnings", "forbidden_string_metadata",
             f"{len(meta_hits)} slice-arg / otherData string(s) (never drawn) contain forbidden display strings; "
             f"first locations: " + "; ".join(w for w, _, _ in meta_hits[:3]))
    for m in R.banner_env_mismatches(banner_text, doc):
        _add(rep, "warnings", "banner_env_mismatch",
             f"banner states {m['knob']}={m['banner']} but the trace capture env records {m['knob']}={m['trace']}")

    if render_check and not rep["errors"]:
        try:
            full = R.parse(dict(cfg), doc=doc)
            res = R.render(cfg, full, render_out, banner, repro)
            res.pop("png_data", None)
            rep["checks"]["render"] = res
        except (R.TraceRedLineError, RuntimeError) as exc:
            _add_exc(rep, exc)
    rep["ok"] = not rep["errors"]
    return rep


def validate_trace(path, **kwargs):
    path = Path(path)
    try:
        doc = load_doc(path)
    except (OSError, ValueError) as exc:
        rep = dict(trace=str(path), ok=False, platform=None, errors=[], warnings=[], checks={})
        _add(rep, "errors", "load", f"cannot read trace: {exc}")
        return rep
    rep = validate_doc(doc, name=str(path), **kwargs)
    raw = path.read_bytes()
    rep["trace_sha256"] = hashlib.sha256(raw).hexdigest()
    rep["trace_bytes"] = len(raw)
    return rep


def print_report(rep, out=sys.stdout):
    status = "PASS" if rep["ok"] else "FAIL"
    print(f"{status}  [{rep.get('platform') or '?'}]  {rep['trace']}", file=out)
    hw = rep["checks"].get("hw")
    if hw:
        rdma = f"  rdma_max={hw['rdma_max']}" if hw.get("rdma_max") is not None else ""
        if hw.get("comm_tx_max") is not None:
            rdma = f"  comm_tx_max={hw['comm_tx_max']}" + rdma
        print(f"      hw: {hw['tool']}  samples/curve={hw['samples_per_curve']}  comm_max={hw['comm_max']}{rdma}"
              f"  evidence={hw['evidence']}", file=out)
    if rep["checks"].get("panel_a_rdma_unavailable_note"):
        first = rep["checks"]["panel_a_rdma_unavailable_note"].split("\n")[0]
        print(f"      panel A: {first}", file=out)
    red = rep["checks"].get("reduce")
    if red:
        print(f"      reduce: tail_combine_start_us={red['tail_combine_start_us']}  "
              f"streaming_slices={red['streaming_slices']} on {red['streaming_units']} unit(s)  "
              f"unit0_streaming_tracks={red['unit0_streaming_tracks']}", file=out)
    if rep["checks"].get("panel_c_tracks"):
        print(f"      panel C: {rep['checks']['panel_c_tracks']}", file=out)
    for e in rep["errors"]:
        print(f"      ERROR {e['message']}", file=out)
    for w in rep["warnings"]:
        print(f"      WARN  {w['message']}", file=out)


def _json_ready(rep):
    out = {k: v for k, v in rep.items() if k != "trace"}
    out["trace_path_b64"] = base64.b64encode(str(rep["trace"]).encode("utf-8")).decode("ascii")
    return out


# ---------------------------------------------------------------- self-test fixtures
# SYNTHETIC, in-memory only: these documents exercise the rules and are never written
# to disk or reported as measurements.
# sm103 NVLink is full duplex: RX (In) and TX (Out) are two separate tracks (never summed).
SM103_CURVES = ["Tensor Active / MMA Pipe (%)", "HBM / TMA Bandwidth (%)", "NVLink RX User Data (%)",
                "NVLink TX User Data (%)", "Active SMs (%)"]


def _curve_events(pid, names, *, n=60, period=5.0, comm_zero=False, bad=None):
    evs = []
    for ci, nm in enumerate(names):
        for i in range(n):
            v = 40.0 + 30.0 * math.sin(0.15 * i + ci)
            if comm_zero and ci == 2:    # the (first) interconnect track: sm103 NVLink RX
                v = 0.0
            if bad is not None and ci == 1 and i == n // 2:
                v = bad
            evs.append({"ph": "C", "pid": pid, "ts": i * period, "name": nm, "args": {"value": v}})
    return evs


def _x(pid, tid, name, ts, dur, **args):
    return {"ph": "X", "pid": pid, "tid": tid, "name": name, "ts": float(ts), "dur": float(dur), "args": args}


def fx_sm103(*, streaming=True, new_reduce_name=False, curve_prefix="", source=None, provenance=True,
             comm_zero=False, bad=None, drop_active=False, title_hit=None,
             streaming_before_middle=False):
    names = [curve_prefix + c for c in SM103_CURVES]
    if drop_active:
        names = [nm for nm in names if not nm.endswith("Active SMs (%)")]
    meta = {"card": "sm103", "arch": "sm_103a", "num_sms": 148}
    evs = []
    if provenance:
        meta["curve_sources"] = {"compute": "Tensor Active [Throughput %]", "hbm": "DRAM Read Bandwidth [Throughput %]",
                                 "nvlink_rx": "NVLink RX Requests User Data [Throughput %]",
                                 "nvlink_tx": "NVLink TX Requests User Data [Throughput %]",
                                 "active": "SMs Active [Throughput %]"}
        evs.append({"ph": "M", "pid": 90000, "name": "process_name", "args": {"name": "Hardware Counters (nsys gb11x)"}})
    evs += _curve_events(90000, names, comm_zero=comm_zero, bad=bad)
    red_name = "8b. Reduce Tile [Epilogue Warps W08..W15]: Top-K Reduce & TMA Store"
    for sm, tail in ((0, 260.0), (1, 258.0)):
        pid = 100000 + sm
        evs.append({"ph": "M", "pid": pid, "name": "process_name",
                    "args": {"name": f"SM {sm:03d}" + (title_hit if (title_hit and sm == 0) else "")}})
        for w in range(4):   # the long 8a. sync slice sits UNDER any streaming Reduce slice (as in real traces)
            evs += [_x(pid, w, "1. Opening: Top-K Histogram & Route Publish", 0.5, 4, warp=w, stage_id=1),
                    _x(pid, w, "2. Dispatch: Cross-Rank Sync & Rendezvous", 5, 3, warp=w, stage_id=2),
                    _x(pid, w, "3. Dispatch Tile: Remote Token & SF Pull", 8, 20, warp=w, stage_id=3),
                    _x(pid, w, "8a. Pre-Combine: TMEM Release / Cleanup & Sync", 30, 225, warp=w, stage_id=13)]
        for t in range(10):   # Middle GEMM / Scatter window: 30 .. 244 us
            a = 30 + 22 * t
            evs += [_x(pid, 4, "5a. FC1 Tile: TMA-A & SFA Load [L1 GEMM]" if t < 5 else "6a. FC2 Tile: TMA-A & SFA Load [L2 GEMM]", a, 5, warp=4, stage_id=5),
                    _x(pid, 5, "5b. FC1 Tile: TMA-B & SFB Load [L1 GEMM]" if t < 5 else "6b. FC2 Tile: TMA-B & SFB Load [L2 GEMM]", a, 5, warp=5, stage_id=6),
                    _x(pid, 6, "5c. FC1 Tile: UMMA Issue & Commit [L1 GEMM]" if t < 5 else "6c. FC2 Tile: UMMA Issue & Commit [L2 GEMM]", a + 5, 5, warp=6, stage_id=7)]
            for w in range(8, 16):
                evs.append(_x(pid, w, "5d. FC1 Epilogue Tile: SwiGLU Quant & TMA Store" if t < 5 else
                              "7a. Scatter Tile [FC2 Epilogue]: BF16 Convert & NVLink Scatter", a + 10, 6, warp=w, stage_id=11))
        evs.append(_x(pid, 7, "4. CLC Scheduler: Dynamic Tile Dispatch", 28, 220, warp=7, stage_id=4))
        for w in range(8, 16):
            evs += [_x(pid, w, "8a. Pre-Combine: TMEM Release / Cleanup & Sync", 152 + 105, 1, warp=w, stage_id=13),
                    _x(pid, w, red_name, tail + (w - 8) * 0.01, 12, warp=w, stage_id=14)]
        for w in (2, 3):
            if streaming:
                nm = "9z. Streaming Reduce Tile (new recorder name)" if new_reduce_name else red_name
                a, step = (6.0, 1.0) if streaming_before_middle else (60.0, 10.0)
                evs += [_x(pid, w, nm, a + step * i, 3, warp=w, stage_id=14) for i in range(18)]
            else:   # tail-only: comm warps reduce after the CTA barrier (sm 1 skew probe: 0.1 us early)
                evs.append(_x(pid, w, red_name, tail - (0.1 if sm == 1 else -2.0), 10, warp=w, stage_id=14))
    other = {"meta": meta, "fixture": "SYNTHETIC SELF-TEST FIXTURE (in-memory only)"}
    if source is not None:
        other["source"] = source
    return {"traceEvents": evs, "otherData": other}


H20_CURVES = ["compute_util [FP8 Tensor Core % of 296T]", "memory_util [HBM3 Bandwidth % of 4.0TB/s]",
              "communication_rx_util [NVLink4 RX (In) % of 450GB/s]",
              "communication_tx_util [NVLink4 TX (Out) % of 450GB/s]", "sm_active [Active H20 SMs % of 78]"]
H20_THREADS = {1: "Dispatch Warp 0", 8: "Dispatch Warp Last", 2: "TMA Warp A (Acts+SF)", 3: "TMA Warp B (Weights)",
               4: "Math+Epi WG0", 5: "Math+Epi WG1", 6: "Math+Epi WG2", 7: "Math+Epi WG3"}


def fx_h20(*, reduce_in_wait=False, extra_tid=False):
    evs = [{"ph": "M", "pid": 1, "name": "process_name",
            "args": {"name": "00 · H20 nsys GPU Metrics Hardware Counters"}}]
    evs += _curve_events(1, H20_CURVES)
    for sm in (0, 1):
        pid = 100 + sm
        for tid, tn in H20_THREADS.items():
            evs.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": f"SM{sm:02d} · {tn}"}})
        for tid in (1, 8):
            evs += [_x(pid, tid, "Opening_Meta", 0.5, 3), _x(pid, tid, "Dispatch_Pull", 4, 20)]
        for t in range(10):
            a = 30 + 20 * t
            for tid in (2, 3):
                evs.append(_x(pid, tid, "L1_TMA_Load" if t < 5 else "L2_TMA_Load", a, 4))
            for tid in (4, 5, 6, 7):
                evs += [_x(pid, tid, "L1_GEMM_SwiGLU" if t < 5 else "L2_GEMM", a + 4, 8),
                        _x(pid, tid, "L2_Scatter", a + 12, 4)]
        for tid in (4, 5, 6, 7, 8):
            evs.append(_x(pid, tid, "Combine_Reduce", 262.0 + sm, 10, stage="Combine_Reduce"))
        evs += [_x(pid, 1, "Combine_Reduce", 80 + 30 * i, 4, stage="Combine_Reduce") for i in range(5)]
        if reduce_in_wait:
            evs.append(_x(pid, 2, "wait", 120, 5, stage="Combine_Reduce"))
        if extra_tid:   # a recorded track that has no Panel C row
            evs.append(_x(pid, 9, "Combine_Reduce", 150, 5, stage="Combine_Reduce"))
    other = {"meta": {"card": "h20", "arch": "sm_90"}, "fixture": "SYNTHETIC SELF-TEST FIXTURE (in-memory only)"}
    return {"traceEvents": evs, "otherData": other}


MI_CURVES = ["compute_util [FP8 Matrix Core % of 606.2T]", "memory_util [HBM3 Bandwidth % of 3684GB/s]",
             "communication_util [xGMI P2P % of 287.2GB/s]", "sm_active [Active MI308X CUs % of 80]"]


def fx_mi308x(*, source="rocprofv3 --pmc SQ_BUSY_CU TCC_EA0_RDREQ_sum GRBM_GUI_ACTIVE hardware counters, same-run kernel window"):
    evs = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "00 · MI308X Hardware Utilization"}}]
    evs += _curve_events(1, MI_CURVES)
    for cu in (0, 1):
        pid = 100 + cu
        for w in range(8):
            comm = w >= 4
            evs.append(_x(pid, w + 1, "Local_Ballot_Plan" if comm else "Route_Plan_80CTA", 0.5, 4, wave=w))
            if comm:
                evs.append(_x(pid, w + 1, "Dispatch_xGMI_A2A", 5, 20, wave=w))
            for t in range(10):
                a = 30 + 20 * t
                evs += [_x(pid, w + 1, "GEMM1_Tile_SwiGLU" if t < 5 else "GEMM2_Tile_MFMA", a, 12, wave=w),
                        _x(pid, w + 1, "GEMM2_Wait_Stage1", a + 12, 2, wave=w)]
            evs.append(_x(pid, w + 1, "Scatter_xGMI", 232, 8, wave=w))
            evs.append(_x(pid, w + 1, "Home_Combine", 262.0 + cu, 10, wave=w))
            if comm:
                evs += [_x(pid, w + 1, "Home_Combine", 90 + 20 * i, 3, wave=w) for i in range(6)]
    other = {"meta": {"card": "mi308x", "arch": "gfx942"}, "fixture": "SYNTHETIC SELF-TEST FIXTURE (in-memory only)"}
    if source is not None:
        other["source"] = source
    return {"traceEvents": evs, "otherData": other}


MI_RDMA_CURVE = "rdma_util [Inter-Node RDMA NIC % of 25GB/s]"
EP16_NIC_SOURCE = (
    "Linux NIC hardware counters /sys/class/infiniband/mlx5_0/ports/1/hw_counters/tx_bytes + "
    "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/rx_bytes (raw byte deltas / dt), link rate "
    "200 Gb/s from /sys/class/infiniband/mlx5_0/ports/1/rate, normalized to 25GB/s")


def fx_mi308x_ep16(*, rdma=True, rdma_zero=False, rdma_bad=None, nic_prov=True, ep=16,
                   source="rocprofiler-sdk device_counting_service SQ_BUSY_CU_CYCLES GRBM_GUI_ACTIVE "
                          "TCC_EA0_WRREQ_DRAM hardware counters, same-run kernel window, plus RDMA NIC "
                          "sysfs hardware counters in the same sampling loop"):
    """SYNTHETIC 2-node EP16 fixture: role-marked waves, EP16 slice names, and the
    OPTIONAL 5th real Inter-Node RDMA NIC curve.  W00..W03 are hybrid compute waves
    (short front-end LocalStage + GEMM tiles + tail HomeCombine); W04..W07 are comm
    waves (xGMI Pull front-end, ScatterPush, streaming Partial_Reduce in Middle,
    tail HomeCombine) -- the geometry the V2 exporter stamps and Panel C labels
    per wave (W00..W07 each on its own row)."""
    names = list(MI_CURVES) + ([MI_RDMA_CURVE] if rdma else [])
    evs = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "00 · MI308X Hardware Utilization"}}]
    evs += _curve_events(1, names)
    if rdma:
        pts = [e for e in evs if e.get("ph") == "C" and e.get("name") == MI_RDMA_CURVE]
        if rdma_zero:
            for e in pts:
                e["args"]["value"] = 0.0
        if rdma_bad is not None:
            pts[len(pts) // 2]["args"]["value"] = rdma_bad
    for cu in (0, 1):
        pid = 100 + cu
        for w in range(8):
            comm = w >= 4
            role = "[Comm]" if comm else "[Compute]"
            evs.append({"ph": "M", "pid": pid, "tid": w + 1, "name": "thread_name",
                        "args": {"name": f"CU{cu:02d} - Wave {w} (W{w:02d}) {role}"}})
            evs.append(_x(pid, w + 1, "Local_Ballot_Plan" if comm else "Route_Plan_Local", 0.5, 4, wave=w))
            evs.append(_x(pid, w + 1, "IntraNode_xGMI_Pull" if comm else "IntraNode_LocalStage", 5, 20 if comm else 10, wave=w))
            if not comm:
                for t in range(10):
                    a = 30 + 20 * t
                    evs += [_x(pid, w + 1, "Compute_GEMM1_SwiGLU" if t < 5 else "Compute_GEMM2_MFMA", a, 12, wave=w),
                            _x(pid, w + 1, "GEMM2_Wait_Stage1", a + 12, 2, wave=w)]
            if comm:
                evs.append(_x(pid, w + 1, "IntraNode_xGMI_ScatterPush", 232, 8, wave=w))
                evs += [_x(pid, w + 1, "InterNode_Partial_Reduce", 90 + 20 * i, 3, wave=w) for i in range(6)]
            evs.append(_x(pid, w + 1, "IntraNode_HomeCombine", 262.0 + cu, 10, wave=w))
    meta = {"card": "mi308x", "arch": "gfx942", "ep": ep}
    if rdma and nic_prov:
        meta["curve_sources"] = {MI_RDMA_CURVE: EP16_NIC_SOURCE}
        meta["nic_capture"] = {
            "nic_raw_per_sample": 2, "nic_errors": 0, "total_rate_gbps": 200.0,
            "nic_ports": [{"device": "mlx5_0", "port": 1, "link_layer": "InfiniBand",
                           "state": "4: ACTIVE", "source": "infiniband_hw_counters", "rate_gbps": 200.0,
                           "rate_path": "/sys/class/infiniband/mlx5_0/ports/1/rate",
                           "counters": [
                               {"path": "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/tx_bytes",
                                "kind": "hw_tx_bytes", "scale": 1.0},
                               {"path": "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/rx_bytes",
                                "kind": "hw_rx_bytes", "scale": 1.0}]}],
        }
    other = {"meta": meta, "fixture": "SYNTHETIC SELF-TEST FIXTURE (in-memory only)"}
    if source is not None:
        other["source"] = source
    return {"traceEvents": evs, "otherData": other}


# The real 2-node MI308X cluster's Inter-Node RDMA HCAs: ACTIVE ports that expose no
# counters at all, so the optional 5th curve is unavailable (never zero).  Used here
# only as synthetic fixture data for the Panel A honesty note.
EP16_RDMA_UNAVAILABLE_HCA = {
    "status": "unavailable", "fabric": "Inter-Node RDMA (NIC)",
    "devices": ["fic2_soe_bond0", "fic2_soe_bond1", "fic2_soe_bond2", "fic2_soe_bond3"],
    "node_desc": "Alibaba EIC SOE Driver HCA",
    "port": {"port": "1", "state": "4: ACTIVE", "rate": "300 Gb/sec (12X EDR)",
             "link_layer": "Ethernet", "counters": [], "hw_counters": []},
    "rdma_statistic": {"cmd": "rdma statistic show link fic2_soe_bond0/1", "rc": 1,
                       "stderr": "error: Invalid argument"},
}
EP16_RDMA_SKIPPED_REASON = (
    "RDMA NIC sysfs counters were readable (nic_capture.ports) but zero bytes moved inside the "
    "FI window: the EIC/SolarRDMA GPU-initiated data path (caller-owned SQ/CQ rings, EIC_ALLOC_GPU) "
    "is not counted by any host-readable sysfs / ethtool / rdma-statistic counter on this cluster; "
    "the optional 5th curve is omitted instead of drawing a dead zero line")


def fx_mi308x_ep16_rdma_unavailable(*, evidence=True, reason=True):
    """SYNTHETIC EP16 fixture with NO rdma curve: Panel A must label the xGMI curve
    Intra-Node and document the Inter-Node RDMA NIC counter as UNAVAILABLE."""
    d = fx_mi308x_ep16(rdma=False)
    meta = d["otherData"]["meta"]
    if evidence:
        meta["nic_rdma_hca"] = copy.deepcopy(EP16_RDMA_UNAVAILABLE_HCA)
    if reason:
        meta["nic_capture"] = {"ports": [{"device": "reth1", "port": 0, "rate_gbps": 200.0}],
                               "total_rate_gbps": 800.0, "rdma_curve_name": None,
                               "rdma_skipped_reason": EP16_RDMA_SKIPPED_REASON}
    return d


def fx_numeric_name():
    """A trace whose slice name is the number 3050 (a forbidden display token) instead of text."""
    d = fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "name": 3050, "ts": 1.0, "dur": 1.0, "args": {}})
    return d


def fx_malformed_ts():
    """A physical-unit slice with no ts: the validator must report it, never crash."""
    d = fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "dur": 2.0,
                             "name": "1. Opening: Top-K Histogram & Route Publish"})
    return d


def fx_negative_dur():
    """A slice with a negative duration: a sw_window error, never a crash."""
    d = fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "ts": 400.0, "dur": -2.0,
                             "name": "1. Opening: Top-K Histogram & Route Publish"})
    return d


def fx_non_object_event():
    """A non-object traceEvents entry: a sw_stage_map error, never an AttributeError."""
    d = fx_sm103()
    d["traceEvents"].append("garbage")
    return d


def run_self_test(renderer_path=RENDERER_PATH) -> int:
    R = load_renderer(renderer_path)
    bad_banner = "Spatial: " + R.FORBIDDEN[0] + " preview"
    title_hit = " " + R.FORBIDDEN[1]
    want_sm = "W02..W03 (Dispatch -> Streaming Reduce)"
    can_draw = R.plt is not None
    cases = [
        ("sm103 dedicated Reduce warps stream Combine_Reduce in Middle", fx_sm103(), {}, dict(
            ok=True, label=want_sm, label2="W00..W01 (Dispatch)", streaming=72, render=can_draw)),
        ("sm103 stage_id 14 under a brand-new slice name stays Combine / Reduce", fx_sm103(new_reduce_name=True), {},
         dict(ok=True, label=want_sm, streaming=72, render=can_draw)),
        ("sm103 tail-only Reduce on comm warps (incl. 0.1 us barrier skew) is not streaming", fx_sm103(streaming=False),
         {}, dict(ok=True, label="W00..W03 (Dispatch)", streaming=0)),
        ("sm103 Reduce that starts before Middle is an early combine, not streaming",
         fx_sm103(streaming_before_middle=True), {}, dict(
             ok=True, label="W00..W03 (Dispatch)", streaming=0)),
        ("mi308x comm waves W04..W07 stream Home_Combine in Middle", fx_mi308x(), {}, dict(
            ok=True, label="W04..W07 (Comm / Scatter -> Streaming Reduce)", label2="W00..W03 (Compute / Combine)",
            streaming=48, render=can_draw)),
        ("mi308x EP16 per-wave W00..W07 Panel C labels + 5th RDMA NIC curve", fx_mi308x_ep16(), {}, dict(
            ok=True, label="W04 (Comm / Scatter -> Streaming Reduce)", label2="W00 (Compute / Combine)",
            streaming=48, render=can_draw)),
        ("mi308x EP16 4-curve trace still gets per-wave W00..W07 labels", fx_mi308x_ep16(rdma=False), {}, dict(
            ok=True, label="W07 (Comm / Scatter -> Streaming Reduce)", label2="W03 (Compute / Combine)",
            streaming=48)),
        ("mi308x EP16 Inter-Node RDMA NIC counter unavailable is documented on Panel A, never zero",
         fx_mi308x_ep16_rdma_unavailable(), {}, dict(
             ok=True, label="W04 (Comm / Scatter -> Streaming Reduce)", streaming=48, render=can_draw,
             note_contains=["Inter-Node RDMA NIC counters unavailable on this HCA", "omitted, not zero-filled"],
             note_excludes=["\n", "SKILL.md", "nic_capture", "fic2_soe_bond"], fabric=True)),
        ("mi308x EP16 Panels B + C: Inter-Node RDMA vs Intra-Node xGMI on distinct fabric swatches",
         fx_mi308x_ep16(), {}, dict(ok=True, streaming=48, render=can_draw, fabric=True)),
        ("mi308x EP16 flat-zero RDMA NIC curve", fx_mi308x_ep16(rdma_zero=True), {}, dict(ok=False, rule="hw_rdma_zero")),
        ("mi308x EP16 RDMA sample 100.5 %", fx_mi308x_ep16(rdma_bad=100.5), {}, dict(ok=False, rule="hw_range")),
        ("mi308x EP16 RDMA curve without NIC sysfs provenance", fx_mi308x_ep16(nic_prov=False), {},
         dict(ok=False, rule="hw_provenance")),
        ("mi308x RDMA curve on a single-node EP8 trace is rejected", fx_mi308x_ep16(ep=8), {},
         dict(ok=False, rule="hw_curves")),
        ("h20 Dispatch W0 streams Combine_Reduce in Middle", fx_h20(), {}, dict(
            ok=True, label="Producer: Dispatch W0 -> Streaming Reduce", label2="Producer: Dispatch W1", streaming=10,
            render=can_draw)),
        ("Model estimate counter names", fx_sm103(curve_prefix="Model estimate: "), {}, dict(ok=False, rule="hw_provenance")),
        ("otherData.source = Model estimates", fx_sm103(source="Model estimates from routing, configuration and software intervals"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("tile-volume reconstruction after ERR_NVGPUCTRPERM", fx_sm103(
            source="nsys CUPTI kernel window + %globaltimer exact tile/route volume (PMU sampling returned ERR_NVGPUCTRPERM)"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("hyphenated tile-volume reconstruction", fx_sm103(
            source="nsys CUPTI kernel window + tile-volume reconstruction of FLOPs"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("underscore per_tile volume reconstruction", fx_sm103(
            source="nsys kernel window + per_tile_volume model"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("modelled NVLink provenance", fx_sm103(source="nsys kernel window, modelled NVLink load"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("real counters merely interpolated for display stay accepted", fx_sm103(
            source="nsys GPU metrics sampled at 3.9 us, linearly interpolated to a 5 us grid"), {}, dict(ok=True)),
        ("mi308x per-tile FLOP / HBM / xGMI volume curves", fx_mi308x(
            source="rocprofv3 kernel window + exact per-tile FLOP / HBM / xGMI volume (5.0 us Hann window)"),
         {}, dict(ok=False, rule="hw_provenance")),
        ("no recorded real-counter provenance", fx_sm103(provenance=False), {}, dict(ok=False, rule="hw_provenance")),
        ("flat-zero NVLink curve", fx_sm103(comm_zero=True), {}, dict(ok=False, rule="hw_comm_zero")),
        ("HBM sample 100.5 %", fx_sm103(bad=100.5), {}, dict(ok=False, rule="hw_range")),
        ("HBM sample -0.25 %", fx_sm103(bad=-0.25), {}, dict(ok=False, rule="hw_range")),
        ("HBM sample NaN", fx_sm103(bad=float("nan")), {}, dict(ok=False, rule="hw_range")),
        ("missing active-SM curve", fx_sm103(drop_active=True), {}, dict(ok=False, rule="hw_curves")),
        ("forbidden string in banner", fx_sm103(), dict(config_banner=bad_banner), dict(ok=False, rule="forbidden_string")),
        ("forbidden string in a Perfetto track title", fx_sm103(title_hit=title_hit), {}, dict(ok=False, rule="forbidden_string")),
        ("numeric Perfetto slice name carrying a forbidden token", fx_numeric_name(), {},
         dict(ok=False, rule="forbidden_string")),
        ("malformed slice timestamp is reported, not crashed", fx_malformed_ts(), {},
         dict(ok=False, rule="sw_stage_map")),
        ("negative slice duration is a sw_window error, not a crash", fx_negative_dur(), {},
         dict(ok=False, rule="sw_window")),
        ("non-object trace event is a sw_stage_map error, not a crash", fx_non_object_event(), {},
         dict(ok=False, rule="sw_stage_map")),
        ("Combine_Reduce slice hidden in a wait span", fx_h20(reduce_in_wait=True), {}, dict(ok=False, rule="sw_reduce_dropped")),
        ("SM 0 track without a Panel C row", fx_h20(extra_tid=True), {}, dict(ok=False, rule="sw_track_hidden")),
    ]
    failed = 0
    for title, doc, kw, exp in cases:
        rep = validate_doc(doc, name=f"<fixture: {title}>", renderer=R, render_check=exp.get("render", False), **kw)
        problems = []
        if rep["ok"] != exp["ok"]:
            problems.append(f"ok={rep['ok']} (want {exp['ok']}); errors={[e['message'] for e in rep['errors']]}")
        if "rule" in exp and exp["rule"] not in {e["rule"] for e in rep["errors"]}:
            problems.append(f"missing rule {exp['rule']!r}; got {[e['rule'] for e in rep['errors']]}")
        labels = rep["checks"].get("panel_c_tracks", [])
        for key in ("label", "label2"):
            if key in exp and exp[key] not in labels:
                problems.append(f"Panel C label {exp[key]!r} not in {labels}")
        if "streaming" in exp and rep["checks"].get("reduce", {}).get("streaming_slices") != exp["streaming"]:
            problems.append(f"streaming_slices={rep['checks'].get('reduce', {}).get('streaming_slices')} "
                            f"(want {exp['streaming']})")
        if "note_contains" in exp:
            note = rep["checks"].get("panel_a_rdma_unavailable_note", "")
            missing = [s for s in exp["note_contains"] if s not in note]
            if missing:
                problems.append(f"Panel A RDMA unavailable note missing {missing}; note={note!r}")
            leaked = [s for s in exp.get("note_excludes", []) if s in note]
            if leaked:
                problems.append(f"Panel A RDMA note leaks internal strings {leaked}; note={note!r}")
        if exp.get("fabric") and exp.get("render") and "render" in rep["checks"]:
            fab = rep["checks"]["render"].get("fabric") or {}
            want_r, want_x = {R.RDMA_TX, R.RDMA_CB}, {R.XGMI_TX, R.XGMI_CB}
            if not fab.get("rdma_xgmi_disjoint"):
                problems.append(f"Inter-Node RDMA / Intra-Node xGMI colors not disjoint: {fab}")
            for where in ("panel_b_legend", "panel_c_legend"):
                if not (want_r | want_x) <= set(fab.get(where, [])):
                    problems.append(f"{where} lacks separate RDMA / xGMI swatches: {fab.get(where)}")
        if exp.get("render") and "render" not in rep["checks"]:
            problems.append("in-memory render check did not run")
        failed += bool(problems)
        extra = ""
        if exp.get("render") and "render" in rep["checks"]:
            r = rep["checks"]["render"]
            extra = (f"  [render {r['px'][0]}x{r['px'][1]}, {r['panel_c_reduce_pixel_probes_ok']} "
                     f"{'fabric-colored' if r.get('fabric') else 'purple'} pixel probes ok]")
        print(f"{'ok  ' if not problems else 'FAIL'}  {title}{extra}")
        for p in problems:
            print(f"        {p}")
    n_cases = len(cases) + 1
    if can_draw:   # sabotage the z-order: the rendered-pixel probe must catch hidden purple slices
        n_cases += 1
        saved = list(R.STAGES)
        R.STAGES[:] = [R.REDUCE_STAGE] + [s for s in saved if s != R.REDUCE_STAGE]
        try:
            rep = validate_doc(fx_sm103(), name="<fixture: z-order sabotage>", renderer=R, render_check=True)
        finally:
            R.STAGES[:] = saved
        rules = [e["rule"] for e in rep["errors"]]
        ok = "sw_reduce_hidden" in rules
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  Combine / Reduce drawn UNDER the 8a. sync slice is caught by the pixel probe"
              + ("" if ok else f"  (errors: {rules})"))
    if can_draw:   # sabotage the EP16 palette: RDMA Combine painted xGMI Combine blue must be rejected
        n_cases += 1
        saved_c = dict(R.FABRIC_COLOR)
        R.FABRIC_COLOR[R.RDMA_CB] = R.FABRIC_COLOR[R.XGMI_CB]
        try:
            rep = validate_doc(fx_mi308x_ep16_rdma_unavailable(), name="<fixture: fabric color sabotage>",
                               renderer=R, render_check=True)
        finally:
            R.FABRIC_COLOR.clear()
            R.FABRIC_COLOR.update(saved_c)
        rules = [e["rule"] for e in rep["errors"]]
        ok = "fabric_palette" in rules
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  EP16 Inter-Node RDMA sharing the Intra-Node xGMI color is rejected"
              + ("" if ok else f"  (errors: {rules})"))
    # the comm-zero red line must surface as a ValueError when called directly
    doc = fx_sm103(comm_zero=True)
    cfg = R.platform_cfg("sm103")
    pn, tn = R.trace_names(doc["traceEvents"])
    try:
        R.check_hw_red_lines(doc, cfg, R.collect_counters(doc["traceEvents"], cfg), pn, tn)
        print("FAIL  flat-zero NVLink did not raise")
        failed += 1
    except ValueError as exc:
        print(f"ok    flat-zero NVLink raises ValueError ({type(exc).__name__}: {str(exc)[:70]}...)")
    print(f"self-test: {n_cases - failed}/{n_cases} passed"
          + ("" if can_draw else " (matplotlib missing: render checks skipped)"))
    return 0 if failed == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", type=Path, help="*.sw_hw.perfetto.json.gz (or plain .json) traces")
    ap.add_argument("--platform", choices=["auto", "sm103", "h20", "mi308x"], default="auto")
    ap.add_argument("--config-banner", default=None, help="banner to audit (default: the renderer preset banner)")
    ap.add_argument("--repro-script", default=None, help="repro line to audit (default: the renderer preset script)")
    ap.add_argument("--no-banner", action="store_true",
                    help="audit / draw the banner-free figure (empty banner and repro line, same 3520x2244 geometry)")
    ap.add_argument("--json-out", type=Path, default=None, help="write the per-trace reports as JSON")
    ap.add_argument("--render-check", action="store_true", help="also draw in memory and run the pixel / text gates")
    ap.add_argument("--render-out", type=Path, default=None, help="save the --render-check PNG (one trace only)")
    ap.add_argument("--renderer", type=Path, default=RENDERER_PATH)
    ap.add_argument("--require-streaming-reduce", action="store_true",
                    help="hard gate: fail when no comm track streams Combine_Reduce inside Middle "
                         "(dedicated WEAVE_REDUCE_WARPS overlap absent)")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic in-memory rule fixtures")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test(args.renderer)
    if not args.traces:
        if args.json_out is not None or args.render_check or args.render_out is not None:
            ap.error("--json-out / --render-check / --render-out need at least one trace")
        # Bare invocation (the repository gate): run the synthetic rule fixtures.
        print("[INFO] no trace given: running the built-in --self-test only "
              "(pass *.sw_hw.perfetto.json.gz traces to validate them)")
        return run_self_test(args.renderer)
    if args.render_out is not None and len(args.traces) != 1:
        ap.error("--render-out needs exactly one trace")
    R = load_renderer(args.renderer)
    banner = "" if args.no_banner else args.config_banner
    repro = "" if args.no_banner else args.repro_script
    reports = []
    for p in args.traces:
        rep = validate_trace(p, platform=args.platform, config_banner=banner,
                             repro_script=repro, render_check=args.render_check or args.render_out is not None,
                             render_out=args.render_out, renderer=R,
                             require_streaming_reduce=args.require_streaming_reduce)
        print_report(rep)
        reports.append(rep)
    n_ok = sum(r["ok"] for r in reports)
    print(f"summary: {n_ok}/{len(reports)} trace(s) passed every red line")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([_json_ready(r) for r in reports], ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"report -> {args.json_out}")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
