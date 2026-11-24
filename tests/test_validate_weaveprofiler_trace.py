#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge-case tests for tests/validate_weaveprofiler_trace.py (the P9 validator).

Edge-case suite for the WeaveProfiler trace validator.  Only tiny
in-memory synthetic fixtures (reused from the validator's own --self-test fixtures) are
used; nothing here is a measurement.  The two file-level tests write their fixtures into
a repository-local scratch directory (tests/.scratch_validate_trace/, removed afterwards)
instead of the system temp folder.

Run:  python tests/test_validate_weaveprofiler_trace.py      (standalone)
      pytest tests/test_validate_weaveprofiler_trace.py      (pytest)
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import io
import json
import random
import subprocess
import sys
import shutil
import uuid
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p9_validator_under_test", ROOT / "tests" / "validate_weaveprofiler_trace.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
R = V.load_renderer()

SM103_COMM = "NVLink RX User Data (%)"
SM103_COMM_TX = "NVLink TX User Data (%)"
SM103_MERGED = "NVLink RX/TX User Data (%)"
SM103_HBM = "HBM / TMA Bandwidth (%)"
SM103_COMPUTE = "Tensor Active / MMA Pipe (%)"
SM103_ACTIVE = "Active SMs (%)"


def _counters(doc, name):
    return [e for e in doc["traceEvents"] if e.get("ph") == "C" and e.get("name") == name]


def _rules(rep):
    return {e["rule"] for e in rep["errors"]}


def _check(doc, **kw):
    return V.validate_doc(doc, name="<test>", **kw)


def _assert_fails(doc, rule, **kw):
    rep = _check(doc, **kw)
    assert not rep["ok"], f"expected failure [{rule}] but trace passed"
    assert rule in _rules(rep), (f"want rule {rule!r}, got {sorted(_rules(rep))}: "
                                 f"{[e['message'] for e in rep['errors']][:2]}")
    return rep


def _write_gz(doc, path):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return Path(path)


SCRATCH_ROOT = ROOT / "tests" / ".scratch_validate_trace"


@contextmanager
def _scratch_dir():
    """Repository-local throw-away directory (never the system temp folder)."""
    path = SCRATCH_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            SCRATCH_ROOT.rmdir()          # only succeeds when no other run is using it
        except OSError:
            pass


# ------------------------------------------------------------ valid baselines (all platforms)
def test_valid_baselines_pass_on_every_platform():
    for fx, plat in ((V.fx_sm103, "sm103"), (V.fx_h20, "h20"), (V.fx_mi308x, "mi308x")):
        rep = _check(fx())
        assert rep["ok"], (plat, rep["errors"])
        assert rep["platform"] == plat
        assert rep["checks"]["hw"]["comm_max"] > 0.0


def test_platform_specific_metric_names():
    # curve names of one platform forced onto another platform must not pass
    for fx, wrong in ((V.fx_h20, "sm103"), (V.fx_sm103, "h20"), (V.fx_mi308x, "h20"), (V.fx_h20, "mi308x")):
        rep = _check(fx(), platform=wrong)
        assert not rep["ok"] and rep["errors"], (fx.__name__, wrong)


def test_auto_detect_failure_is_reported():
    doc = {"traceEvents": [{"ph": "X", "pid": 5, "tid": 1, "ts": 1.0, "dur": 1.0, "name": "x"}], "otherData": {}}
    rep = _check(doc)
    assert not rep["ok"] and rep["errors"] and rep["platform"] is None


# ------------------------------------------------------------ counter value gates
def test_counter_range_boundaries_accepted():
    d = V.fx_sm103()
    pts = _counters(d, SM103_HBM)
    pts[0]["args"]["value"] = 0.0
    pts[1]["args"]["value"] = 100.0
    assert _check(d)["ok"]


def test_counter_over_100_each_role_rejected():
    for nm, v in ((SM103_HBM, 100.5), (SM103_COMPUTE, 101.0), (SM103_COMM, 120.0)):
        d = V.fx_sm103()
        _counters(d, nm)[10]["args"]["value"] = v
        _assert_fails(d, "hw_range")


def test_counter_negative_nan_inf_rejected():
    for bad in (-0.25, float("nan"), float("inf"), float("-inf")):
        d = V.fx_sm103()
        _counters(d, SM103_COMPUTE)[5]["args"]["value"] = bad
        _assert_fails(d, "hw_range")


def test_non_finite_timestamp_rejected():
    d = V.fx_sm103()
    _counters(d, SM103_COMPUTE)[5]["ts"] = float("nan")
    rep = _check(d)
    assert not rep["ok"]


def test_none_or_missing_counter_value_is_error_not_crash():
    d = V.fx_sm103()
    _counters(d, SM103_HBM)[3]["args"]["value"] = None
    _assert_fails(d, "hw_curves")
    d = V.fx_sm103()
    _counters(d, SM103_HBM)[3]["args"] = {}
    _assert_fails(d, "hw_curves")


def test_boolean_counter_value_rejected():      # renderer collect_counters() bool guard (HR4 edge case)
    d = V.fx_sm103()
    _counters(d, SM103_HBM)[3]["args"]["value"] = True
    _assert_fails(d, "hw_curves")



def test_unsorted_counter_events_accepted():
    d = V.fx_sm103()
    random.Random(7).shuffle(d["traceEvents"])
    rep = _check(d)
    assert rep["ok"], rep["errors"]


# ------------------------------------------------------------ curve set / interconnect
def test_missing_each_curve_rejected():
    for nm in (SM103_COMPUTE, SM103_HBM, SM103_COMM, SM103_COMM_TX, SM103_ACTIVE):
        d = V.fx_sm103()
        d["traceEvents"] = [e for e in d["traceEvents"] if not (e.get("ph") == "C" and e.get("name") == nm)]
        _assert_fails(d, "hw_curves")


def test_sm103_nvlink_rx_tx_are_two_separate_curves():
    rep = _check(V.fx_sm103())
    assert rep["ok"], rep["errors"]
    hw = rep["checks"]["hw"]
    assert hw["comm_max"] > 0.0 and hw["comm_tx_max"] > 0.0
    assert hw["curves"]["comm"]["name"] == SM103_COMM and hw["curves"]["comm_tx"]["name"] == SM103_COMM_TX
    assert hw["curves"]["comm"]["label"] == "NVLink RX (In) %"
    assert hw["curves"]["comm_tx"]["label"] == "NVLink TX (Out) %"
    assert hw["curves"]["hbm"]["label"].endswith("DRAM Read + Write %")
    labels = [lab for _, lab in R.PLATFORMS["sm103"]["curves"].values()]
    legend = [lab for lab in labels if lab.startswith("NVLink")]
    assert legend == ["NVLink RX (In) %", "NVLink TX (Out) %"], legend
    assert R.curve_order(R.PLATFORMS["sm103"]) == ["compute", "hbm", "comm", "comm_tx", "active"]
    assert not any("RX/TX" in lab or "RX+TX" in lab for lab in legend), legend


def test_sm103_merged_nvlink_rx_plus_tx_track_rejected():
    # the legacy merged track (RX % + TX % summed) is a red line on sm103
    d = V.fx_sm103()
    for e in d["traceEvents"]:
        if e.get("ph") == "C" and e.get("name") == SM103_COMM:
            e["name"] = SM103_MERGED
    d["traceEvents"] = [e for e in d["traceEvents"] if not (e.get("ph") == "C" and e.get("name") == SM103_COMM_TX)]
    _assert_fails(d, "hw_nvlink_merged")
    d = V.fx_sm103()   # an extra merged track next to the two direction tracks is rejected too
    d["traceEvents"] += [{"ph": "C", "pid": 90000, "ts": i * 5.0, "name": "NVLink RX+TX (%)",
                          "args": {"value": 10.0}} for i in range(60)]
    _assert_fails(d, "hw_nvlink_merged")


def test_sm103_flat_zero_nvlink_tx_rejected():
    d = V.fx_sm103()
    for e in _counters(d, SM103_COMM_TX):
        e["args"]["value"] = 0.0
    _assert_fails(d, "hw_comm_zero")


def test_unexpected_extra_counter_track_rejected():
    d = V.fx_sm103()
    d["traceEvents"] += [{"ph": "C", "pid": 90000, "ts": float(i), "name": "Rogue Counter (%)", "args": {"value": 5.0}}
                         for i in range(5)]
    _assert_fails(d, "hw_curves")


def test_two_tracks_for_one_role_rejected():
    d = V.fx_sm103()
    d["traceEvents"] += [{"ph": "C", "pid": 90000, "ts": i * 5.0, "name": "SM Warp Issue Activity (%)",
                          "args": {"value": 33.0}} for i in range(60)]   # alias of "Active SMs (%)"
    _assert_fails(d, "hw_curves")


def test_too_few_samples_rejected():
    d = V.fx_sm103()
    keep = _counters(d, SM103_HBM)[:2]
    d["traceEvents"] = [e for e in d["traceEvents"] if e.get("name") != SM103_HBM] + keep
    _assert_fails(d, "hw_curves")


def test_unequal_sample_counts_rejected():
    d = V.fx_sm103()
    d["traceEvents"].remove(_counters(d, SM103_HBM)[-1])
    _assert_fails(d, "hw_curves")


def test_flat_zero_interconnect_rejected_and_tiny_positive_accepted():
    _assert_fails(V.fx_sm103(comm_zero=True), "hw_comm_zero")
    d = V.fx_sm103(comm_zero=True)
    _counters(d, SM103_COMM)[17]["args"]["value"] = 0.01
    assert _check(d)["ok"]


def test_mi308x_zero_xgmi_rejected():
    d = V.fx_mi308x()
    for e in _counters(d, "communication_util [xGMI P2P % of 287.2GB/s]"):
        e["args"]["value"] = 0.0
    _assert_fails(d, "hw_comm_zero")


def test_hw_curves_not_covering_sw_window_rejected():
    d = V.fx_sm103()
    for e in d["traceEvents"]:
        if e.get("ph") == "C":
            e["ts"] = float(e["ts"]) / 10.0     # curves end at ~30 us, software window ~270 us
    _assert_fails(d, "hw_alignment")


# ------------------------------------------------------------ provenance ("Model estimate" gate)
def test_model_estimate_markers_case_and_location():
    _assert_fails(V.fx_sm103(curve_prefix="MODEL ESTIMATE: "), "hw_provenance")      # counter names
    d = V.fx_sm103()
    d["otherData"]["hw_counter_source"] = "model_estimate"                           # any *source* key
    _assert_fails(d, "hw_provenance")
    d = V.fx_sm103()
    for e in d["traceEvents"]:
        if e.get("ph") == "M" and e["pid"] == 90000:
            e["args"]["name"] = "Hardware Counters (synthetic)"                      # counter-track title
    _assert_fails(d, "hw_provenance")
    _assert_fails(V.fx_sm103(source="analytic roofline curves"), "hw_provenance")
    _assert_fails(V.fx_sm103(source="simulated utilization"), "hw_provenance")
    _assert_fails(V.fx_sm103(source="values extrapolated past the kernel window"), "hw_provenance")


def test_missing_or_modelled_provenance_rejected_on_amd():
    _assert_fails(V.fx_mi308x(source=None), "hw_provenance")
    _assert_fails(V.fx_mi308x(source="rocprofv3 model estimate of SQ_BUSY_CU"), "hw_provenance")


def test_forbidden_word_in_slice_args_is_only_a_warning():
    d = V.fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 0, "ts": 1.0, "dur": 1.0,
                             "name": "1. Opening: Top-K Histogram & Route Publish",
                             "args": {"warp": 0, "stage_id": 1, "note": "WEAVE_HOT_COLD_MODE=1"}})
    rep = _check(d)
    assert rep["ok"], rep["errors"]
    assert any(w["rule"] == "forbidden_string_metadata" for w in rep["warnings"])


# ------------------------------------------------------------ forbidden display strings
def test_every_forbidden_token_rejected_in_banner_repro_and_title():
    for tok in R.FORBIDDEN:
        _assert_fails(V.fx_sm103(), "forbidden_string", config_banner=f"Spatial: {tok.upper()} preview")
        _assert_fails(V.fx_sm103(), "forbidden_string", repro_script=f"examples/{tok}.sh")
        _assert_fails(V.fx_sm103(title_hit=" " + tok), "forbidden_string")


def test_forbidden_in_thread_name():
    d = V.fx_h20()
    for e in d["traceEvents"]:
        if e.get("name") == "thread_name" and e["pid"] == 100 and e["tid"] == 1:
            e["args"]["name"] += " " + R.FORBIDDEN[2]
    _assert_fails(d, "forbidden_string")


# ------------------------------------------------------------ Combine_Reduce / stage_id == 14
def test_stage_id_14_float_and_string_forms_are_reduce():
    for raw in (14, 14.0, "14"):
        d = V.fx_sm103(new_reduce_name=True)
        for e in d["traceEvents"]:
            if e.get("ph") == "X" and str(e["name"]).startswith("9z."):
                e["args"]["stage_id"] = raw
        rep = _check(d)
        assert rep["ok"], (raw, rep["errors"])
        assert rep["checks"]["sw"]["stage_counts"].get("Combine / Reduce", 0) > 0


def test_stage_id_14_5_is_not_reduce_so_unknown_name_is_unmapped():
    d = V.fx_sm103(new_reduce_name=True)
    for e in d["traceEvents"]:
        if e.get("ph") == "X" and str(e["name"]).startswith("9z."):
            e["args"]["stage_id"] = 14.5
    _assert_fails(d, "sw_stage_map")


def test_stage_id_14_on_a_non_reduce_name_is_a_contradiction():
    d = V.fx_sm103()
    for e in d["traceEvents"]:
        if e.get("ph") == "X" and e["name"].startswith("3. Dispatch Tile"):
            e["args"]["stage_id"] = 14
            break
    _assert_fails(d, "sw_stage_map")


def test_trace_without_reduce_slices_shows_zero_reduce_in_report():
    d = V.fx_h20()
    d["traceEvents"] = [e for e in d["traceEvents"] if e.get("name") != "Combine_Reduce"]
    rep = _check(d)
    assert rep["checks"]["sw"]["stage_counts"].get("Combine / Reduce", 0) == 0
    assert rep["checks"]["reduce"]["comm_reduce_slices"] == 0


def test_streaming_vs_tail_vs_early_reduce_report_counts():
    rep = _check(V.fx_sm103(streaming=True))
    assert rep["ok"] and rep["checks"]["reduce"]["streaming_slices"] == 72
    assert rep["checks"]["reduce"]["unit0_streaming_tracks"] == [2, 3]
    assert "W02..W03 (Dispatch -> Streaming Reduce)" in rep["checks"]["panel_c_tracks"]
    for kw in (dict(streaming=False), dict(streaming_before_middle=True)):
        rep = _check(V.fx_sm103(**kw))
        assert rep["ok"] and rep["checks"]["reduce"]["streaming_slices"] == 0
        assert not any("Streaming" in t for t in rep["checks"]["panel_c_tracks"])


def test_reduce_slice_hidden_in_drop_span_rejected():
    _assert_fails(V.fx_h20(reduce_in_wait=True), "sw_reduce_dropped")


# ------------------------------------------------------------ malformed / degenerate documents
def test_degenerate_documents_report_errors_not_exceptions():
    for bad in ({}, {"traceEvents": None}, {"traceEvents": "x"}, [], "x", None):
        rep = V.validate_doc(bad, name="<bad>")
        assert not rep["ok"] and rep["errors"]
    rep = V.validate_doc({"traceEvents": [], "otherData": {"meta": {"card": "sm103"}}}, name="<empty>")
    assert not rep["ok"] and rep["errors"]          # empty tracks: no slices, no counters


def test_malformed_slices_and_non_object_events():
    d = V.fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "ts": 5.0, "dur": float("inf"),
                             "name": "1. Opening: Top-K Histogram & Route Publish"})
    _assert_fails(d, "sw_window")
    d = V.fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "ts": "abc", "dur": 1.0,
                             "name": "1. Opening: Top-K Histogram & Route Publish"})
    _assert_fails(d, "sw_stage_map")
    d = V.fx_sm103()
    d["traceEvents"].append(None)
    _assert_fails(d, "sw_stage_map")


def test_unknown_slice_name_rejected():
    d = V.fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 100000, "tid": 1, "ts": 1.0, "dur": 1.0,
                             "name": "Totally Unknown Stage", "args": {"warp": 1}})
    _assert_fails(d, "sw_stage_map")


def test_slices_outside_unit_pid_range_are_ignored():
    d = V.fx_sm103()
    d["traceEvents"].append({"ph": "X", "pid": 5, "tid": 1, "ts": 1.0, "dur": 1.0, "name": "Totally Unknown Stage"})
    assert _check(d)["ok"]


def test_duplicate_process_metadata_and_duplicate_slices_tolerated():
    d = V.fx_sm103()
    d["traceEvents"] += [e for e in d["traceEvents"] if e.get("ph") == "M"]      # duplicated pid metadata
    d["traceEvents"] += [copy.deepcopy(e) for e in d["traceEvents"]
                         if e.get("ph") == "X" and e["pid"] == 100001][:5]
    rep = _check(d)
    assert rep["ok"], rep["errors"]


def test_sm0_track_without_panel_c_row_rejected():
    _assert_fails(V.fx_h20(extra_tid=True), "sw_track_hidden")


def test_h20_missing_thread_name_rejected():
    d = V.fx_h20()
    d["traceEvents"] = [e for e in d["traceEvents"] if not (e.get("name") == "thread_name" and e.get("tid") == 2)]
    _assert_fails(d, "sw_roles")


def test_launch_skew_on_sm103_rejected():
    d = V.fx_sm103()
    for e in d["traceEvents"]:
        if e.get("ph") == "X" and e["pid"] == 100000 and e["name"].startswith("2. Dispatch"):
            e["dur"] = 40.0
    _assert_fails(d, "sw_window")


# ------------------------------------------------------------ file handling, report JSON, CLI
def test_gz_plain_missing_and_corrupt_files():
    with _scratch_dir() as td:
        doc = V.fx_sm103()
        gz = _write_gz(doc, td / "a.sw_hw.perfetto.json.gz")
        plain = td / "a.json"
        plain.write_text(json.dumps(doc), encoding="utf-8")
        r1, r2 = V.validate_trace(gz), V.validate_trace(plain)
        assert r1["ok"] and r2["ok"] and r1["platform"] == r2["platform"] == "sm103"
        assert len(r1["trace_sha256"]) == 64 and r1["trace_bytes"] == gz.stat().st_size
        miss = V.validate_trace(td / "nope.json.gz")
        assert not miss["ok"] and miss["errors"][0]["rule"] == "load"
        bad_gz = td / "bad.json.gz"
        bad_gz.write_bytes(b"not gzip")
        assert V.validate_trace(bad_gz)["errors"][0]["rule"] == "load"
        bad_json = td / "bad.json"
        bad_json.write_text("{truncated", encoding="utf-8")
        assert V.validate_trace(bad_json)["errors"][0]["rule"] == "load"


def test_cli_exit_codes_and_json_report():
    """Real subprocess invocations: 0 = all pass (bare call = self-test), 1 = any red line, 2 = usage error."""
    script = ROOT / "tests" / "validate_weaveprofiler_trace.py"

    def run(*args):
        return subprocess.run([sys.executable, "-I", str(script), *map(str, args)], capture_output=True,
                              text=True, encoding="utf-8", timeout=300)

    with _scratch_dir() as td:
        good = _write_gz(V.fx_sm103(), td / "good.json.gz")
        bad = _write_gz(V.fx_sm103(comm_zero=True), td / "bad.json.gz")
        rep = td / "sub" / "report.json"
        r = run(good, "--json-out", rep)
        assert r.returncode == 0 and "PASS" in r.stdout and "1/1 trace(s) passed" in r.stdout, r.stdout + r.stderr
        data = json.loads(rep.read_text(encoding="utf-8"))
        assert data[0]["ok"] is True and data[0]["platform"] == "sm103" and "trace_path_b64" in data[0]
        r = run(good, bad)
        assert r.returncode == 1 and "FAIL" in r.stdout and "hw_comm_zero" in r.stdout and "1/2" in r.stdout
        assert run(bad, "--platform", "sm103").returncode == 1
        assert run(td / "missing.gz").returncode == 1
        bare = run()
        assert bare.returncode == 0 and "self-test:" in bare.stdout, bare.stdout + bare.stderr
        assert run("--json-out", td / "r.json").returncode == 2
        assert run(good, bad, "--render-out", td / "x.png").returncode == 2


def test_validator_builtin_self_test_passes():
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert V.run_self_test() == 0
    assert "self-test:" in buf.getvalue()


# ------------------------------------------------------------ EP16 per-wave Panel C + 5th RDMA NIC curve
EP16_PER_WAVE_LABELS = [
    "W00 (Compute / Combine)", "W01 (Compute / Combine)",
    "W02 (Compute / Combine)", "W03 (Compute / Combine)",
    "W04 (Comm / Scatter -> Streaming Reduce)", "W05 (Comm / Scatter -> Streaming Reduce)",
    "W06 (Comm / Scatter -> Streaming Reduce)", "W07 (Comm / Scatter -> Streaming Reduce)",
]


def test_ep16_per_wave_labels_and_rdma_curve():
    rep = _check(V.fx_mi308x_ep16(), render_check=True)
    assert rep["ok"], rep["errors"]
    assert rep["checks"]["panel_c_tracks"] == EP16_PER_WAVE_LABELS, rep["checks"]["panel_c_tracks"]
    hw = rep["checks"]["hw"]
    assert hw["samples_per_curve"] == 60 and hw["comm_max"] > 0.0
    assert hw["rdma_max"] is not None and hw["rdma_max"] > 0.0
    assert "rdma" in hw["curves"] and hw["curves"]["rdma"]["points"] == 60
    assert rep["checks"]["reduce"]["streaming_slices"] == 48
    assert rep["checks"]["reduce"]["unit0_streaming_tracks"] == [4, 5, 6, 7]
    r = rep["checks"]["render"]          # full 3520x2244 draw: no text overlap, crisp purple probes
    assert r["px"] == [3520, 2244] and r["panel_c_reduce_pixel_probes_ok"] == r["panel_c_reduce_rects"]


def test_ep16_four_curve_trace_still_per_wave():
    rep = _check(V.fx_mi308x_ep16(rdma=False))
    assert rep["ok"], rep["errors"]
    assert rep["checks"]["panel_c_tracks"] == EP16_PER_WAVE_LABELS
    assert rep["checks"]["hw"]["rdma_max"] is None
    assert "rdma" not in rep["checks"]["hw"]["curves"]


def test_ep8_mi308x_merged_labels_unchanged():
    rep = _check(V.fx_mi308x())
    assert rep["ok"], rep["errors"]
    assert rep["checks"]["panel_c_tracks"] == [
        "W00..W03 (Compute / Combine)", "W04..W07 (Comm / Scatter -> Streaming Reduce)"]


def test_ep16_rdma_zero_rejected():
    _assert_fails(V.fx_mi308x_ep16(rdma_zero=True), "hw_rdma_zero")


def test_ep16_rdma_range_rejected():
    for bad in (100.5, -0.25, float("nan"), float("inf")):
        _assert_fails(V.fx_mi308x_ep16(rdma_bad=bad), "hw_range")


def test_ep16_rdma_requires_nic_sysfs_provenance():
    _assert_fails(V.fx_mi308x_ep16(nic_prov=False), "hw_provenance")
    d = V.fx_mi308x_ep16()
    d["otherData"]["meta"]["curve_sources"][V.MI_RDMA_CURVE] = "NIC utilization from routing volumes"
    _assert_fails(d, "hw_provenance")


def test_ep16_rdma_rejected_on_single_node_ep8():
    _assert_fails(V.fx_mi308x_ep16(ep=8), "hw_curves")


def test_ep16_rdma_unequal_sample_counts_rejected():
    d = V.fx_mi308x_ep16()
    rdma_evs = _counters(d, V.MI_RDMA_CURVE)
    d["traceEvents"].remove(rdma_evs[-1])
    _assert_fails(d, "hw_curves")


H20_RX = "communication_rx_util [NVLink4 RX (In) % of 450GB/s]"
H20_TX = "communication_tx_util [NVLink4 TX (Out) % of 450GB/s]"
H20_MERGED = "communication_util [NVLink4 % of 450GB/s]"


def test_h20_nvlink_rx_tx_are_two_separate_panel_a_curves():
    """NVLink4 is full duplex: H20 Panel A draws RX (In) and TX (Out) as two lines."""
    d = V.fx_h20()
    cfg = R.trace_cfg(R.platform_cfg("h20"), d)
    pn, tn = R.trace_names(d["traceEvents"])
    hw = R.check_hw_red_lines(d, cfg, R.collect_counters(d["traceEvents"], cfg), pn, tn)
    legend = dict((role, lab) for role, _, lab in R.panel_a_legend(cfg, hw))
    assert legend["comm"].startswith("NVLink4 RX (In)"), legend
    assert legend["comm_tx"].startswith("NVLink4 TX (Out)"), legend
    assert hw["role_src"]["comm"] == H20_RX and hw["role_src"]["comm_tx"] == H20_TX
    assert R.CURVE_COLOR["comm"] != R.CURVE_COLOR["comm_tx"]
    assert set(hw["comm_maxes"]) == {"comm", "comm_tx"}


def test_h20_merged_nvlink_curve_rejected():
    d = V.fx_h20()
    for e in d["traceEvents"]:
        if e.get("ph") == "C" and e.get("name") == H20_RX:
            e["name"] = H20_MERGED
    d["traceEvents"] = [e for e in d["traceEvents"] if not (e.get("ph") == "C" and e.get("name") == H20_TX)]
    _assert_fails(d, "hw_curves")


def test_h20_missing_tx_curve_rejected():
    d = V.fx_h20()
    d["traceEvents"] = [e for e in d["traceEvents"] if not (e.get("ph") == "C" and e.get("name") == H20_TX)]
    _assert_fails(d, "hw_curves")


def test_h20_flat_zero_tx_curve_rejected():
    d = V.fx_h20()
    for e in _counters(d, H20_TX):
        e["args"]["value"] = 0.0
    _assert_fails(d, "hw_comm_zero")


def test_ep16_rdma_legend_distinguishes_the_two_fabrics():
    d = V.fx_mi308x_ep16()
    cfg = R.trace_cfg(R.platform_cfg("mi308x"), d)
    pn, tn = R.trace_names(d["traceEvents"])
    hw = R.check_hw_red_lines(d, cfg, R.collect_counters(d["traceEvents"], cfg), pn, tn)
    legend = dict((role, lab) for role, _, lab in R.panel_a_legend(cfg, hw))
    assert legend["comm"].startswith("Intra-Node xGMI"), legend
    assert legend["rdma"] == "Inter-Node RDMA (NIC) \u2014 % of 25GB/s", legend
    drawn = [t for w, t in R.display_strings(cfg, dict(hw=hw)) if w == "panel A legend"]
    assert any("Intra-Node xGMI" in t for t in drawn) and any("Inter-Node RDMA (NIC)" in t for t in drawn)
    # EP8 (no rdma) keeps the original xGMI legend verbatim
    d8 = V.fx_mi308x()
    cfg8 = R.trace_cfg(R.platform_cfg("mi308x"), d8)
    pn8, tn8 = R.trace_names(d8["traceEvents"])
    hw8 = R.check_hw_red_lines(d8, cfg8, R.collect_counters(d8["traceEvents"], cfg8), pn8, tn8)
    legend8 = dict((role, lab) for role, _, lab in R.panel_a_legend(cfg8, hw8))
    assert legend8["comm"] == "xGMI \u2014 P2P % of 287.2GB/s" and "rdma" not in legend8


def _note_cfg_hw(doc):
    cfg = R.trace_cfg(R.platform_cfg("mi308x"), doc)
    pn, tn = R.trace_names(doc["traceEvents"])
    hw = R.check_hw_red_lines(doc, cfg, R.collect_counters(doc["traceEvents"], cfg), pn, tn)
    return cfg, hw


PANEL_A_INTERNAL_STRINGS = ("SKILL.md", "nic_capture", "fic2_soe_bond")


def test_ep16_rdma_unavailable_note_documents_fic2_soe():
    """No collectable Inter-Node RDMA NIC counter: Panel A labels the xGMI curve
    Intra-Node and draws ONE clean footnote line; the real device evidence stays in
    the validator report (panel_a_rdma_evidence), never on the figure."""
    d = V.fx_mi308x_ep16_rdma_unavailable()
    cfg, hw = _note_cfg_hw(d)
    legend = dict((role, lab) for role, _, lab in R.panel_a_legend(cfg, hw))
    assert legend["comm"].startswith("Intra-Node xGMI"), legend
    assert "rdma" not in legend and hw["rdma_max"] is None
    note = R.panel_a_rdma_note(cfg, hw, d)
    assert note == "Inter-Node RDMA NIC counters unavailable on this HCA (omitted, not zero-filled)", note
    evidence = R.panel_a_rdma_evidence(cfg, hw, d)
    for token in ("UNAVAILABLE", "fic2_soe_bond0..3", "Alibaba EIC SOE Driver HCA", "4: ACTIVE",
                  "300 Gb/sec (12X EDR)", "hw_counters=[]", "error: Invalid argument", "EIC/SolarRDMA"):
        assert token in evidence, (token, evidence)
    drawn = [t for w, t in R.display_strings(cfg, dict(hw=hw, doc=d)) if w == "panel A RDMA fabric note"]
    assert drawn == [note], drawn
    rep = _check(d, render_check=True)
    assert rep["ok"] and not rep["errors"] and not rep["warnings"], rep
    assert rep["checks"]["panel_a_rdma_unavailable_note"] == note
    assert rep["checks"]["panel_a_rdma_evidence"] == evidence
    r = rep["checks"]["render"]
    assert r["px"] == [3520, 2244] and r["panel_a_rdma_note_font"] is not None, r
    assert r["panel_c_reduce_pixel_probes_ok"] == r["panel_c_reduce_rects"]
    assert rep["checks"]["panel_c_tracks"] == EP16_PER_WAVE_LABELS


def test_ep16_rdma_present_draws_no_unavailable_note():
    d = V.fx_mi308x_ep16()
    cfg, hw = _note_cfg_hw(d)
    assert R.panel_a_rdma_note(cfg, hw, d) is None
    rep = _check(d, render_check=True)
    assert rep["ok"], rep["errors"]
    assert "panel_a_rdma_unavailable_note" not in rep["checks"]
    assert rep["checks"]["render"]["panel_a_rdma_note_font"] is None


def test_ep8_has_no_rdma_note_and_keeps_xgmi_legend():
    d = V.fx_mi308x()
    cfg, hw = _note_cfg_hw(d)
    assert R.panel_a_rdma_note(cfg, hw, d) is None
    legend = dict((role, lab) for role, _, lab in R.panel_a_legend(cfg, hw))
    assert legend["comm"] == "xGMI \u2014 P2P % of 287.2GB/s", legend


def test_ep16_rdma_unavailable_note_without_evidence_stays_honest():
    """Even with no NIC evidence recorded at all the note must not claim a measurement
    and must never offer a zero curve; the trace still renders and passes."""
    for kw in (dict(evidence=False, reason=False), dict(evidence=False), dict(reason=False)):
        d = V.fx_mi308x_ep16_rdma_unavailable(**kw)
        cfg, hw = _note_cfg_hw(d)
        note = R.panel_a_rdma_note(cfg, hw, d)
        assert "\n" not in note and note.endswith("(omitted, not zero-filled)"), note
        if not kw.get("evidence", True) and not kw.get("reason", True):
            # no HCA / NIC fact recorded at all -> no "on this HCA" claim is invented
            assert note == R.RDMA_NOTE_NONE and "HCA" not in note, note
        else:
            assert note == R.RDMA_NOTE_HCA, note
        rep = _check(d, render_check=True)
        assert rep["ok"] and not rep["warnings"], (kw, rep)


def test_panel_a_note_band_contains_no_real_counter_sample():
    d = V.fx_mi308x_ep16_rdma_unavailable()
    cfg, hw = _note_cfg_hw(d)
    data = R.parse(dict(cfg), doc=d)
    bands = R._panel_a_free_bands(data)
    vals = [v for pts in data["curves"].values() for _, v in pts]
    for lo, hi in bands:
        assert not any(lo < v < hi for v in vals), (lo, hi)
    assert bands[0] == (max(vals), 105.0), bands[:3]
    assert R.panel_a_rdma_note(cfg, hw, d) is not None


def test_sm103_title_follows_captured_qwen38max_workload():
    d = V.fx_sm103()
    d.setdefault("otherData", {}).setdefault("meta", {})["capture_meta"] = dict(model="qwen38max", tokens_global=1024)
    cfg = R.trace_cfg(dict(R.PLATFORMS["sm103"], platform="sm103"), d)
    assert cfg["title"] == "Weave on NVIDIA SM103 (148-SM, 4x EP4, Qwen3.8-Max E=512 top-10, S=1024)"


def test_renderer_preset_titles_match_delivered_figures():
    """The sm103 / h20 presets title the delivered 4x EP4 figures, not an EP8 cluster."""
    assert R.PLATFORMS["sm103"]["title"] == "Weave on NVIDIA SM103 (148-SM, 4x EP4, Qwen3.8-Max E=512 top-10, S=4096)"
    assert R.PLATFORMS["h20"]["title"] == "Weave on NVIDIA H20 (4x EP4, Qwen3.8-Max, S=1024)"
    for key in ("sm103", "h20"):
        assert "EP8" not in R.PLATFORMS[key]["title"], key
        assert "ep4_s1024" in R.REPRO_SCRIPTS[key], key
        assert (ROOT / R.REPRO_SCRIPTS[key]).is_file(), R.REPRO_SCRIPTS[key]


# ------------------------------------------------------------ V3: EP16 dual-color fabric palette
EP16_FABRIC_HEX = {
    "Inter-Node RDMA Dispatch": "#D62728", "Inter-Node RDMA Combine": "#F28E2B",
    "Intra-Node xGMI Dispatch": "#009FB7", "Intra-Node xGMI Combine": "#2563EB",
    "GEMM1 (FC1+SwiGLU)": "#16A34A", "GEMM2 (FC2)": "#166534",
    "Route / Sync": "#94A3B8", "Local Stage (prep)": "#8B5CF6",
}
RDMA_CLASSES = ("Inter-Node RDMA Dispatch", "Inter-Node RDMA Combine")
XGMI_CLASSES = ("Intra-Node xGMI Dispatch", "Intra-Node xGMI Combine")


def _with_rdma_dispatch(doc):
    """Fixture copy that also carries a real-named InterNode_RDMA_Dispatch slice on CU 1."""
    d = copy.deepcopy(doc)
    d["traceEvents"].append({"ph": "X", "pid": 101, "tid": 6, "name": "InterNode_RDMA_Dispatch",
                             "ts": 26.0, "dur": 4.0, "args": {"wave": 5}})
    return d


def test_ep16_fabric_palette_hex_and_rdma_xgmi_disjoint():
    assert R.FABRIC_CLASSES == list(EP16_FABRIC_HEX), R.FABRIC_CLASSES
    assert R.FABRIC_COLOR == EP16_FABRIC_HEX, R.FABRIC_COLOR
    rdma = {EP16_FABRIC_HEX[c] for c in RDMA_CLASSES}
    xgmi = {EP16_FABRIC_HEX[c] for c in XGMI_CLASSES}
    assert len(rdma) == 2 and len(xgmi) == 2 and not (rdma & xgmi)
    assert len(set(EP16_FABRIC_HEX.values())) == len(EP16_FABRIC_HEX)
    assert R.check_fabric_palette() >= R.FABRIC_MIN_RGB_DIST


def test_ep16_slice_names_map_to_their_fabric():
    cfg = R.trace_cfg(R.platform_cfg("mi308x"), V.fx_mi308x_ep16())
    assert cfg["fabric_palette"] is True
    want = {
        "InterNode_RDMA_Dispatch": "Inter-Node RDMA Dispatch", "InterNode_RDMA_Return": "Inter-Node RDMA Combine",
        "InterNode_Partial_Reduce": "Inter-Node RDMA Combine", "IntraNode_xGMI_Pull": "Intra-Node xGMI Dispatch",
        "IntraNode_xGMI_ScatterPush": "Intra-Node xGMI Combine", "IntraNode_HomeCombine": "Intra-Node xGMI Combine",
        "Compute_GEMM1_SwiGLU": "GEMM1 (FC1+SwiGLU)", "Compute_GEMM2_MFMA": "GEMM2 (FC2)",
        "Route_Plan_Local": "Route / Sync", "Route_Plan_Remote": "Route / Sync",
        "Local_Ballot_Plan": "Route / Sync", "IntraNode_LocalStage": "Local Stage (prep)",
    }
    for nm, cls in want.items():
        st = R.classify_stage(cfg, {"name": nm})
        assert R.span_class(cfg, st, nm) == cls, (nm, R.span_class(cfg, st, nm))
    # every name the EP16 exporter can emit has a fabric class
    for nm, st in cfg["stage_map"].items():
        assert R.fabric_class(cfg, nm, st) in R.FABRIC_CLASSES, nm
    try:
        R.fabric_class(cfg, "Mystery_Comm", "Dispatch / Pull")
    except R.TraceRedLineError as exc:
        assert exc.rule == "sw_stage_map"
    else:
        raise AssertionError("an unknown EP16 slice name must not be guessed into a fabric")


def test_ep16_panels_b_and_c_render_distinct_rdma_and_xgmi_swatches():
    for doc in (_with_rdma_dispatch(V.fx_mi308x_ep16_rdma_unavailable()), _with_rdma_dispatch(V.fx_mi308x_ep16())):
        rep = _check(doc, render_check=True)
        assert rep["ok"], rep["errors"]
        fab = rep["checks"]["render"]["fabric"]
        assert fab["rdma_xgmi_disjoint"] is True
        assert fab["rdma_hex"] == ["#D62728", "#F28E2B"] and fab["xgmi_hex"] == ["#009FB7", "#2563EB"], fab
        assert fab["panel_b_legend"] == list(RDMA_CLASSES + XGMI_CLASSES), fab["panel_b_legend"]
        assert fab["panel_c_legend"] == list(EP16_FABRIC_HEX), fab["panel_c_legend"]
        for where in ("panel B", "panel C"):
            for c in RDMA_CLASSES + XGMI_CLASSES:
                assert fab["legend_hex"][where][c] == EP16_FABRIC_HEX[c], (where, c)
        # both fabrics actually drawn on both panels (RDMA Dispatch on CU 1, Partial_Reduce on CU 0)
        assert {"Inter-Node RDMA Dispatch", "Inter-Node RDMA Combine",
                "Intra-Node xGMI Dispatch", "Intra-Node xGMI Combine"} <= set(fab["panel_b_classes_drawn"])
        assert {"Inter-Node RDMA Combine", "Intra-Node xGMI Combine"} <= set(fab["panel_c_classes_drawn"])
        r = rep["checks"]["render"]
        assert r["panel_c_reduce_pixel_probes_ok"] == r["panel_c_reduce_rects"] > 0
        assert rep["checks"]["panel_c_tracks"] == EP16_PER_WAVE_LABELS
        drawn = R.display_strings(R.trace_cfg(R.platform_cfg("mi308x"), doc), dict(doc=doc))
        assert [t for w, t in drawn if w == "panel B legend"] == list(RDMA_CLASSES + XGMI_CLASSES)


def test_panel_a_rdma_note_has_no_internal_strings():
    for kw in (dict(), dict(evidence=False), dict(reason=False), dict(evidence=False, reason=False)):
        d = V.fx_mi308x_ep16_rdma_unavailable(**kw)
        cfg, hw = _note_cfg_hw(d)
        for w, t in R.display_strings(cfg, dict(hw=hw, doc=d)):
            if w == "panel A RDMA fabric note":
                assert "\n" not in t, t
                for bad in PANEL_A_INTERNAL_STRINGS:
                    assert bad not in t, (bad, t)
        rep = _check(d, render_check=True)
        assert rep["ok"], (kw, rep["errors"])
        note = rep["checks"]["render"]["panel_a_rdma_note"]
        assert note and not any(bad in note for bad in PANEL_A_INTERNAL_STRINGS), note


def test_ep8_and_nvidia_keep_the_seven_stage_palette():
    for fx, plat in ((V.fx_sm103, "sm103"), (V.fx_h20, "h20"), (V.fx_mi308x, "mi308x")):
        d = fx()
        cfg = R.trace_cfg(R.platform_cfg(plat), d)
        assert not cfg.get("fabric_palette"), plat
        assert R.stage_palette(cfg) == (R.STAGES, R.STAGE_COLOR), plat
        assert R.panel_b_legend_classes(cfg) == [], plat
        assert not any(w == "panel B legend" for w, _ in R.display_strings(cfg, dict(doc=d)))
    rep = _check(V.fx_mi308x(), render_check=True)
    assert rep["ok"] and rep["checks"]["render"]["fabric"] is None, rep["errors"]


def test_fabric_palette_collision_rejected():
    saved = dict(R.FABRIC_COLOR)
    try:
        for a, b in ((RDMA_CLASSES[1], XGMI_CLASSES[1]), (RDMA_CLASSES[0], XGMI_CLASSES[0])):
            R.FABRIC_COLOR.clear(); R.FABRIC_COLOR.update(saved)
            R.FABRIC_COLOR[a] = R.FABRIC_COLOR[b]
            try:
                R.check_fabric_palette()
            except R.TraceRedLineError as exc:
                assert exc.rule == "fabric_palette"
            else:
                raise AssertionError(f"{a} sharing {b}'s color must be rejected")
        R.FABRIC_COLOR.clear(); R.FABRIC_COLOR.update(saved)
        R.FABRIC_COLOR[XGMI_CLASSES[0]] = "#00A0B8"      # two distinct hex strings, but only
        R.FABRIC_COLOR[XGMI_CLASSES[1]] = "#00A4BC"      # 4 RGB levels apart -> indistinct, rejected
        try:
            R.check_fabric_palette()
        except R.TraceRedLineError as exc:
            assert exc.rule == "fabric_palette"
        else:
            raise AssertionError("near-identical fabric colors must be rejected")
    finally:
        R.FABRIC_COLOR.clear(); R.FABRIC_COLOR.update(saved)
    rep = _check(V.fx_mi308x_ep16(), render_check=False)
    assert rep["ok"] and rep["checks"]["fabric_palette_min_rgb_dist"] >= R.FABRIC_MIN_RGB_DIST


# ------------------------------------------------------------ standalone runner
def _collect():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def main() -> int:
    failed = 0
    tests = _collect()
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:                     # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc(limit=3, file=sys.stdout)
    print(f"test_validate_weaveprofiler_trace: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
