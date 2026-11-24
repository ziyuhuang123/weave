#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU unit tests for the V2 Inter-Node RDMA NIC export chain (no GPU needed).

Synthetic FILITE02 dumps + PMC .bin/.json records (with and without the V2 NIC
raw fields) are pushed through ablation/export_mi308x_pmc_swhw.py and the
exported traces are re-validated by tests/validate_weaveprofiler_trace.py
(including the in-memory 3520x2244 render gates).  Nothing here is a measurement.

Run:  python tests/test_export_mi308x_nic.py      (standalone)
      pytest tests/test_export_mi308x_nic.py      (pytest)
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import struct
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("exporter_under_test", ROOT / "ablation" / "export_mi308x_pmc_swhw.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
_vspec = importlib.util.spec_from_file_location("validator_for_nic_test", ROOT / "tests" / "validate_weaveprofiler_trace.py")
V = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(V)

SCRATCH = ROOT / "tests" / ".scratch_export_nic"
MAGIC = 0x504D4353414D504C
OPEN, END, CLOSE = 0x40000000, 0x80000000, 0x08000000
TIMER_HZ = 100_000_000
PASS_A = ["GRBM_GUI_ACTIVE", "SQ_BUSY_CU_CYCLES", "TCC_EA0_RDREQ_DRAM", "TCC_EA0_WRREQ_DRAM"]
PASS_B = ["GRBM_GUI_ACTIVE", "SQ_VALU_MFMA_BUSY_CYCLES", "TCC_EA0_RDREQ_DRAM", "TCC_EA0_WRREQ_DRAM"]
NIC_PORTS = [{"device": "mlx5_0", "port": 1, "link_layer": "InfiniBand", "state": "4: ACTIVE",
              "source": "infiniband_hw_counters", "rate_gbps": 200.0, "rate_raw": "200 Gb/sec (4X EDR)",
              "rate_path": "/sys/class/infiniband/mlx5_0/ports/1/rate",
              "counters": [{"path": "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/tx_bytes",
                            "kind": "hw_tx_bytes", "scale": 1.0},
                           {"path": "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/rx_bytes",
                            "kind": "hw_rx_bytes", "scale": 1.0}]}]


def _fi_dump(path: Path, ep16: bool):
    """80 CTAs x 8 waves of leaf spans; CTA 0 is a 4/4 hybrid whose comm waves
    stream Partial_Reduce inside Middle before the global tail HomeCombine."""
    slots = []
    for cta in range(80):
        xcc, se, cu = cta // 20, (cta // 5) % 4, cta % 5
        key = (xcc << 12) | (se << 8) | cu   # decode_hw_aux: xcc[15:12] se[9:8] cu[3:0]
        for w in range(8):
            us = lambda v: int(v * 1000 * TIMER_HZ // 1_000_000_000)   # us -> ticks
            recs = [(us(0), OPEN | 17, key), (us(0), 17, key), (us(10), END | 17, key), (us(10), CLOSE | 17, key)]
            if w < 4:
                recs += [(us(100), 26, key), (us(200), END | 26, key),
                         (us(200), 29, key), (us(300), END | 29, key),
                         (us(900), 12, key), (us(950), END | 12, key)]
            else:
                recs += [(us(10), 16, key), (us(50), END | 16, key),
                         (us(250), 9, key), (us(300), END | 9, key)]
                for a in (120, 150, 180):
                    recs += [(us(a), 41, key), (us(a + 10), END | 41, key)]
                    # eid 12 (Home_Combine / IntraNode_HomeCombine) streams in
                    # Middle in BOTH EP worlds (EP8 does not map eid 41).
                    recs += [(us(a + 20), 12, key), (us(a + 30), END | 12, key)]
                recs += [(us(900), 12, key), (us(950), END | 12, key)]
            recs.sort(key=lambda r: r[0])
            slots.append(recs)
    meta = json.dumps({"timer_hz": TIMER_HZ, "ep": 16 if ep16 else 8}).encode()
    out = bytearray(b"FILITE02")
    out += struct.pack("<6I", len(slots), 64, 8, 0, 0, len(meta))
    out += meta
    out += struct.pack(f"<{len(slots)}I", *(len(s) for s in slots))
    for s in slots:
        for t, tag, aux in s:
            out += struct.pack("<QII", t, tag, aux)
    path.write_bytes(bytes(out))
    return path


def _pmc(path: Path, counters, nic, n=50, period_us=20.0, nic_delta=True):
    """Cumulative PMC counters + (optionally) raw NIC reads, one record per period.
    NIC tx/rx deltas alternate 50000/75000 B -> 100000/150000 B per interval ->
    5 / 7.5 GB/s = 20% / 30% of the 25 GB/s (200 Gb/s) link rate.  nic_delta=False
    keeps the raw NIC reads constant (readable counters, zero observed traffic)."""
    stride = len(counters) + (2 if nic else 0)
    recs, tx, rx = bytearray(), 0, 0
    for i in range(n):
        hsa = int(i * period_us * 1000)
        gui, sq_cu, mfma = i * 4000, i * 40000, i * 96000
        rd, wr = i * 100000, i * 50000
        vals = []
        for nm in counters:
            vals.append({"GRBM_GUI_ACTIVE": gui, "SQ_BUSY_CU_CYCLES": sq_cu,
                         "SQ_VALU_MFMA_BUSY_CYCLES": mfma,
                         "TCC_EA0_RDREQ_DRAM": rd, "TCC_EA0_WRREQ_DRAM": wr}[nm])
        if nic:
            if i and nic_delta:
                d = 50000 if i % 2 else 75000
                tx += d
                rx += d
            vals += [tx, rx]
        recs += struct.pack("<4Q", hsa + 5_000_000, hsa, i, stride)
        recs += struct.pack(f"<{stride}d", *vals)
    path.with_suffix(".bin").write_bytes(struct.pack("<4Q", MAGIC, stride, n, 32) + bytes(recs))
    meta = {"magic": "PMCSAMPL", "gpu_agent_index": 0, "period_us_target": period_us,
            "samples": n, "errors": 0, "record_header_bytes": 32, "counters": counters,
            "requested": counters, "marks": [], "last_record_count": 0,
            "unmatched_record_counters": [], "sampler": "synthetic unit-test records"}
    if nic:
        meta.update(nic_raw_per_sample=2, nic_errors=0, nic_ports=NIC_PORTS)
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return path.with_suffix(".bin"), path.with_suffix(".json")


def _sidecar(dump: Path):
    Path(str(dump) + ".json").write_text(json.dumps(
        {"mark_before": {"mono_ns": 5_000_000, "hsa_ns": 1000},
         "mark_after": {"mono_ns": 6_000_000, "hsa_ns": 951_000},
         "clock_probe": {"hsa_minus_memrealtime_ns": 1000}}), encoding="utf-8")


def _build(td: Path, ep: int, nic: bool, tag: str, nic_delta: bool = True):
    dump = _fi_dump(td / f"fi_{tag}.bin", ep16=(ep == 16))
    _sidecar(dump)
    ab, aj = _pmc(td / f"pmcA_{tag}", PASS_A, nic, nic_delta=nic_delta)
    bb, bj = _pmc(td / f"pmcB_{tag}", PASS_B, nic, nic_delta=nic_delta)
    dumpB = _fi_dump(td / f"fiB_{tag}.bin", ep16=(ep == 16))
    _sidecar(dumpB)
    out = td / f"trace_{tag}.sw_hw.perfetto.json.gz"
    rep, hwrep = E.build_trace(
        dump=str(dump), pmc_bin=str(ab), pmc_json=str(aj), out_gz=str(out),
        model="qwen38max", seqlen=16384, uninstrumented_us=11578.969, rank=0, arm="nic_unit_test",
        env_cfg={}, comm_mode="p2p_landing", comm_counter="", n_xcc=4, n_cu=80, pad_us=0.0,
        co_sha256="0" * 64, extra_meta={"fixture": "SYNTHETIC UNIT-TEST TRACE (not a measurement)"},
        compute_pass=(str(bb), str(bj), str(dumpB)), compute_mode="none",
        compute_pass_mode="mfma_busy", ep=ep)
    return rep, hwrep, out


def _with_scratch(fn):
    def wrap():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        td = SCRATCH / f"case_{abs(hash(fn.__name__)) % 10**8}"
        td.mkdir(exist_ok=True)
        try:
            fn(td)
        finally:
            import shutil
            shutil.rmtree(td, ignore_errors=True)
            try:
                SCRATCH.rmdir()
            except OSError:
                pass
    wrap.__name__ = fn.__name__
    return wrap


@_with_scratch
def test_v1_format_records_still_decode(td):
    ab, aj = _pmc(td / "pmc_v1", PASS_A, nic=False)
    meta, samples = E.load_pmc(str(ab), str(aj))
    assert len(samples) == 50 and all(s[3] is None for s in samples)
    assert len(samples[0][2]) == 4
    ser, rep = E.build_hw_series(meta, samples, 0, 950_000, 4, 80, "p2p_landing", "", 0.0, 1000, "none")
    assert set(ser) == {"compute", "hbm", "comm_rx", "active"} and rep["nic"] is None
    assert all(0.0 <= v <= 100.0 for pts in ser.values() for _, v in pts)


@_with_scratch
def test_v2_nic_split_and_rdma_formula(td):
    ab, aj = _pmc(td / "pmc_v2", PASS_A, nic=True)
    meta, samples = E.load_pmc(str(ab), str(aj))
    assert all(s[3] is not None and len(s[3]) == 2 for s in samples)
    ser, rep = E.build_hw_series(meta, samples, 0, 950_000, 4, 80, "p2p_landing", "", 0.0, 1000, "none")
    # V4 full-duplex iron rule: TX and RX MUST stay separate (NEVER summed against unidirectional 25 GB/s)!
    assert "rdma_tx" in ser and "rdma_rx" in ser and "rdma" not in ser
    assert len(ser["rdma_tx"]) == len(ser["compute"]) == len(ser["rdma_rx"])
    # hand arithmetic: 50000 B / 20 us = 2.5 GB/s = 10% of 25 GB/s; 75000 B -> 15% (each direction independently)
    for k in ("rdma_tx", "rdma_rx"):
        vals = [v for _, v in ser[k]]
        assert all(abs(v - (10.0 if i % 2 else 15.0)) < 1e-6 for i, v in enumerate(vals)), (k, vals[:6])
    assert rep["nic"]["total_rate_gbps"] == 200.0 and rep["nic"]["rdma_rate_gb_per_s"] == 25.0
    assert all(r["rdma_tx_bytes"] in (50000.0, 75000.0) and r["rdma_rx_bytes"] in (50000.0, 75000.0) for r in rep["raw"])
    assert E.rdma_curve_name(200.0, "tx") == "rdma_tx_util [Inter-Node RDMA TX (Out) % of 25GB/s]"
    assert E.rdma_curve_name(200.0, "rx") == "rdma_rx_util [Inter-Node RDMA RX (In) % of 25GB/s]"
    assert "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/tx_bytes" in E.nic_provenance_text(rep["nic"], "tx")
    assert "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/rx_bytes" in E.nic_provenance_text(rep["nic"], "rx")


@_with_scratch
def test_nic_counter_reset_never_negative(td):
    ab, aj = _pmc(td / "pmc_rst", PASS_A, nic=True)
    data = bytearray(ab.read_bytes())
    # corrupt one raw tx read to simulate a counter reset (raw value drops)
    off = 32 + 10 * (32 + 6 * 8) + 32 + 4 * 8
    struct.pack_into("<d", data, off, 0.0)
    ab.write_bytes(bytes(data))
    meta, samples = E.load_pmc(str(ab), str(aj))
    ser, rep = E.build_hw_series(meta, samples, 0, 950_000, 4, 80, "p2p_landing", "", 0.0, 1000, "none")
    assert all(v >= 0.0 for _, v in ser["rdma_tx"]) and all(v >= 0.0 for _, v in ser["rdma_rx"])


@_with_scratch
def test_no_link_rate_skips_rdma_with_reason(td):
    ab, aj = _pmc(td / "pmc_norate", PASS_A, nic=True)
    meta = json.loads(aj.read_text())
    meta["nic_ports"][0]["rate_gbps"] = 0.0
    aj.write_text(json.dumps(meta))
    meta, samples = E.load_pmc(str(ab), str(aj))
    ser, rep = E.build_hw_series(meta, samples, 0, 950_000, 4, 80, "p2p_landing", "", 0.0, 1000, "none")
    assert "rdma_rx" not in ser and "rdma_tx" not in ser and "rdma_skipped_reason" in rep["nic"]


@_with_scratch
def test_ep16_end_to_end_five_curves_per_wave_labels(td):
    rep, hwrep, out = _build(td, 16, nic=True, tag="ep16nic")
    assert "rdma_rx" in rep["curve_stats"] and rep["curve_stats"]["rdma_rx"]["max"] > 0.0
    assert "rdma_tx" in rep["curve_stats"] and rep["curve_stats"]["rdma_tx"]["max"] > 0.0
    assert "comm_rx" in rep["curve_stats"] and "rdma" not in rep["curve_stats"]
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    vrep = V.validate_doc(doc, name="<nic-test-ep16>", render_check=True)
    assert vrep["ok"], vrep["errors"]
    assert vrep["checks"]["hw"]["rdma_max"] > 0.0
    assert vrep["checks"]["panel_c_tracks"] == [
        "W00 (Compute / Combine)", "W01 (Compute / Combine)",
        "W02 (Compute / Combine)", "W03 (Compute / Combine)",
        "W04 (Comm / Scatter -> Streaming Reduce)", "W05 (Comm / Scatter -> Streaming Reduce)",
        "W06 (Comm / Scatter -> Streaming Reduce)", "W07 (Comm / Scatter -> Streaming Reduce)"]
    assert doc["otherData"]["meta"]["nic_capture"]["rdma_curve_names"] == {
        "rdma_rx": "rdma_rx_util [Inter-Node RDMA RX (In) % of 25GB/s]",
        "rdma_tx": "rdma_tx_util [Inter-Node RDMA TX (Out) % of 25GB/s]",
    }
    assert vrep["checks"]["render"]["px"] == [3520, 2244]


@_with_scratch
def test_ep16_end_to_end_without_nic_stays_four_curves(td):
    rep, hwrep, out = _build(td, 16, nic=False, tag="ep16nonic")
    assert "rdma_rx" not in rep["curve_stats"] and "rdma_tx" not in rep["curve_stats"]
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    vrep = V.validate_doc(doc, name="<nic-test-ep16-nonnic>")
    assert vrep["ok"], vrep["errors"]
    assert vrep["checks"]["hw"]["rdma_max"] is None
    assert len(vrep["checks"]["panel_c_tracks"]) == 8   # per-wave labels even without RDMA curves


@_with_scratch
def test_zero_nic_deltas_stay_zero_at_series_level(td):
    ab, aj = _pmc(td / "pmc_zero", PASS_A, nic=True, nic_delta=False)
    meta, samples = E.load_pmc(str(ab), str(aj))
    ser, rep = E.build_hw_series(meta, samples, 0, 950_000, 4, 80, "p2p_landing", "", 0.0, 1000, "none")
    assert "rdma_rx" in ser and "rdma_tx" in ser
    assert max(v for _, v in ser["rdma_rx"]) == 0.0 and max(v for _, v in ser["rdma_tx"]) == 0.0
    assert all(r["rdma_rx_bytes"] == 0.0 and r["rdma_tx_bytes"] == 0.0 for r in rep["raw"])
    assert rep["nic"]["nic_errors"] == 0 and rep["nic"]["total_rate_gbps"] == 200.0


@_with_scratch
def test_ep16_zero_traffic_nic_skips_rdma_with_reason(td):
    rep, hwrep, out = _build(td, 16, nic=True, tag="ep16nic0", nic_delta=False)
    assert "rdma_rx" not in rep["curve_stats"] and "rdma_tx" not in rep["curve_stats"]
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    nic = doc["otherData"]["meta"]["nic_capture"]
    assert nic["rdma_curve_name"] is None and nic["ports"]
    assert "rdma_skipped_reason" in nic and "zero bytes" in nic["rdma_skipped_reason"]
    vrep = V.validate_doc(doc, name="<nic-test-ep16-zero>", render_check=True)
    assert vrep["ok"], vrep["errors"]
    assert vrep["checks"]["hw"]["rdma_max"] is None
    assert len(vrep["checks"]["panel_c_tracks"]) == 8   # per-wave labels kept
    assert vrep["checks"]["render"]["px"] == [3520, 2244]


@_with_scratch
def test_ep8_ignores_nic_curve_but_keeps_provenance(td):
    rep, hwrep, out = _build(td, 8, nic=True, tag="ep8nic")
    assert "rdma_rx" not in rep["curve_stats"] and "rdma_tx" not in rep["curve_stats"]
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert doc["otherData"]["meta"]["nic_capture"] is not None
    vrep = V.validate_doc(doc, name="<nic-test-ep8>")
    assert vrep["ok"], vrep["errors"]
    assert vrep["checks"]["panel_c_tracks"] == [
        "W00..W03 (Compute / Combine)", "W04..W07 (Comm / Scatter -> Streaming Reduce)"]


@_with_scratch
def test_full_duplex_gmi_ea_never_sums_rx_tx(td):
    """Full-duplex xGMI gmi_ea mode MUST export comm_rx and comm_tx separately, never summed."""
    counters = ["GRBM_GUI_ACTIVE", "SQ_BUSY_CU_CYCLES",
                "TCC_EA0_RDREQ_DRAM", "TCC_EA0_WRREQ_DRAM",
                "TCC_EA0_RDREQ", "TCC_EA0_WRREQ"]
    stride = len(counters) + 2
    recs, tx, rx = bytearray(), 0, 0
    for i in range(50):
        hsa = int(i * 20.0 * 1000)
        vals = [i * 4000, i * 40000,
                i * 100000, i * 50000,
                i * 120000, i * 60000]   # rd_gmi delta = 20000 reqs, wr_gmi delta = 10000 reqs
        if i:
            tx += 50000; rx += 75000
        vals += [tx, rx]
        recs += struct.pack("<4Q", hsa + 5_000_000, hsa, i, stride)
        recs += struct.pack(f"<{stride}d", *vals)
    base = td / "pmc_fd"
    base.with_suffix(".bin").write_bytes(struct.pack("<4Q", MAGIC, stride, 50, 32) + bytes(recs))
    meta = {"magic": "PMCSAMPL", "gpu_agent_index": 0, "period_us_target": 20.0,
            "samples": 50, "errors": 0, "record_header_bytes": 32, "counters": counters,
            "requested": counters, "marks": [], "last_record_count": 0,
            "unmatched_record_counters": [], "sampler": "synthetic full-duplex unit-test",
            "nic_raw_per_sample": 2, "nic_errors": 0, "nic_ports": NIC_PORTS}
    base.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    m, s = E.load_pmc(str(base.with_suffix(".bin")), str(base.with_suffix(".json")))
    ser, rep = E.build_hw_series(m, s, 0, 950_000, 4, 80, "gmi_ea", "", 0.0, 1000, "none")
    assert set(ser) == {"compute", "hbm", "comm_rx", "comm_tx", "rdma_rx", "rdma_tx", "active"}
    # rd_gmi=20000*64B / 20us = 64 GB/s -> 22.2841%; wr_gmi=10000*64B / 20us = 32 GB/s -> 11.1421%
    assert abs(ser["comm_rx"][0][1] - (20000 * 64.0 / 20e-6 / 287.2e9 * 100.0)) < 1e-3
    assert abs(ser["comm_tx"][0][1] - (10000 * 64.0 / 20e-6 / 287.2e9 * 100.0)) < 1e-3
    # RDMA TX=50000B -> 10.0%, RDMA RX=75000B -> 15.0% (distinct, never summed!)
    assert abs(ser["rdma_tx"][0][1] - 10.0) < 1e-4
    assert abs(ser["rdma_rx"][0][1] - 15.0) < 1e-4
    # Full 7-curve end-to-end build + 3520x2244 render check
    dump = _fi_dump(td / "fi_fd.bin", ep16=True); _sidecar(dump)
    bb, bj = _pmc(td / "pmcB_fd", PASS_B, nic=True)
    dumpB = _fi_dump(td / "fiB_fd.bin", ep16=True); _sidecar(dumpB)
    out = td / "trace_fd.sw_hw.perfetto.json.gz"
    E.build_trace(str(dump), str(base.with_suffix(".bin")), str(base.with_suffix(".json")), str(out),
                  "qwen38max", 16384, 3000.0, 0, "fd_test", {}, "gmi_ea", "", 4, 80, 0.0, "0" * 64,
                  {"fixture": "SYNTHETIC UNIT-TEST TRACE"}, compute_pass=(str(bb), str(bj), str(dumpB)),
                  compute_mode="none", compute_pass_mode="mfma_busy", ep=16)
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    vrep = V.validate_doc(doc, name="<full-duplex-7curve>", render_check=True)
    assert vrep["ok"], vrep["errors"]
    assert vrep["checks"]["render"]["px"] == [3520, 2244]


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
            traceback.print_exc(limit=4, file=sys.stdout)
    print(f"test_export_mi308x_nic: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
