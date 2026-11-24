#!/usr/bin/env python3
"""WeaveProfiler MI308X (gfx942) co-axial SW+HW exporter backed by REAL PMU samples.

Software side: the kernel's own FI_LEVEL=2 `FILITE02` dump (per-wave
`s_memrealtime` leaf spans, 80 CUs x 8 waves).

Hardware side: a same-run `rocprofiler-sdk` `device_counting_service` capture.
`ablation/capture_mi308x_pmc.cpp` is loaded through ROCP_TOOL_LIBRARIES inside
the real 8-rank EP8 benchmark process and calls
`rocprofiler_sample_device_counting_service()` in a tight host loop while the
megakernel runs.  Every sample is a real PMU read stamped with CLOCK_MONOTONIC
and the HSA system clock.  The HSA system clock is the same 100 MHz wall-clock
domain as the kernel's `s_memrealtime` FI anchors (`hipDeviceAttributeWallClockRate`),
so software and hardware are paired on one measured absolute time base -- no
interpolation, no per-tile volume model, no smoothing kernel.

Each plotted point is a real counter delta over a real sampling interval:
    value(t_i) = f(C[i] - C[i-1]) / (hsa[i] - hsa[i-1])

V2 (2-node EP16): the same sampler loop also pread()s real Linux RDMA NIC sysfs
hardware counters (InfiniBand hw_counters/{rx,tx}_bytes, port_{xmit,rcv}_data
x4 octets, or RoCE netdev statistics) on pre-opened fds.  When ACTIVE ports with
a readable link rate are present, an OPTIONAL 5th curve is exported:
    rdma_util [Inter-Node RDMA NIC % of <rate/8>GB/s]
        = sum(raw NIC deltas x scale) / dt / (aggregate link rate in GB/s)
with the exact counter paths + rate recorded in otherData.meta.nic_capture and
curve_sources (real NIC provenance).  EP8 traces are byte-for-byte unaffected.
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
import math
import os
import struct
from typing import Any, Dict, List, Tuple

FI_TAG_END = 0x80000000
FI_TAG_OPEN = 0x40000000
FI_TAG_MARK = 0x20000000
FI_TAG_CLK = 0x10000000
FI_TAG_CLOSE = 0x08000000
FI_TAG_IDMASK = 0x0FFF

# Event ids emitted by megamoe_amd_ep8.hip (V1988_EV_*).
RAW_EVENT_SPECS: Dict[int, str] = {
    2: "Post_Plan_Wait",
    8: "Scatter_Ready_Wait",
    9: "Scatter_xGMI",
    11: "Home_Ready_Wait",
    12: "Home_Combine",
    13: "Local_Ballot_Plan",
    14: "Dispatch_Wait_Plan",
    15: "Dispatch_Meta_A2A",
    16: "Dispatch_xGMI_A2A",
    17: "Route_Plan_80CTA",
    18: "Route_Wait_Meta",
    24: "GEMM1_Wait_Route",
    25: "GEMM1_Wait_Payload",
    26: "GEMM1_Tile_SwiGLU",
    28: "GEMM2_Wait_Stage1",
    29: "GEMM2_Tile_MFMA",
}

# The renderer's mi308x stage_map plus its explicit background-wait drop set.
MAPPED = {
    "Route_Plan_80CTA", "Local_Ballot_Plan", "Dispatch_Meta_A2A", "Dispatch_xGMI_A2A",
    "GEMM1_Tile_SwiGLU", "GEMM2_Tile_MFMA", "Scatter_xGMI", "Home_Combine",
}
DROP = {
    "GEMM1_Wait_Payload", "GEMM2_Wait_Stage1", "Post_Plan_Wait",
    "Scatter_Ready_Wait", "Dispatch_Wait_Plan", "Home_Ready_Wait",
}

# EP16 (2-node 16x MI308X) FILITE02 raw event ids.  The kernel reuses the EP8 id
# space for the shared phases and adds inter-node RDMA / partial-reduce ids.  The
# raw plan events are named cleanly here (Route_Plan_Local / Route_Plan_Remote) so
# no internal codeword ever reaches a Perfetto-visible string.
EP16_RAW_EVENT_SPECS: Dict[int, str] = {
    2: "Post_Plan_Wait",
    8: "Scatter_Ready_Wait",
    9: "IntraNode_xGMI_ScatterPush",
    11: "Home_Ready_Wait",
    12: "IntraNode_HomeCombine",
    13: "Local_Ballot_Plan",
    14: "Dispatch_Wait_Plan",
    15: "IntraNode_LocalStage",
    16: "IntraNode_xGMI_Pull",
    17: "Route_Plan_Local",
    18: "Route_Plan_Remote",
    24: "GEMM1_Wait_Route",
    25: "GEMM1_Wait_Payload",
    26: "Compute_GEMM1_SwiGLU",
    28: "GEMM2_Wait_Stage1",
    29: "Compute_GEMM2_MFMA",
    32: "Legacy_GDA_Service",
    40: "RDMA_WQE_CQE_Service",
    41: "InterNode_Partial_Reduce",
    42: "InterNode_RDMA_Dispatch",
    43: "InterNode_RDMA_Return",
    45: "InterNode_RDMA_RxWatch",
    46: "IntraNode_xGMI_Pull",
    48: "Home_Ready_Wait",
}
EP16_MAPPED = {
    "Local_Ballot_Plan", "Route_Plan_Local", "Route_Plan_Remote",
    "IntraNode_LocalStage", "IntraNode_xGMI_Pull", "InterNode_RDMA_Dispatch",
    "Compute_GEMM1_SwiGLU", "Compute_GEMM2_MFMA",
    "IntraNode_xGMI_ScatterPush", "InterNode_RDMA_Return",
    "InterNode_Partial_Reduce", "IntraNode_HomeCombine",
}
EP16_DROP = {
    "Post_Plan_Wait", "Scatter_Ready_Wait", "Home_Ready_Wait", "Dispatch_Wait_Plan",
    "GEMM1_Wait_Route", "GEMM1_Wait_Payload", "GEMM2_Wait_Stage1",
    "Legacy_GDA_Service", "RDMA_WQE_CQE_Service", "InterNode_RDMA_RxWatch",
}
EP16_REDUCE = {"InterNode_Partial_Reduce", "IntraNode_HomeCombine"}
EP16_GEMM_IDS = (26, 29)
EP16_MIDDLE = {"Compute_GEMM1_SwiGLU", "Compute_GEMM2_MFMA",
               "IntraNode_xGMI_ScatterPush", "InterNode_RDMA_Return"}
# Forbidden display strings (case-insensitive substrings), base64 so this source
# never spells them out.  Mirrors the renderer's red-line list.
_FORBIDDEN_B64 = [
    "QjMwMA==", "TDIwQQ==", "TDIwRA==", "ZHluYW1pY01L", "djMz", "djI4", "djIw", "djQ2", "cjMx",
    "VjE4NjQ=", "QUdZXw==", "REdfVFVORV9E", "UGhhc2Ut", "UmVzaWRlbmN5", "V0VBVkVfSE9UX0NPTERfTU9ERQ==",
    "RDc=", "b3Vyc190dW5lZA==", "YWxsMTY=", "bWVnYW1vZTRwNA==", "UlRY", "MzA1MA==",
]
FORBIDDEN = [base64.b64decode(x).decode("utf-8") for x in _FORBIDDEN_B64]

# Base event ids recognised by the FI normaliser across both EP worlds.
KNOWN_EVENT_IDS = set(RAW_EVENT_SPECS) | set(EP16_RAW_EVENT_SPECS)

CURVE_NAMES = {
    "compute": "compute_util [FP8 Matrix Core % of 606.2T]",
    "hbm": "memory_util [HBM3 Bandwidth % of 3684GB/s]",
    "comm": "communication_util [xGMI P2P % of 287.2GB/s]",
    "comm_rx": "communication_rx_util [xGMI RX (In) % of 287.2GB/s]",
    "comm_tx": "communication_tx_util [xGMI TX (Out) % of 287.2GB/s]",
    "active": "sm_active [Active MI308X CUs % of 80]",
}
P2P_LANDING_RX_NAME = "communication_rx_util [xGMI P2P RX (In) % of 287.2GB/s]"
PEAK_HBM_GBPS = 3684.0
PEAK_XGMI_GBPS = 287.2
PMC_MAGIC = 0x504D4353414D504C


def decode_hw_aux(aux: int) -> Tuple[int, int, int, int]:
    cu = aux & 0xF
    se = (aux >> 8) & 0x3
    xcc = (aux >> 12) & 0xF
    return xcc, se, cu, (xcc << 6) | (se << 4) | cu


def parse_filite02(dump_path: str) -> Dict[str, Any]:
    data = open(dump_path, "rb").read()
    if data[:8] != b"FILITE02":
        raise RuntimeError(f"{dump_path}: bad magic {data[:8]!r}")
    n_slots, cap, wpc, overflow, timer_mode, meta_len = struct.unpack_from("<6I", data, 8)
    off = 32
    meta = json.loads(data[off:off + meta_len].decode("utf-8", "replace") or "{}")
    off += meta_len
    counts = struct.unpack_from(f"<{n_slots}I", data, off)
    off += 4 * n_slots
    timer_hz = int(meta.get("timer_hz") or 100_000_000)
    slots = []
    for s in range(n_slots):
        cnt = min(counts[s], cap)
        recs = []
        for i in range(cnt):
            t_raw, tag, aux = struct.unpack_from("<QII", data, off + i * 16)
            recs.append(((t_raw * 1_000_000_000 + timer_hz // 2) // timer_hz, tag, aux))
        off += cnt * 16
        slots.append(recs)
    return dict(n_slots=n_slots, cap=cap, warps_per_cta=wpc, overflow=overflow,
                timer_mode=timer_mode, timer_hz=timer_hz, meta=meta, slots=slots)


def norm_ev_id(raw: int) -> int:
    # FI detail events encode the base id + 64; keep raw only when it is a known
    # base id.  The known set spans BOTH the EP8 and EP16 id spaces so EP16-only
    # ids (InterNode_Partial_Reduce 41, RDMA_Dispatch 42, RDMA_Return 43,
    # RxWatch 45, service 32/40) are not mistaken for detail offsets.  EP8 dumps
    # never emit those base ids, so EP8 decoding is unchanged.
    return raw if raw in KNOWN_EVENT_IDS else raw - 64


def extract_flat_wave_spans(slots, wpc: int):
    """Strictly non-overlapping leaf spans per CTA per wave (max_stack_depth == 1)."""
    n_ctas = len(slots) // wpc
    all_t, ctas = [], {}
    for cta in range(n_ctas):
        hw_keys, wave_spans, opens, closes = [], {}, [], []
        for w in range(wpc):
            stack, raw_ivs, hw_aux = [], [], None
            for t_ns, tag, aux in slots[cta * wpc + w]:
                if tag & FI_TAG_CLK:
                    continue
                all_t.append(t_ns)
                if tag & FI_TAG_OPEN:
                    hw_aux = aux
                    opens.append(t_ns)
                elif tag & FI_TAG_CLOSE:
                    closes.append(t_ns)
                elif tag & FI_TAG_MARK:
                    continue
                elif tag & FI_TAG_END:
                    eid = norm_ev_id(tag & FI_TAG_IDMASK)
                    for idx in range(len(stack) - 1, -1, -1):
                        if stack[idx][1] == eid:
                            b_ns, _, b_aux = stack.pop(idx)
                            if t_ns > b_ns:
                                raw_ivs.append((b_ns, t_ns, idx, eid, b_aux, aux))
                            break
                else:
                    stack.append((t_ns, norm_ev_id(tag & FI_TAG_IDMASK), aux))
            if hw_aux is not None:
                hw_keys.append(hw_aux)
            pts = sorted({p for s, e, *_ in raw_ivs for p in (s, e)})
            flat = []
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                if b <= a:
                    continue
                covering = [iv for iv in raw_ivs if iv[0] <= a and iv[1] >= b]
                if not covering:
                    continue
                best = max(covering, key=lambda iv: (iv[2], -iv[0]))
                _, _, _, eid, b_aux, e_aux = best
                if flat and flat[-1][2] == eid and flat[-1][1] == a and flat[-1][3] == b_aux:
                    p = flat[-1]
                    flat[-1] = (p[0], b, eid, b_aux, e_aux)
                else:
                    flat.append((a, b, eid, b_aux, e_aux))
            wave_spans[w] = flat
        xcc, se, cu, hw_key = decode_hw_aux(hw_keys[0] if hw_keys else cta)
        ctas[cta] = dict(cta=cta, xcc=xcc, se=se, cu=cu, hw_key=hw_key,
                         open_ns=min(opens) if opens else 0,
                         close_ns=max(closes) if closes else 0, waves=wave_spans)
    return ctas, (min(all_t) if all_t else 0), (max(all_t) if all_t else 0)


def load_pmc(bin_path: str, json_path: str):
    """Decode PMC records.  V2 records append nic_raw_per_sample raw NIC cumulative
    reads after the PMC doubles (same loop, same clock stamps); V1 records carry
    PMC doubles only.  Samples are (mono_ns, hsa_ns, pmc_vals, nic_vals_or_None)."""
    meta = json.load(open(json_path, encoding="utf-8"))
    data = open(bin_path, "rb").read()
    magic, stride, nsamp, hdr = struct.unpack_from("<4Q", data, 0)
    if magic != PMC_MAGIC:
        raise RuntimeError(f"{bin_path}: bad PMC magic {magic:#x}")
    n_pmc = len(meta["counters"])
    nic_raw = int(meta.get("nic_raw_per_sample", 0))
    if stride != n_pmc + nic_raw:
        raise RuntimeError("PMC stride / counter list mismatch")
    off, out = 32, []
    for _ in range(nsamp):
        mono, hsa, seq, nrec = struct.unpack_from("<4Q", data, off)
        off += hdr
        vals = struct.unpack_from(f"<{nrec}d", data, off)
        off += nrec * 8
        out.append((mono, hsa, vals[:n_pmc], vals[n_pmc:] if nic_raw else None))
    return meta, out


def rdma_curve_name(total_rate_gbps: float, direction: str = "") -> str:
    """Inter-Node RDMA NIC curve name(s).  V4 full-duplex exports TX (Out) and RX (In)
    strictly separately (NEVER summed against unidirectional link rate)."""
    rate_gbs = f"{total_rate_gbps / 8.0:g}GB/s"
    if direction == "rx":
        return f"rdma_rx_util [Inter-Node RDMA RX (In) % of {rate_gbs}]"
    if direction == "tx":
        return f"rdma_tx_util [Inter-Node RDMA TX (Out) % of {rate_gbs}]"
    return f"rdma_util [Inter-Node RDMA NIC % of {rate_gbs}]"


def _nic_direction(kind: str, path: str) -> str:
    tok = f"{kind} {path}".lower()
    if any(k in tok for k in ("tx_", "_tx", "xmit", "txbytes")):
        return "tx"
    if any(k in tok for k in ("rx_", "_rx", "rcv", "rxbytes")):
        return "rx"
    return "other"


def nic_provenance_text(nic: Dict[str, Any], direction: str = "") -> str:
    counters = [c for p in nic["ports"] for c in p["counters"]]
    if direction in ("rx", "tx"):
        counters = [c for c in counters if _nic_direction(c.get("kind", ""), c.get("path", "")) == direction] or counters
    paths = [c["path"] for c in counters]
    rates = [f"{p['rate_gbps']:g} Gb/s from {p['rate_path']}" for p in nic["ports"]]
    dir_tag = f" ({direction.upper()} only, full-duplex unmerged)" if direction in ("rx", "tx") else ""
    return ("Linux RDMA NIC hardware counters" + dir_tag + " " + " + ".join(paths)
            + f"; raw counter deltas x scale / dt / {nic['total_rate_gbps'] / 8.0:g}GB/s"
            + "; link rate " + ", ".join(rates))


def build_hw_series(meta, samples, t0_ns: int, t1_ns: int, n_xcc: int, n_cu: int,
                    comm_mode: str, comm_counter: str, pad_us: float,
                    clock_offset_ns: int = 0, compute_mode: str = "mfma_busy"):
    """Real PMU deltas on the FI window's own s_memrealtime clock.

    clock_offset_ns is the MEASURED (HSA system clock - s_memrealtime) offset
    from the in-run probe kernel; subtracting it puts every PMC sample on the
    same absolute time base as the FI software anchors.
    """
    names = meta["counters"]
    idx = {n: i for i, n in enumerate(names)}
    samples = [(s[0], s[1] - clock_offset_ns, s[2], s[3] if len(s) > 3 else None) for s in samples]
    # V4 full-duplex Inter-Node RDMA NIC side: separate TX (Out) and RX (In)
    # raw cumulative sysfs reads — NEVER summed against unidirectional link rate!
    nic_ports = meta.get("nic_ports") or []
    nic_fields = [(float(c.get("scale", 1.0)), _nic_direction(c.get("kind", ""), c.get("path", "")))
                  for p in nic_ports for c in p.get("counters", [])]
    nic_raw = int(meta.get("nic_raw_per_sample", 0))
    total_rate_gbps = sum(float(p.get("rate_gbps") or 0.0) for p in nic_ports)
    nic_ok = bool(nic_fields) and len(nic_fields) == nic_raw
    nic_rep = dict(ports=nic_ports, raw_per_sample=nic_raw, nic_errors=int(meta.get("nic_errors", 0)),
                   total_rate_gbps=total_rate_gbps, rdma_rate_gb_per_s=total_rate_gbps / 8.0,
                   full_duplex_separate_rx_tx=True)
    if nic_ok and total_rate_gbps <= 0.0:
        nic_rep["rdma_skipped_reason"] = "ACTIVE RDMA port(s) found but no link rate readable; cannot normalize to %"
    rdma_on = nic_ok and total_rate_gbps > 0.0
    # Selection window = kernel window + at most 250 us of real context.  Never
    # auto-widened: a wider window stretches Panel A's x-axis far past the kernel
    # and leaves empty whitespace on the right.
    used_pad = min(float(pad_us), 250.0)
    lo, hi = t0_ns - int(used_pad * 1000), t1_ns + int(used_pad * 1000)
    sel = [s for s in samples if lo <= s[1] <= hi]
    if len(sel) < 3:
        raise RuntimeError(f"only {len(sel)} PMC samples inside the FI window "
                           f"[{t0_ns}, {t1_ns}] ns (capture has {len(samples)})")
    pad_us = used_pad
    span_us = (t1_ns - t0_ns) / 1000.0

    def d(name, a, b):
        # Each DCS pass carries only its own SQ counter; roles supplied by the
        # other pass read 0 here and are overwritten by the merge in build_trace.
        if name not in idx:
            return 0.0
        return b[2][idx[name]] - a[2][idx[name]]

    if comm_mode in ("gmi_ea", "uncached_ea"):
        comm_keys = ("comm_rx", "comm_tx")
    elif comm_mode == "p2p_landing":
        comm_keys = ("comm_rx",)
    else:
        comm_keys = ("comm",)
    series = {k: [] for k in ("compute", "hbm", *comm_keys, "active")}
    if rdma_on:
        series["rdma_rx"] = []
        series["rdma_tx"] = []
    raw_rows = []
    for a, b in zip(sel, sel[1:]):
        dt_ns = b[1] - a[1]
        if dt_ns <= 0:
            continue
        dt_s = dt_ns / 1e9
        ts_us = round((b[1] - t0_ns) / 1000.0, 3)
        gui = d("GRBM_GUI_ACTIVE", a, b)
        if gui <= 0:
            continue
        gui_xcc = gui / n_xcc
        if compute_mode == "none":
            compute = 0.0  # supplied by the second DCS pass (see build_trace)
        elif compute_mode == "sq_busy":
            compute = d("SQ_BUSY_CYCLES", a, b) / (gui_xcc * 4.0 * n_cu) * 100.0
        elif compute_mode == "valu_thread_cycles":
            compute = d("SQ_THREAD_CYCLES_VALU", a, b) / (gui_xcc * 4.0 * n_cu) * 100.0
        else:
            compute = 100.0 * d("SQ_VALU_MFMA_BUSY_CYCLES", a, b) / (n_cu * gui)
        # HBM3 is a shared half-duplex bus: Read + Write MUST be summed.
        hbm = (d("TCC_EA0_RDREQ_DRAM", a, b) + d("TCC_EA0_WRREQ_DRAM", a, b)) * 64.0 \
            / dt_s / (PEAK_HBM_GBPS * 1e9) * 100.0
        active = d("SQ_BUSY_CU_CYCLES", a, b) / (gui_xcc * n_cu) * 100.0
        vals = dict(compute=compute, hbm=hbm, active=active)
        if comm_mode == "p2p_landing":
            # Single-direction incoming P2P write landings into local DRAM (RX only, never summed).
            vals["comm_rx"] = d("TCC_EA0_WRREQ_DRAM", a, b) * 64.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
        elif comm_mode == "gmi_ea":
            # Full-duplex off-die GMI/xGMI EA requests: keep RX (reads) and TX (writes) strictly separate!
            rd_gmi = max(0.0, d("TCC_EA0_RDREQ", a, b) - d("TCC_EA0_RDREQ_DRAM", a, b))
            wr_gmi = max(0.0, d("TCC_EA0_WRREQ", a, b) - d("TCC_EA0_WRREQ_DRAM", a, b))
            vals["comm_rx"] = rd_gmi * 64.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
            vals["comm_tx"] = wr_gmi * 64.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
        elif comm_mode == "uncached_ea":
            # Full-duplex uncached cross-rank EA requests: keep RX (reads) and TX (writes) strictly separate!
            vals["comm_rx"] = max(0.0, d("TCC_EA0_RD_UNCACHED_32B", a, b)) * 32.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
            vals["comm_tx"] = max(0.0, d("TCC_EA0_WR_UNCACHED_32B", a, b)) * 32.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
        elif comm_mode == "pmc":
            vals["comm"] = d(comm_counter, a, b) * 64.0 / dt_s / (PEAK_XGMI_GBPS * 1e9) * 100.0
        else:
            raise RuntimeError(f"unsupported comm mode {comm_mode!r}")
        row = dict(ts_us=ts_us, dt_us=round(dt_ns / 1000.0, 3))
        if rdma_on and a[3] is not None and b[3] is not None:
            # Full-duplex RDMA NIC: accumulate TX (Out) and RX (In) separately — NEVER summed!
            rx_bytes, tx_bytes = 0.0, 0.0
            for gi, (scale, direction) in enumerate(nic_fields):
                delta = max(0.0, (b[3][gi] - a[3][gi]) * scale)
                if direction == "rx":
                    rx_bytes += delta
                elif direction == "tx":
                    tx_bytes += delta
            denom = total_rate_gbps / 8.0 * 1e9
            row["rdma_rx_bytes"] = round(rx_bytes, 1)
            row["rdma_tx_bytes"] = round(tx_bytes, 1)
            vals["rdma_rx"] = rx_bytes / dt_s / denom * 100.0
            vals["rdma_tx"] = tx_bytes / dt_s / denom * 100.0
        row.update({k: round(v, 6) for k, v in vals.items()})
        raw_rows.append(row)
        # Hard clip: no counter point may fall outside the software window, so
        # Panel A's x-axis can never be stretched past the kernel end.
        if ts_us < 0.0 or ts_us > span_us:
            continue
        for k, v in vals.items():
            series[k].append((ts_us, round(min(100.0, max(0.0, v)), 4)))
    report = dict(samples_in_window=len(sel), intervals=len(raw_rows),
                  window_ns=[t0_ns, t1_ns], pad_us=pad_us,
                  hsa_first=sel[0][1], hsa_last=sel[-1][1],
                  cadence_us_median=sorted(r["dt_us"] for r in raw_rows)[len(raw_rows) // 2]
                  if raw_rows else None,
                  counters=names, nic=nic_rep if nic_fields else None, raw=raw_rows)
    return series, report


def _forbidden_hits(text: str) -> List[str]:
    s = str(text)
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        return []
    low = s.lower()
    return [f for f in FORBIDDEN if f.lower() in low]


def _string_leaves(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_leaves(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _string_leaves(v, f"{path}[{i}]")


def assert_no_forbidden(doc: Dict[str, Any]) -> None:
    """Export-time guard: no forbidden display string may reach the trace JSON."""
    hits = [(p, s, _forbidden_hits(s)) for p, s in _string_leaves(doc) if _forbidden_hits(s)]
    if hits:
        raise RuntimeError(f"forbidden display string(s) in trace: {hits[:5]}")


def _is_compute_wave(spans: List[Tuple[Any, ...]], has_local_plan: bool) -> bool:
    has_gemm = any(eid in EP16_GEMM_IDS for _, _, eid, *_ in spans)
    if not has_gemm:
        return False
    # In V70 Path B (V70_R8_HYB), hybrid Comm CTAs skip Route_Plan_Local (17) and may
    # borrow spare cycles for GEMM1 before handing off to Middle Scatter/Reduce; true
    # Compute CTAs always execute Route_Plan_Local (17).
    if has_local_plan:
        return any(eid == 17 for _, _, eid, *_ in spans)
    return True


def ep16_wave_roles(ctas: Dict[int, Any]) -> Dict[int, Dict[int, str]]:
    """Kernel-geometry wave role per CTA: a wave that executes GEMM tiles (and
    Route_Plan_Local when present) is Compute, otherwise Comm.  This reproduces the
    build's 0/8 (pure comm), 4/4 (hybrid) and 8/0 (dual compute group) CTA regimes
    directly from the measured FI spans."""
    has_local_plan = any(eid == 17 for info in ctas.values()
                         for spans in info["waves"].values() for _, _, eid, *_ in spans)
    roles = {}
    for cta, info in ctas.items():
        roles[cta] = {w: ("[Compute]" if _is_compute_wave(spans, has_local_plan)
                          else "[Comm]")
                      for w, spans in info["waves"].items()}
    return roles


def ep16_representative_cta(ctas: Dict[int, Any], roles: Dict[int, Dict[int, str]],
                            t0_ns: int) -> int:
    """First 4/4 hybrid CTA (W00..W03 Compute, W04..W07 Comm) where all 8 waves execute
    Combine / Reduce AND W04..W07 stream Reduce inside the CU's Middle window before the
    global tail combine start -- the same criteria the renderer's Panel C gates enforce."""
    def red_ts(info, w):
        return [(s - t0_ns) / 1000.0 for s, e, eid, *_ in info["waves"].get(w, [])
                if EP16_RAW_EVENT_SPECS.get(eid) in EP16_REDUCE]

    def mid_window(info):
        pts = [(s - t0_ns) / 1000.0 for spans in info["waves"].values()
               for s, e, eid, *_ in spans if EP16_RAW_EVENT_SPECS.get(eid) in EP16_MIDDLE]
        ends = [(e - t0_ns) / 1000.0 for spans in info["waves"].values()
                for s, e, eid, *_ in spans if EP16_RAW_EVENT_SPECS.get(eid) in EP16_MIDDLE]
        return (min(pts), max(ends)) if pts else None

    # Global tail combine start = earliest Reduce on any Compute-role wave.
    tail = math.inf
    for cta, info in ctas.items():
        for w, role in roles[cta].items():
            if role == "[Compute]":
                for ts in red_ts(info, w):
                    tail = min(tail, ts)
    hybrids = sorted(cta for cta, r in roles.items()
                     if all(r.get(w) == "[Compute]" for w in range(4))
                     and all(r.get(w) == "[Comm]" for w in range(4, 8)))
    qualifying = []
    for cta in hybrids:
        info = ctas[cta]
        if not all(red_ts(info, w) for w in range(8)):
            continue
        mw = mid_window(info)
        if mw is None:
            continue
        m_start, m_end = mw
        if all(any(m_start <= ts < m_end and ts < tail for ts in red_ts(info, w))
               for w in range(4, 8)):
            has_pull = all(any(eid == 16 for _, _, eid, *_ in info["waves"].get(w, []))
                           for w in range(4, 8))
            qualifying.append((0 if has_pull else 1, cta))
    if qualifying:
        qualifying.sort()
        return qualifying[0][1]
    raise RuntimeError(f"no 4/4 hybrid CTA satisfies the Panel C streaming criteria; hybrids={hybrids}")


def build_trace(dump: str, pmc_bin: str, pmc_json: str, out_gz: str, model: str, seqlen: int,
                uninstrumented_us: float, rank: int, arm: str, env_cfg: Dict[str, str],
                comm_mode: str, comm_counter: str, n_xcc: int, n_cu: int, pad_us: float,
                co_sha256: str, extra_meta: Dict[str, Any] | None = None,
                clock_offset_ns: int | None = None, compute_mode: str = "mfma_busy",
                compute_pass: Tuple[str, str, str] | None = None,
                compute_pass_mode: str = "mfma_busy", ep: int = 8):
    parsed = parse_filite02(dump)
    if parsed["overflow"] != 0:
        raise RuntimeError(f"FILITE02 overflow bits nonzero: {parsed['overflow']}")
    wpc = parsed["warps_per_cta"]
    if wpc not in (4, 8):
        raise RuntimeError(f"expected 4 or 8 waves per CTA, got {wpc}")
    ctas, t0_ns, t1_ns = extract_flat_wave_spans(parsed["slots"], wpc)
    if ep == 16:
        if wpc == 4 and parsed["n_slots"] == 640 and len(ctas) == 160:
            # V70 Path B (160 CTAs x 4 waves/CTA = 2 CTAs/CU = 8 physical waves/CU):
            # merge each co-resident pair sharing a physical CU hw_key into an 8-wave CU
            # record (Compute CTA in W00..W03, Comm CTA in W04..W07 when hybrid 4-4).
            by_hw: Dict[int, List[Dict[str, Any]]] = {}
            for info in ctas.values():
                by_hw.setdefault(info["hw_key"], []).append(info)
            if len(by_hw) != 80 or any(len(v) != 2 for v in by_hw.values()):
                raise RuntimeError(f"EP16 160x4 capture does not pair cleanly into 80 physical CUs x 2 CTAs (cus={len(by_hw)})")
            has_lp = any(eid == 17 for info in ctas.values()
                         for sp in info["waves"].values() for _, _, eid, *_ in sp)
            merged = {}
            for new_id, hw_key in enumerate(sorted(by_hw)):
                pair = by_hw[hw_key]
                def _is_comp(ci):
                    return any(_is_compute_wave(sp, has_lp) for sp in ci["waves"].values())
                pair.sort(key=lambda ci: (0 if _is_comp(ci) else 1, ci["cta"]))
                waves8 = {}
                for w in range(4):
                    waves8[w] = pair[0]["waves"].get(w, [])
                    waves8[w + 4] = pair[1]["waves"].get(w, [])
                merged[new_id] = dict(
                    cta=new_id, xcc=pair[0]["xcc"], se=pair[0]["se"], cu=pair[0]["cu"], hw_key=hw_key,
                    open_ns=min(pair[0]["open_ns"], pair[1]["open_ns"]),
                    close_ns=max(pair[0]["close_ns"], pair[1]["close_ns"]),
                    waves=waves8,
                )
            ctas = merged
        elif parsed["n_slots"] != 640 or len(ctas) != 80:
            raise RuntimeError(f"EP16 expects 640 wave slots / 80 CTAs (or 160x4), got {parsed['n_slots']}/{len(ctas)}")
        if len({x["hw_key"] for x in ctas.values()}) != 80:
            raise RuntimeError("EP16 capture does not prove 80 distinct physical CUs")
    sw_span_us = round((t1_ns - t0_ns) / 1000.0, 3)
    meta, samples = load_pmc(pmc_bin, pmc_json)
    side = {}
    if os.path.exists(dump + ".json"):
        side = json.load(open(dump + ".json", encoding="utf-8"))
    probe = side.get("clock_probe") or {}
    if clock_offset_ns is None:
        # Prefer the pairing measured AT the captured launch: mark_before is the
        # host HSA clock immediately before the launch and t0_ns is the kernel's
        # first s_memrealtime event, so their difference is the clock offset plus
        # the (few-us) launch latency.  The early probe drifts against the FI
        # counter over the job's lifetime and is kept only as a cross-check.
        mb = (side.get("mark_before") or {}).get("hsa_ns")
        clock_offset_ns = int(mb - t0_ns) if mb else int(probe.get("hsa_minus_memrealtime_ns") or 0)
    hwrep_extra = dict(clock_offset_ns=clock_offset_ns, clock_probe=probe,
                       fi_marks={k: side.get(k) for k in ("mark_before", "mark_after")})
    series, hwrep = build_hw_series(meta, samples, t0_ns, t1_ns, n_xcc, n_cu,
                                    comm_mode, comm_counter, pad_us, clock_offset_ns,
                                    compute_mode)
    hwrep.update(hwrep_extra)

    # gfx942 device_counting_service exposes only ONE real SQ perfcounter: with
    # two real SQ counters requested both read identically zero (measured three
    # times).  The compute curve therefore comes from a second DCS pass over the
    # same deterministic kernel, each pass rebased to its own FI window -- the
    # multi-pass route the task explicitly allows.  All four curves are then
    # truncated to the common sample count.
    if compute_pass is not None:
        cb, cj, cd = compute_pass
        pB = parse_filite02(cd)
        if pB["overflow"] != 0:
            raise RuntimeError("compute-pass FI overflow nonzero")
        _, b0, b1 = extract_flat_wave_spans(pB["slots"], pB["warps_per_cta"])
        sideB = json.load(open(cd + ".json", encoding="utf-8")) if os.path.exists(cd + ".json") else {}
        mbB = (sideB.get("mark_before") or {}).get("hsa_ns")
        offB = int(mbB - b0) if mbB else int((sideB.get("clock_probe") or {}).get("hsa_minus_memrealtime_ns") or 0)
        metaB, sampB = load_pmc(cb, cj)
        serB, repB = build_hw_series(metaB, sampB, b0, b1, n_xcc, n_cu, comm_mode,
                                     comm_counter, pad_us, offB, compute_pass_mode)
        series["compute"] = serB["compute"]
        n = min(len(v) for v in series.values())
        series = {k: v[:n] for k, v in series.items()}
        hwrep["compute_pass"] = dict(pmc_bin=cb, fi_dump=cd, clock_offset_ns=offB,
                                     window_ns=[b0, b1], samples_in_window=repB["samples_in_window"],
                                     counters=metaB["counters"], mode=compute_pass_mode,
                                     cadence_us_median=repB["cadence_us_median"])
        hwrep["common_sample_count"] = n
        compute_mode = compute_pass_mode  # provenance now describes pass B

    # V4 full-duplex Inter-Node RDMA NIC hardware curves: TX (Out) and RX (In)
    # exported strictly separately (NEVER summed).  Omitted on EP8 or when zero
    # bytes moved inside the FI window (never drawn as a dead zero line).
    rdma_names: Dict[str, str] = {}
    if "rdma_rx" in series or "rdma_tx" in series:
        rx_max = max((v for _, v in series.get("rdma_rx", [])), default=0.0)
        tx_max = max((v for _, v in series.get("rdma_tx", [])), default=0.0)
        if ep == 16 and (rx_max >= 0.05 or tx_max >= 0.05):
            rate_g = hwrep["nic"]["total_rate_gbps"]
            if rx_max >= 0.05:
                rdma_names["rdma_rx"] = rdma_curve_name(rate_g, "rx")
            else:
                series.pop("rdma_rx", None)
            if tx_max >= 0.05:
                rdma_names["rdma_tx"] = rdma_curve_name(rate_g, "tx")
            else:
                series.pop("rdma_tx", None)
        else:
            if ep == 16 and hwrep.get("nic"):
                hwrep["nic"]["rdma_skipped_reason"] = (
                    "RDMA NIC sysfs counters were readable (nic_capture.ports) but zero bytes "
                    "moved inside the FI window: the EIC/SolarRDMA GPU-initiated data path "
                    "(caller-owned SQ/CQ rings, EIC_ALLOC_GPU) is not counted by any host-readable "
                    "sysfs / ethtool / rdma-statistic counter on this cluster; the optional RDMA "
                    "curves are omitted instead of drawing dead zero lines")
            series.pop("rdma_rx", None)
            series.pop("rdma_tx", None)
    rdma_name = rdma_names.get("rdma_tx") or rdma_names.get("rdma_rx")

    ordered = sorted(ctas.values(), key=lambda x: (x["xcc"], x["se"], x["cu"], x["cta"]))
    rep_cta = None
    roles = {}
    if ep == 16:
        roles = ep16_wave_roles(ctas)
        rep_cta = ep16_representative_cta(ctas, roles, t0_ns)
        # Representative hybrid CU first (pid 100 = "CU 00"); the other 79 physical
        # CUs keep their (XCC, SE, CU) order on pid 101..179 so all 80 appear in Panel B.
        rep_info = ctas[rep_cta]
        ordered = [rep_info] + [x for x in ordered if x["cta"] != rep_cta]
    events: List[Dict[str, Any]] = []
    events.append({"ph": "M", "pid": 1, "tid": 0, "name": "process_name",
                   "args": {"name": "00 - MI308X rocprofiler-sdk Hardware PMC (device_counting_service)"}})
    events.append({"ph": "M", "pid": 1, "tid": 0, "name": "process_sort_index", "args": {"sort_index": -1000}})
    events.append({"ph": "M", "pid": 1, "tid": 1, "name": "thread_name",
                   "args": {"name": "MI308X Spec & Launch Audit (click for args)"}})
    events.append({"ph": "M", "pid": 1, "tid": 2, "name": "thread_name",
                   "args": {"name": "rocprofiler-sdk device_counting_service - real PMU delta samples"}})
    launch_grid_x = 160 if (ep == 16 and wpc == 4) else len(ordered)
    launch_block_x = 256 if (ep == 16 and wpc == 4) else wpc * 64
    events.append({"ph": "X", "pid": 1, "tid": 1, "ts": 0.0, "dur": sw_span_us,
                   "name": f"AMD Instinct MI308X - {n_cu} CUs x 8 Waves - {model} S={seqlen} "
                           f"(EP={ep}, Rank {rank}) - Uninstrumented={uninstrumented_us:.2f} us",
                   "cat": "info",
                   "args": {"card": "mi308x", "arch": "gfx942", "model": model, "S_global": seqlen,
                            "rank": rank, "gridX": launch_grid_x, "blockX": launch_block_x, "waves_per_cu": 8,
                            "uninstrumented_steady_state_us": round(uninstrumented_us, 2),
                            "s_memrealtime_span_us": sw_span_us, "lds_bytes_per_cu": 65536,
                            "co_sha256": co_sha256}})
    events.append({"ph": "X", "pid": 1, "tid": 2, "ts": 0.0, "dur": sw_span_us,
                   "name": f"Weave persistent 8-wave MegaMoE (EP{ep}, 80 CU x 512 threads)",
                   "cat": "rocm_kernel", "args": {"grid": [launch_grid_x, 1, 1], "block": [launch_block_x, 1, 1], "arch": "gfx942"}})

    role_curve_name: Dict[str, str] = {}
    for role in ("compute", "hbm", "comm", "comm_rx", "comm_tx", "active"):
        if role not in series:
            continue
        cname = P2P_LANDING_RX_NAME if (role == "comm_rx" and comm_mode == "p2p_landing") else CURVE_NAMES[role]
        role_curve_name[role] = cname
    role_curve_name.update(rdma_names)

    npts = 0
    for role, cname in role_curve_name.items():
        for ts_us, val in series[role]:
            events.append({"ph": "C", "pid": 1, "name": cname, "ts": ts_us, "args": {"value": val}})
            npts += 1

    events.append({"ph": "M", "pid": 2, "tid": 0, "name": "process_name",
                   "args": {"name": "01 - Macro Software Semantic Stage Progression"}})
    events.append({"ph": "M", "pid": 2, "tid": 0, "name": "process_sort_index", "args": {"sort_index": -900}})
    if ep == 16:
        macro = [(10, "1. Opening: Local & Remote Route Plan", (13, 17, 18)),
                 (11, "2. Dispatch / Pull (Local Stage + xGMI Pull + RDMA)", (15, 16, 42)),
                 (12, "3. GEMM1 + SwiGLU", (26,)),
                 (13, "4. GEMM2 MFMA", (29,)),
                 (14, "5. Scatter Push / RDMA Return", (9, 43)),
                 (15, "6. Partial Reduce / Home Combine", (41, 12))]
    else:
        macro = [(10, "1. Opening: Local Ballot & 80-CTA Route Plan", (13, 17)),
                 (11, "2. xGMI Dispatch A2A", (15, 16)),
                 (12, "3. GEMM1 + SwiGLU", (26,)),
                 (13, "4. GEMM2 MFMA", (29,)),
                 (14, "5. xGMI Scatter", (9,)),
                 (15, "6. Home Combine / Reduce", (12,))]
    for tid_m, label, eids in macro:
        starts, ends, cnt = [], [], 0
        for info in ordered:
            for spans in info["waves"].values():
                for s_ns, e_ns, eid, *_ in spans:
                    if eid in eids:
                        starts.append((s_ns - t0_ns) / 1000.0)
                        ends.append((e_ns - t0_ns) / 1000.0)
                        cnt += 1
        if starts:
            events.append({"ph": "M", "pid": 2, "tid": tid_m, "name": "thread_name",
                           "args": {"name": f"{label} ({cnt} wave-slices)"}})
            events.append({"ph": "M", "pid": 2, "tid": tid_m, "name": "thread_sort_index",
                           "args": {"sort_index": tid_m}})
            events.append({"ph": "X", "pid": 2, "tid": tid_m, "ts": round(min(starts), 3),
                           "dur": round(max(0.01, max(ends) - min(starts)), 3),
                           "name": label, "cat": "software.macro", "args": {"wave_slices": cnt}})

    stage_counts: Dict[str, int] = collections.Counter()
    reduce_waves: Dict[int, set] = collections.defaultdict(set)
    sw_slices = 0
    specs = EP16_RAW_EVENT_SPECS if ep == 16 else RAW_EVENT_SPECS
    mapped = EP16_MAPPED if ep == 16 else MAPPED
    drop = EP16_DROP if ep == 16 else DROP
    reduce_names = EP16_REDUCE if ep == 16 else {"Home_Combine"}
    for cu_ord, info in enumerate(ordered):
        pid = 100 + cu_ord
        is_rep = (ep == 16 and info["cta"] == rep_cta)
        suffix = ", 4/4 Hybrid" if is_rep else ""
        events.append({"ph": "M", "pid": pid, "tid": 0, "name": "process_name",
                       "args": {"name": f"CU {cu_ord:02d} (XCC{info['xcc']}-SE{info['se']}-CU{info['cu']}, "
                                        f"CTA {info['cta']:02d}{suffix})"}})
        events.append({"ph": "M", "pid": pid, "tid": 0, "name": "process_sort_index", "args": {"sort_index": pid}})
        for w in range(8):
            tid = w + 1
            role_tag = f" {roles.get(info['cta'], {}).get(w, '')}" if ep == 16 else ""
            events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                           "args": {"name": f"CU{cu_ord:02d} - Wave {w} (W{w:02d}){role_tag}"}})
            events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": tid}})
            last_end = -1.0
            for s_ns, e_ns, eid, b_aux, e_aux in info["waves"].get(w, []):
                nm = specs.get(eid, f"Event_{eid}")
                if nm not in mapped and nm not in drop:
                    continue
                ts_us = round((s_ns - t0_ns) / 1000.0, 3)
                end_us = round((e_ns - t0_ns) / 1000.0, 3)
                if ts_us < last_end:
                    ts_us = last_end
                dur = round(end_us - ts_us, 3)
                if dur <= 0.0:
                    continue
                last_end = round(ts_us + dur, 3)
                events.append({"ph": "X", "pid": pid, "tid": tid, "ts": ts_us, "dur": dur,
                               "name": nm, "cat": "software.wave",
                               "args": {"cu": cu_ord, "xcc": info["xcc"], "se": info["se"],
                                        "hw_cu": info["cu"], "cta": info["cta"], "wave": w}})
                stage_counts[nm] += 1
                sw_slices += 1
                if nm in reduce_names:
                    reduce_waves[cu_ord].add(w)

    all8 = sorted(cu for cu, ws in reduce_waves.items() if len(ws) == 8)
    prov = (
        "rocprofiler-sdk 1.1.0 device_counting_service "
        "(rocprofiler_configure_device_counting_service + rocprofiler_sample_device_counting_service), "
        "ROCm 7.2.0, gfx942 MI308X; real PMU counters "
        + ", ".join(meta["counters"])
        + "; per-interval hardware counter deltas on the HSA system clock, which is the same "
          "100 MHz s_memrealtime domain as the FI_LEVEL=2 software anchors"
    )
    if rdma_names:
        prov += "; Inter-Node RDMA NIC full-duplex curves (RX & TX separate, never summed) from " + nic_provenance_text(hwrep["nic"])
    src_desc = {
        "compute": ("SQ_THREAD_CYCLES_VALU (SQ event 118) / (GRBM_GUI_ACTIVE/4 * 4 SIMD * "
                    f"{n_cu} CU) * 100"
                    if compute_mode == "valu_thread_cycles" else
                    f"100 * SQ_VALU_MFMA_BUSY_CYCLES / ({n_cu} * GRBM_GUI_ACTIVE)"),
        "hbm": "(TCC_EA0_RDREQ_DRAM + TCC_EA0_WRREQ_DRAM) * 64B / dt / 3684GB/s (half-duplex shared HBM3 bus)",
        "comm_rx": (f"TCC_EA0_WRREQ_DRAM * 64B / dt / {PEAK_XGMI_GBPS}GB/s (incoming xGMI P2P RX landing only, unmerged)"
                    if comm_mode == "p2p_landing" else
                    f"(TCC_EA0_RDREQ - TCC_EA0_RDREQ_DRAM) * 64B / dt / {PEAK_XGMI_GBPS}GB/s (off-die GMI/xGMI RX reads only, unmerged)"
                    if comm_mode == "gmi_ea" else
                    f"TCC_EA0_RD_UNCACHED_32B * 32B / dt / {PEAK_XGMI_GBPS}GB/s (uncached cross-rank RX reads only, unmerged)"),
        "comm_tx": (f"(TCC_EA0_WRREQ - TCC_EA0_WRREQ_DRAM) * 64B / dt / {PEAK_XGMI_GBPS}GB/s (off-die GMI/xGMI TX writes only, unmerged)"
                    if comm_mode == "gmi_ea" else
                    f"TCC_EA0_WR_UNCACHED_32B * 32B / dt / {PEAK_XGMI_GBPS}GB/s (uncached cross-rank TX writes only, unmerged)"),
        "comm": f"{comm_counter} * 64B / dt / {PEAK_XGMI_GBPS}GB/s",
        "active": f"SQ_BUSY_CU_CYCLES / (GRBM_GUI_ACTIVE/{n_xcc} * {n_cu})",
    }
    curve_sources = {cname: src_desc[role] for role, cname in role_curve_name.items() if role in src_desc}
    for rkey, rname in rdma_names.items():
        curve_sources[rname] = nic_provenance_text(hwrep["nic"], "rx" if rkey == "rdma_rx" else "tx")
    other = {
        "schema_version": "fi-perfetto-swhw/v1",
        "tool": "weaveprofiler",
        "platform": "amd",
        "arch": "gfx942",
        "group": "cu",
        "gran": 0,
        "time_zero": "kernel s_memrealtime first FI event; hardware PMC samples share the HSA system clock",
        "source": prov,
        "meta": {
            "card": "mi308x", "arch": "gfx942", "platform": "amd", "model": model, "S": seqlen,
            "rank": rank, "arm": arm, "ep": ep, "gridX": len(ordered), "waves_per_cu": 8,
            "unique_physical_cus": len({x["hw_key"] for x in ordered}),
            "uninstrumented_steady_state_us": round(uninstrumented_us, 2),
            "s_memrealtime_span_us": sw_span_us,
            "hw_counter_source": prov,
            "curve_sources": curve_sources,
            "env_cfg": env_cfg,
            "co_sha256": co_sha256,
            "pmc_capture": {k: v for k, v in hwrep.items() if k != "raw"},
            "pmc_sampler_meta": meta,
            "nic_capture": (dict(hwrep["nic"], rdma_curve_name=rdma_name, rdma_curve_names=rdma_names)
                            if hwrep.get("nic") else None),
            "all_wave_reduce": {
                "cus_with_home_combine_on_all_8_waves": len(all8),
                "cus_with_any_home_combine": len(reduce_waves),
                "expected_cus": len(ordered),
                "cu0_reduce_waves": sorted(reduce_waves.get(0, set())),
            },
        },
        "stage_slice_counts": dict(stage_counts),
    }
    if ep == 16:
        n08 = sum(1 for c, r in roles.items() if all(r.get(w) == "[Comm]" for w in range(8)))
        n44 = sum(1 for c, r in roles.items()
                  if all(r.get(w) == "[Compute]" for w in range(4))
                  and all(r.get(w) == "[Comm]" for w in range(4, 8)))
        n80 = len(roles) - n08 - n44
        other["meta"]["cta_regimes"] = {"N08_pure_comm": n08, "N44_hybrid": n44, "N80_dual_compute": n80}
        other["meta"]["representative_cta"] = rep_cta
        other["meta"]["representative_cu"] = {
            "pid": 100, "cta": rep_cta,
            "xcc": ctas[rep_cta]["xcc"], "se": ctas[rep_cta]["se"], "cu": ctas[rep_cta]["cu"],
            "reduce_waves": sorted(reduce_waves.get(0, set())),
        }
        other["meta"]["all_wave_reduce"]["reduce_events"] = sorted(EP16_REDUCE)
    if extra_meta:
        other["meta"].update(extra_meta)
    doc = {"displayTimeUnit": "ns", "traceEvents": events, "otherData": other}
    assert_no_forbidden(doc)
    os.makedirs(os.path.dirname(os.path.abspath(out_gz)), exist_ok=True)
    with gzip.open(out_gz, "wt", encoding="utf-8", compresslevel=6) as gf:
        json.dump(doc, gf, ensure_ascii=False)
    rep = dict(out_gz=out_gz, bytes=os.path.getsize(out_gz),
               sha256=hashlib.sha256(open(out_gz, "rb").read()).hexdigest(),
               model=model, S=seqlen, rank=rank, arm=arm,
               uninstrumented_us=round(uninstrumented_us, 2), s_memrealtime_span_us=sw_span_us,
               sw_slices=sw_slices, counter_points=npts,
               curve_stats={k: dict(points=len(v), min=min(x[1] for x in v),
                                    max=max(x[1] for x in v),
                                    mean=round(sum(x[1] for x in v) / len(v), 4),
                                    std=round(math.sqrt(sum((x[1] - sum(y[1] for y in v) / len(v)) ** 2
                                                            for x in v) / len(v)), 4))
                            for k, v in series.items()},
               all_wave_reduce=other["meta"]["all_wave_reduce"],
               stage_slice_counts=dict(stage_counts),
               pmc_capture=other["meta"]["pmc_capture"])
    return rep, hwrep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_bin")
    ap.add_argument("--pmc-bin", required=True)
    ap.add_argument("--pmc-json", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="qwen38max")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--uninstrumented-us", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--arm", default="pure21")
    ap.add_argument("--co-sha256", default="")
    ap.add_argument("--comm-mode", default="pmc")
    ap.add_argument("--comm-counter", default="")
    ap.add_argument("--n-xcc", type=int, default=4)
    ap.add_argument("--n-cu", type=int, default=80)
    ap.add_argument("--pad-us", type=float, default=0.0)
    ap.add_argument("--env-cfg", default="{}", help="JSON dict of the public WEAVE_* knobs used")
    ap.add_argument("--extra-meta", default="{}")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--raw-out", default="")
    ap.add_argument("--ep", type=int, default=8, choices=(8, 16),
                    help="expert-parallel world size of the capture (8 = EP8, 16 = 2-node EP16)")
    ap.add_argument("--compute-pass", nargs=3, metavar=("BIN", "JSON", "DUMP"), default=None,
                    help="second DCS pass (pmc bin, pmc json, FI dump) supplying the MFMA compute curve")
    ap.add_argument("--compute-pass-mode", default="mfma_busy",
                    choices=("mfma_busy", "sq_busy", "valu_thread_cycles", "none"))
    a = ap.parse_args()
    compute_mode = "none" if (a.ep == 16 and a.compute_pass) else "mfma_busy"
    rep, hwrep = build_trace(a.dump_bin, a.pmc_bin, a.pmc_json, a.out, a.model, a.seqlen,
                             a.uninstrumented_us, a.rank, a.arm, json.loads(a.env_cfg),
                             a.comm_mode, a.comm_counter, a.n_xcc, a.n_cu, a.pad_us,
                             a.co_sha256, json.loads(a.extra_meta),
                             compute_pass=tuple(a.compute_pass) if a.compute_pass else None,
                             compute_mode=compute_mode, compute_pass_mode=a.compute_pass_mode,
                             ep=a.ep)
    if a.raw_out:
        with open(a.raw_out, "w", encoding="utf-8") as f:
            json.dump(hwrep, f, indent=1)
    if a.summary_out:
        with open(a.summary_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
