#!/usr/bin/env python3
"""FI_LEVEL=2 per-wave capture + same-run rocprofiler-sdk PMC sampling wrapper.

Imports the repo's own `bench_ep8.py` unmodified and wraps only its launcher
pool, so the measured kernel, kernarg and launch geometry are exactly what the
production harness uses.  Around one chosen launch it adds:

  megakernel_fi_upload    bind FiBuf into the module's fi_lite_global
  megakernel_fi_reset     zero the record arena before the captured launch
  pmc_mark(1) / pmc_mark(2)  host anchors in CLOCK_MONOTONIC + HSA clock
  megakernel_fi_download  write the FILITE02 dump after it

The PMC sampler itself is attached by ROCP_TOOL_LIBRARIES before HSA init and
samples continuously; the exporter selects the samples inside the FI window
using the shared HSA / s_memrealtime clock.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
import time

# Default work root: the repository checkout this script lives in.  The PMC /
# memrealtime sampler .so files are expected under <root>/tools_pmc unless
# PMC_SO / MR_SO override them.
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EP8 = os.environ.get("FI_EP8_DIR", W + "/deep_gemm/include/deep_gemm/impls/amd/ep8")
DUMP_DIR = os.environ["FI_DUMP_DIR"]
CAP = int(os.environ.get("FI_CAP", "8192"))
WPC = int(os.environ.get("FI_WARPS_PER_CTA", "8"))
GRID = int(os.environ.get("FI_GRID", "80"))
TIMER_HZ = int(os.environ.get("FI_TIMER_HZ", "100000000"))
CAPTURE_INDEX = int(os.environ.get("FI_CAPTURE_INDEX", "2"))
CAPTURE_RANKS = {int(x) for x in os.environ.get("FI_CAPTURE_RANKS", "0").split(",") if x != ""}
PMC_SO = os.environ.get("PMC_SO", W + "/tools_pmc/libpmc_sampler.so")

RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
ACTIVE = RANK in CAPTURE_RANKS

sys.path.insert(0, EP8)
import bench_ep8 as B  # noqa: E402

STATE = {"launches": 0, "captured": False, "dump": None}
PMC = None
if ACTIVE and os.path.exists(PMC_SO):
    try:
        PMC = ctypes.CDLL(PMC_SO)
        PMC.pmc_mark.argtypes = [ctypes.c_uint64]
        PMC.pmc_mark.restype = None
        PMC.pmc_now.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        PMC.pmc_now.restype = None
        PMC.pmc_sample_count.restype = ctypes.c_uint64
        PMC.pmc_error_count.restype = ctypes.c_uint64
        PMC.pmc_flush.argtypes = []
        PMC.pmc_flush.restype = None
        PMC.pmc_start.argtypes = []
        PMC.pmc_start.restype = ctypes.c_int
    except OSError as exc:
        print(json.dumps({"event": "pmc_load_failed", "error": str(exc)}), flush=True)
        PMC = None


def _log(event, **kw):
    kw["event"] = event
    kw["rank"] = RANK
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def _now():
    if PMC is None:
        return None
    m, h = ctypes.c_uint64(0), ctypes.c_uint64(0)
    PMC.pmc_now(ctypes.byref(m), ctypes.byref(h))
    return {"mono_ns": int(m.value), "hsa_ns": int(h.value)}


MR_SO = os.environ.get("MR_SO", W + "/tools_pmc/libmemrealtime_probe.so")
MR = None
if ACTIVE and os.path.exists(MR_SO):
    try:
        MR = ctypes.CDLL(MR_SO)
        MR.mr_read.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)]
        MR.mr_read.restype = ctypes.c_int
    except OSError:
        MR = None


def _clock_probe():
    """Read s_memrealtime on the device and the host clocks back to back, so the
    FI (s_memrealtime) and PMC (HSA system clock) time bases can be paired by a
    measured offset instead of an assumed one."""
    if MR is None:
        return None
    ticks, khz = ctypes.c_uint64(0), ctypes.c_int(0)
    rc = int(MR.mr_read(ctypes.byref(ticks), ctypes.byref(khz)))
    host = _now()
    if rc != 0 or host is None:
        return {"rc": rc}
    hz = int(khz.value) * 1000
    mr_ns = (int(ticks.value) * 1_000_000_000 + hz // 2) // hz if hz else None
    return {"rc": rc, "wall_khz": int(khz.value), "wall_hz": hz,
            "memrealtime_ticks": int(ticks.value), "memrealtime_ns": mr_ns,
            "host": host,
            "hsa_minus_memrealtime_ns": (host["hsa_ns"] - mr_ns) if mr_ns else None}


def _bind_fi(entry):
    import torch
    torch.cuda.init()
    lib = entry["lib"]
    for name, argtypes in (
        ("megakernel_fi_upload", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]),
        ("megakernel_fi_reset", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]),
        ("megakernel_fi_download", [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.c_size_t]),
    ):
        if not hasattr(lib, name):
            raise RuntimeError(f"launcher lacks {name}; use aux_fi_launcher/launcher_marked.so")
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    n_slots = GRID * WPC
    rec_bytes = n_slots * CAP * 16
    dev = torch.cuda.current_device()
    rec = torch.zeros(rec_bytes, dtype=torch.uint8, device=f"cuda:{dev}")
    ovf = torch.zeros(1, dtype=torch.uint32, device=f"cuda:{dev}")
    arch = os.environ.get("FI_ARCH", "gfx942").encode()
    packet = struct.pack("<QQQIII32s4x", rec.data_ptr(), ovf.data_ptr(), TIMER_HZ,
                         CAP, WPC, n_slots, arch)
    assert len(packet) == 72
    err = entry["err"]
    rc = int(lib.megakernel_fi_upload(packet, err, len(err)))
    if rc:
        raise RuntimeError(f"fi_upload rc={rc}: {err.value.decode('replace')}")
    entry["fi"] = {"rec": rec, "ovf": ovf, "n_slots": n_slots}
    _log("fi_bound", n_slots=n_slots, cap=CAP, timer_hz=TIMER_HZ, co_sha256=entry.get("co_sha256"))


_orig_get = B.PersistentLauncherPool.get


def _get(self, co_path, symbol):
    entry = _orig_get(self, co_path, symbol)
    if ACTIVE and "fi" not in entry:
        _bind_fi(entry)
    return entry


_orig_launch = B.PersistentLauncherPool.launch


def _launch(self, entry, stream_handle):
    idx = STATE["launches"]
    fi = entry.get("fi") if ACTIVE else None
    if fi is not None and not STATE["captured"] and idx <= CAPTURE_INDEX:
        err = entry["err"]
        rc = int(entry["lib"].megakernel_fi_reset(ctypes.c_void_p(stream_handle), err, len(err)))
        if rc:
            raise RuntimeError(f"fi_reset rc={rc}: {err.value.decode('replace')}")
    if fi is not None and not STATE["captured"] and idx == CAPTURE_INDEX - 1 and PMC is not None:
        # Attach DCS one launch before the measured one.  Attaching tens of
        # seconds earlier freezes the SQ-block counters for the whole megamoe
        # window (verified in pmc_back/{za,zb,pa7}); attaching with a host sleep
        # right before the launch desynchronises the 8 ranks and hangs the tail.
        # One launch of lead time needs no host sleep and keeps SQ live.
        if not STATE.get("pmc_started"):
            rc = int(PMC.pmc_start())
            STATE["pmc_started"] = True
            _log("pmc_start", rc=rc, launch_index=idx)
    if fi is not None and not STATE["captured"] and idx == CAPTURE_INDEX and PMC is not None:
        PMC.pmc_mark(1)
        STATE["mark_before"] = _now()
    _orig_launch(self, entry, stream_handle)
    STATE["launches"] = idx + 1
    if fi is not None and not STATE["captured"] and idx == CAPTURE_INDEX:
        os.makedirs(DUMP_DIR, exist_ok=True)
        path = os.path.join(DUMP_DIR, f"fi_megamoe_rank{RANK}_launch{idx}.bin")
        if os.path.exists(path):
            os.remove(path)
        meta = json.dumps({
            "capture": "megamoe-ep8-per-wave-fi2-pmc",
            "driver": "repo bench_ep8.py (imported unmodified)",
            "rank": RANK, "world_size": int(os.environ.get("WORLD_SIZE", "8")),
            "grid_x": GRID, "block_x": WPC * 64, "warps_per_cta": WPC,
            "launch_index": idx, "co_sha256": entry.get("co_sha256"),
            "timer_hz_source": "hipDeviceAttributeWallClockRate (s_memrealtime domain)",
        }, ensure_ascii=False)
        ovf = ctypes.c_uint32(0)
        err = entry["err"]
        rc = int(entry["lib"].megakernel_fi_download(path.encode(), meta.encode(),
                                                     ctypes.byref(ovf), err, len(err)))
        if rc:
            raise RuntimeError(f"fi_download rc={rc}: {err.value.decode('replace')}")
        if PMC is not None:
            PMC.pmc_mark(2)
            STATE["mark_after"] = _now()
            PMC.pmc_flush()
        STATE["clock_probe"] = _clock_probe()
        _log("clock_probe", **(STATE["clock_probe"] or {}))
        if PMC is not None:
            # Stop sampling immediately: leaving DCS running perturbs the
            # subsequent timing rounds (latched device_error 210/120).
            PMC.pmc_stop()
            _log("pmc_stop", samples=int(PMC.pmc_sample_count()),
                 errors=int(PMC.pmc_error_count()))
        STATE["captured"] = True
        STATE["dump"] = path
        info = {"dump": path, "bytes": os.path.getsize(path), "overflow_bits": int(ovf.value),
                "launch_index": idx, "rank": RANK, "cap": CAP, "timer_hz": TIMER_HZ,
                "mark_before": STATE.get("mark_before"), "mark_after": STATE.get("mark_after"),
                "clock_probe": STATE.get("clock_probe"),
                "pmc_samples": int(PMC.pmc_sample_count()) if PMC else None,
                "pmc_errors": int(PMC.pmc_error_count()) if PMC else None,
                "meta": json.loads(meta)}
        with open(path + ".json", "w") as f:
            json.dump(info, f, indent=1)
        _log("fi_dump_written", **{k: info[k] for k in
                                   ("dump", "bytes", "overflow_bits", "launch_index",
                                    "pmc_samples", "pmc_errors")})


B.PersistentLauncherPool.get = _get
B.PersistentLauncherPool.launch = _launch

if __name__ == "__main__":
    _log("fi_pmc_wrapper_start", active=ACTIVE, capture_index=CAPTURE_INDEX, cap=CAP,
         dump_dir=DUMP_DIR, pmc=bool(PMC))
    code = B.main()
    _log("fi_pmc_wrapper_done", launches=STATE["launches"], captured=STATE["captured"],
         dump=STATE["dump"], driver_rc=code)
    raise SystemExit(code)
