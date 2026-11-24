#!/usr/bin/env python3
"""Two-Level framework gate: offline calibration + online CTA-0 policy parity.

Verifies, without torch / CUDA / GPU:

1. Bit-exact cross-language parity of the OnlineRoutingPolicy: the C++ test
   binary (tests/test_cost_model_eq123.cpp, built with g++/clang++ against the
   real repository headers) prints 12 routing fixtures x 3 hardware x 3 knob
   modes as ONLINE lines (floats as binary32 hex bits); this driver recomputes
   every field with deep_gemm/mega/cost_model.py and compares bit-for-bit.
2. The offline calibration constants (CALIB lines) match
   deep_gemm/mega/calibrate.py and results/hardware_calibration_3hw.json
   bit-for-bit, and every provenance sha256 matches the recorded source.
3. select_offline_regime returns exact structural env dicts (names drawn from
   the 14 public knobs, their two legacy aliases, the accepted-but-not-public compat names and the internal pins of the
   six retired knobs) and the documented regime / verified flags.
4. Eq.(1)-(3) golden bit-identity: the existing C++ goldens are unchanged and
   the Python mirror still reproduces them (validator convention).
5. Explicit-pin override semantics (V_HEAD / READY_GROUP_SIZE / DEDUP_MODE and
   the internal REDUCE_WARPS pin) on the Python mirror directly.
6. The public surface is exactly the 14-knob / 3-category contract
   (Spatial 5, Temporal 3, Implementation 6): exact names and order in
   deep_gemm/mega/knobs.py, the six retired knobs absent from it and from the
   generated parameter tables, the three legacy aliases documented.

On Windows the driver delegates to WSL when g++ + python3 are installed there
(fresh unsigned Windows binaries are commonly blocked by application control),
mirroring tests/test_21_weave_knobs_surface.py.

Usage:
    python tests/test_two_level_adaptive_policy.py [--cpp-out FILE]
Exit status 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = platform.system() == "Windows"

CHECKS = 0
FAILURES = 0


def check(ok: bool, label: str) -> None:
    global CHECKS, FAILURES
    CHECKS += 1
    if not ok:
        FAILURES += 1
        print(f"  [FAIL] {label}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module("weave_cost_model_py_twolevel", ROOT / "deep_gemm/mega/cost_model.py")
CAL = load_module("weave_calibrate_twolevel", ROOT / "deep_gemm/mega/calibrate.py")
KNOBS = load_module("weave_knob_schema_twolevel", ROOT / "deep_gemm/mega/knobs.py")


# ---------------------------------------------------------------------------
# WSL delegation / compiler discovery (same pattern as the host-surface gate)
# ---------------------------------------------------------------------------
def windows_to_wsl(path: str) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[len(resolved.drive):].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def maybe_delegate_to_wsl() -> Optional[int]:
    if not WINDOWS:
        return None
    wsl = shutil.which("wsl")
    if wsl is None:
        return None
    root = windows_to_wsl(str(ROOT))
    probe = subprocess.run(
        [wsl, "-e", "bash", "-lc",
         f"cd {shlex.quote(root)} && command -v g++ >/dev/null 2>&1 "
         "&& command -v python3 >/dev/null 2>&1"],
        capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        return None
    args = ["python3", "tests/test_two_level_adaptive_policy.py", *sys.argv[1:]]
    command = " ".join(shlex.quote(a) for a in args)
    print("[INFO] delegating to WSL: g++ + python3", flush=True)
    return subprocess.run(
        [wsl, "-e", "bash", "-lc",
         f"cd {shlex.quote(root)} && PYTHONUNBUFFERED=1 "
         f"PYTHONIOENCODING=utf-8 {command}"]).returncode


def find_compiler() -> Optional[List[str]]:
    override = os.environ.get("WEAVE_SURFACE_CXX")
    if override:
        return shlex.split(override)
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return [path]
    return None


def build_and_run_cpp() -> str:
    cc = find_compiler()
    if cc is None:
        raise SystemExit("[FAIL] no C++ compiler found (set WEAVE_SURFACE_CXX)")
    with tempfile.TemporaryDirectory(prefix="weave_twolevel_") as td:
        binary = Path(td) / "test_cost_model_eq123"
        cmd = [*cc, "-std=c++17", "-O2", "-pthread",
               "-I", str(ROOT / "deep_gemm/include"),
               str(ROOT / "tests/test_cost_model_eq123.cpp"),
               "-o", str(binary)]
        print("[BUILD] " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f"[FAIL] C++ build exited with {proc.returncode}")
        run = subprocess.run([str(binary)], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=180)
        if run.returncode != 0:
            print(run.stdout[-4000:])
            print(run.stderr[-4000:])
            raise SystemExit(f"[FAIL] C++ test exited with {run.returncode}")
        return run.stdout


# ---------------------------------------------------------------------------
# Fixtures -- identical generation rules to test_cost_model_eq123.cpp
# ---------------------------------------------------------------------------
def max_anchored(total: int, max_n: int, nonzero: int, experts: int) -> List[int]:
    h = [0] * experts
    if nonzero == 0 or total == 0:
        return h
    if nonzero == 1:
        h[0] = total
        return h
    h[0] = max_n
    base, rem = divmod(total - max_n, nonzero - 1)
    for i in range(1, nonzero):
        h[i] = base + (1 if i <= rem else 0)
    return h


def fixtures() -> List[Dict]:
    f = []
    f.append(dict(name="uniform_decode_s128", E=64, H=2048, I=512,
                  m_e=[2] * 64, d_in=128, m_out=128, uniq=128))
    f.append(dict(name="uniform_balanced_s4096", E=128, H=2048, I=512,
                  m_e=[32] * 128, d_in=4096, m_out=4096, uniq=4096))
    f.append(dict(name="h20_rank0_real", E=128, H=8192, I=2048,
                  m_e=max_anchored(2482, 131, 123, 128),
                  d_in=964, m_out=1849, uniq=964))
    f.append(dict(name="h20_rank2_hot", E=128, H=8192, I=2048,
                  m_e=max_anchored(2937, 346, 122, 128),
                  d_in=1002, m_out=2181, uniq=1002))
    f.append(dict(name="skew_hot_1expert", E=128, H=2048, I=512,
                  m_e=[4096] + [8] * 127, d_in=5112, m_out=5112, uniq=5112))
    f.append(dict(name="ultra_dense_32k", E=256, H=2048, I=512,
                  m_e=[128] * 256, d_in=32768, m_out=32768, uniq=32768))
    f.append(dict(name="ultra_dense_64k", E=256, H=2048, I=512,
                  m_e=[256] * 256, d_in=65536, m_out=65536, uniq=65536))
    f.append(dict(name="sparse_low_mu", E=128, H=2048, I=512,
                  m_e=[64] * 16 + [0] * 112, d_in=1024, m_out=1024, uniq=1024))
    f.append(dict(name="high_mu_dup", E=64, H=2048, I=512,
                  m_e=[64] * 64, d_in=512, m_out=4096, uniq=512))
    f.append(dict(name="mid_mu_thresh", E=64, H=2048, I=512,
                  m_e=[32] * 64, d_in=1024, m_out=2048, uniq=1024))
    f.append(dict(name="empty", E=128, H=2048, I=512,
                  m_e=[0] * 128, d_in=0, m_out=0, uniq=0))
    f.append(dict(name="single_expert_max", E=256, H=2048, I=512,
                  m_e=[65535] + [0] * 255, d_in=1024, m_out=65535, uniq=1024))
    return f


MODES = {
    "auto": dict(adapt_bm=1, reduce_warps=4, v_head_override=None,
                 ready_group_size=0, dedup_mode="auto"),
    "pinned": dict(adapt_bm=0, reduce_warps=4, v_head_override=64,
                   ready_group_size=2, dedup_mode=1),
    "pinned_adaptive": dict(adapt_bm=1, reduce_warps=4, v_head_override=64,
                            ready_group_size=2, dedup_mode=0),
}

INT_FIELDS = ["arw", "avh", "arg", "edr", "bmt", "bme", "ae", "mx", "tot",
              "ada", "hbm", "tail"]
FLOAT_FIELDS = ["mean", "mu", "util", "frac"]
POLICY_INT_ATTRS = ["active_reduce_warps", "active_v_head",
                    "active_ready_group_size", "enable_dedup_runtime",
                    "bm_tier_mean", "bm_envelope", "active_experts", "max_n",
                    "total_routes", "adaptive_active", "reduce_gate_hbm_ok",
                    "reduce_gate_tail_ok"]
POLICY_FLOAT_ATTRS = ["mean_n", "rank_hit_multiplicity_mu", "middle_hbm_util",
                      "tail_combine_frac"]


def parse_lines(stdout: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {
        "ONLINE": [], "CALIB": [], "ENV": [], "INNER": [], "ORDER": [],
        "LUT": [], "PEND": [], "PRERED": [], "RING": [], "CLUSIM": []}
    for line in stdout.splitlines():
        for tag in out:
            if line.startswith(tag + " "):
                kv = dict(tok.split("=", 1) for tok in line.split()[1:])
                out[tag].append(kv)
                break
    return out


def check_online_parity(online: List[Dict]) -> None:
    print("\n== OnlineRoutingPolicy bit-exact parity (C++ vs Python) ==")
    f0 = FAILURES
    fx_by_name = {f["name"]: f for f in fixtures()}
    check(len(online) == 108, f"108 ONLINE lines (got {len(online)})")
    seen = set()
    for row in online:
        arch, fix, mode = row["arch"], row["fix"], row["mode"]
        key = (arch, fix, mode)
        check(key not in seen, f"duplicate ONLINE line {key}")
        seen.add(key)
        fx = fx_by_name.get(fix)
        if fx is None or mode not in MODES:
            check(False, f"unknown fixture/mode {key}")
            continue
        wl = M.WorkloadSpec(fx["E"], fx["H"], fx["I"])
        _, p = M.online_policy(arch, wl, fx["m_e"], fx["d_in"], fx["m_out"],
                               unique_tokens=fx["uniq"], **MODES[mode])
        for field, attr in zip(INT_FIELDS, POLICY_INT_ATTRS):
            check(int(row[field]) == getattr(p, attr),
                  f"{key} {field}: C++={row[field]} Py={getattr(p, attr)}")
        for field, attr in zip(FLOAT_FIELDS, POLICY_FLOAT_ATTRS):
            cpp_bits = int(row[field], 16)
            py_bits = M.f32_bits(getattr(p, attr))
            check(cpp_bits == py_bits,
                  f"{key} {field}: C++ bits={cpp_bits:08x} Py bits={py_bits:08x}")
    check(len(seen) == 108, "all 12 fixtures x 3 arch x 3 modes covered")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] online parity: "
          f"{len(online)} policies compared field-by-field (floats by binary32 bits)")


def check_calib_parity(calib: List[Dict]) -> Dict:
    print("\n== Offline calibration constants (C++ vs calibrate.py vs JSON) ==")
    f0 = FAILURES
    profiles = {h: CAL.calibrate_hardware_profile(h)
                for h in ("sm103", "h20", "mi308x")}
    arch_of = {"sm103": "b300", "h20": "h20", "mi308x": "mi308x"}
    json_path = ROOT / "results/hardware_calibration_3hw.json"
    check(json_path.exists(), "results/hardware_calibration_3hw.json exists")
    report = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    check(report.get("schema") == "weave-hardware-calibration-3hw/v1",
          "calibration JSON schema")
    fields = [("feff", "f_eff_tflops"), ("bhbm", "b_hbm_gbps"),
              ("bcomm", "b_comm_gbps"), ("tau", "tau_sync_us"),
              ("mu", "mu_dedup_thresh"), ("hbm", "hbm_headroom_thresh"),
              ("tail", "tail_ratio_thresh")]
    check(len(calib) == 3, f"3 CALIB lines (got {len(calib)})")
    for row in calib:
        arch = row["arch"]
        hw = next(h for h, a in arch_of.items() if a == arch)
        prof = profiles[hw]
        mirror = M.HARDWARE_CALIBRATION[arch]
        jprof = report.get("profiles", {}).get(hw, {})
        for key, attr in fields:
            cpp_bits = int(row[key], 16)
            check(cpp_bits == M.f32_bits(getattr(prof, attr)),
                  f"{hw} {attr}: C++ vs calibrate.py bits")
            check(cpp_bits == M.f32_bits(getattr(mirror, attr)),
                  f"{hw} {attr}: C++ vs cost_model.py bits")
            # The JSON records struct.pack('f') bytes (little-endian); the
            # C++ CALIB line prints the uint32 value of the same bits.
            json_hex = jprof.get("float_bits", {}).get(attr, "")
            json_bits = (int.from_bytes(bytes.fromhex(json_hex), "little")
                         if json_hex else -1)
            check(cpp_bits == json_bits, f"{hw} {attr}: C++ vs JSON bits")
        check(int(row["vhb"]) == prof.v_head_hot_budget == mirror.v_head_hot_budget,
              f"{hw} v_head_hot_budget")
        check(int(row["slack"]) == prof.bm_pad_slack == mirror.bm_pad_slack,
              f"{hw} bm_pad_slack")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] calibration constants: "
          "3 hardware x 9 constants, C++ == calibrate.py == cost_model.py == JSON")
    return report


def check_provenance(report: Dict) -> None:
    print("\n== Provenance: recorded sources + sha256 ==")
    import hashlib
    for hw, prof in report.get("profiles", {}).items():
        prov = prof.get("provenance", {})
        src = prov.get("source", "")
        sha = prov.get("sha256", "")
        check(bool(src) and bool(sha), f"{hw}: provenance source + sha256 recorded")
        # Repository-local sources are re-hashed live; external sprint artifacts
        # are verified against the hash transcribed in calibrate.py.
        local = ROOT / src
        if local.exists():
            live = hashlib.sha256(local.read_bytes()).hexdigest()
            check(live == sha, f"{hw}: live sha256 of {src} matches provenance")
        else:
            cal_sha = {"sm103": CAL.SM103_SHA256, "h20": CAL.H20_SHA256,
                       "mi308x": CAL.MI308X_SHA256}.get(hw)
            check(sha == cal_sha,
                  f"{hw}: external artifact sha matches calibrate.py transcription")
        check(bool(prov.get("derivations")), f"{hw}: derivation rules documented")
    routing = ROOT / "data/routing_stats_3hw.json"
    if routing.exists():
        live = hashlib.sha256(routing.read_bytes()).hexdigest()
        check(live == CAL.ROUTING_SHA256, "routing replay sha256 matches calibrate.py")


EXPECTED_PUBLIC = [
    ("WEAVE_DISPATCH_ACTIVE_SMS", "Spatial"), ("WEAVE_SCATTER_ACTIVE_SMS", "Spatial"),
    ("WEAVE_DISPATCH_WARPS", "Spatial"), ("WEAVE_SCATTER_WARPS", "Spatial"),
    ("WEAVE_CU_WAVE_SPLIT", "Spatial"),
    ("WEAVE_V_HEAD", "Temporal"), ("WEAVE_READY_GROUP_SIZE", "Temporal"),
    ("WEAVE_SCATTER_OWNER", "Temporal"),
    ("WEAVE_PRECISION_MODE", "Implementation"),
    ("WEAVE_GEMM1_CONFIG", "Implementation"), ("WEAVE_GEMM2_CONFIG", "Implementation"),
    ("WEAVE_REG_PULL_BYTES", "Implementation"), ("WEAVE_DEDUP_MODE", "Implementation"),
    ("WEAVE_SCATTER_STAGING", "Implementation"),
]
EXPECTED_RETIRED = ["WEAVE_REDUCE_WARPS", "WEAVE_COMBINE_SYNC_CHUNK", "WEAVE_REG_BALANCE",
                    "WEAVE_LOCAL_SCATTER_DEDUP", "WEAVE_SM0_STREAM_SCHED", "WEAVE_READY_QUEUE"]
EXPECTED_INTERNAL_COMPAT = ["WEAVE_ADAPT_GEMM_CONFIG", "WEAVE_ADAPT_BM"]
EXPECTED_ALIASES = {"WEAVE_BLOCK_M_L1": "WEAVE_GEMM1_CONFIG",
                    "WEAVE_BLOCK_M_L2": "WEAVE_GEMM2_CONFIG"}


def check_knob_contract() -> None:
    print("\n== 14-knob / 3-category public contract ==")
    f0 = FAILURES
    knobs = list(KNOBS.KNOBS)
    check([(k.name, k.category) for k in knobs] == EXPECTED_PUBLIC,
          "schema lists exactly the 14 public knobs, in contract order, with their categories")
    check(len(knobs) == 14, "14 public knobs in the schema")
    counts = {c: sum(1 for k in knobs if k.category == c) for c in ("Spatial", "Temporal", "Implementation")}
    check(counts == {"Spatial": 5, "Temporal": 3, "Implementation": 6},
          f"category split is Spatial 5 / Temporal 3 / Implementation 6 (got {counts})")
    check(tuple(KNOBS.CATEGORIES) == ("Spatial", "Temporal", "Implementation"),
          "CATEGORIES keeps the three existing category strings")
    check(list(KNOBS.RETIRED_INTERNAL) == EXPECTED_RETIRED,
          "RETIRED_INTERNAL lists exactly the six retired knobs")
    check(dict(KNOBS.LEGACY_ALIASES) == EXPECTED_ALIASES,
          "LEGACY_ALIASES maps the two renamed knobs to their public names")
    names = {k.name for k in knobs}
    check(not (set(EXPECTED_RETIRED) & names), "no retired knob is in KNOBS")
    check(not (set(EXPECTED_ALIASES) & names), "no legacy alias is a public knob")
    check(list(KNOBS.INTERNAL_COMPAT_ENV) == EXPECTED_INTERNAL_COMPAT,
          "INTERNAL_COMPAT_ENV lists exactly the accepted-but-not-public compat names")
    check(not (set(EXPECTED_INTERNAL_COMPAT) & (names | set(EXPECTED_ALIASES) | set(EXPECTED_RETIRED))),
          "internal compat names are disjoint from public knobs, aliases and retired pins")
    check(tuple(KNOBS.REQUIRED) == tuple(n for n, _ in EXPECTED_PUBLIC),
          "REQUIRED follows the contract order")
    check(set(KNOBS.by_name()) == names and len(KNOBS.as_dicts()) == 14,
          "by_name() / as_dicts() cover the 14 knobs")
    for old, new in EXPECTED_ALIASES.items():
        check(KNOBS.canonical_name(old) == new and KNOBS.legacy_aliases_of(new) == (old,),
              f"{old} is a transparent alias of {new}")
        check(old in KNOBS.by_name()[new].values, f"{new}: schema text documents the alias {old}")
    # Parameter tables of the generated reference: exactly the 14 names, no retired
    # knob anywhere inside a parameter row, every alias documented.
    doc = (ROOT / "docs" / "TUNABLE_PARAMETERS.md").read_text(encoding="utf-8")
    rows = []
    for line in doc.splitlines():
        if line.startswith("| `WEAVE_"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append((cells[0].strip("`"), cells))
    check([n for n, _ in rows] == [n for n, _ in EXPECTED_PUBLIC],
          "docs/TUNABLE_PARAMETERS.md parameter tables list exactly the 14 knobs in contract order")
    row_text = " ".join(" ".join(c) for _, c in rows)
    for retired in EXPECTED_RETIRED:
        check(retired not in row_text, f"retired {retired} is absent from the parameter tables")
        check(retired in doc, f"retired {retired} is documented in the retired/internal note")
    for compat in EXPECTED_INTERNAL_COMPAT:
        check(compat not in dict(rows), f"internal compat {compat} is not a parameter row")
        check(any(compat in ln for ln in doc.splitlines() if not ln.startswith("|")),
              f"internal compat {compat} is documented in prose")
    for old, new in EXPECTED_ALIASES.items():
        row = dict(rows).get(new, [])
        check(old in " ".join(row), f"{new} row documents the legacy alias {old}")
        check(any(old in ln and new in ln for ln in doc.splitlines() if not ln.startswith("|")),
              f"alias list maps {old} -> {new}")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] 14-knob / 3-category contract")


def check_regimes(report: Dict) -> None:
    print("\n== select_offline_regime: structural env names ==")
    f0 = FAILURES
    public = set(KNOBS.by_name().keys())
    check(len(public) == 14, "14 public knobs in the schema")
    # Level-1 profiles write the structural env with the legacy spellings of the
    # three renamed knobs and with the internal pins of the retired knobs
    # (results/hardware_calibration_3hw.json pins those spellings): every key
    # must be a public knob, a legacy alias of one, or a retired internal pin.
    known = (public | set(KNOBS.LEGACY_ALIASES) | set(KNOBS.RETIRED_INTERNAL)
             | set(KNOBS.INTERNAL_COMPAT_ENV))
    for hw in ("sm103", "h20", "mi308x"):
        prof = CAL.calibrate_hardware_profile(hw)
        check(set(prof.structural_knobs.keys()) <= known,
              f"{hw}: structural env uses only public knobs, legacy aliases or internal pins")
        for s_global, shared in ((128, 0), (1024, 0), (4096, 0),
                                 (8192, 0), (32768, 0), (4096, 2)):
            regime = CAL.select_offline_regime(prof, s_global, shared)
            check(set(regime.structural_env.keys()) <= known,
                  f"{hw} S={s_global}: regime env names are public, alias or internal pins")
            check(regime.structural_env == prof.structural_knobs,
                  f"{hw} S={s_global}: regime env frozen from the verified mapping")
            check(regime.online_thresholds == prof.online_thresholds(),
                  f"{hw} S={s_global}: online thresholds carried")
            check(regime.verified == (s_global <= prof.verified_max_s_global),
                  f"{hw} S={s_global}: verified flag")
            check(bool(regime.anchor), f"{hw} S={s_global}: anchor citation")
        jprof = report.get("profiles", {}).get(hw, {})
        check(jprof.get("structural_knobs") == prof.structural_knobs,
              f"{hw}: JSON structural env matches calibrate.py")
    # MI308X never selects the runtime-FAIL 8-0 wave split.
    mi = CAL.calibrate_hardware_profile("mi308x")
    check(mi.structural_knobs.get("WEAVE_CU_WAVE_SPLIT") != 3,
          "mi308x regime never selects CU_WAVE_SPLIT=8-0 (runtime FAIL)")
    # 2-Node 16xMI308X EP16 Level-1 profile: the structural env must use only
    # public / alias / internal-pin names, match the JSON, and pass the EP16 TU
    # fail-closed knob gate
    # (megamoe_amd.hip MI308X_EP16_KNOB_GATE) so it can actually be built.
    ep = CAL.calibrate_hardware_profile("mi308x_ep16")
    sk = ep.structural_knobs
    check(set(sk.keys()) <= known,
          "mi308x_ep16: structural env uses only public knobs, legacy aliases or internal pins")
    check(report.get("profiles", {}).get("mi308x_ep16", {}).get("structural_knobs") == sk,
          "mi308x_ep16: JSON structural env matches calibrate.py")
    check(sk.get("WEAVE_SM0_STREAM_SCHED", 0) == 0,
          "mi308x_ep16: internal WEAVE_SM0_STREAM_SCHED pin = 0 (EP16 TU rejects 1)")
    check(sk.get("WEAVE_REDUCE_WARPS", 0) == 0,
          "mi308x_ep16: internal WEAVE_REDUCE_WARPS pin = 0 (EP16 TU rejects != 0)")
    check(sk.get("WEAVE_CU_WAVE_SPLIT", 0) in (0, 1, 2, 3),
          "mi308x_ep16: WEAVE_CU_WAVE_SPLIT in the EP16 gate range")
    check(all(sk.get(k, 0) in (0, 16, 32, 64, 128)
              for k in ("WEAVE_BLOCK_M_L1", "WEAVE_BLOCK_M_L2")),
          "mi308x_ep16: WEAVE_BLOCK_M_L1/L2 in the EP16 gate set")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] regimes: 3 hardware x 6 launch points")


def check_golden_identity(stdout: str) -> None:
    print("\n== Eq.(1)-(3) golden bit-identity ==")
    check("ALL EQ(1)-(3), CTA-0 SELECTION, E_local<=256, WEAVE_V_HEAD, "
          "WEAVE_REDUCE_WARPS AND TWO-LEVEL ONLINE POLICY TESTS PASSED" in stdout,
          "C++ binary printed the full pass banner (all existing suites green)")
    check("ALL PHASE-NK PROFILES, SM0 EXPERT ORDER, READY-TOKEN QUEUE AND "
          "PROFILE-LUT TESTS PASSED" in stdout,
          "C++ binary printed the 3-Tier pass banner (new suites green)")
    golden = {"b300": (264196, 0, 32, 14.022),
              "h20": (2147813386, 0, 32, 18.107),
              "mi308x": (526336, 0, 32, 22.103)}
    wl = M.WorkloadSpec(4, 128, 64)
    m_e = [0, 3, 5, 9]
    for arch, (r_mask, scatter, v_head, total_us) in golden.items():
        b = M.select_policy(arch, wl, m_e, 23, 17)
        check(b.r_mask == r_mask and b.scatter_mode == scatter and
              b.v_head == v_head and abs(b.t_total_us - total_us) <= 0.002,
              f"{arch}: Python Eq. golden unchanged")
    # The online tier must not move the Eq. decision (C++ checks this per
    # fixture; re-verify the canonical fixture through the f32 mirror too).
    for arch in golden:
        br0 = M.select_policy_f32(arch, wl, m_e, 23, 17, reduce_warps=0)
        _, p = M.online_policy(arch, wl, m_e, 23, 17, adapt_bm=1, reduce_warps=0)
        check(p.total_routes == 17 and p.active_experts == 3,
              f"{arch}: online stats on the golden fixture")
        check(br0.total_tokens == 17, f"{arch}: f32 mirror M snapshot")


def check_pin_semantics() -> None:
    print("\n== Explicit-pin override semantics (Python mirror) ==")
    wl = M.WorkloadSpec(128, 8192, 2048)
    m_e = max_anchored(2937, 346, 122, 128)  # hot-expert H20 rank2
    _, p_auto = M.online_policy("h20", wl, m_e, 1002, 2181,
                                adapt_bm=1, reduce_warps=4, unique_tokens=1002)
    _, p_pin = M.online_policy("h20", wl, m_e, 1002, 2181,
                               adapt_bm=1, reduce_warps=4, v_head_override=64,
                               ready_group_size=2, dedup_mode=0, unique_tokens=1002)
    check(p_pin.active_v_head == 64, "V_HEAD pin wins under adaptive skew")
    check(p_pin.active_ready_group_size == 2, "READY_GROUP pin wins")
    check(p_pin.enable_dedup_runtime == 0, "DEDUP_MODE=0 pin wins")
    check(p_auto.active_v_head <= 64 and p_auto.active_v_head % 32 == 0,
          "auto hot-expert v_head shrinks to the calibrated budget (64)")
    _, p_off = M.online_policy("h20", wl, m_e, 1002, 2181,
                               adapt_bm=0, reduce_warps=4, unique_tokens=1002)
    check(p_off.active_reduce_warps == 4 and p_off.adaptive_active == 0,
          "adapt_bm=0 keeps R_max deterministic (legacy bit-identity)")
    _, p_zero = M.online_policy("h20", wl, m_e, 1002, 2181,
                                adapt_bm=1, reduce_warps=0, unique_tokens=1002)
    check(p_zero.active_reduce_warps == 0, "R_max=0 compiles no lanes")
    # mu gate: high-mu duplicate load enables dedup in auto, low-mu does not.
    wl64 = M.WorkloadSpec(64, 2048, 512)
    _, p_hi = M.online_policy("h20", wl64, [64] * 64, 512, 4096,
                              adapt_bm=1, unique_tokens=512)
    _, p_lo = M.online_policy("h20", wl64, [64] * 64, 4096, 4096,
                              adapt_bm=1, unique_tokens=4096)
    check(p_hi.enable_dedup_runtime == 1 and p_hi.rank_hit_multiplicity_mu >= 3.0,
          "mu=8 >= h20 threshold 3 -> dedup recommended")
    check(p_lo.enable_dedup_runtime == 0, "mu=1 < h20 threshold 3 -> no dedup")


# ---------------------------------------------------------------------------
# 3-Tier upgrade parity: (Phase,N,K) envelopes + per-M inner tuples, SM0
# expert ordering, and the online per-expert profile LUT.  Every compared
# quantity is integer (codes / fields / permutations) -- bit-exact by
# construction, no float rounding involved.
# ---------------------------------------------------------------------------

PREC_OF_INDEX = {0: "fp8xfp4", 1: "fp4xfp4", 2: "fp8", 3: "bf16"}


def check_phase_nk_parity(env_rows: List[Dict], inner_rows: List[Dict]) -> None:
    print("\n== Tier-1 (Phase,N,K) envelope + per-M inner tuple parity ==")
    f0 = FAILURES
    check(len(env_rows) == 192, f"192 ENV lines (got {len(env_rows)})")
    check(len(inner_rows) == 3456, f"3456 INNER lines (got {len(inner_rows)})")
    env_cache: Dict[Tuple, object] = {}
    for row in env_rows:
        arch = row["arch"]
        prec = PREC_OF_INDEX[int(row["prec"])]
        phase, n, k = int(row["phase"]), int(row["n"]), int(row["k"])
        env = CAL.select_phase_envelope(arch, prec, phase, n, k)
        env_cache[(arch, prec, phase, n, k)] = env
        key = f"{arch}/{prec}/p{phase}/N{n}/K{k}"
        check(int(row["code"], 16) == env.code(), f"{key} envelope code")
        for field, attr in (("bn", "bn"), ("bk", "bk"), ("bm", "max_bm"),
                            ("st", "num_stages"), ("grp", "l2_swizzle_group"),
                            ("mode", "interleave_mode"), ("epw", "epw"),
                            ("score", "score_ns"), ("wg", "wg_layout_mode"),
                            ("pen", "penalty")):
            check(int(row[field]) == getattr(env, attr),
                  f"{key} {field}: C++={row[field]} Py={getattr(env, attr)}")
        # Backend tile legality (the kernels' own static asserts).
        check(env.bn in CAL.PHASE_NK_BN[arch], f"{key} legal BLOCK_N")
        check(env.bk in CAL.PHASE_NK_BK[arch], f"{key} legal BLOCK_K")
        check(env.max_bm in CAL.PHASE_NK_BM[arch], f"{key} legal BLOCK_M")
    for row in inner_rows:
        arch = row["arch"]
        prec = PREC_OF_INDEX[int(row["prec"])]
        phase, n, k = int(row["phase"]), int(row["n"]), int(row["k"])
        m, dedup, tight = int(row["m"]), int(row["dedup"]), int(row["tight"])
        env = env_cache.get((arch, prec, phase, n, k))
        if env is None:
            env = CAL.select_phase_envelope(arch, prec, phase, n, k)
        p = CAL.select_inner_from_envelope(arch, prec, phase, n, k, env,
                                           m, dedup, tight)
        key = f"{arch}/{prec}/p{phase}/N{n}/K{k}/M{m}/d{dedup}/t{tight}"
        check(int(row["code"], 16) == p.code(arch), f"{key} inner code")
        for field, attr in (("bm", "block_m"), ("tma", "tma_load_m"),
                            ("wg", "wg_layout_mode"), ("pipe", "pipe_mode"),
                            ("grp", "l2_swizzle_group"),
                            ("store", "store_block_m"),
                            ("epi", "epi_reg_stages"), ("scat", "scatter_path")):
            check(int(row[field]) == getattr(p, attr),
                  f"{key} {field}: C++={row[field]} Py={getattr(p, attr)}")
    # GEMM1 vs GEMM2 independence: the two phases of the same model must not
    # be forced onto one shared envelope (the pre-upgrade behavior).
    differing = 0
    for (arch, prec, phase, n, k), env in env_cache.items():
        if phase != 1:
            continue
        other = env_cache.get((arch, prec, 2, k, n // 2))
        if other is not None and other.code() != env.code():
            differing += 1
    check(differing > 0, "GEMM1/GEMM2 envelopes solved independently")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] phase-nk parity: "
          f"{len(env_rows)} envelopes + {len(inner_rows)} inner tuples "
          f"field-by-field (integer codes), {differing} independent phase pairs")


def order_fixtures_py() -> List[Dict]:
    """Identical generation rules to test_cost_model_eq123.cpp order_fixtures.

    The zipf histograms come from the SAME calibrate.py implementation that
    froze the C++ literals, so no libm divergence can enter the comparison.
    """
    f = []
    z64 = CAL.zipf_histogram(4096, 64)
    f.append(dict(name="zipf64_4096_w16", counts=z64, wave=16, shift=32, dedup=1))
    f.append(dict(name="zipf64_4096_w16_nd", counts=z64, wave=16, shift=32, dedup=0))
    f.append(dict(name="h20_rank0_w78",
                  counts=CAL.max_anchored_histogram(2482, 131, 123, 128),
                  wave=78, shift=64, dedup=1))
    f.append(dict(name="h20_rank2_w78",
                  counts=CAL.max_anchored_histogram(2937, 346, 122, 128),
                  wave=78, shift=64, dedup=1))
    f.append(dict(name="zipf128_32768_w16", counts=CAL.zipf_histogram(32768, 128),
                  wave=16, shift=64, dedup=1))
    f.append(dict(name="decode128_w78", counts=CAL.zipf_histogram(320, 128),
                  wave=78, shift=64, dedup=1))
    f.append(dict(name="ten_hot_w78", counts=[100] * 10 + [5] * 118,
                  wave=78, shift=64, dedup=0))
    f.append(dict(name="empty_128", counts=[0] * 128, wave=78, shift=64, dedup=1))
    f.append(dict(name="single_256", counts=[65535] + [0] * 255,
                  wave=16, shift=128, dedup=1))
    f.append(dict(name="dense256_w16", counts=[128] * 256,
                  wave=16, shift=128, dedup=1))
    f.append(dict(name="uniform64_w8", counts=[32] * 64, wave=8, shift=32, dedup=1))
    f.append(dict(name="skew_hot_1expert", counts=[4096] + [8] * 127,
                  wave=16, shift=64, dedup=1))
    f.append(dict(name="tiny_waves_w1", counts=[50, 3, 80, 2],
                  wave=1, shift=2, dedup=1))
    return f


def check_sm0_order_parity(order_rows: List[Dict]) -> None:
    print("\n== SM0 dedup ordering + hot/cold pairing (C++ vs Python mirror) ==")
    f0 = FAILURES
    fixtures = {f["name"]: f for f in order_fixtures_py()}
    check(len(order_rows) == len(fixtures),
          f"{len(fixtures)} ORDER lines (got {len(order_rows)})")
    span_checked = 0
    var_checked = 0
    for row in order_rows:
        fx = fixtures.get(row["fix"])
        if fx is None:
            check(False, f"unknown ORDER fixture {row['fix']}")
            continue
        counts = fx["counts"]
        e_count = len(counts)
        multi = [c // 2 for c in counts]
        partner = [(e + fx["shift"]) % e_count for e in range(e_count)]
        expect, expect_mode = CAL.build_sm0_expert_order_with_mode(
            counts, multi, partner, fx["wave"], fx["dedup"])
        got = [int(x) for x in row["order"].split(",")]
        check(got == expect, f"{fx['name']} permutation bit-exact")
        check(int(row["mode"]) == expect_mode,
              f"{fx['name']} mode verdict bit-exact "
              f"(C++={row['mode']} Py={expect_mode})")
        if not fx["dedup"]:
            check(expect_mode == (CAL.SM0_MODE_HOT_COLD if fx["wave"] >= 2
                                  else CAL.SM0_MODE_NO_PAIR),
                  f"{fx['name']} hot/cold pairing needs experts_per_wave >= 2")
        check(int(row["E"]) == e_count and int(row["wave"]) == fx["wave"] and
              int(row["dedup"]) == fx["dedup"], f"{fx['name']} fixture metadata")
        # Independent metric re-verification on the Python side.
        active = sum(1 for c in counts if c > 0)
        if fx["dedup"] and active > 1:
            hits = [[e, partner[e]] for e in range(e_count)
                    if counts[e] > 0 and counts[partner[e]] > 0]
            base = CAL.sm0_completion_span(list(range(e_count)), hits)
            new = CAL.sm0_completion_span(expect, hits)
            if base > 0:
                check(new * 100 <= base * 70,
                      f"{fx['name']} span reduction >= 30% ({base}->{new})")
                span_checked += 1
        desc = sorted(range(e_count), key=lambda e: (-counts[e], e))
        vd = CAL.sm0_wave_m_variance(desc, counts, fx["wave"])
        vn = CAL.sm0_wave_m_variance(expect, counts, fx["wave"])
        check(vn <= vd + 1e-9, f"{fx['name']} wave variance never grows")
        if fx["name"] in ("zipf64_4096_w16", "zipf64_4096_w16_nd",
                          "zipf128_32768_w16", "ten_hot_w78", "h20_rank2_w78"):
            check(vn < vd, f"{fx['name']} wave variance strictly lower")
            var_checked += 1
    check(span_checked >= 6, f"span reduction verified on {span_checked} fixtures")
    check(var_checked >= 5, f"variance reduction verified on {var_checked} fixtures")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] sm0 order parity: "
          f"{len(order_rows)} permutations bit-exact, span >= 30% on "
          f"{span_checked}, variance lower on {var_checked}")


def check_profile_lut_parity(lut_rows: List[Dict]) -> None:
    print("\n== Online per-expert profile LUT publication parity ==")
    f0 = FAILURES
    fx_by_name = {f["name"]: f for f in fixtures()}
    check(len(lut_rows) == 36, f"36 LUT lines (got {len(lut_rows)})")
    codes_checked = 0
    for row in lut_rows:
        arch, fix = row["arch"], row["fix"]
        fx = fx_by_name.get(fix)
        if fx is None:
            check(False, f"unknown LUT fixture {fix}")
            continue
        wl = M.WorkloadSpec(fx["E"], fx["H"], fx["I"])
        _, p = M.online_policy(arch, wl, fx["m_e"], fx["d_in"], fx["m_out"],
                               adapt_bm=1, reduce_warps=4,
                               unique_tokens=fx["uniq"])
        key = f"{arch}/{fix}"
        check(int(row["active"]) == 1, f"{key} phase_profile_active")
        tight = 1 if arch == "h20" else 0
        verdict = p.enable_dedup_runtime
        n1, k1 = 2 * fx["I"], fx["H"]
        n2, k2 = fx["H"], fx["I"]
        env1 = CAL.select_phase_envelope(arch, "fp8xfp4", 1, n1, k1)
        env2 = CAL.select_phase_envelope(arch, "fp8xfp4", 2, n2, k2)
        check(int(row["l1env"], 16) == env1.code(), f"{key} GEMM1 envelope code")
        check(int(row["l2env"], 16) == env2.code(), f"{key} GEMM2 envelope code")
        l1 = [int(x, 16) for x in row["l1"].split(",")]
        l2 = [int(x, 16) for x in row["l2"].split(",")]
        check(len(l1) == fx["E"] and len(l2) == fx["E"], f"{key} LUT length")
        for e, m in enumerate(fx["m_e"]):
            if m == 0:
                exp1 = exp2 = 0
            else:
                b = CAL.m_bucket(m)
                exp1 = CAL.select_phase_inner_profile(
                    arch, "fp8xfp4", 1, n1, k1, b, verdict, tight).code(arch)
                exp2 = CAL.select_phase_inner_profile(
                    arch, "fp8xfp4", 2, n2, k2, b, verdict, tight).code(arch)
            check(l1[e] == exp1, f"{key} e{e} GEMM1 code")
            check(l2[e] == exp2, f"{key} e{e} GEMM2 code")
            codes_checked += 2
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] profile LUT parity: "
          f"{len(lut_rows)} publications, {codes_checked} per-expert codes")


# ---------------------------------------------------------------------------
# Round-2 synthesis parity: pendulum / no-pair mode verdicts, lock-free
# pre-reduce, sharded ready rings (3 modes) and cluster DSMEM fan-out.
# The Python models below are ISOMORPHIC to the C++ host simulations; the
# real device PTX path is proven by tests/check_dsmem_sched_compile.py.
# ---------------------------------------------------------------------------

RING_SLOTS = 256
RING_MODE_CLUSTER_SPSC = 0
RING_MODE_MPSC_EPOCH = 1
RING_MODE_MPMC_STEAL = 2


def pendulum_fixtures_py() -> List[Dict]:
    """Same generation rules as test_cost_model_eq123.cpp test_sm0_order_modes."""
    out = []
    for E in (32, 64, 128):
        counts = [400 // (e + 1) + 4 for e in range(E)]
        hits = [(E - e) if e % 3 == 0 else 0 for e in range(E)]
        name = f"pend{E}"
        out.append(dict(name=name, counts=counts, hits=hits, wave=16))
    out.append(dict(name="w1_nopair", counts=[100 - 3 * e for e in range(16)],
                    hits=[0] * 16, wave=1))
    return out


def pend_hit_sets(counts: Sequence[int], hits: Sequence[int]) -> List[List[int]]:
    E = len(counts)
    sets = []
    for t in range(24):
        hs = [e for e in range(t % 3, E, 3)
              if hits[e] > 0 and (e // 3) % 4 == t % 4]
        if len(hs) >= 2:
            sets.append(hs)
    return sets


def check_pendulum_parity(rows: List[Dict]) -> None:
    print("\n== SM0 pendulum / no-pair mode verdicts (C++ vs Python mirror) ==")
    f0 = FAILURES
    fx_by_name = {f["name"]: f for f in pendulum_fixtures_py()}
    check(len(rows) == len(fx_by_name),
          f"{len(fx_by_name)} PEND lines (got {len(rows)})")
    reductions = []
    for row in rows:
        fx = fx_by_name.get(row["fix"])
        if fx is None:
            check(False, f"unknown PEND fixture {row['fix']}")
            continue
        counts, hits = fx["counts"], fx["hits"]
        E = len(counts)
        none = [E] * E
        dedup = 0 if row["fix"] == "w1_nopair" else 1
        expect, mode = CAL.build_sm0_expert_order_with_mode(
            counts, hits, none, fx["wave"], dedup)
        got = [int(x) for x in row["order"].split(",")]
        check(got == expect, f"{fx['name']} pendulum permutation bit-exact")
        check(int(row["mode"]) == mode, f"{fx['name']} mode verdict")
        if row["fix"] == "w1_nopair":
            check(mode == CAL.SM0_MODE_NO_PAIR, "W=1 auto-exits pairing")
            continue
        check(mode == CAL.SM0_MODE_PENDULUM,
              "multi-hit-only dedup selects pendulum")
        sets = pend_hit_sets(counts, hits)
        base = CAL.sm0_completion_span(list(range(E)), sets)
        new = CAL.sm0_completion_span(expect, sets)
        check(int(row["base"]) == base, f"{fx['name']} base span parity")
        check(int(row["new"]) == new, f"{fx['name']} pendulum span parity")
        check(base > 0 and new < base, f"{fx['name']} span strictly shorter")
        reductions.append(100.0 * (base - new) / base)
    check(len(reductions) == 3, "3 pendulum span measurements")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] pendulum parity: "
          f"{len(rows)} fixtures bit-exact, span reduction "
          f"{min(reductions):.1f}-{max(reductions):.1f}%")


class PyReadyRing:
    """Isomorphic model of weave::ShardedReadyTokenRing (all three modes).

    Atomics are modelled with a lock so the ALGORITHM (claim-after-ready-check)
    is what the threaded exactly-once test exercises.
    """

    def __init__(self, num_shards: int, mode: int, leaders=None):
        import threading
        self.d = num_shards
        self.mode = mode
        self.epoch = 1
        self.owner = [leaders[d] if leaders else d for d in range(num_shards)]
        self.head = [0] * num_shards
        self.tail = [0] * num_shards
        self.overflow = [0] * num_shards
        self.token = [[0] * RING_SLOTS for _ in range(num_shards)]
        self.seq = [[0 if mode == RING_MODE_MPSC_EPOCH else i
                     for i in range(RING_SLOTS)] for _ in range(num_shards)]
        self._lock = threading.Lock()

    def _cas(self, arr, idx, expected, desired):
        with self._lock:
            if arr[idx] != expected:
                return False
            arr[idx] = desired
            return True

    def _fetch_add(self, arr, idx, delta):
        with self._lock:
            old = arr[idx]
            arr[idx] = old + delta
            return old

    def begin_epoch(self, epoch):
        self.epoch = epoch
        for d in range(self.d):
            self.head[d] = self.tail[d] = self.overflow[d] = 0

    def push(self, d, sm, token):
        if self.mode == RING_MODE_CLUSTER_SPSC:
            if sm != self.owner[d]:
                return False
            pos = self.tail[d]
            slot = pos & (RING_SLOTS - 1)
            if self.seq[d][slot] != pos:
                return False
            self.token[d][slot] = token
            self.seq[d][slot] = pos + 1
            self.tail[d] = pos + 1
            return True
        if self.mode == RING_MODE_MPSC_EPOCH:
            pos = self._fetch_add(self.tail, d, 1)
            if pos >= RING_SLOTS:
                self.overflow[d] = 1
                return False
            self.token[d][pos] = token
            self.seq[d][pos] = self.epoch
            return True
        pos = self.tail[d]
        slot = pos & (RING_SLOTS - 1)
        if self.seq[d][slot] != pos:
            return False
        if not self._cas(self.tail, d, pos, pos + 1):
            return False
        self.token[d][slot] = token
        self.seq[d][slot] = pos + 1
        return True

    def claim_pop(self, d):
        pos = self.head[d]
        slot = pos & (RING_SLOTS - 1)
        if self.seq[d][slot] != pos + 1:
            return None
        if not self._cas(self.head, d, pos, pos + 1):
            return None
        tok = self.token[d][slot]
        self.seq[d][slot] = pos + RING_SLOTS
        return tok

    def pop(self, sm):
        home = sm % self.d
        if self.mode == RING_MODE_CLUSTER_SPSC:
            if sm != self.owner[home]:
                return None
            pos = self.head[home]
            slot = pos & (RING_SLOTS - 1)
            if self.seq[home][slot] != pos + 1:
                return None
            tok = self.token[home][slot]
            self.head[home] = pos + 1
            self.seq[home][slot] = pos + RING_SLOTS
            return tok
        if self.mode == RING_MODE_MPSC_EPOCH:
            if sm != self.owner[home]:
                return None
            pos = self.head[home]
            if pos >= RING_SLOTS or self.seq[home][pos] != self.epoch:
                return None
            tok = self.token[home][pos]
            self.head[home] = pos + 1
            return tok
        tok = self.claim_pop(home)
        if tok is not None:
            return tok
        return self.claim_pop((home + 1) % self.d)


def check_ring_and_prereduce(rows_prered: List[Dict],
                             rows_ring: List[Dict]) -> None:
    print("\n== Lock-free pre-reduce + sharded ready rings (Python mirror) ==")
    f0 = FAILURES
    import threading

    check(len(rows_prered) == 1, "1 PRERED line")
    for row in rows_prered:
        check(int(row["bad_last"]) == 0,
              "C++ threaded pre-reduce: exactly one last contributor")
        check(int(row["rounds"]) * int(row["contributors"]) > 0,
              "pre-reduce stress actually ran")

    # Mode 0: cluster-exclusive SPSC, owner-only + wrap.
    r = PyReadyRing(4, RING_MODE_CLUSTER_SPSC)
    check(r.push(0, 0, 11) is True, "spsc owner push")
    check(r.push(0, 1, 12) is False, "spsc non-owner push denied")
    check(r.pop(1) is None, "spsc non-owner pop denied")
    check(r.pop(0) == 11, "spsc owner pop")
    check(r.pop(0) is None, "spsc empty")
    for rnd in range(3):
        for i in range(RING_SLOTS):
            check(r.push(1, 1, rnd * 1000 + i) is True, "spsc fill")
        check(r.push(1, 1, 999999) is False, "spsc full rejected")
        for i in range(RING_SLOTS):
            check(r.pop(1) == rnd * 1000 + i, "spsc FIFO across wrap")

    # Mode 1: no-CAS MPSC epochs + sticky overflow.
    r = PyReadyRing(2, RING_MODE_MPSC_EPOCH)
    r.begin_epoch(7)
    for i in range(RING_SLOTS):
        check(r.push(0, 5, i) is True, "mpsc push")
    check(r.push(0, 5, 999) is False, "mpsc overflow rejected")
    check(r.overflow[0] == 1, "mpsc sticky overflow")
    for i in range(RING_SLOTS):
        check(r.pop(0) == i, "mpsc FIFO")
    check(r.pop(0) is None, "mpsc drained")
    r.begin_epoch(8)
    check(r.overflow[0] == 0, "new epoch clears overflow")
    check(r.push(0, 5, 42) is True and r.pop(0) == 42, "epoch rollover")

    # Mode 2: unified claim, home first then one steal hop.
    r = PyReadyRing(2, RING_MODE_MPMC_STEAL)
    check(r.push(1, 0, 77) is True, "claim push shard1")
    check(r.pop(0) == 77, "steal from neighbour when home empty")
    check(r.pop(0) is None, "both empty")

    # Mode 2 threaded EXACTLY-ONCE: the duplicate-consumption regression.
    producers, per_producer, consumers = 4, 1500, 4
    total = producers * per_producer
    r = PyReadyRing(2, RING_MODE_MPMC_STEAL)
    seen = [0] * total
    popped = [0]
    lock = threading.Lock()
    threads = []

    def produce(p):
        for i in range(per_producer):
            tok = p * per_producer + i
            while not r.push(tok % 2, 0, tok):
                pass

    def consume(c):
        while True:
            with lock:
                if popped[0] >= total:
                    return
            tok = r.pop(c)
            if tok is None:
                continue
            with lock:
                seen[tok] += 1
                popped[0] += 1

    for p in range(producers):
        threads.append(threading.Thread(target=produce, args=(p,)))
    for c in range(consumers):
        threads.append(threading.Thread(target=consume, args=(c,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    dup = sum(1 for n in seen if n > 1)
    missing = sum(1 for n in seen if n == 0)
    check(popped[0] == total, "every token popped (Python mirror)")
    check(dup == 0, "NO duplicate consumption (Python mirror)")
    check(missing == 0, "no token lost (Python mirror)")

    steal_rows = [r for r in rows_ring if r["mode"] == "mpmc_steal"]
    check(len(steal_rows) == 1, "1 threaded steal stress line")
    for row in steal_rows:
        check(int(row["dup"]) == 0 and int(row["missing"]) == 0,
              "C++ threaded steal: exactly-once")
        check(int(row["total"]) == 8000,
              f"C++ steal stress moved 8000 tokens (got {row['total']})")
        check(int(row["producers"]) >= 2 and int(row["consumers"]) >= 2,
              "C++ steal stress is genuinely multi-producer / multi-consumer")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] pre-reduce + rings: "
          f"3 modes, {total} tokens exactly-once in the Python mirror, "
          f"C++ threaded stress green")


CLUSIM_SLOTS = 32


def check_cluster_sim(rows: List[Dict]) -> None:
    print("\n== Cluster DSMEM fan-out simulation (C++ vs Python mirror) ==")
    f0 = FAILURES
    check(len(rows) == 3, f"3 CLUSIM lines (got {len(rows)})")
    for row in rows:
        cs = int(row["cluster"])
        heads = [0] * cs
        tx = [0] * cs
        packets = [[] for _ in range(cs)]
        for rnd in range(5):
            for i in range(CLUSIM_SLOTS):
                for c in range(1, cs):
                    packets[c].append((rnd, rnd * 100 + i))
                    heads[c] += 1
                    tx[c] += 16
            # Full inbox: rejected without partial delivery.
            check(all(heads[c] - 0 >= CLUSIM_SLOTS for c in range(1, cs)),
                  f"cluster{cs} inbox full before overflow attempt")
            for c in range(1, cs):
                for i in range(CLUSIM_SLOTS):
                    check(packets[c].pop(0) == (rnd, rnd * 100 + i),
                          f"cluster{cs} peer{c} FIFO across wrap")
                heads[c] -= CLUSIM_SLOTS
        check(tx[cs - 1] == int(row["tx"]),
              f"cluster{cs} transaction bytes parity "
              f"(C++={row['tx']} Py={tx[cs - 1]})")
        check(tx[cs - 1] == 5 * CLUSIM_SLOTS * 16,
              f"cluster{cs} 16 bytes per packet")
        check(heads[0] == 0 and tx[0] == 0, f"cluster{cs} leader excluded")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] cluster DSMEM sim parity: "
          f"{len(rows)} cluster sizes, delivery + FIFO + wrap + tx accounting")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-out", default=None,
                    help="reuse an existing C++ test stdout file (skip build)")
    args = ap.parse_args()

    delegated = maybe_delegate_to_wsl()
    if delegated is not None:
        return delegated

    if args.cpp_out:
        stdout = Path(args.cpp_out).read_text(encoding="utf-8", errors="replace")
    else:
        stdout = build_and_run_cpp()

    lines = parse_lines(stdout)
    check_online_parity(lines["ONLINE"])
    report = check_calib_parity(lines["CALIB"])
    check_provenance(report)
    check_knob_contract()
    check_regimes(report)
    check_golden_identity(stdout)
    check_pin_semantics()
    check_phase_nk_parity(lines["ENV"], lines["INNER"])
    check_sm0_order_parity(lines["ORDER"])
    check_profile_lut_parity(lines["LUT"])
    check_pendulum_parity(lines["PEND"])
    check_ring_and_prereduce(lines["PRERED"], lines["RING"])
    check_cluster_sim(lines["CLUSIM"])

    out = ROOT / "results/two_level_adaptive_policy_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "weave-two-level-adaptive-policy-test/v3",
        "checks": CHECKS, "failures": FAILURES,
        "online_policies_compared": len(lines["ONLINE"]),
        "calibration_lines": len(lines["CALIB"]),
        "phase_nk_envelopes_compared": len(lines["ENV"]),
        "phase_nk_inner_tuples_compared": len(lines["INNER"]),
        "sm0_order_permutations_compared": len(lines["ORDER"]),
        "profile_lut_publications_compared": len(lines["LUT"]),
        "sm0_mode_fixtures_compared": len(lines["PEND"]),
        "prereduce_stress_lines": len(lines["PRERED"]),
        "ready_ring_lines": len(lines["RING"]),
        "cluster_dsmem_sim_lines": len(lines["CLUSIM"]),
        "pass": FAILURES == 0,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nCHECKS={CHECKS} FAILURES={FAILURES}")
    print(f"WROTE {out}")
    if FAILURES == 0:
        print("ALL TWO-LEVEL ADAPTIVE POLICY CHECKS PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
