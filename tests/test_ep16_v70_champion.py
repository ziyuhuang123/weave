#!/usr/bin/env python3
"""CPU-only gate for the V70 Path-B (160x256, 32 KiB dynamic LDS) EP16 champion & all 17 active MI308X knobs."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMD = ROOT / "deep_gemm" / "include" / "deep_gemm" / "impls" / "amd"
V70 = AMD / "ep16" / "v70"
BUILD_V70 = V70 / "build_v70.py"
BUILD_PORT = AMD / "ep16" / "build_port.py"
PRESET = V70 / "v70_champion_preset.json"
KNOBS_PY = ROOT / "deep_gemm" / "mega" / "knobs.py"

ACTIVE_17_KNOBS = [
    "WEAVE_CU_WAVE_SPLIT",
    "WEAVE_DISPATCH_ACTIVE_SMS",
    "WEAVE_SCATTER_ACTIVE_SMS",
    "WEAVE_SM0_STREAM_SCHED",
    "WEAVE_DISPATCH_WARPS",
    "WEAVE_SCATTER_WARPS",
    "WEAVE_REDUCE_WARPS",
    "WEAVE_V_HEAD",
    "WEAVE_READY_GROUP_SIZE",
    "WEAVE_SCATTER_OWNER",
    "WEAVE_COMBINE_SYNC_CHUNK",
    "WEAVE_ADAPT_BM",
    "WEAVE_BLOCK_M_L1",
    "WEAVE_BLOCK_M_L2",
    "WEAVE_REG_PULL_BYTES",
    "WEAVE_DEDUP_MODE",
    "WEAVE_SCATTER_STAGING",
]

CHECKS = 0
FAILURES = 0


def check(ok: bool, label: str) -> None:
    global CHECKS, FAILURES
    CHECKS += 1
    if not ok:
        FAILURES += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_env(extra=None) -> dict:
    pub = {k.name for k in load_module(KNOBS_PY, "weave_knob_schema_v70").KNOBS}
    env = {k: v for k, v in os.environ.items() if k not in pub and k != "WEAVE_PROFILER_LEVEL"}
    env.update(extra or {})
    return env


def dry_run(script: Path, args, extra_env=None):
    p = subprocess.run([sys.executable, str(script), *args], env=clean_env(extra_env),
                       capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    obj = None
    start = p.stdout.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(p.stdout, start)
        except json.JSONDecodeError:
            obj = None
    return p.returncode, obj, p.stdout + p.stderr


def main() -> int:
    preset = json.loads(PRESET.read_text(encoding="utf-8"))
    b = load_module(BUILD_V70, "weave_build_v70")

    print("\n== V70 preset & source manifest ==")
    check(preset.get("schema") == "weave-ep16-v70-champion-preset/v1", "preset schema")
    check(preset.get("arm") == "v70_champion", "arm = v70_champion")
    errs = b.verify_manifest(preset)
    check(not errs, f"live include closure == manifest ({len(preset['source_manifest'])} files)"
          + ("" if not errs else f": {errs[:4]}"))
    closure, external = b.include_closure()
    check(not any(rel.startswith("ep8/") for rel in closure), "V70 closure never reads EP8 sources")
    check(all(name.startswith("ck/") or name.endswith(".cuh") for name in external),
          f"external includes are aiter/CK only ({len(external)})")

    print("\n== V70 build_port.py --champion v70_champion delegation & FI0/FI2 ==")
    rc0, fi0, out0 = dry_run(BUILD_PORT, ["--champion", "v70_champion", "--dry-run"])
    check(rc0 == 0 and fi0 is not None and fi0["fi"] == 0 and "BUILD_V70_DRY_RUN_DONE" in out0,
          f"build_port.py --champion v70_champion --dry-run FI0 exits 0 (rc={rc0})")
    check(fi0 is not None and fi0["launch"] == {"grid": 160, "block": 256, "dynamic_lds_bytes": 32768},
          "Path B launch geometry 160x256 + 32 KiB dynamic LDS")
    rc2, fi2, out2 = dry_run(BUILD_PORT, ["--champion", "v70_champion", "--dry-run", "--fi", "2"])
    check(rc2 == 0 and fi2 is not None and fi2["fi"] == 2 and fi2["defines"]["FI_LEVEL"] == "2",
          f"build_port.py --champion v70_champion --fi 2 resolves FI_LEVEL=2 (rc={rc2})")

    print("\n== V70 all 3 CU modes & all 17 active MI308X knobs ==")
    check(all(k in fi0["defines"] for k in ACTIVE_17_KNOBS), "all 17 active WEAVE_* knobs present in defines")
    for mode_name, code in [("0-8", "1"), ("4-4", "2"), ("8-0", "3"), ("auto", "2")]:
        rcm, objm, _ = dry_run(BUILD_V70, ["--dry-run"], {"WEAVE_CU_WAVE_SPLIT": mode_name})
        check(rcm == 0 and objm is not None and objm["defines"]["WEAVE_CU_WAVE_SPLIT"] == code,
              f"CU mode {mode_name} -> WEAVE_CU_WAVE_SPLIT={code}")
    rc_k, obj_k, _ = dry_run(BUILD_V70, ["--dry-run"], {
        "WEAVE_DISPATCH_ACTIVE_SMS": "2", "WEAVE_SCATTER_ACTIVE_SMS": "2",
        "WEAVE_SM0_STREAM_SCHED": "0", "WEAVE_DISPATCH_WARPS": "4",
        "WEAVE_SCATTER_WARPS": "8", "WEAVE_REDUCE_WARPS": "4",
        "WEAVE_V_HEAD": "0", "WEAVE_READY_GROUP_SIZE": "1",
        "WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_COMBINE_SYNC_CHUNK": "512",
        "WEAVE_ADAPT_BM": "0", "WEAVE_BLOCK_M_L1": "32", "WEAVE_BLOCK_M_L2": "64",
        "WEAVE_REG_PULL_BYTES": "1024", "WEAVE_DEDUP_MODE": "0",
        "WEAVE_SCATTER_STAGING": "smem",
    })
    check(rc_k == 0 and obj_k is not None
          and obj_k["defines"]["WEAVE_DISPATCH_ACTIVE_SMS"] == "2"
          and obj_k["defines"]["WEAVE_SCATTER_ACTIVE_SMS"] == "2"
          and obj_k["defines"]["WEAVE_SM0_STREAM_SCHED"] == "0"
          and obj_k["defines"]["WEAVE_DISPATCH_WARPS"] == "4"
          and obj_k["defines"]["WEAVE_SCATTER_WARPS"] == "8"
          and obj_k["defines"]["WEAVE_REDUCE_WARPS"] == "4"
          and obj_k["defines"]["WEAVE_V_HEAD"] == "0"
          and obj_k["defines"]["WEAVE_READY_GROUP_SIZE"] == "1"
          and obj_k["defines"]["WEAVE_SCATTER_OWNER"] == "1"
          and obj_k["defines"]["WEAVE_COMBINE_SYNC_CHUNK"] == "512"
          and obj_k["defines"]["WEAVE_ADAPT_BM"] == "0"
          and obj_k["defines"]["WEAVE_BLOCK_M_L1"] == "32"
          and obj_k["defines"]["WEAVE_BLOCK_M_L2"] == "64"
          and obj_k["defines"]["WEAVE_REG_PULL_BYTES"] == "1024"
          and obj_k["defines"]["WEAVE_DEDUP_MODE"] == "0"
          and obj_k["defines"]["WEAVE_SCATTER_STAGING"] == "1",
          "all 17 active knobs override cleanly on compile line")

    for name, val in [("WEAVE_CU_WAVE_SPLIT", "2-6"), ("WEAVE_DEDUP_MODE", "2"),
                      ("WEAVE_SCATTER_STAGING", "bad"), ("WEAVE_REG_PULL_BYTES", "512"),
                      ("WEAVE_PRECISION_MODE", "bf16")]:
        rcb, _, errb = dry_run(BUILD_V70, ["--dry-run"], {name: val})
        check(rcb != 0 and name in errb, f"invalid {name}={val} fails closed (rc={rcb})")

    status = "PASS" if FAILURES == 0 else "FAIL"
    print(f"\n[{status}] EP16 V70 champion gate: {CHECKS - FAILURES}/{CHECKS} checks passed")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
