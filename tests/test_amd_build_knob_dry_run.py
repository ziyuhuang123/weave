#!/usr/bin/env python3
"""CPU-only gate for the MI308X build-script knob parsing (EP8 + EP16).

Both AMD build entry points resolve the public WEAVE_* knobs into -D macros
before any ROCm toolchain runs:

  * deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh  (WEAVE_DRY_RUN=1)
  * deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py (--dry-run)

This test drives both dry runs and checks the merged contract:
  - strict fail-fast parsing: an invalid public value exits non-zero and names
    the offending knob (no ROCm toolchain, no build directory is touched);
  - WEAVE_ADAPT_BM / WEAVE_DEDUP_MODE are the only public spellings: the
    legacy env aliases WEAVE_BM_TIER_MODE / WEAVE_SELECTIVE_DEDUP are ignored;
  - every knob-derived macro the compile line receives is reported, including
    the owner / local-dedup _EXPLICIT flags, WEAVE_COMBINE_SYNC_CHUNK,
    WEAVE_REDUCE_WARPS and WEAVE_V_HEAD.

The shell half needs a bash (Git Bash or WSL); without one it is reported as
SKIP and only the Python half gates.  Exit status 0 only when every executed
check passes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
AMD = ROOT / "deep_gemm" / "include" / "deep_gemm" / "impls" / "amd"
EP8_SH = AMD / "ep8" / "build_one.sh"
EP16_PY = AMD / "ep16" / "build_port.py"

# Every public / legacy name this test sets; cleared from the inherited env so
# a developer shell cannot change the expected defaults.
KNOB_ENV = [
    "WEAVE_SCATTER_OWNER", "WEAVE_SCATTER_MODE", "WEAVE_ADAPT_BM",
    "WEAVE_BM_TIER_MODE", "WEAVE_DEDUP_MODE", "WEAVE_SELECTIVE_DEDUP",
    "WEAVE_CU_WAVE_SPLIT", "WEAVE_SCATTER_ACTIVE_SMS", "WEAVE_DISPATCH_ACTIVE_SMS",
    "WEAVE_BLOCK_M_L1", "WEAVE_BLOCK_M_L2", "WEAVE_V_HEAD", "WEAVE_PRECISION_MODE",
    "WEAVE_SM0_STREAM_SCHED", "WEAVE_LOCAL_SCATTER_DEDUP", "WEAVE_COMBINE_SYNC_CHUNK",
    "WEAVE_REDUCE_WARPS", "WEAVE_SCATTER_STAGING", "WEAVE_DISPATCH_WARPS",
    "WEAVE_SCATTER_WARPS", "WEAVE_READY_GROUP_SIZE", "WEAVE_REG_PULL_BYTES",
    "WEAVE_ABLATION_PRESET", "WEAVE_DRY_RUN",
]

CHECKS = 0
FAILURES = 0


def check(ok: bool, label: str) -> None:
    global CHECKS, FAILURES
    CHECKS += 1
    if not ok:
        FAILURES += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def clean_env(extra: Dict[str, str]) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in KNOB_ENV}
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# EP8 build_one.sh (bash)
# ---------------------------------------------------------------------------
def find_bash() -> Optional[List[str]]:
    """Return a bash invocation that reads a script from stdin, or None."""
    candidates: List[List[str]] = []
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        if git_bash.exists():
            candidates.append([str(git_bash)])
        if shutil.which("wsl"):
            candidates.append(["wsl", "-e", "bash"])
    else:
        path = shutil.which("bash")
        if path:
            candidates.append([path])
    for cand in candidates:
        try:
            probe = subprocess.run(cand + ["-c", "echo ok"], capture_output=True,
                                   text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return cand
    return None


def run_ep8(bash: List[str], env_kv: Dict[str, str]) -> Tuple[int, Dict[str, str], str]:
    # The checkout may carry CRLF line endings (core.autocrlf); bash needs LF.
    script = EP8_SH.read_text(encoding="utf-8").replace("\r\n", "\n")
    # Environment is passed inline so it also crosses the WSL boundary.
    assigns = " ".join(f"{k}={v}" for k, v in {"WEAVE_DRY_RUN": "1", **env_kv}.items())
    unset = " ".join(f"-u {k}" for k in KNOB_ENV)
    cmd = bash + ["-c", f"env {unset} {assigns} bash -s -- dry_run_test 0"]
    proc = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=120)
    defines: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = re.fullmatch(r"-D([A-Za-z0-9_]+)=(\S*)", line.strip())
        if m:
            defines[m.group(1)] = m.group(2)
    return proc.returncode, defines, proc.stderr


def ep8_suite(bash: List[str]) -> None:
    print(f"\n== EP8 build_one.sh dry run ({' '.join(bash)}) ==")
    rc, d, err = run_ep8(bash, {})
    check(rc == 0, f"defaults exit 0 (rc={rc}, stderr={err.strip()[:120]!r})")
    check(d.get("WEAVE_SELECTIVE_DEDUP") == "1", "default WEAVE_DEDUP_MODE -> SELECTIVE_DEDUP=1")
    check(d.get("WEAVE_BM_TIER_MODE") == "2", "default WEAVE_ADAPT_BM -> BM_TIER_MODE=2")
    check(d.get("WEAVE_SCATTER_OWNER_EXPLICIT") == "0" and "WEAVE_SCATTER_OWNER" not in d,
          "unset owner -> only WEAVE_SCATTER_OWNER_EXPLICIT=0")
    check(d.get("WEAVE_LOCAL_SCATTER_DEDUP_EXPLICIT") == "0", "unset local dedup -> _EXPLICIT=0")
    check(d.get("WEAVE_REDUCE_WARPS") == "0u" and d.get("WEAVE_COMBINE_SYNC_CHUNK") == "0u",
          "default REDUCE_WARPS / COMBINE_SYNC_CHUNK = 0u")
    check("WEAVE_V_HEAD" not in d, "auto WEAVE_V_HEAD is not passed")

    rc, d, err = run_ep8(bash, {
        "WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_REDUCE_WARPS": "4",
        "WEAVE_COMBINE_SYNC_CHUNK": "32", "WEAVE_LOCAL_SCATTER_DEDUP": "1",
        "WEAVE_V_HEAD": "3", "WEAVE_ADAPT_BM": "0", "WEAVE_DEDUP_MODE": "0",
        "WEAVE_CU_WAVE_SPLIT": "8-0", "WEAVE_BLOCK_M_L1": "64",
        "WEAVE_SCATTER_STAGING": "smem", "WEAVE_DISPATCH_WARPS": "8",
        "WEAVE_SCATTER_WARPS": "4", "WEAVE_READY_GROUP_SIZE": "4",
        "WEAVE_REG_PULL_BYTES": "2048"})
    check(rc == 0, f"explicit knobs exit 0 (rc={rc})")
    expect = {"WEAVE_SCATTER_OWNER": "1u", "WEAVE_SCATTER_OWNER_EXPLICIT": "1",
              "WEAVE_SCATTER_MODE": "2", "WEAVE_REDUCE_WARPS": "4u",
              "WEAVE_COMBINE_SYNC_CHUNK": "32u", "WEAVE_LOCAL_SCATTER_DEDUP": "1",
              "WEAVE_LOCAL_SCATTER_DEDUP_EXPLICIT": "1", "WEAVE_V_HEAD": "3u",
              "WEAVE_BM_TIER_MODE": "0", "WEAVE_SELECTIVE_DEDUP": "0",
              "WEAVE_CU_WAVE_SPLIT": "3", "WEAVE_BLOCK_M_L1": "64u",
              "WEAVE_SCATTER_STAGING": "1u", "WEAVE_DISPATCH_WARPS": "8u",
              "WEAVE_SCATTER_WARPS": "4u", "WEAVE_READY_GROUP_SIZE": "4u",
              "WEAVE_REG_PULL_BYTES": "2048u"}
    for key, value in expect.items():
        check(d.get(key) == value, f"explicit {key}={value} (got {d.get(key)})")

    rc, d, _ = run_ep8(bash, {"WEAVE_SELECTIVE_DEDUP": "0", "WEAVE_BM_TIER_MODE": "0"})
    check(rc == 0 and d.get("WEAVE_SELECTIVE_DEDUP") == "1" and d.get("WEAVE_BM_TIER_MODE") == "2",
          "legacy WEAVE_SELECTIVE_DEDUP / WEAVE_BM_TIER_MODE env aliases are ignored")

    for name, value in [("WEAVE_DEDUP_MODE", "2"), ("WEAVE_ADAPT_BM", "yes"),
                        ("WEAVE_SCATTER_OWNER", "mixed"), ("WEAVE_REDUCE_WARPS", "3"),
                        ("WEAVE_COMBINE_SYNC_CHUNK", "24"), ("WEAVE_BLOCK_M_L2", "48"),
                        ("WEAVE_SCATTER_ACTIVE_SMS", "81"), ("WEAVE_V_HEAD", "-1"),
                        ("WEAVE_PRECISION_MODE", "bf16"), ("WEAVE_CU_WAVE_SPLIT", "2-6"),
                        ("WEAVE_SCATTER_STAGING", "bad"), ("WEAVE_DISPATCH_WARPS", "6"),
                        ("WEAVE_SCATTER_WARPS", "2"), ("WEAVE_READY_GROUP_SIZE", "3"),
                        ("WEAVE_REG_PULL_BYTES", "512")]:
        rc, _, err = run_ep8(bash, {name: value})
        check(rc != 0 and name in err, f"invalid {name}={value} fails closed (rc={rc})")


# ---------------------------------------------------------------------------
# EP16 build_port.py (python)
# ---------------------------------------------------------------------------
def run_ep16(env_kv: Dict[str, str]) -> Tuple[int, List[dict], str]:
    proc = subprocess.run([sys.executable, str(EP16_PY), "--dry-run"],
                          env=clean_env(env_kv), capture_output=True, text=True,
                          timeout=300, cwd=str(ROOT))
    variants: List[dict] = []
    decoder = json.JSONDecoder()
    text = proc.stdout
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and "defines" in obj:
            variants.append(obj)
        pos = end
    return proc.returncode, variants, proc.stderr + proc.stdout[-400:]


def ep16_suite() -> None:
    print("\n== EP16 build_port.py --dry-run ==")
    rc, variants, err = run_ep16({})
    check(rc == 0 and variants, f"defaults exit 0 with >=1 variant (rc={rc}, n={len(variants)})")
    for v in variants:
        d = v["defines"]
        check(d.get("WEAVE_SELECTIVE_DEDUP") == "1" and d.get("WEAVE_BM_TIER_MODE") == "2",
              f"{v['variant']}: default SELECTIVE_DEDUP=1, BM_TIER_MODE=2")

    rc, variants, _ = run_ep16({"WEAVE_SELECTIVE_DEDUP": "0", "WEAVE_BM_TIER_MODE": "0"})
    check(rc == 0 and variants and all(
        v["defines"].get("WEAVE_SELECTIVE_DEDUP") == "1" and
        v["defines"].get("WEAVE_BM_TIER_MODE") == "2" for v in variants),
        "legacy WEAVE_SELECTIVE_DEDUP / WEAVE_BM_TIER_MODE env aliases are ignored")

    rc, variants, _ = run_ep16({"WEAVE_ADAPT_BM": "0", "WEAVE_DEDUP_MODE": "0",
                                "WEAVE_SCATTER_STAGING": "hbm_dedup",
                                "WEAVE_DISPATCH_WARPS": "4", "WEAVE_SCATTER_WARPS": "8",
                                "WEAVE_READY_GROUP_SIZE": "8", "WEAVE_REG_PULL_BYTES": "4096"})
    check(rc == 0 and variants and all(
        v["defines"].get("WEAVE_SELECTIVE_DEDUP") == "0" and
        v["defines"].get("WEAVE_BM_TIER_MODE") == "0" and
        v["defines"].get("WEAVE_SCATTER_STAGING") == "2u" and
        v["defines"].get("WEAVE_LOCAL_SCATTER_DEDUP") == "1" and
        v["defines"].get("WEAVE_SCATTER_OWNER") == "1u" and
        v["defines"].get("WEAVE_DISPATCH_WARPS") == "4u" and
        v["defines"].get("WEAVE_SCATTER_WARPS") == "8u" and
        v["defines"].get("WEAVE_READY_GROUP_SIZE") == "8u" and
        v["defines"].get("WEAVE_REG_PULL_BYTES") == "4096u" for v in variants),
        "WEAVE_ADAPT_BM / DEDUP / SCATTER_STAGING / WARPS / READY_GROUP / REG_PULL reach internal macros")

    for name, value in [("WEAVE_DEDUP_MODE", "2"), ("WEAVE_ADAPT_BM", "yes"),
                        ("WEAVE_SCATTER_OWNER", "mixed"), ("WEAVE_BLOCK_M_L1", "48"),
                        ("WEAVE_DISPATCH_ACTIVE_SMS", "81"), ("WEAVE_PRECISION_MODE", "bf16"),
                        ("WEAVE_SCATTER_STAGING", "invalid"), ("WEAVE_DISPATCH_WARPS", "2"),
                        ("WEAVE_SCATTER_WARPS", "6"), ("WEAVE_READY_GROUP_SIZE", "32"),
                        ("WEAVE_REG_PULL_BYTES", "512")]:
        rc, _, err = run_ep16({name: value})
        check(rc != 0 and name in err, f"invalid {name}={value} fails closed (rc={rc})")


def ep16_v62_champion_suite() -> None:
    """build_port.py --champion v62_champion: the vendored V62 2-Node 16xMI308X
    champion TU (ep16/v62/) builds from its recorded -D macros only, so public
    WEAVE_* knobs -- even invalid ones -- must not reach or break it."""
    print("\n== EP16 build_port.py --champion v62_champion --dry-run ==")
    preset = json.loads((AMD / "ep16" / "v62" / "v62_champion_preset.json").read_text(encoding="utf-8"))
    exact = dict(t[2:].split("=", 1) for t in preset["exact_v62_extra_flags"].split())

    def champ(env_kv: Dict[str, str]) -> Tuple[int, Optional[dict]]:
        proc = subprocess.run([sys.executable, str(EP16_PY), "--champion", "v62_champion", "--dry-run"],
                              env=clean_env(env_kv), capture_output=True, text=True,
                              timeout=300, cwd=str(ROOT))
        start = proc.stdout.find("{")
        obj = json.JSONDecoder().raw_decode(proc.stdout, start)[0] if start >= 0 else None
        return proc.returncode, obj

    rc, obj = champ({})
    check(rc == 0 and obj is not None and obj.get("variant") == "v62_champion",
          f"champion dry run exits 0 with variant v62_champion (rc={rc})")
    check(obj is not None and all(obj["defines"].get(k) == v for k, v in exact.items()),
          f"all {len(exact)} exact V62 macros reach the compile line")
    check(obj is not None and obj["include_closure"]["manifest_ok"] and
          obj.get("expected_co_sha256") == preset["expected_co"]["sha256"],
          "sha256 source manifest OK; expected P.co sha256 reported")
    rc2, obj2 = champ({"WEAVE_BLOCK_M_L1": "48", "WEAVE_DEDUP_MODE": "2"})
    check(rc2 == 0 and obj2 is not None and obj is not None and obj2["defines"] == obj["defines"],
          "invalid public WEAVE_* values do not touch the champion compile line")


def main() -> int:
    for path in (EP8_SH, EP16_PY):
        if not path.exists():
            print(f"[FAIL] missing {path}")
            return 1
    ep16_suite()
    ep16_v62_champion_suite()
    bash = find_bash()
    if bash is None:
        print("\n[SKIP] EP8 build_one.sh dry run: no bash (Git Bash / WSL) found")
    else:
        ep8_suite(bash)
    status = "PASS" if FAILURES == 0 else "FAIL"
    print(f"\n[{status}] AMD build knob dry run: {CHECKS - FAILURES}/{CHECKS} checks passed")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
