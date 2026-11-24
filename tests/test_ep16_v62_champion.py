#!/usr/bin/env python3
"""CPU-only gate for the vendored V62 2-Node 16xMI308X EP16 champion (arm v62_champion).

Checks, without any ROCm toolchain or GPU:
  * preset provenance: Nebula job 01a204b684c8479453de03934774988b, production
    P.co sha256 e915ce1f..., 375960 bytes, 0-spill / 1-s_barrier ISA gate;
  * source integrity: the live quoted-include closure of ep16/v62/tree/megamoe_amd.hip
    equals the preset sha256 manifest file-for-file (LF-normalised), the V62 TU
    resolves its own pre-v59 sub6_6models_generalize.hpp, and
    common/weave_knobs.hpp carries the build-host verified sha256 3612153d...;
  * honest scope wording: source SHA-verified + local dry run only, ROCm rebuild
    SHA check still pending, no absolute build-host paths in the preset;
  * exact flags: every -D macro of the recorded V62 command line is consumed by
    the closure, except the three documented harness-only macros;
  * knob isolation: build_v62.py reads only the pre-existing toolchain/path
    overrides and zero public WEAVE_* tunables; public knobs (even invalid
    values) do not change the champion compile line;
  * build_port.py --champion v62_champion --dry-run delegates and the default
    policy-wired port dry run is unchanged;
  * evidence ledger results/ep16_mi308x_v62_16gpu_champion.json: 6 cells x 5
    arms, 30/30 16-rank FP64 PASS, v62_champion wins every cell, geomean
    recomputed from the per-cell ratios, champion co sha == preset sha, no
    infrastructure identifiers.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMD = ROOT / "deep_gemm" / "include" / "deep_gemm" / "impls" / "amd"
V62 = AMD / "ep16" / "v62"
BUILD_V62 = V62 / "build_v62.py"
BUILD_PORT = AMD / "ep16" / "build_port.py"
PRESET = V62 / "v62_champion_preset.json"
LEDGER = ROOT / "results" / "ep16_mi308x_v62_16gpu_champion.json"
KNOBS_PY = ROOT / "deep_gemm" / "mega" / "knobs.py"

JOB = "01a204b684c8479453de03934774988b"
CO_SHA = "e915ce1f51f53cbba0b4739b1e6497f578a5126590a2f0e4bd238fd59386e8b2"
# sha256sum R/common/weave_knobs.hpp on the V62 build host (2026-11-22).
KNOBS_HPP_SHA = "3612153d33ba3531fa1eb4dd510f81ce1c604c485937f1b6553d803677a16e5b"
SCOPE_PROVENANCE = ("Source files SHA-verified against Nebula job 01a204b684c8479453de03934774988b "
                    "(local build_port.py --champion v62_champion dry-run verified; ROCm rebuild SHA "
                    "check pending on MI308X host)")
TOOL_ENV = {"WEAVE_CLANG", "WEAVE_LLD", "WEAVE_PY", "WEAVE_SNAP_DIR", "WEAVE_V3225_DIR",
            "WEAVE_AITER_SITE", "WEAVE_PY_SITE"}
INERT = {"V61_N08", "V61_N44", "V61_COMM"}

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


def public_knobs() -> set:
    return {k.name for k in load_module(KNOBS_PY, "weave_knob_schema_v62").KNOBS}


def surface_names() -> set:
    """Every WEAVE_* spelling the champion build must ignore: the 14 public knobs,
    their two legacy aliases, the two accepted-but-not-public compat names
    (WEAVE_ADAPT_GEMM_CONFIG / WEAVE_ADAPT_BM) and the six retired (internal) knobs."""
    mod = load_module(KNOBS_PY, "weave_knob_schema_v62")
    return ({k.name for k in mod.KNOBS} | set(mod.LEGACY_ALIASES) | set(mod.RETIRED_INTERNAL)
            | set(mod.INTERNAL_COMPAT_ENV))


def clean_env(extra=None) -> dict:
    surface = surface_names()
    env = {k: v for k, v in os.environ.items() if k not in surface and k not in TOOL_ENV}
    env.update(extra or {})
    return env


def dry_run(script: Path, args, extra_env=None):
    p = subprocess.run([sys.executable, str(script), *args], env=clean_env(extra_env),
                       capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    out = p.stdout
    obj = None
    start = out.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(out, start)
        except json.JSONDecodeError:
            obj = None
    return p.returncode, obj, out + p.stderr


def preset_suite(preset: dict) -> None:
    print("\n== preset provenance ==")
    check(preset.get("schema") == "weave-ep16-v62-champion-preset/v1", "preset schema")
    check(preset.get("arm") == "v62_champion", "arm = v62_champion")
    check(preset["provenance"].get("benchmark_nebula_job") == JOB, f"benchmark Nebula job {JOB}")
    co = preset["expected_co"]
    check(co.get("sha256") == CO_SHA and co.get("bytes") == 375960 and co.get("fi_level") == 0,
          "expected production P.co sha256 e915ce1f... / 375960 B / FI_LEVEL=0")
    g = co.get("static_isa_gate", {})
    check(g.get("vgpr_spill_count_max") == 0 and g.get("s_barrier") == 1 and
          g.get("private_segment_fixed_size_max") == 0 and g.get("group_segment_fixed_size_max") <= 65536
          and g.get("vgpr_count_max") <= 256,
          f"recorded ISA gate: {g.get('vgpr_count_max')} VGPR, 0 spill, 0 scratch, 1 s_barrier, <=64 KiB LDS")
    for p in (BUILD_V62, BUILD_PORT):
        txt = p.read_text(encoding="utf-8")
        check(JOB in txt and "e915ce1f" in txt, f"{p.name}: provenance comment cites job + sha256")
    scope = preset.get("scope", "")
    check(SCOPE_PROVENANCE in scope and "silicon-verified" not in scope.lower(),
          "scope states source-SHA + dry-run verification only (ROCm rebuild SHA check pending)")
    raw = PRESET.read_text(encoding="utf-8")
    check("/home/" not in raw and not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|pangu://|e01-cn-", raw),
          "no absolute build-host paths / IPs / node names in the preset")


def source_suite(preset: dict, b) -> None:
    print("\n== source integrity (sha256 manifest vs live include closure) ==")
    errs = b.verify_manifest(preset)
    check(not errs, f"live closure == manifest ({len(preset['source_manifest'])} files)"
          + ("" if not errs else f": {errs[:4]}"))
    closure, external = b.include_closure()
    man = preset["source_manifest"]
    check(all(rel.startswith("ep16/v62/tree/") for rel, v in man.items()
              if v["origin"] == "build_host_v62_pull"), "build-host V62 files live under ep16/v62/tree/")
    check(sum(v["origin"] == "build_host_v62_pull" for v in man.values()) == 25,
          "25 verbatim build-host V62 files (megamoe_amd.hip + 24 headers)")
    shared = sorted(rel for rel, v in man.items() if v["origin"] == "repo_shared_identical")
    check(shared == ["include/gda_mesh.hpp", "include/v1262_subwave_barrier.hpp",
                     "include/weave_ck_512_configs.hpp"],
          "exactly 3 shared byte-identical amd/include headers")
    knobs = man.get("ep16/v62/common/weave_knobs.hpp", {})
    ver = knobs.get("verified_against", {})
    check(knobs.get("origin") == "build_host_sha256_verified" and knobs.get("sha256_lf") == KNOBS_HPP_SHA
          and ver.get("sha256") == KNOBS_HPP_SHA and ver.get("host") == "v62-build-host"
          and ver.get("path") == "R/common/weave_knobs.hpp",
          "common/weave_knobs.hpp sha256 3612153d... verified against the V62 build host R/common/weave_knobs.hpp")
    check(not any(v["origin"] == "candidate_unverified" for v in man.values()),
          "no header left in candidate_unverified state")
    check("ep16/v62/tree/include/sub6_6models_generalize.hpp" in closure and
          "include/sub6_6models_generalize.hpp" not in closure,
          "V62 TU resolves its own pre-v59 sub6 header (v59 shared-experts nibble stays on the port)")
    check(not any(rel.startswith("ep8/") for rel in closure), "V62 closure never reads EP8 sources")
    check(all(name.startswith("ck/") or name.endswith(".cuh") for name in external),
          f"external includes are aiter/CK only ({len(external)})")
    lines = [ln for p in closure.values() for ln in p.read_bytes().split(b"\n")]
    check(not any(b"\r" in ln for ln in lines) or os.name == "nt",
          "LF sources (CRLF tolerated only on a Windows autocrlf checkout)")


def flags_suite(preset: dict, b) -> None:
    print("\n== exact V62 flags ==")
    toks = preset["exact_v62_extra_flags"].split()
    check(len(toks) == 36 and all(t.startswith("-D") and "=" in t for t in toks),
          f"36 -DNAME=VALUE tokens (got {len(toks)})")
    names = [t[2:].split("=", 1)[0] for t in toks]
    check(len(set(names)) == len(names), "no duplicated macro")
    closure, _ = b.include_closure()
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in closure.values())
    unused = {n for n in names if not re.search(r"\b" + re.escape(n) + r"\b", text)}
    check(unused == INERT, f"every flag macro is consumed except the 3 harness-only ones (unused={sorted(unused)})")
    check(set(preset["inert_in_production_tu"]["macros"]) == INERT, "inert macros documented in the preset")
    cmd = b.clang_cmd(Path("OUT"), preset)
    d = b.defines_of(cmd)
    check(all(d.get(n) == t.split("=", 1)[1] for n, t in zip(names, toks)),
          "compile line carries every exact flag verbatim")
    check(d.get("FI_LEVEL") == "0" and d.get("WEAVE_KNOB_BM_TIER_MODE") == "2" and
          d.get("WEAVE_KNOB_SELECTIVE_DEDUP") == "1" and d.get("WEAVE_SCATTER_MODE") == "2",
          "recorded V62 base flags (FI_LEVEL=0, BM_TIER 2, SELECTIVE_DEDUP 1, SCATTER_MODE 2)")
    check(cmd.index(str(b.SRC)) > cmd.index(toks[-1]), "extra flags precede the source like the recorded V62 builder")


def isolation_suite(preset: dict) -> None:
    print("\n== knob isolation (14 public WEAVE_* knobs, legacy aliases, compat names and retired pins untouched) ==")
    pub = public_knobs()
    surface = surface_names()
    check(len(pub) == 14, "14 public knobs in the schema")
    check(len(surface) == 14 + 2 + 2 + 6, "14 public + 2 legacy alias + 2 compat + 6 retired internal spellings")
    src = BUILD_V62.read_text(encoding="utf-8")
    env_reads = set(re.findall(r'os\.environ(?:\.get)?\(\s*"([A-Z0-9_]+)"', src))
    env_reads |= set(re.findall(r'os\.environ\[\s*"([A-Z0-9_]+)"\s*\]', src)) - {
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE"}
    check(env_reads <= TOOL_ENV, f"build_v62.py reads only toolchain/path overrides ({sorted(env_reads)})")
    check(not (env_reads & surface), "build_v62.py reads zero public / legacy / retired WEAVE_* tunables")
    port_src = BUILD_PORT.read_text(encoding="utf-8")
    port_reads = set(re.findall(r'"(WEAVE_[A-Z0-9_]+)"', port_src))
    check("WEAVE_CHAMPION" not in port_src and not any("V62" in n for n in port_reads),
          "build_port.py gained a CLI flag, not an env knob")

    rc0, base, _ = dry_run(BUILD_V62, ["--dry-run"])
    check(rc0 == 0 and base is not None and base["include_closure"]["manifest_ok"],
          f"build_v62.py --dry-run exits 0 with manifest_ok (rc={rc0})")
    rc1, bad, _ = dry_run(BUILD_V62, ["--dry-run"], {"WEAVE_BLOCK_M_L1": "48", "WEAVE_ADAPT_BM": "0",
                                                      "WEAVE_CU_WAVE_SPLIT": "3"})
    check(rc1 == 0 and bad is not None and base is not None and bad["defines"] == base["defines"],
          "public knobs and legacy aliases (incl. invalid values) do not change the champion compile line")
    rc2, dele, out2 = dry_run(BUILD_PORT, ["--champion", "v62_champion", "--dry-run"])
    check(rc2 == 0 and dele is not None and base is not None and dele["defines"] == base["defines"]
          and "BUILD_V62_DRY_RUN_DONE" in out2,
          "build_port.py --champion v62_champion --dry-run delegates to build_v62.py")
    rc3, _, _ = dry_run(BUILD_PORT, ["--champion", "v99", "--dry-run"])
    check(rc3 != 0, f"unknown --champion value is rejected (rc={rc3})")
    rc4, port, out4 = dry_run(BUILD_PORT, ["--dry-run"])
    check(rc4 == 0 and port is not None and port.get("variant") != "v62_champion" and
          "BUILD_DRY_RUN_DONE" in out4 and "WEAVE_BLOCK_M_L1" in port.get("defines", {}),
          "default policy-wired port dry run unchanged")


def ledger_suite(preset: dict) -> None:
    print("\n== evidence ledger results/ep16_mi308x_v62_16gpu_champion.json ==")
    raw = LEDGER.read_text(encoding="utf-8")
    led = json.loads(raw)
    check(led.get("schema") == "ep16-mi308x-v62-16gpu-champion/v1", "ledger schema")
    check(led["provenance"].get("benchmark_nebula_job") == JOB, "ledger cites the benchmark job")
    models = led["models"]
    check(len(models) == 6, "6 model cells")
    arms = {a["arm"]: a for a in led["arms"]}
    check(len(arms) == 5 and arms.get("v62_champion", {}).get("co_sha256") == preset["expected_co"]["sha256"],
          "5 arms; champion co sha256 == preset sha256")
    fp = [a["fp64"] for m in models.values() for a in m["arms"].values()]
    check(len(fp) == 30 and all(f["pass"] and f["world_size"] == 16 and f["rank_set_complete"]
                                and f["all_elements_checked_on_every_rank"] for f in fp),
          "30/30 16-rank full-element FP64 PASS")
    check(led["aggregate"].get("fp64_pass_cells") == "30/30", "aggregate fp64_pass_cells = 30/30")
    check(all(m["winner"] == "v62_champion" and m["timing_rank_count"] == 16 for m in models.values()),
          "v62_champion wins every cell, 16 timing ranks")
    for arm in arms:
        ratios = []
        for m in models.values():
            a = m["arms"][arm]
            ratios.append(m["arms"]["v58_ctrl"]["median_iter_max_us"] / a["median_iter_max_us"])
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        rec = led["aggregate"]["geometric_mean_speedup_vs_v58_ctrl"][arm]
        check(abs(gm - rec) < 1e-3, f"{arm}: geomean vs V58 recomputed {gm:.4f} == recorded {rec}")
    q = models["qwen38max_S16384_EP16"]
    r = q["rccl_aiter"]["median_iter_max_us"] / q["arms"]["v62_champion"]["median_iter_max_us"]
    check(abs(r - q["winner_vs_rccl_speedup"]) < 1e-3 and r > 2.27,
          f"qwen38max (2.4T) v62_champion vs RCCL+AITER recomputed {r:.4f}x")
    check(not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|pangu://|e01-cn-", raw),
          "no host IPs / node names / cluster log paths in the ledger")


def main() -> int:
    for p in (BUILD_V62, BUILD_PORT, PRESET, LEDGER, KNOBS_PY):
        if not p.exists():
            print(f"[FAIL] missing {p}")
            return 1
    preset = json.loads(PRESET.read_text(encoding="utf-8"))
    b = load_module(BUILD_V62, "weave_build_v62")
    preset_suite(preset)
    source_suite(preset, b)
    flags_suite(preset, b)
    isolation_suite(preset)
    ledger_suite(preset)
    status = "PASS" if FAILURES == 0 else "FAIL"
    print(f"\n[{status}] EP16 V62 champion gate: {CHECKS - FAILURES}/{CHECKS} checks passed")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
