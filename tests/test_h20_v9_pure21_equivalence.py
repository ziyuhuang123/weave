#!/usr/bin/env python3
"""Pure WEAVE_* env <-> WEAVE_H20_CONFIG equivalence gate for the H20 V9 engine.

("pure21" is the historical name of the env-only side of the pair, from the 21-knob
surface.  The public surface is now 15 knobs; the cells also set internal pins of
the six retired knobs (WEAVE_REDUCE_WARPS, WEAVE_COMBINE_SYNC_CHUNK, ...), which the
host still accepts as internal env diagnostics.  The renamed knobs are exercised
under both spellings: WEAVE_ADAPT_GEMM_CONFIG / WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG
and their legacy aliases WEAVE_ADAPT_BM / WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2.)

For every arm of the w12v9q sweep (batches b00..b08: 15 pure-21 cells plus the
120-cell interleave{0,5} x epw{8,16,32,64} x COMBINE_SYNC_CHUNK{0,16,32,64,128}
x REDUCE_WARPS{0,1,2} grid, all on comm_only + smem + ADAPT_BM=1, plus the
p21 / v8base anchors), this test runs the LIVE SM90 host heuristics twice at
S=1024 and S=4096 through tests/h20_v9_host_emitter.cpp:

  legacy  -- the arm exactly as the sweep launches it (private WEAVE_H20_CONFIG
             spec string carrying interleave / epw / comb_sync_chunk);
  pure21  -- the same cell expressed with public WEAVE_* knobs only, no
             WEAVE_H20_CONFIG:  WEAVE_CU_WAVE_SPLIT = epw (interleave 5) or
             "0-epw" (interleave off), WEAVE_COMBINE_SYNC_CHUNK, WEAVE_REDUCE_WARPS.

and requires the complete nvcc input -- the JIT #define block plus the kernel
template instantiation that SM90FP8MegaMoERuntime::compile_only would compile --
to be byte-identical.  Identical nvcc input implies identical ptxas output, so
the measured arms are reproducible without the private config.

It also re-runs the golden-geometry host checks merged from 7e33e16
(tests/verify_sm90_golden_host.cpp there) against the V9 engine, the decoupled
GEMM1/GEMM2 BLOCK_M / BLOCK_N / BLOCK_K profiles, and the fail-closed spellings.

Usage: python tests/test_h20_v9_pure21_equivalence.py [--jobs N]
Writes results/h20_v9_pure21_equivalence.json; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h20_v9_host_emit as emit  # noqa: E402

ROOT = emit.ROOT
OUT_JSON = ROOT / "results/h20_v9_pure21_equivalence.json"
BUILD = ROOT / "scratch/h20_v9_pure21_equivalence"

# The measured H20 V8 winner spec (prior EP4 sweep, results/h20_ep4_prior_verified.json shape), as
# the w12v9q sweep (mkarms_q.py) composes it.
V8SPEC = ("fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=64,adapt_bm=1,bf16_acc=0,"
          "bn=256,bk=128,wgs=4,swap=1,stages=0,swap_pipe=0,fast_swap_epi=1,"
          "l1_nmajor=1,l2_nmajor=1,rank_cache=1,fast_clean=1,fast_pull=1,dedup_disp=1,"
          "dedup_tier=1,reg_comb=1,fast_comb=1,all_idle_comb=1")
BASE = {"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem",
        "WEAVE_ADAPT_BM": "1"}
ILS = (0, 5)
EPWS = (8, 16, 32, 64)
CHUNKS = (0, 16, 32, 64, 128)
RWS = (0, 1, 2)
TOKENS = (1024, 4096)


def wave_split(il: int, epw: int) -> str:
    return str(epw) if il == 5 else f"{il}-{epw}"


def build_pairs() -> list[dict]:
    pairs = []
    # b01..b08: the 120-cell grid.
    for il in ILS:
        for epw in EPWS:
            for ch in CHUNKS:
                for rw in RWS:
                    legacy = {**BASE, "WEAVE_REDUCE_WARPS": str(rw),
                              "WEAVE_H20_CONFIG": V8SPEC + f",interleave={il},epw={epw},comb_sync_chunk={ch}"}
                    pure = {**BASE, "WEAVE_REDUCE_WARPS": str(rw),
                            "WEAVE_COMBINE_SYNC_CHUNK": str(ch),
                            "WEAVE_CU_WAVE_SPLIT": wave_split(il, epw)}
                    pairs.append({"name": f"i{il}e{epw}c{ch}r{rw}", "batch": "grid",
                                  "legacy": legacy, "pure21": pure})
    # b00: the pure-21 cells already carry no private config.  Under the engine
    # default (interleave off, epw = E_local; the measured V8 schedule) each must
    # equal the V8 spec with only comb_sync_chunk={ch} added.
    for ch in CHUNKS:
        for rw in RWS:
            pure = {**BASE, "WEAVE_COMBINE_SYNC_CHUNK": str(ch), "WEAVE_REDUCE_WARPS": str(rw)}
            legacy = {**BASE, "WEAVE_REDUCE_WARPS": str(rw),
                      "WEAVE_H20_CONFIG": V8SPEC + f",comb_sync_chunk={ch}"}
            pairs.append({"name": f"p_c{ch}r{rw}", "batch": "b00", "legacy": legacy, "pure21": pure})
    # Anchors: p21 (the unset-knob pure-21 default) must now BE v8base (V8
    # skeleton, Reduce off, chunk 0, no interleave, epw resolved to E_local =
    # 128); so must WEAVE_CU_WAVE_SPLIT=0 (schema: 0 = auto).
    V8BASE_LEGACY = {**BASE, "WEAVE_REDUCE_WARPS": "0",
                     "WEAVE_H20_CONFIG": V8SPEC + ",comb_sync_chunk=0"}
    pairs.append({"name": "p21", "batch": "anchor", "pure21": dict(BASE),
                  "legacy": dict(V8BASE_LEGACY)})
    pairs.append({"name": "p21_ws0", "batch": "anchor",
                  "pure21": {**BASE, "WEAVE_CU_WAVE_SPLIT": "0"},
                  "legacy": dict(V8BASE_LEGACY)})
    pairs.append({"name": "v8base", "batch": "anchor",
                  "legacy": {**BASE, "WEAVE_REDUCE_WARPS": "0",
                             "WEAVE_H20_CONFIG": V8SPEC + ",comb_sync_chunk=0"},
                  "pure21": {**BASE, "WEAVE_REDUCE_WARPS": "0", "WEAVE_COMBINE_SYNC_CHUNK": "0",
                             "WEAVE_CU_WAVE_SPLIT": "0-128"}})
    return pairs


def nvcc_inputs(res: dict) -> dict[str, str]:
    return {k: p.read_text(encoding="utf-8") for k, p in sorted(res["sources"].items())}


def host_rejected(r: dict) -> bool:
    """A fail-closed probe passes only on a real DG_HOST_ASSERT, never on a bare
    non-zero exit code."""
    return r["returncode"] != 0 and "Assertion error" in r.get("stderr", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    binary = emit.build_emitter(BUILD)
    checks: list[dict] = []

    def check(ok: bool, label: str, **detail) -> None:
        checks.append({"label": label, "pass": bool(ok), **detail})
        if not ok:
            print(f"[FAIL] {label} {json.dumps(detail)[:600]}")

    jobs = []
    for pair in build_pairs():
        for tokens in TOKENS:
            for side in ("legacy", "pure21"):
                jobs.append((pair, tokens, side))

    def run(job):
        pair, tokens, side = job
        prefix = BUILD / f"{pair['name']}_s{tokens}_{side}"
        return job, emit.run_emitter(binary, tokens, pair[side], prefix, check=False)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, jobs))
    by_key = {(j[0]["name"], j[1], j[2]): r for j, r in results}

    rows = []
    for pair in build_pairs():
        for tokens in TOKENS:
            lg = by_key[(pair["name"], tokens, "legacy")]
            pu = by_key[(pair["name"], tokens, "pure21")]
            ok_run = lg["returncode"] == 0 and pu["returncode"] == 0
            same = ok_run and nvcc_inputs(lg) == nvcc_inputs(pu)
            if ok_run:
                four_wg = all(pu[f"{ph}_EPI"] == "512" for ph in ("L1", "L2"))
            else:
                four_wg = False
            label = f"{pair['name']} S={tokens}: pure-21 nvcc input == WEAVE_H20_CONFIG arm"
            check(same, label, legacy_rc=lg["returncode"], pure_rc=pu["returncode"],
                  legacy_err=lg.get("stderr", "")[-300:], pure_err=pu.get("stderr", "")[-300:])
            check(four_wg, f"{pair['name']} S={tokens}: 4 compute warpgroups (512 epilogue threads)")
            no_private = "WEAVE_H20_CONFIG" not in pair["pure21"]
            check(no_private, f"{pair['name']}: pure-21 side sets no WEAVE_H20_CONFIG")
            if ok_run:
                rows.append({"arm": pair["name"], "batch": pair["batch"], "tokens": tokens,
                             "pure21_env": pair["pure21"], "identical": same,
                             "fused": pu["FUSED"], "kernels": pu["kernels"],
                             "bm": [pu["L1_BM"], pu["L2_BM"]], "bn": [pu["L1_BN"], pu["L2_BN"]],
                             "bk": [pu["L1_BK"], pu["L2_BK"]], "stages": pu["L1_STAGES"],
                             "threads": [pu["L1_DISP"], pu["L1_NONEPI"], pu["L1_EPI"]],
                             "epw": pu["L1_EPW"], "comb_warps": pu["COMB_WARPS"],
                             "defines_sha": __import__("hashlib").sha256(
                                 "".join(nvcc_inputs(pu).values()).encode()).hexdigest()[:16]})

    # Cell-level semantics that the equivalence alone does not pin.
    def single(env: dict, tokens: int = 1024, name: str = "probe") -> dict:
        return emit.run_emitter(binary, tokens, env, BUILD / f"probe_{name}_s{tokens}", check=False)

    r = single({**BASE, "WEAVE_COMBINE_SYNC_CHUNK": "0", "WEAVE_REDUCE_WARPS": "2"}, name="c0r2")
    check(r["returncode"] == 0 and "AGY_FINE_COMB_G" not in r["defines"],
          "explicit WEAVE_COMBINE_SYNC_CHUNK=0 wins over the Reduce-on chunk-64 default")
    r = single({**BASE, "WEAVE_REDUCE_WARPS": "2"}, name="r2default")
    check(r["returncode"] == 0 and "#define AGY_FINE_COMB_G 4" in r["defines"],
          "unset WEAVE_COMBINE_SYNC_CHUNK keeps the chunk-64 default (AGY_FINE_COMB_G 4)")
    for value, ilv, epw in (("8", 5, 8), ("16", 5, 16), ("32", 5, 32), ("64", 5, 64),
                            ("0-8", 0, 8), ("0-64", 0, 64), ("4-8", 4, 8), ("6-16", 6, 16)):
        r = single({**BASE, "WEAVE_CU_WAVE_SPLIT": value}, name=f"ws{value}")
        ok = r["returncode"] == 0 and r["L1_EPW"] == str(epw) and \
            ((f"#define WEAVE_INTERLEAVE_L1_L2 {ilv}" in r["defines"]) if ilv else
             ("WEAVE_INTERLEAVE_L1_L2" not in r["defines"]))
        check(ok, f"WEAVE_CU_WAVE_SPLIT={value} -> interleave={ilv}, epw={epw}",
              epw_got=r.get("L1_EPW"), err=r.get("stderr", "")[-200:])
    for value in ("auto", "0"):
        r = single({**BASE, "WEAVE_CU_WAVE_SPLIT": value}, name=f"ws_{value}")
        check(r["returncode"] == 0 and r["L1_EPW"] == "128" and
              "WEAVE_INTERLEAVE_L1_L2" not in r["defines"],
              f"WEAVE_CU_WAVE_SPLIT={value} keeps the engine default: interleave off, epw=E_local",
              epw_got=r.get("L1_EPW"), err=r.get("stderr", "")[-200:])
    r = single(dict(BASE), name="p21_default")
    check(r["returncode"] == 0 and r["COMB_WARPS"] == "0" and r["L1_EPW"] == "128" and
          "WEAVE_INTERLEAVE_L1_L2" not in r["defines"] and "WEAVE_REDUCE_WARPS" not in r["defines"],
          "unset-knob pure-21 default = V8 schedule: interleave off, epw=E_local, Reduce warps 0",
          got={k: r.get(k) for k in ("COMB_WARPS", "L1_EPW")}, err=r.get("stderr", "")[-200:])
    r = single({**BASE, "WEAVE_H20_CONFIG": V8SPEC}, name="legacy_rw_default")
    check(r["returncode"] == 0 and r["COMB_WARPS"] == "2",
          "private WEAVE_H20_CONFIG engine keeps its R=2 default (only the pure skeleton changed)",
          got=r.get("COMB_WARPS"), err=r.get("stderr", "")[-200:])
    # v11 lag-W expert task queue: "7-W" -> interleave 7, WEAVE_TASKQ_LAG W, and the
    # engine keeps epw = E_local (no expert waves).
    for lag in (2, 4, 6, 8, 12):
        r = single({**BASE, "WEAVE_CU_WAVE_SPLIT": f"7-{lag}"}, name=f"ws7-{lag}")
        check(r["returncode"] == 0 and r["L1_EPW"] == "128" and
              "#define WEAVE_INTERLEAVE_L1_L2 7" in r["defines"] and
              f"#define WEAVE_TASKQ_LAG {lag}u" in r["defines"],
              f"WEAVE_CU_WAVE_SPLIT=7-{lag} -> interleave=7, taskq lag={lag}, epw=E_local",
              epw_got=r.get("L1_EPW"), err=r.get("stderr", "")[-200:])
    for bad in ("1", "2", "3", "8-0", "8-8", "9-8", "x", "5-", "7-0"):
        r = single({**BASE, "WEAVE_CU_WAVE_SPLIT": bad}, name=f"wsbad{bad}")
        check(host_rejected(r), f"WEAVE_CU_WAVE_SPLIT={bad} fails closed on SM90")
    r = single({"WEAVE_CU_WAVE_SPLIT": "16"}, name="ws_only")
    check(r["returncode"] == 0 and r.get("SPEC_ENABLED") == "1" and r["L1_EPW"] == "16",
          "WEAVE_CU_WAVE_SPLIT alone selects the pure 21-knob engine")
    r = single({**BASE, "WEAVE_REDUCE_WARPS": "8"}, name="rw8")
    check(host_rejected(r), "WEAVE_REDUCE_WARPS=8 fails closed on the comm_only SMEM engine")
    r = single({**BASE, "WEAVE_COMBINE_SYNC_CHUNK": "256"}, name="c256")
    check(host_rejected(r), "WEAVE_COMBINE_SYNC_CHUNK=256 is rejected")

    # Golden geometry (merged from 7e33e16's verify_sm90_golden_host.cpp, V9 engine).
    def golden(env: dict, name: str, disp: str | None = None, stages: str | None = None) -> None:
        r = single(env, name=name)
        ok = (r["returncode"] == 0 and r["FUSED"] == "1" and
              all(r[f"{ph}_BM"] == "64" and r[f"{ph}_BN"] == "256" and r[f"{ph}_BK"] == "128" and
                  r[f"{ph}_EPI"] == "512" and r[f"{ph}_SWAP"] == "1" and r[f"{ph}_NMAJOR"] == "1"
                  for ph in ("L1", "L2")) and r["BF16_ACC"] == "0")
        if disp is not None:
            ok = ok and r["L1_DISP"] == disp
        if stages is not None:
            ok = ok and r["L1_STAGES"] == stages
        check(ok, f"golden geometry ({name}): fused BM64/BN256/BK128 4-WG swap N-major fp32",
              got={k: r.get(k) for k in ("FUSED", "L1_BM", "L1_BN", "L1_EPI", "L1_DISP", "L1_STAGES")},
              err=r.get("stderr", "")[-200:])

    golden({**BASE}, "default_r0", disp="64", stages="4")
    # v11: on the comm_only SMEM engine with WEAVE_SCATTER_WARPS unset the Reduce
    # warps are the gap-Reduce subset of the scatter warps, so R=2 reuses W0/W1
    # (2-warp group, 4 stages) and R=4 needs a 4-warp group (was 4 / 8 warps).
    golden({**BASE, "WEAVE_REDUCE_WARPS": "2"}, "winner_r2", disp="64", stages="4")
    golden({**BASE, "WEAVE_REDUCE_WARPS": "1"}, "winner_r1", disp="64", stages="4")
    golden({**BASE, "WEAVE_REDUCE_WARPS": "0"}, "winner_r0", disp="64", stages="4")
    golden({**BASE, "WEAVE_REDUCE_WARPS": "4"}, "winner_r4", disp="128")
    golden({"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem",
            "WEAVE_DEDUP_MODE": "1"}, "comm_only_smem_dedup")
    golden({"WEAVE_DISPATCH_WARPS": "4"}, "dispatch_warps4", disp="128")
    golden({"WEAVE_LOCAL_SCATTER_DEDUP": "1"}, "local_scatter_dedup")
    golden({**BASE, "WEAVE_BLOCK_M_L1": "64", "WEAVE_BLOCK_M_L2": "64"}, "bm64_explicit")
    r = single({**BASE, "WEAVE_ADAPT_BM": "0"}, name="adapt0")
    check(r["returncode"] == 0 and "WEAVE_ADAPT_BM" not in r["defines"],
          "explicit WEAVE_ADAPT_BM=0 turns the baked adapt_bm off")
    r = single({**BASE, "WEAVE_DEDUP_MODE": "0"}, name="dedup0")
    check(r["returncode"] == 0 and "WEAVE_DEDUP_DISPATCH" not in r["defines"],
          "explicit WEAVE_DEDUP_MODE=0 turns the baked dedup_disp off")
    r = single({**BASE, "WEAVE_H20_CONFIG": V8SPEC}, name="h20cfg_verbatim")
    check(r["returncode"] == 0 and r["FUSED"] == "1" and r["L1_EPW"] == "128" and
          "WEAVE_INTERLEAVE_L1_L2" not in r["defines"],
          "legacy WEAVE_H20_CONFIG still parses verbatim (V8 skeleton: no interleave, epw=E_local)")

    # BLOCK_M_L1 / _L2: fused (equal) and decoupled split launches.
    def bm_case(l1: str | None, l2: str | None, expect: dict, name: str, env_extra=None) -> dict:
        env = {"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem"}
        if l1: env["WEAVE_BLOCK_M_L1"] = l1
        if l2: env["WEAVE_BLOCK_M_L2"] = l2
        env.update(env_extra or {})
        r = single(env, name=name)
        ok = r["returncode"] == 0 and all(r.get(k) == v for k, v in expect.items())
        check(ok, f"BLOCK_M case {name}: {expect}",
              got={k: r.get(k) for k in expect}, err=r.get("stderr", "")[-300:])
        return r

    four = lambda bm1, bm2, fused: {"FUSED": fused, "L1_BM": bm1, "L2_BM": bm2,
                                    "L1_BN": "256", "L2_BN": "256", "L1_BK": "128", "L2_BK": "128",
                                    "L1_EPI": "512", "L2_EPI": "512", "L1_SWAP": "1", "L2_SWAP": "1"}
    bm_case("32", None, four("32", "32", "1"), "l1_32_only")
    bm_case(None, "16", four("16", "16", "1"), "l2_16_only")
    bm_case("32", "32", four("32", "32", "1"), "both_32")
    for a, b in (("32", "64"), ("16", "64"), ("32", "16")):
        bm_case(a, b, four(a, b, "0"), f"split_{a}_{b}")
    # The split L1 kernel at BLOCK_M=64 spills 12 B at the 4-WG cap: fail closed.
    for a, b in (("64", "16"), ("64", "32")):
        r = single({"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem",
                    "WEAVE_BLOCK_M_L1": a, "WEAVE_BLOCK_M_L2": b}, name=f"split_{a}_{b}_rejected")
        check(host_rejected(r), f"decoupled WEAVE_BLOCK_M_L1={a}/L2={b} (spilling split L1) fails closed")
    # GEMM1/GEMM2 decoupled BN/BK: a BLOCK_M=128 phase locks BN=128 (non-swap,
    # 2 WGs by construction) while the other phase keeps the BN=256 4-WG tile.
    bm_case("128", "64", {"FUSED": "0", "L1_BM": "128", "L1_BN": "128", "L1_BK": "128",
                          "L1_SWAP": "0", "L2_BM": "64", "L2_BN": "256", "L2_BK": "128",
                          "L2_SWAP": "1", "L2_EPI": "512"}, "split_128_64")
    bm_case("128", None, {"FUSED": "1", "L1_BM": "128", "L1_BN": "128", "L2_BN": "128",
                          "L1_SWAP": "0"}, "fused_128")
    # v13: WEAVE_ADAPT_GEMM_CONFIG (legacy alias WEAVE_ADAPT_BM) is a host-only
    # whole-tuple selector; explicit GEMM1/GEMM2 pins always win over it, and it
    # never emits the device per-expert BLOCK_M sub-tiling define.
    r = bm_case("32", "16", four("32", "16", "0"), "split_adapt1_pins_win",
                env_extra={"WEAVE_ADAPT_BM": "1"})
    check(r["returncode"] == 0 and "WEAVE_ADAPT_BM" not in r.get("defines", ""),
          "decoupled pins + explicit WEAVE_ADAPT_BM=1: pins win, no device adapt-BM define")
    # New public names are exact aliases of the legacy ones (identical nvcc input).
    scat = {"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem"}
    for toks in (256, 1024):
        for new_env, old_env, tag in (
                ({"WEAVE_GEMM1_CONFIG": "128", "WEAVE_GEMM2_CONFIG": "128"},
                 {"WEAVE_BLOCK_M_L1": "128", "WEAVE_BLOCK_M_L2": "128"}, "g128"),
                ({"WEAVE_GEMM1_CONFIG": "32", "WEAVE_GEMM2_CONFIG": "64"},
                 {"WEAVE_BLOCK_M_L1": "32", "WEAVE_BLOCK_M_L2": "64"}, "g32_64"),
                ({"WEAVE_ADAPT_GEMM_CONFIG": "0"}, {"WEAVE_ADAPT_BM": "0"}, "adapt0"),
                ({"WEAVE_ADAPT_GEMM_CONFIG": "1"}, {"WEAVE_ADAPT_BM": "1"}, "adapt1")):
            a = emit.run_emitter(binary, toks, {**scat, **new_env}, BUILD / f"v13_new_{tag}_s{toks}", check=False)
            b = emit.run_emitter(binary, toks, {**scat, **old_env}, BUILD / f"v13_old_{tag}_s{toks}", check=False)
            same = (a["returncode"] == 0 and b["returncode"] == 0 and a.get("kernels") == b.get("kernels") and
                    all(a["sources"][k].read_bytes() == b["sources"][k].read_bytes() for k in a["kernels"]))
            check(same, f"v13 alias {tag} tokens={toks}: new name == legacy name (identical nvcc input)")
    # Host whole-tuple selection on Qwen3.8-Max (E=512, K=10, I=2048):
    # tokens/rank=256 -> M_expected=20 (<=24): decoupled 32/64 split launch, or the
    # fused canonical 64 tuple when a fused-only knob (CU_WAVE_SPLIT) is set;
    # tokens/rank=1024 -> M_expected=80: fused canonical 64 tuple.
    r = single(dict(scat), tokens=256, name="v13_adapt_auto_m20")
    check(r["returncode"] == 0 and r["FUSED"] == "0" and r["L1_BM"] == "32" and r["L2_BM"] == "64" and
          r["L1_SWAP"] == "1" and r["L2_SWAP"] == "1",
          "v13 ADAPT_GEMM_CONFIG default, M_expected=20: decoupled GEMM1 BM32 / GEMM2 BM64 Swap-AB",
          got={k: r.get(k) for k in ("FUSED", "L1_BM", "L2_BM")})
    r = single({**scat, "WEAVE_CU_WAVE_SPLIT": "7-32"}, tokens=256, name="v13_adapt_auto_m20_fused")
    check(r["returncode"] == 0 and r["FUSED"] == "1" and r["L1_BM"] == "64" and r["L2_BM"] == "64" and
          r["L1_BN"] == "256" and r["L1_EPI"] == "512" and r["L1_SWAP"] == "1",
          "v13 ADAPT_GEMM_CONFIG default, M_expected=20 + CU_WAVE_SPLIT: fused Swap-AB 4-WG BM64",
          got={k: r.get(k) for k in ("FUSED", "L1_BM", "L2_BM")})
    for toks in (256, 1024):
        r = single({**scat, "WEAVE_ADAPT_GEMM_CONFIG": "0"}, tokens=toks, name=f"v13_adapt0_{toks}")
        check(r["returncode"] == 0 and r["FUSED"] == "1" and r["L1_BM"] == "64" and r["L1_BN"] == "256" and
              r["L1_EPI"] == "512" and r["L1_SWAP"] == "1" and r["L1_STAGES"] == "4",
              f"v13 ADAPT_GEMM_CONFIG=0 tokens={toks}: fixed canonical BM64/BN256/BK128 4-WG 4-stage tuple",
              got={k: r.get(k) for k in ("FUSED", "L1_BM", "L1_BN", "L1_STAGES")})
    r = single(dict(scat), tokens=1024, name="v13_adapt_auto_m80")
    check(r["returncode"] == 0 and r["FUSED"] == "1" and r["L1_BM"] == "64" and r["L2_BM"] == "64",
          "v13 ADAPT_GEMM_CONFIG default, M_expected=80 (I=2048): fused canonical BM64 tuple")

    n_pass = sum(c["pass"] for c in checks)
    report = {"schema": "weave-h20-v9-pure21-equivalence/v1",
              "checks_total": len(checks), "checks_pass": n_pass,
              "pairs": len(build_pairs()), "tokens": list(TOKENS),
              "rows": rows, "failures": [c for c in checks if not c["pass"]],
              "pass": n_pass == len(checks)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    identical = sum(1 for r in rows if r["identical"])
    print(f"[INFO] pairs={len(build_pairs())} x tokens={len(TOKENS)}: identical nvcc input {identical}/{len(rows)}")
    print(f"CHECKS={len(checks)} FAILURES={len(checks) - n_pass}")
    print("ALL H20 V9 PURE-21 EQUIVALENCE CHECKS PASS" if report["pass"]
          else "H20 V9 PURE-21 EQUIVALENCE CHECKS FAILED")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
