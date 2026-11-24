#!/usr/bin/env python3
"""3-Tier Tier-1 offline engine gate: (HW, Precision, Phase, N, K) profiles.

Pure Python (no torch / CUDA / compiler).  Verifies:

1. Coverage: results/gemm_profiles_3hw.json holds all 6 standard models x
   3 hardware x 2 phases x the 9-bucket M ladder, and every committed cell
   equals the live engine output (regeneration is bit-identical).
2. Analytical legality: N % BN == 0, N % (2*BN) == 0 (2-CTA cluster /
   SwiGLU-half alignment), K % BK == 0, SMEM/LDS stage budget, zero-spill
   accumulator budget, K-loop stage bound and the short-K (<= 1024) cap.
3. L1 intra-expert 2D L2-swizzle preservation: multi-block M with the weight
   inside half of L2 keeps swizzle group >= 2 (never tight-mode eviction).
4. Per-M inner tuples share the phase envelope (BM <= max_BM, 32-row align).
5. Diversity: different (N, K) models select genuinely different envelopes;
   GEMM1 and GEMM2 are solved independently; inner tuples vary across M.
6. SM0 ordering edge cases + the dedup-span / hot-cold metrics on realistic
   Zipf / Qwen routing histograms (Python mirror properties).

Usage:
    python tests/test_phase_nk_profiles.py
Exit status 0 only when every check passes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

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


CAL = load_module("weave_calibrate_phase_nk", ROOT / "deep_gemm/mega/calibrate.py")

ARCH_OF = {"sm103": "b300", "h20": "h20", "mi308x": "mi308x"}
PREC_OF = {"sm103": "fp8xfp4", "h20": "fp8", "mi308x": "fp8"}


def check_coverage_and_regeneration() -> Dict:
    print("\n== Coverage: 6 models x 3 hw x 2 phases x M ladder ==")
    f0 = FAILURES
    path = ROOT / "results/gemm_profiles_3hw.json"
    check(path.exists(), "results/gemm_profiles_3hw.json exists")
    committed = json.loads(path.read_text(encoding="utf-8"))
    check(committed.get("schema") == "weave-gemm-profiles-3hw/v2", "schema tag")
    live = CAL.build_gemm_profiles_report()
    check(committed == live, "committed JSON == live engine (bit-identical)")
    check(committed.get("m_ladder") == list(CAL.M_LADDER), "M ladder recorded")
    check(set(committed.get("models", {})) == set(CAL.STANDARD_MODELS),
          "all 6 standard models present")
    check(set(committed.get("profiles", {})) == {"sm103", "h20", "mi308x"},
          "all 3 hardware present")
    cells = 0
    for hw, models in committed["profiles"].items():
        check(set(models) == set(CAL.STANDARD_MODELS), f"{hw}: 6 models")
        for model, phases in models.items():
            check(set(phases) == {"gemm1", "gemm2"}, f"{hw}/{model}: 2 phases")
            for ph, cell in phases.items():
                cells += 1
                check(set(cell["inner_lut"]) == {str(m) for m in CAL.M_LADDER},
                      f"{hw}/{model}/{ph}: 9 M buckets")
                check(set(cell["inner_codes"]) == {str(m) for m in CAL.M_LADDER},
                      f"{hw}/{model}/{ph}: 9 inner codes")
    check(cells == 36, f"36 phase cells (got {cells})")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] coverage: {cells} cells, "
          "regeneration bit-identical")
    return committed


def check_legality(committed: Dict) -> None:
    print("\n== Analytical legality (pruning rules, every cell) ==")
    f0 = FAILURES
    for hw, models in committed["profiles"].items():
        arch = ARCH_OF[hw]
        hwc = CAL.PHASE_NK_HW[arch]
        prec = PREC_OF[hw]
        act32, w32 = CAL.PHASE_NK_BYTES32[prec]
        for model, phases in models.items():
            for ph, cell in phases.items():
                phase = int(ph[-1])
                n, k = cell["n"], cell["k"]
                env = cell["envelope"]
                bn, bk, bm = env["bn"], env["bk"], env["max_bm"]
                st = env["num_stages"]
                key = f"{hw}/{model}/{ph}"
                check(n % bn == 0, f"{key}: N % BN == 0 (zero tail waste)")
                check(n % (2 * bn) == 0, f"{key}: N % 2BN == 0 (cluster/SwiGLU)")
                check(k % bk == 0, f"{key}: K % BK == 0")
                # Backend tile legality: the kernels' own static asserts.
                check(bn in CAL.PHASE_NK_BN[arch], f"{key}: legal BLOCK_N")
                check(bk in CAL.PHASE_NK_BK[arch], f"{key}: legal BLOCK_K")
                check(bm in CAL.PHASE_NK_BM[arch], f"{key}: legal BLOCK_M")
                if arch == "b300":
                    check(bm % 16 == 0, f"{key}: SM100 BLOCK_M % 16 == 0")
                elif arch == "h20":
                    check(bm % 64 == 0, f"{key}: SM90 BLOCK_M % 64 == 0")
                    check(bk % 128 == 0, f"{key}: SM90 BLOCK_K % 128 == 0")
                check(env["wg_layout_mode"] in (0, 1, 2),
                      f"{key}: warpgroup layout legal")
                check(env["penalty"] >= 0, f"{key}: structural penalty recorded")
                check(bm * bn <= hwc["acc_tile_limit"],
                      f"{key}: zero-spill acc budget")
                stage_bytes = (bm * bk * act32 + bn * bk * w32) // 32
                check(st * stage_bytes <= hwc["smem_bytes"] - hwc["smem_reserve"],
                      f"{key}: stages fit SMEM/LDS")
                check(2 <= st <= 8, f"{key}: stages in 2..8")
                check(st <= max(2, k // bk), f"{key}: stages <= K iterations")
                if k <= 1024:
                    check(st <= 4, f"{key}: short-K stage cap")
                # Envelope code round trip through the live engine.
                live_env = CAL.select_phase_envelope(arch, prec, phase, n, k)
                check(live_env.code() == cell["envelope_code"],
                      f"{key}: envelope code")
                check(env["interleave_mode"] in (0, 1, 2),
                      f"{key}: interleave mode legal")
                check(1 <= env["l2_swizzle_group"] <= 8,
                      f"{key}: swizzle group 1..8")
                # Inner tuples share the envelope.
                for m, inner in cell["inner_lut"].items():
                    check(inner["block_m"] <= bm, f"{key}/M{m}: BM <= max_BM")
                    check(inner["block_m"] % 16 == 0, f"{key}/M{m}: BM 16-align")
                    if arch == "h20":
                        check(inner["block_m"] % 64 == 0 or
                              inner["block_m"] in (16, 32),
                              f"{key}/M{m}: WGMMA::M or swapAB height")
                    elif arch == "mi308x":
                        check(inner["block_m"] in (16, 32, 64, 128),
                              f"{key}/M{m}: legal CK tile height")
                    check(inner["tma_load_m"] <= inner["block_m"],
                          f"{key}/M{m}: tma <= BM")
                    check(inner["block_m"] % inner["store_block_m"] == 0,
                          f"{key}/M{m}: store divides BM")
                    check(inner["epi_reg_stages"] in (1, 2),
                          f"{key}/M{m}: epi stages 1/2")
                    if phase == 1:
                        check(inner["scatter_path"] == 0,
                              f"{key}/M{m}: GEMM1 no scatter")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] legality: all pruning "
          "rules hold on every committed cell")


def check_l1_swizzle(committed: Dict) -> None:
    print("\n== L1 intra-expert 2D L2-swizzle preservation ==")
    f0 = FAILURES
    protected = 0
    for hw, models in committed["profiles"].items():
        arch = ARCH_OF[hw]
        hwc = CAL.PHASE_NK_HW[arch]
        prec = PREC_OF[hw]
        _, w32 = CAL.PHASE_NK_BYTES32[prec]
        for model, phases in models.items():
            for ph, cell in phases.items():
                n, k = cell["n"], cell["k"]
                weight = n * k * w32 // 32
                for m, inner in cell["inner_lut"].items():
                    m_blocks = CAL._pnk_ceil_div(int(m), inner["block_m"])
                    if m_blocks >= 2 and weight * 2 <= hwc["l2_bytes"]:
                        check(inner["l2_swizzle_group"] >= 2,
                              f"{hw}/{model}/{ph}/M{m}: multi-block L1 keeps "
                              "2D L2 swizzle (group >= 2)")
                        protected += 1
                    else:
                        check(inner["l2_swizzle_group"] == 1,
                              f"{hw}/{model}/{ph}/M{m}: single-block group 1")
                # Envelope-level verdict at the STRUCTURAL prefill anchor
                # (top of the M ladder, so max_BM is actually exercised).
                env = cell["envelope"]
                anchor = CAL.PNK_PREFILL_M
                bm_d = CAL.M.select_online_expert_bm(anchor, env["max_bm"], 0)
                tm_d = CAL._pnk_ceil_div(anchor, bm_d)
                if tm_d >= 2 and weight * 2 <= hwc["l2_bytes"]:
                    check(env["interleave_mode"] == 1,
                          f"{hw}/{model}/{ph}: envelope preserves expert-major "
                          "2D swizzle (mode 1, not tight mode)")
    check(protected > 0, "multi-block L1 protection exercised")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] L1 swizzle: {protected} "
          "multi-block inner tuples keep group >= 2; envelopes mode 1")


def check_diversity(committed: Dict) -> None:
    print("\n== Diversity: (N,K) models get genuinely different profiles ==")
    f0 = FAILURES
    codes = committed["envelope_codes"]
    check(len(set(codes.values())) >= 20,
          f">= 20 distinct envelopes across 36 cells (got {len(set(codes.values()))})")
    check(committed["distinct_envelopes"] == len(set(codes.values())),
          "distinct count consistent")
    check(committed["distinct_tile_tuples"] >= 12,
          f">= 12 distinct (BN,BK,BM,stages) tuples "
          f"(got {committed['distinct_tile_tuples']})")
    # There are only 12 distinct (phase,N,K) geometries among the 6 models, so
    # the per-M LUT signature count is bounded well below the envelope count
    # (the envelope additionally carries the per-cell swizzle / epw verdicts).
    check(committed["distinct_lut_signatures"] >= 12,
          f">= 12 distinct per-M LUT signatures "
          f"(got {committed['distinct_lut_signatures']})")
    # GEMM1 vs GEMM2 independence per (hw, model): MOST pairs must differ.
    indep = 0
    for hw, models in committed["profiles"].items():
        for model, phases in models.items():
            if phases["gemm1"]["envelope_code"] != phases["gemm2"]["envelope_code"]:
                indep += 1
    check(indep >= 10, f"GEMM1/GEMM2 differ on {indep}/18 model-hw cells")
    check(committed["independent_phase_pairs"] == indep,
          "independent-pair audit consistent")
    check(len(committed["independent_phase_pair_cells"]) == indep,
          "independent-pair cell list consistent")
    # No hardware may collapse every model+phase onto one constant envelope.
    for hw, variants in committed["per_hw_envelope_variants"].items():
        check(variants >= 3, f"{hw}: {variants} envelope variants (not a constant)")
    # Per-M inner tuples vary across the ladder (not one fixed tuple).
    varying = 0
    for hw, models in committed["profiles"].items():
        for model, phases in models.items():
            for ph, cell in phases.items():
                if len(set(cell["inner_codes"].values())) >= 3:
                    varying += 1
    check(varying >= 30, f"per-M LUT varies on {varying}/36 cells")
    # Same model, different hardware -> different envelopes somewhere.
    cross_hw = 0
    for model in CAL.STANDARD_MODELS:
        for ph in ("gemm1", "gemm2"):
            cs = {committed["profiles"][hw][model][ph]["envelope_code"]
                  for hw in ARCH_OF}
            if len(cs) >= 2:
                cross_hw += 1
    check(cross_hw >= 8, f"hardware changes the envelope on {cross_hw}/12 cells")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] diversity: "
          f"{len(set(codes.values()))} envelopes, {indep} independent phase "
          f"pairs, {varying} varying LUTs, {cross_hw} cross-hw differences")


def check_sm0_edge_cases() -> None:
    print("\n== SM0 ordering edge cases + realistic-histogram metrics ==")
    f0 = FAILURES
    # Edge cases: permutation validity + determinism on degenerate inputs.
    cases = [
        ("single", [7], 1, 1),
        ("two_hot", [100, 100], 1, 1),
        ("all_zero", [0] * 64, 16, 1),
        ("wave_one", [50, 3, 80, 2, 1], 1, 1),
        ("wave_gt_E", [10, 20, 30], 128, 1),
        ("e256", list(range(1, 257)), 16, 1),
        ("partner_self", [10] * 8, 4, 1),
        ("partner_inactive", [10, 0, 10, 0], 2, 1),
    ]
    for name, counts, wave, dedup in cases:
        e_count = len(counts)
        multi = [c // 2 for c in counts]
        partner = [(e + e_count // 2) % e_count for e in range(e_count)]
        if name == "partner_self":
            partner = list(range(e_count))
        if name == "partner_inactive":
            partner = [1, 0, 3, 2]
        o1 = CAL.build_sm0_expert_order(counts, multi, partner, wave, dedup)
        o2 = CAL.build_sm0_expert_order(counts, multi, partner, wave, dedup)
        check(sorted(o1) == list(range(e_count)), f"{name}: permutation")
        check(o1 == o2, f"{name}: deterministic")
        check(o1[-sum(1 for c in counts if c == 0):] ==
              sorted(e for e in range(e_count) if counts[e] == 0)
              if any(c == 0 for c in counts) else True,
              f"{name}: inactive experts last")
    # Realistic histograms: span reduction + wave variance + cold-wave ban.
    hist_cases = [
        ("zipf64", CAL.zipf_histogram(4096, 64), 16, 32),
        ("zipf128", CAL.zipf_histogram(32768, 128), 16, 64),
        ("qwen_h20_rank2", CAL.max_anchored_histogram(2937, 346, 122, 128), 78, 64),
        ("ten_hot", [100] * 10 + [5] * 118, 78, 64),
    ]
    for name, counts, wave, shift in hist_cases:
        e_count = len(counts)
        multi = [c // 2 for c in counts]
        partner = [(e + shift) % e_count for e in range(e_count)]
        hits = [[e, partner[e]] for e in range(e_count)
                if counts[e] > 0 and counts[partner[e]] > 0]
        order = CAL.build_sm0_expert_order(counts, multi, partner, wave, 1)
        base = CAL.sm0_completion_span(list(range(e_count)), hits)
        new = CAL.sm0_completion_span(order, hits)
        check(base > 0 and new * 100 <= base * 70,
              f"{name}: dedup span >= 30% shorter ({base} -> {new})")
        desc = sorted(range(e_count), key=lambda e: (-counts[e], e))
        vd = CAL.sm0_wave_m_variance(desc, counts, wave)
        vn = CAL.sm0_wave_m_variance(order, counts, wave)
        check(vn < vd, f"{name}: hot/cold pairing lowers wave M variance")
        # Cluster adjacency for mutual pairs.
        pos = {e: i for i, e in enumerate(order)}
        for e in range(e_count):
            p = partner[e]
            if counts[e] > 0 and counts[p] > 0 and partner[p] == e:
                check(abs(pos[e] - pos[p]) == 1,
                      f"{name}: pair ({e},{p}) adjacent")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] sm0 edge cases + metrics")


def check_sm0_mode_boundaries() -> None:
    """The hot/cold pairing boundary must be explicit and honoured."""
    print("\n== SM0 pairing-mode boundary (dedup / pendulum / W=1) ==")
    f0 = FAILURES
    counts = [400 // (e + 1) + 4 for e in range(64)]
    hits = [(64 - e) if e % 3 == 0 else 0 for e in range(64)]
    none = [64] * 64
    partner = [(e + 32) % 64 for e in range(64)]
    # Real co-occurrence edges -> clustering wins over hot/cold interleaving.
    _, mode = CAL.build_sm0_expert_order_with_mode(counts, hits, partner, 16, 1)
    check(mode == CAL.SM0_MODE_DEDUP_CLUSTER,
          "co-occurrence edges select dedup clustering")
    # Per-expert multi-hit only -> pendulum centring.
    order, mode = CAL.build_sm0_expert_order_with_mode(counts, hits, none, 16, 1)
    check(mode == CAL.SM0_MODE_PENDULUM, "multi-hit-only selects pendulum")
    pos = {e: i for i, e in enumerate(order)}
    best = max(range(64), key=lambda e: (hits[e], counts[e], -e))
    check(pos[best] == (64 - 1) // 2, "most-shared expert occupies the centre")
    sets = []
    for t in range(24):
        hs = [e for e in range(t % 3, 64, 3)
              if hits[e] > 0 and (e // 3) % 4 == t % 4]
        if len(hs) >= 2:
            sets.append(hs)
    base = CAL.sm0_completion_span(list(range(64)), sets)
    new = CAL.sm0_completion_span(order, sets)
    check(base > 0 and new * 100 <= base * 50,
          f"pendulum span >= 50% shorter ({base} -> {new})")
    # No multi-hit information at all -> hot/cold pairing when W >= 2.
    _, mode = CAL.build_sm0_expert_order_with_mode(counts, [0] * 64, none, 16, 1)
    check(mode == CAL.SM0_MODE_HOT_COLD, "no multi-hit -> hot/cold pairing")
    _, mode = CAL.build_sm0_expert_order_with_mode(counts, [0] * 64, none, 16, 0)
    check(mode == CAL.SM0_MODE_HOT_COLD, "dedup off + W>=2 -> hot/cold pairing")
    # W = 1: one expert fills the whole card -> pairing auto-exits.
    order, mode = CAL.build_sm0_expert_order_with_mode(counts, [0] * 64, none, 1, 0)
    check(mode == CAL.SM0_MODE_NO_PAIR, "W=1 auto-exits pairing")
    check(all(counts[order[i]] >= counts[order[i + 1]] for i in range(63)),
          "no-pair order is load-descending")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] mode boundary: cluster / "
          "pendulum / hot-cold / W=1 all honoured")


def check_no_new_knobs(committed: Dict) -> None:
    print("\n== 14-knob invariant: profiles introduce no WEAVE_* surface ==")
    f0 = FAILURES
    text = json.dumps(committed)
    check("WEAVE_" not in text, "profile report carries no WEAVE_* knob names")
    knobs = load_module("weave_knobs_phase_nk", ROOT / "deep_gemm/mega/knobs.py")
    check(len(knobs.KNOBS) == 14, "public knob schema has 14 entries")
    counts = {c: sum(1 for k in knobs.KNOBS if k.category == c)
              for c in ("Spatial", "Temporal", "Implementation")}
    check(counts == {"Spatial": 5, "Temporal": 3, "Implementation": 6},
          f"public knob schema is Spatial 5 / Temporal 3 / Implementation 6 (got {counts})")
    print(f"[{'PASS' if FAILURES == f0 else 'FAIL'}] no new public knobs")


def main() -> int:
    committed = check_coverage_and_regeneration()
    check_legality(committed)
    check_l1_swizzle(committed)
    check_diversity(committed)
    check_sm0_edge_cases()
    check_sm0_mode_boundaries()
    check_no_new_knobs(committed)
    out = ROOT / "results/phase_nk_profiles_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "weave-phase-nk-profiles-test/v1",
        "checks": CHECKS, "failures": FAILURES,
        "pass": FAILURES == 0,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nCHECKS={CHECKS} FAILURES={FAILURES}")
    print(f"WROTE {out}")
    if FAILURES == 0:
        print("ALL PHASE-NK OFFLINE PROFILE CHECKS PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
