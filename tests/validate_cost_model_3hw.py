#!/usr/bin/env python3
"""Validate Weave Eq.(1)-(3) and CTA-0 selection across B300/H20/MI308X.

Inputs:
  * data/routing_stats_3hw.json: real recorded routing aggregates copied from
    the B300 routing replay.  The per-local-expert histogram m_e is explicitly
    a deterministic proxy because the source artifact only stores aggregates.
  * golden values emitted by tests/test_cost_model_eq123.cpp.

The script does not import deep_gemm.__init__ (which requires torch/_C).  It
loads deep_gemm/mega/cost_model.py directly so it works on any machine.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "deep_gemm" / "mega" / "cost_model.py"


def load_model():
    spec = importlib.util.spec_from_file_location("weave_cost_model_py", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_model()

INTERMEDIATE = {
    "qwen38max": 2048,
    "minimax_m3": 3072,
    "DeepSeek-V4-Flash": 1024,
}

# Values printed by tests/test_cost_model_eq123.cpp for
# wl={4,128,64}, m_e={0,3,5,9}, D_in=23, M_out=17.
CPP_GOLDEN = {
    "b300": (264196, 0, 32, 14.022),
    "h20": (2147813386, 0, 32, 18.107),
    "mi308x": (526336, 0, 32, 22.103),
}

# Predicted total microseconds printed by test_reduce_warps() in the same C++
# test for the same fixture, indexed by WEAVE_REDUCE_WARPS = 0, 2, 4.
CPP_REDUCE_GOLDEN = {
    "b300": (14.022, 14.022, 14.022),
    "h20": (18.107, 18.097, 18.097),
    "mi308x": (22.103, 22.083, 22.078),
}


def zipf_histogram(total: int, bins: int) -> List[int]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    weights = [1.0 / ((i + 1) ** 0.9) for i in range(bins)]
    norm = sum(weights)
    raw = [total * w / norm for w in weights]
    out = [int(x) for x in raw]
    out[-1] += total - sum(out)
    assert sum(out) == total
    return out


def candidate_by_mask(arch: str, mask: int):
    hits = [c for c in M.CANDIDATES[arch] if c.r_mask == mask]
    if not hits:
        raise AssertionError(f"selection returned unknown r_mask {mask} for {arch}")
    return hits[0]


def run() -> Dict:
    with open(ROOT / "data" / "routing_stats_3hw.json", "r", encoding="utf-8") as f:
        routing = json.load(f)

    configs = [
        ("b300", 8, "fp8xfp4"),
        ("b300", 8, "fp4xfp4"),
        ("h20", 8, "fp8xfp4"),
        ("mi308x", 8, "fp8xfp4"),
        ("mi308x", 16, "fp8xfp4"),
    ]
    report = {
        "schema": "weave-cost-model-validation-3hw/v4",
        "model": "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "python_mirror": "deep_gemm/mega/cost_model.py",
        "routing_source": routing["source"],
        "routing_reality_boundary": routing["reality_boundary"],
        "cpp_golden_match": False,
        "precision_modes": {},
        "max_runtime_experts": M.MAX_RUNTIME_EXPERTS,
        "reduce_warps": {},
        "platforms": [],
        "pass": True,
    }

    # 1. C++ golden cross-check on the synthetic equation fixture.
    golden_ok = True
    wl = M.WorkloadSpec(4, 128, 64)
    m_e = [0, 3, 5, 9]
    for arch, (r_mask, scatter, v_head, total_us) in CPP_GOLDEN.items():
        b = M.select_policy(arch, wl, m_e, 23, 17)
        ok = (b.r_mask == r_mask and b.scatter_mode == scatter and
              b.v_head == v_head and abs(b.t_total_us - total_us) <= 0.002)
        if not ok:
            golden_ok = False
            print(f"[FAIL] C++ golden mismatch {arch}: {b}")
    report["cpp_golden_match"] = golden_ok
    report["pass"] = report["pass"] and golden_ok

    # 1b. Precision knob: the default must stay bit-identical, while NVFP4 must
    # change both byte widths and tensor-core rate in the documented direction.
    precision_report = {}
    base = M.select_policy("b300", wl, m_e, 23, 17)
    for mode in ("fp8xfp4", "fp4xfp4", "fp8", "bf16"):
        hw = M.hardware_for("b300", mode)
        br = M.select_policy("b300", wl, m_e, 23, 17, precision_mode=mode)
        precision_report[mode] = {
            "weight_bytes": hw.weight_bytes,
            "act_in_bytes": hw.act_in_bytes,
            "p_tc_full_tflops": hw.p_tc_full_tflops,
            "predicted_total_us": round(br.t_total_us, 4),
        }
    default_ok = (precision_report["fp8xfp4"]["predicted_total_us"] ==
                  round(base.t_total_us, 4))
    nv_ok = (precision_report["fp4xfp4"]["weight_bytes"] == 0.5625 and
             precision_report["fp4xfp4"]["p_tc_full_tflops"] ==
             2.0 * precision_report["fp8xfp4"]["p_tc_full_tflops"])
    bf16_ok = (precision_report["bf16"]["weight_bytes"] == 2.0 and
               precision_report["bf16"]["p_tc_full_tflops"] ==
               0.5 * precision_report["fp8xfp4"]["p_tc_full_tflops"])
    if not (default_ok and nv_ok and bf16_ok):
        report["pass"] = False
        print("[FAIL] precision knob invariants")
    report["precision_modes"] = precision_report
    report["precision_invariants"] = {
        "default_bit_identical": default_ok,
        "nvfp4_bytes_and_rate": nv_ok,
        "bf16_bytes_and_rate": bf16_ok,
    }

    # 1c. WEAVE_REDUCE_WARPS: 0 stays bit-identical to Eq.(1)-(3); 2/4 lanes
    # overlap the Combine reduction with Middle on the spare HBM bandwidth and
    # carve the middle comm-queue scatter group down to max(1, N_c - N) units.
    reduce_ok = True
    reduce_report: Dict[str, Dict] = {}
    for arch in ("b300", "h20", "mi308x"):
        hw = M.hardware_for(arch)
        golden = CPP_REDUCE_GOLDEN[arch]
        totals = {}
        for idx, rw in enumerate((0, 2, 4)):
            b = M.select_policy(arch, wl, m_e, 23, 17, reduce_warps=rw)
            totals[str(rw)] = round(b.t_total_us, 4)
            if abs(b.t_total_us - golden[idx]) > 0.002:
                reduce_ok = False
                print(f"[FAIL] C++ reduce-warps golden mismatch {arch} "
                      f"rw={rw}: {b.t_total_us:.4f} vs {golden[idx]:.3f}")
        for cand in M.CANDIDATES[arch]:
            for s in (0, 1):
                base = M.evaluate(arch, wl, cand, s, m_e, 23, 17)
                prev_tail = base.t_tail_reduce_us
                for rw in (2, 4):
                    br = M.evaluate(arch, wl, cand, s, m_e, 23, 17,
                                    reduce_warps=rw)
                    checks = [
                        br.f_gemm_flops == base.f_gemm_flops,
                        br.v_mem_bytes == base.v_mem_bytes,
                        br.v_epi_bytes == base.v_epi_bytes,
                        br.t_mma_us == base.t_mma_us,
                        br.t_epi_us == base.t_epi_us,
                        br.t_comp_us == base.t_comp_us,
                        br.t_pull_us == base.t_pull_us,
                        br.v_head == base.v_head,
                        br.t_tail_reduce_us <= prev_tail + 1e-9,
                        br.t_total_us == hw.t_head_us + br.t_mid_us + br.t_tail_reduce_us,
                    ]
                    if s == 0:
                        checks += [br.t_scat_us == base.t_scat_us,
                                   br.t_mid_us == base.t_mid_us,
                                   br.t_total_us <= base.t_total_us + 1e-9]
                    else:
                        eff = cand.comm_units or cand.comp_units
                        units = max(1, eff - rw)
                        b_scat = min(hw.b_link_gbps,
                                     max(1.0, float(hw.num_sms_or_cus * units) *
                                         hw.b_scat_comm_per_unit_gbps))
                        expect_scat = 17 * wl.hidden * hw.out_bytes / (b_scat * 1000.0)
                        checks += [abs(br.t_scat_us - expect_scat) <=
                                   1e-9 * max(1.0, expect_scat),
                                   br.t_scat_us >= base.t_scat_us - 1e-12]
                    if not all(checks):
                        reduce_ok = False
                        print(f"[FAIL] reduce-warps invariant {arch} "
                              f"c{cand.comm_units}_{cand.comp_units} s{s} rw={rw}")
                    prev_tail = br.t_tail_reduce_us
        reduce_report[arch] = {
            "predicted_total_us": totals,
            "cpp_golden_total_us": list(golden),
        }
    report["reduce_warps"] = reduce_report
    report["reduce_warps_invariants"] = {
        "zero_bit_identical": M.select_policy("b300", wl, m_e, 23, 17).t_total_us ==
                              M.select_policy("b300", wl, m_e, 23, 17,
                                              reduce_warps=0).t_total_us,
        "cpp_golden_match": reduce_ok,
    }
    report["pass"] = report["pass"] and reduce_ok

    # 1d. kMaxRuntimeExperts = 256: every 128 < E_local <= 256 workload must
    # still select a valid policy (the C++ rail rejects only E_local > 256).
    experts_ok = M.MAX_RUNTIME_EXPERTS == 256
    if not experts_ok:
        print("[FAIL] MAX_RUNTIME_EXPERTS must mirror weave::kMaxRuntimeExperts=256")
    experts_report: Dict[str, List[int]] = {}
    for arch in ("b300", "h20", "mi308x"):
        checked = []
        for experts in (128, 129, 192, 256):
            wl_e = M.WorkloadSpec(experts, 2048, 512)
            m_hist = [e % 5 for e in range(experts)]
            total_e = sum(m_hist)
            for rw in (0, 4):
                b = M.select_policy(arch, wl_e, m_hist, total_e + 13, total_e,
                                    reduce_warps=rw)
                ok = (b.total_tokens == total_e and b.v_head > 0 and
                      b.scatter_mode in (0, 1) and b.t_total_us > 0.0 and
                      math.isfinite(b.t_total_us))
                if not ok:
                    experts_ok = False
                    print(f"[FAIL] E_local={experts} rw={rw} on {arch}: {b}")
            checked.append(experts)
        experts_report[arch] = checked
    report["experts_256"] = {
        "max_runtime_experts": M.MAX_RUNTIME_EXPERTS,
        "local_experts_checked": experts_report,
        "pass": experts_ok,
    }
    report["pass"] = report["pass"] and experts_ok

    # 2. Replay each real routing aggregate across every architecture.
    by_arch: Dict[str, List[Dict]] = {}
    for arch, ep, precision in configs:
        rows = []
        for cell in routing["cells"]:
            if ep == 16 and cell["model"] == "qwen38max":
                local_experts = cell["experts"] // ep
            else:
                local_experts = cell["experts"] // ep
            m_proxy = zipf_histogram(cell["local_contributions"], local_experts)
            wl = M.WorkloadSpec(local_experts, cell["hidden"],
                                INTERMEDIATE[cell["model"]])
            dedup_pulls = cell["local_contributions"] + cell["dedup_remote_scatter_rows"]
            scatter_count = cell["baseline_remote_scatter_rows"]
            b = M.select_policy(arch, wl, m_proxy, dedup_pulls, scatter_count,
                                precision_mode=precision)
            cand = candidate_by_mask(arch, b.r_mask)
            assert b.total_tokens == sum(m_proxy)
            assert b.v_head > 0
            assert b.scatter_mode in (0, 1)
            rows.append({
                "model": cell["model"],
                "S": cell["S"],
                "ep": ep,
                "precision_mode": precision,
                "local_experts": local_experts,
                "total_tokens_M": b.total_tokens,
                "dedup_pulls_Din": dedup_pulls,
                "scatter_rows_Mout": scatter_count,
                "selected_r_mask": b.r_mask,
                "selected_comm_per_sm": cand.comm_units,
                "selected_comp_per_sm": cand.comp_units,
                "selected_block_m": cand.block_m,
                "selected_scatter_mode": b.scatter_mode,
                "selected_v_head": b.v_head,
                "predicted_mid_us": round(b.t_mid_us, 4),
                "predicted_total_us": round(b.t_total_us, 4),
                "m_e_histogram_source": "zipf0.9_proxy_from_real_local_contributions",
            })

        key = f"{arch}_ep{ep}" if precision == "fp8xfp4" else f"{arch}_ep{ep}_{precision}"
        by_arch[key] = rows
        scatter_counts = {0: sum(x["selected_scatter_mode"] == 0 for x in rows),
                          1: sum(x["selected_scatter_mode"] == 1 for x in rows)}
        report["platforms"].append({
            "platform": key,
            "evaluated_cells": len(rows),
            "selection_counts": {str(k): v for k, v in scatter_counts.items()},
            "rows": rows,
        })

    # 3. Paper-consistent qualitative checks where the routing replay is real.
    b300 = by_arch["b300_ep8"]
    small = [r for r in b300 if r["model"] == "qwen38max" and r["S"] == 128]
    large = [r for r in b300 if r["model"] == "qwen38max" and r["S"] >= 1024]
    if not (len(small) == 1 and small[0]["selected_scatter_mode"] == 0):
        report["pass"] = False
        print("[FAIL] B300 small-S paper trend: expected s=epi")
    if not (large and all(r["selected_scatter_mode"] == 1 for r in large)):
        report["pass"] = False
        print("[FAIL] B300 large-S paper trend: expected s=comm")

    mi = by_arch["mi308x_ep16"]
    if not all(r["selected_scatter_mode"] == 1 for r in mi):
        report["pass"] = False
        print("[FAIL] MI308X EP16 expected comm scatter on all replay cells")

    print(f"[PASS] C++ golden cross-check: {golden_ok}")
    print(f"[PASS] WEAVE_REDUCE_WARPS 0/2/4 cross-language golden + invariants: {reduce_ok}")
    for arch, row in reduce_report.items():
        print(f"       {arch}: total_us={row['predicted_total_us']}")
    print(f"[PASS] E_local<=256 (128/129/192/256) selection: {experts_ok}")
    for p in report["platforms"]:
        print(f"[PASS] {p['platform']}: cells={p['evaluated_cells']} "
              f"selection_counts={p['selection_counts']}")
    print(f"OVERALL={'PASS' if report['pass'] else 'FAIL'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "cost_model_validation_3hw.json"))
    args = parser.parse_args()
    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"WROTE {out}")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

