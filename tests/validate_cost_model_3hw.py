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
        "schema": "weave-cost-model-validation-3hw/v3",
        "model": "deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh",
        "python_mirror": "deep_gemm/mega/cost_model.py",
        "routing_source": routing["source"],
        "routing_reality_boundary": routing["reality_boundary"],
        "cpp_golden_match": False,
        "precision_modes": {},
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

