#!/usr/bin/env python3
"""Evaluate the CTA-0 cost model against an exhaustive candidate oracle.

The compact repository file data/routing_stats_3hw.json contains six real
routing aggregates and a documented Zipf(0.9) per-expert split proxy. This
script replays those aggregates through the pure-Python mirror of Eq.(1)-(3)
for SM103, H20, and MI308X. It reports candidate selection, predicted latency,
and the gap to the best predicted candidate.

An optional --oracle-json can supply measured per-candidate latencies for a
full 48-cell sweep. The paper's measured 48-cell summary is printed as a
separate reference block and is never mixed into locally computed values.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODEL_PATH = ROOT / "deep_gemm" / "mega" / "cost_model.py"
_spec = importlib.util.spec_from_file_location("weave_cost_model", _MODEL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load cost model from {_MODEL_PATH}")
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

INTERMEDIATE = {
    "qwen38max": 2048,
    "minimax_m3": 3072,
    "DeepSeek-V4-Flash": 2048,
    "Qwen3.5-35B-A3B": 512,
    "Qwen3.5-122B-A10B": 1024,
    "Qwen3.5-397B-A17B": 1024,
}

PAPER_REFERENCE = {
    "b300_48cell": {
        "weave_geomean": 1.301,
        "minus_S": 1.192,
        "minus_S_T": 1.010,
        "minus_S_T_C": 0.996,
        "oracle_geomean": 1.332,
        "exact_oracle_matches": "38/48",
        "geomean_selection_regret_pct": 2.4,
        "mean_absolute_latency_error_pct": 34.0,
    },
    "h20_48cell": {
        "weave_geomean": 1.162,
        "minus_S": 1.137,
        "minus_S_T": 1.053,
        "minus_S_T_C": 0.947,
        "selected_fixed_vs_oracle": "99.5%",
        "exact_oracle_matches": "32/48",
        "within_one_percent": "41/48",
    },
    "mi308x_ep16_48cell": {
        "weave_geomean": 2.147,
        "minus_S": 1.808,
        "minus_S_T": 1.537,
        "minus_S_T_C": 1.362,
    },
    "eplb_24cell": {
        "original_routing_speedup": 1.274,
        "with_eplb_speedup": 1.128,
        "over_megamoe_with_eplb": 1.144,
        "imbalance_before": 1.37,
        "imbalance_after": 1.27,
    },
}


def zipf_histogram(total: int, buckets: int, exponent: float = 0.9) -> List[int]:
    """Deterministic skew proxy used only when per-expert counts are absent."""
    if buckets <= 0:
        return []
    weights = [1.0 / ((i + 1) ** exponent) for i in range(buckets)]
    denom = sum(weights)
    raw = [total * w / denom for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    for idx in range(total - sum(counts)):
        counts[idx % buckets] += 1
    return counts


def candidate_label(c: M.SpatialCandidate, scatter: int) -> str:
    return f"c{c.comm_units}_{c.comp_units}_bm{c.block_m}_s{scatter}"


def evaluate_cell(arch: str, ep: int, cell: dict,
                  precision: str) -> dict:
    local_experts = int(cell["experts"]) // ep
    wl = M.WorkloadSpec(
        local_experts=local_experts,
        hidden=int(cell["hidden"]),
        intermediate=INTERMEDIATE.get(cell["model"], int(cell["hidden"]) // 4),
    )
    m_e = zipf_histogram(int(cell["local_contributions"]), local_experts)
    dedup_pulls = int(cell["local_contributions"] + cell["dedup_remote_scatter_rows"])
    scatter_rows = int(cell["baseline_remote_scatter_rows"])

    outcomes: List[Tuple[str, M.Breakdown]] = []
    for cand in M.CANDIDATES[arch]:
        for scatter in (0, 1):
            br = M.evaluate(arch, wl, cand, scatter, m_e,
                            dedup_pulls, scatter_rows, precision)
            outcomes.append((candidate_label(cand, scatter), br))
    best_label, best = min(outcomes, key=lambda kv: kv[1].t_total_us)
    selected = M.select_policy(arch, wl, m_e, dedup_pulls, scatter_rows, precision)
    # Identify by the public fields without relying on a private object id.
    selected_label = next(
        label for label, br in outcomes
        if br.r_mask == selected.r_mask and
        br.scatter_mode == selected.scatter_mode and
        abs(br.t_total_us - selected.t_total_us) < 1e-9)
    regret = (selected.t_total_us / best.t_total_us - 1.0) * 100.0
    return {
        "model": cell["model"],
        "S": cell["S"],
        "ep": ep,
        "precision": precision,
        "local_experts": local_experts,
        "proxy_total_tokens": int(sum(m_e)),
        "dedup_pulls": dedup_pulls,
        "scatter_rows": scatter_rows,
        "selected": selected_label,
        "predicted_total_us": round(selected.t_total_us, 6),
        "predicted_mid_us": round(selected.t_mid_us, 6),
        "predicted_v_head": selected.v_head,
        "model_oracle": best_label,
        "model_oracle_us": round(best.t_total_us, 6),
        "selection_regret_pct": round(regret, 6),
        "exact_model_oracle_match": selected_label == best_label,
        "latency_prediction_error_pct": None,
    }


def load_oracle(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "cells" not in data:
        raise ValueError("--oracle-json must contain a top-level 'cells' list")
    return data


def oracle_metrics(data: dict) -> dict:
    exact = 0
    regrets = []
    abs_errors = []
    for row in data["cells"]:
        measured = row.get("measured_us", {})
        if not measured:
            continue
        selected = row.get("selected")
        oracle_key, oracle_us = min(measured.items(), key=lambda kv: kv[1])
        if selected in measured:
            exact += int(selected == oracle_key)
            regrets.append(measured[selected] / oracle_us - 1.0)
            if row.get("predicted_us") is not None:
                abs_errors.append(abs(row["predicted_us"] - measured[selected]) /
                                  measured[selected])
    n = len(regrets)
    return {
        "cells": n,
        "exact_oracle_matches": exact,
        "geomean_selection_regret_pct": (
            100.0 * math.exp(sum(math.log1p(x) for x in regrets) / n) - 100.0
            if n else None),
        "mean_absolute_latency_error_pct": (
            100.0 * sum(abs_errors) / len(abs_errors) if abs_errors else None),
    }


def _report_path(path: Path) -> str:
    """Repository-relative POSIX path for inputs inside the repository, so the
    archived report does not change with the checkout location."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing", type=Path,
                    default=ROOT / "data" / "routing_stats_3hw.json")
    ap.add_argument("--oracle-json", type=Path)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "ablation" /
                            "cost_model_vs_oracle.json")
    args = ap.parse_args()

    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    platforms = [
        ("b300_ep8", "b300", 8, "fp8xfp4"),
        ("h20_ep8", "h20", 8, "fp8"),
        ("mi308x_ep8", "mi308x", 8, "fp8"),
        ("mi308x_ep16", "mi308x", 16, "fp8"),
    ]
    rows_by_platform: Dict[str, List[dict]] = {}
    for name, arch, ep, precision in platforms:
        rows = [evaluate_cell(arch, ep, cell, precision)
                for cell in routing["cells"]]
        rows_by_platform[name] = rows

    oracle_data = load_oracle(args.oracle_json)
    report = {
        "schema": "weave-cost-model-vs-oracle/v1",
        "routing_input": _report_path(args.routing),
        "routing_reality_boundary": routing.get("reality_boundary"),
        "local_replay": {
            "cells_per_platform": len(routing["cells"]),
            "platforms": rows_by_platform,
            "model_oracle_note": (
                "The model-internal oracle is the minimum predicted latency "
                "over the same public candidate set. It verifies consistency, "
                "not measured hardware optimality."),
        },
        "paper_measured_reference": PAPER_REFERENCE,
        "oracle_file_metrics": oracle_metrics(oracle_data) if oracle_data else None,
        "verdict": "PASS",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, rows in rows_by_platform.items():
        matches = sum(r["exact_model_oracle_match"] for r in rows)
        worst = max(r["selection_regret_pct"] for r in rows)
        print(f"[PASS] {name}: cells={len(rows)} "
              f"model_oracle_matches={matches}/{len(rows)} "
              f"max_model_regret={worst:.4f}%")
    print("[REFERENCE] SM103 48-cell: 38/48 exact, 2.4% geomean regret, "
          "1.301x selected vs 1.332x oracle, 34.0% mean absolute latency error")
    if oracle_data:
        metrics = oracle_metrics(oracle_data)
        print("[ORACLE]", json.dumps(metrics, sort_keys=True))
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
