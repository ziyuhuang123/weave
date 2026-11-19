#!/usr/bin/env python3
"""Static gate for the in-kernel CTA-0 policy integrations.

This is a source-level wiring audit, not a compiler or GPU result.  It checks
that each backend contains the publication rail, consumes the selected policy,
and (negatively) that known scratch-buffer aliases are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "b300": ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
    "b300_nvfp4": ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh",
    "h20": ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh",
    "mi308x": ROOT / "deep_gemm/include/deep_gemm/impls/gfx942_fp8_mega_moe.cuh",
    "mi308x_body": ROOT / "deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip",
    "mi308x_route": ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp",
    "mi308x_comm": ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp",
}

# (label, needle, expected_count).  expected_count == 0 is a negative check.
CHECKS = {
    # B300 uses the SM100 precision-selectable MegaMoE scheduler directly.
    # The CTA-0 publication rail is wired on H20 and MI308X, not on these
    # SM100 kernels; keep that architectural split explicit and auditable.
    "b300": [
        ("scheduler_include", "scheduler/mega_moe.cuh", 1),
        ("entry", "sm100_fp8_fp4_mega_moe_impl", 1),
        ("scheduler", "sched::MegaMoEScheduler", 1),
        ("no_policy_rail", "weave_policy_runtime.cuh", 0),
    ],
    "b300_nvfp4": [
        ("scheduler_include", "scheduler/nvfp4_mega_moe.cuh", 1),
        ("entry", "sm100_fp4_fp4_mega_moe_impl", 1),
        ("scheduler", "sched::nvfp4::MegaMoEScheduler", 1),
        ("no_policy_rail", "weave_policy_runtime.cuh", 0),
    ],
    "h20": [
        ("include", "weave_policy_runtime.cuh", 1),
        ("cta0", "cta0_select_from_packed_hist", 1),
        ("live_m_out", "count_remote_expert_sends", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 2),
        ("no_literal_cr", "`r`n", 0),
    ],
    "mi308x_body": [
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("all_cta_read", "g_weave_policy_control", 1),
    ],
    "mi308x_route": [
        ("cta0", "cta0_select_and_publish", 1),
        ("cta0_guard", "if (cta == 0u)", 1),
        ("histogram", "plan_counts", 1),
        ("publish_before_release", "pf + 6u", 1),
        ("no_trace_scratch", "trace[2048u]", 0),
    ],
    "mi308x_comm": [
        ("consume_v_head", "selected_v_head", 1),
        ("policy_valid", "weave_policy_ok", 1),
    ],
}


def audit() -> dict:
    report = {"schema": "weave-kernel-policy-wiring/v3", "pass": True, "files": {}}
    for name, checks in CHECKS.items():
        path = FILES[name]
        if not path.exists():
            report["pass"] = False
            report["files"][name] = {"exists": False, "checks": {}}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = {}
        for label, needle, expected in checks:
            count = text.count(needle)
            ok = (count >= expected) if expected > 0 else (count == 0)
            rows[label] = {
                "needle": needle,
                "expected_count": expected,
                "observed_count": count,
                "ok": ok,
            }
            if not ok:
                report["pass"] = False
        report["files"][name] = {
            "exists": True,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "checks": rows,
        }
    return report


def main() -> None:
    report = audit()
    out = ROOT / "results" / "kernel_policy_wiring.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, row in report["files"].items():
        passed = row.get("exists") and all(v["ok"] for v in row["checks"].values())
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"WROTE {out}")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
