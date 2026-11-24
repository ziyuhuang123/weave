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
    "b300_bf16": ROOT / "deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh",
    "h20": ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh",
    "mi308x": ROOT / "deep_gemm/include/deep_gemm/impls/gfx942_fp8_mega_moe.cuh",
    "mi308x_body": ROOT / "deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip",
    "mi308x_route": ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp",
    "mi308x_comm": ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp",
    "nvfp4_host": ROOT / "csrc/jit_kernels/heuristics/nvfp4_mega_moe.hpp",
    "bf16_host": ROOT / "csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp",
}

# Additional files concatenated into the audited text of a label.
EXTRA_FILES = {
    "mi308x": [
        ROOT / "deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip",
        ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp",
        ROOT / "deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp",
    ],
    "nvfp4_host": [
        ROOT / "csrc/jit_kernels/impls/sm100_fp4_fp4_mega_moe.hpp",
        ROOT / "csrc/apis/nvfp4_mega_moe.hpp",
    ],
    "bf16_host": [
        ROOT / "csrc/jit_kernels/heuristics/mega_moe.hpp",
    ],
}

# (label, needle, expected_count).  expected_count == 0 is a negative check.
CHECKS = {
    # B300 uses the SM100 precision-selectable MegaMoE scheduler. Both SM100
    # precision paths carry the same CTA-0 publication rail as H20/MI308X:
    # one CTA-0 selector, a release/acquire epoch, and per-CTA consumption.
    "b300": [
        ("scheduler_include", "scheduler/mega_moe.cuh", 1),
        ("entry", "sm100_fp8_fp4_mega_moe_impl", 1),
        ("scheduler", "sched::MegaMoEScheduler", 1),
        ("include", "weave_policy_runtime.cuh", 1),
        ("cta0", "cta0_select_from_packed_hist", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 1),
        ("epoch_wait", "wait_and_copy_policy", 1),
    ],
    "b300_nvfp4": [
        ("scheduler_include", "scheduler/nvfp4_mega_moe.cuh", 1),
        ("entry", "sm100_fp4_fp4_mega_moe_impl", 1),
        ("scheduler", "sched::nvfp4::MegaMoEScheduler", 1),
        ("include", "weave_policy_runtime.cuh", 1),
        ("cta0", "cta0_select_from_packed_hist", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 1),
        ("epoch_wait", "wait_and_copy_policy", 1),
        ("consume_sm0_sched", "WEAVE_SM0_STREAM_SCHED", 1),
        ("consume_dispatch_sms", "WEAVE_DISPATCH_ACTIVE_SMS", 1),
        ("assert_dispatch_warps", "WEAVE_DISPATCH_WARPS disagrees with the launch thread layout", 1),
        ("assert_ring_blocks", "WEAVE_RING_BLOCKS must match the allocated ring", 1),
    ],
    "b300_bf16": [
        ("scheduler_include", "scheduler/mega_moe.cuh", 1),
        ("entry", "sm100_bf16_mega_moe_impl", 1),
        ("scheduler", "sched::MegaMoEScheduler", 1),
        ("include", "weave_policy_runtime.cuh", 1),
        ("cta0", "cta0_select_from_packed_hist", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 1),
        ("epoch_wait", "wait_and_copy_policy", 1),
        ("consume_sm0_sched", "WEAVE_SM0_STREAM_SCHED", 1),
        ("consume_dispatch_sms", "WEAVE_DISPATCH_ACTIVE_SMS", 1),
        ("assert_dispatch_warps", "WEAVE_DISPATCH_WARPS disagrees with the launch thread layout", 1),
        ("assert_ring_blocks", "WEAVE_RING_BLOCKS must match the allocated ring", 1),
    ],
    "h20": [
        ("include", "weave_policy_runtime.cuh", 1),
        ("cta0", "cta0_select_from_packed_hist", 1),
        ("cta0_smem", "cta0_select_and_publish", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "WEAVE_SCATTER_OWNER", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 1),
        ("no_literal_cr", "`r`n", 0),
    ],
    # The MI308X wrapper is include-only; audit the wrapper plus the two
    # implementation headers it pulls in so the public backend name has a
    # complete rail check of its own.
    "mi308x": [
        ("include_body", "amd/megamoe_amd.hip", 1),
        ("include_policy", "weave_policy_runtime.cuh", 1),
        ("cta0_u32", "cta0_select_and_publish", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("consume_v_head", "selected_v_head", 1),
        ("all_cta_read", "load_published_policy", 1),
    ],
    "mi308x_body": [
        ("include", "weave_policy_runtime.cuh", 1),
        ("consume_r", "selected_r_mask", 1),
        ("consume_scatter", "selected_scatter_mode", 1),
        ("all_cta_read", "g_weave_policy_control", 1),
        ("load_published", "load_published_policy", 1),
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
    # Host/JIT dispatchers: the unified WEAVE_* request must be resolved from
    # the environment and mirrored into the kernel translation unit, not
    # silently dropped between Python and the compiled kernel.
    "nvfp4_host": [
        ("from_env", "weave_knobs::from_env", 1),
        ("dispatch_warps_resolver", "weave::resolve_dispatch_warps", 1),
        ("block_m_envelope", "resolve_block_m_envelope", 1),
        ("ring_cap", "knobs.ring_blocks", 1),
        ("define_precision", "#define WEAVE_PRECISION_MODE", 1),
        ("define_v_head", "#define WEAVE_V_HEAD", 1),
        ("define_dispatch_warps", "#define WEAVE_DISPATCH_WARPS", 1),
        ("define_sm0_sched", "#define WEAVE_SM0_STREAM_SCHED", 1),
        ("define_dispatch_sms", "#define WEAVE_DISPATCH_ACTIVE_SMS", 1),
        ("define_ring_blocks", "#define WEAVE_RING_BLOCKS", 1),
        ("define_block_m_l1", "#define WEAVE_BLOCK_M_L1", 1),
        ("define_block_m_l2", "#define WEAVE_BLOCK_M_L2", 1),
    ],
    "bf16_host": [
        ("from_env", "weave_knobs::from_env", 1),
        ("define_precision", "#define WEAVE_PRECISION_MODE", 1),
        ("define_v_head", "#define WEAVE_V_HEAD", 1),
        ("define_dispatch_warps", "#define WEAVE_DISPATCH_WARPS", 1),
        ("define_sm0_sched", "#define WEAVE_SM0_STREAM_SCHED", 1),
        ("define_dispatch_sms", "#define WEAVE_DISPATCH_ACTIVE_SMS", 1),
        ("define_ring_blocks", "#define WEAVE_RING_BLOCKS", 1),
        ("define_block_m_l1", "#define WEAVE_BLOCK_M_L1", 1),
        ("define_block_m_l2", "#define WEAVE_BLOCK_M_L2", 1),
    ],
}


def audit() -> dict:
    report = {"schema": "weave-kernel-policy-wiring/v4", "pass": True, "files": {}}
    for name, checks in CHECKS.items():
        path = FILES[name]
        if not path.exists():
            report["pass"] = False
            report["files"][name] = {"exists": False, "checks": {}}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for extra in EXTRA_FILES.get(name, []):
            text += "\n" + extra.read_text(encoding="utf-8", errors="replace")
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
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    for name, row in report["files"].items():
        passed = row.get("exists") and all(v["ok"] for v in row["checks"].values())
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"WROTE {out}")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
