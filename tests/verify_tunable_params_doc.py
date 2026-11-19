#!/usr/bin/env python3
"""Verify every file:line citation in the tunable-parameter documentation.

Each markdown table row has the form

    | `KNOB` / `KERNEL_SYMBOL` | ... | `path:line` | `path:line` |

The verifier resolves every `path:line` citation and requires the cited source
line to contain at least one identifier named in the row's Knob column.  This
keeps the documentation mechanically tied to the implementation instead of
accepting stale or invented line numbers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "TUNABLE_PARAMETERS.md"]

CITATION_RE = re.compile(r"`([^`\s]+):(\d+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Knobs that the Phase-2 contract requires to be documented.  The check is
# intentionally name-based so a row can list a host macro and its kernel-side
# alias together.
REQUIRED = [
    "WEAVE_ABLATION_PRESET", "WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL",
    "WEAVE_KNOB_COMM_UNITS", "WEAVE_KNOB_COMP_UNITS",
    "WEAVE_KNOB_DUALGROUP_MIN_TOKENS", "WEAVE_KNOB_BM_TIER_MODE",
    "WEAVE_KNOB_BN1", "WEAVE_KNOB_BN2_MODE", "WEAVE_KNOB_BK",
    "WEAVE_KNOB_S2_NGROUP_MODE", "WEAVE_KNOB_HETERO_SM_MASK",
    "WEAVE_KNOB_OVERLAP_PIPELINE", "WEAVE_KNOB_SCATTER_MODE",
    "WEAVE_KNOB_GEMM_STAGES", "WEAVE_KNOB_ALL_WAVE_COMBINE",
    "WEAVE_KNOB_SPIN_BEFORE_SLEEP", "WEAVE_KNOB_SELECTIVE_DEDUP",
    "WEAVE_KNOB_REGISTER_DIRECT_COMM", "WEAVE_KNOB_PARALLEL_ROUTE_PLAN",
    "WEAVE_KNOB_LOCAL_FIRST_ORDER", "WEAVE_KNOB_FUSED_SWIGLU_REQUANT",
    "WEAVE_KNOB_TAIL_FINE_GRAINED", "WEAVE_KNOB_V_HEAD",
    "WEAVE_KNOB_PRECISION_MODE", "precision_mode", "mma_type",
    "DG_TUNE_D1_G", "DG_TUNE_D2_MODE", "DG_TUNE_D2_REG_PULL_BYTES",
    "DG_ISOLATED_D2_D8_SM_LIMIT", "DG_ISOLATED_D2_DISP_REGS",
    "DG_ISOLATED_D2_EPI_REGS", "DG_TUNE_D3_BM", "DG_TUNE_D3_BN",
    "DG_TUNE_D3_BK", "DG_TUNE_D3_STORE_M", "DG_TUNE_D3_DYN_EPI",
    "DG_TUNE_D3_DYN_TMA_A", "DG_TUNE_D3_OVERLAP_SMEM",
    "DG_D3_ACCUM_STAGES", "DG_D3_TMA_STORE_STAGES", "DG_D3_SCHED_STAGES",
    "DG_TUNE_D4_EPW", "DG_TUNE_D4_STATIC", "DG_TUNE_D4_READYQ",
    "DG_TUNE_D4_MODE", "DG_TUNE_D5_DEDUP", "DG_TUNE_D5_SCHED",
    "DG_TUNE_D6_G", "DG_TUNE_D6_REDUCE_WARPS", "DG_TUNE_D7_SCATTER_MODE",
    "DG_TUNE_D7_SMEM_ALIAS", "DG_TUNE_D7_WARPS", "DG_TUNE_D7_HANDOFF_WAIT",
    "DG_SHARED_L2_EARLY_BLOCKS", "kEnableMulticast",
    "DG_MOE_SWAPAB_AUTO", "DG_MOE_SWAPAB_EPI_WG", "DG_MOE_SWAPAB_MAX_LOAD",
    "DG_MOE_SWAPAB_PAD_PCT", "DG_MOE_SWAPAB_MIN_LOAD", "kSwapSFSubBlocks",
    "AGY_MCAST", "AGY_SWIZZLE_TILES", "AGY_HOT_COLD_L1",
    "AGY_DEDUP_DISP", "AGY_DEDUP_SCAT", "AGY_REG_COMB",
    "AGY_ALL_IDLE_COMB", "AGY_DYN_SCHED", "AGY_D_MASK_MODE",
    "WEAVE_SCATTER_MODE", "WEAVE_KNOB_BN2_MODE",
    "WEAVE_TH_BM64", "WEAVE_TH_BM128", "WEAVE_PARALLEL_ROUTE_PLAN",
    "V1864Roles", "EP16_SLOT_CHUNK", "ep16t_ret_chunk",
    "meta_chunk_slots", "FI_LEVEL",
]


def knob_tokens(cell: str) -> List[str]:
    tokens = IDENT_RE.findall(cell)
    return [t for t in tokens if t not in {"struct", "field", "Python", "C"}]


def check_file(path: Path) -> Tuple[Dict, List[str]]:
    errors: List[str] = []
    rows = 0
    citations = 0
    seen_knobs = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        knob = cells[0]
        if knob.lower() in {"knob", "hardware"}:
            continue
        tokens = knob_tokens(knob)
        if not tokens:
            continue
        seen_knobs.update(tokens)
        refs = CITATION_RE.findall(line)
        if not refs:
            continue
        rows += 1
        for rel, line_s in refs:
            citations += 1
            target = (ROOT / rel).resolve()
            try:
                target.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{path.name}:{lineno}: citation escapes repository: {rel}")
                continue
            if not target.exists():
                errors.append(f"{path.name}:{lineno}: missing file {rel}")
                continue
            src_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            n = int(line_s)
            if n < 1 or n > len(src_lines):
                errors.append(f"{path.name}:{lineno}: {rel}:{n} out of range (1..{len(src_lines)})")
                continue
            src = src_lines[n - 1]
            # Allow the common -D<MACRO> build-flag spelling.
            if not any(re.search(rf"{re.escape(t)}(?![A-Za-z0-9_])", src)
                       for t in tokens):
                errors.append(
                    f"{path.name}:{lineno}: {rel}:{n} does not contain any of {tokens}; "
                    f"line={src.strip()[:120]!r}")
    missing = [k for k in REQUIRED if k not in seen_knobs]
    for k in missing:
        errors.append(f"{path.name}: required knob {k} is not documented")
    return {"rows": rows, "citations": citations, "knobs": len(seen_knobs)}, errors


def main() -> None:
    all_errors: List[str] = []
    report = {}
    for path in DOCS:
        if not path.exists():
            all_errors.append(f"missing documentation file: {path}")
            continue
        stats, errors = check_file(path)
        report[path.name] = stats
        all_errors.extend(errors)
        print(f"[{'PASS' if not errors else 'FAIL'}] {path.name}: "
              f"rows={stats['rows']} citations={stats['citations']} knobs={stats['knobs']}")
    if all_errors:
        for e in all_errors:
            print("  ERROR:", e)
        print(f"TOTAL ERRORS: {len(all_errors)}")
        raise SystemExit(1)
    print("ALL TUNABLE-PARAMETER file:line CITATIONS VERIFIED")


if __name__ == "__main__":
    main()
