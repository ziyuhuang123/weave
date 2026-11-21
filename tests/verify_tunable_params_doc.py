#!/usr/bin/env python3
"""Publication gate for tunable-parameter documentation.

The gate verifies four independent properties:

1. Every path:line citation in README.md, README_zh.md and
   docs/TUNABLE_PARAMETERS.md resolves to a line containing the parameter or
   documented kernel-side symbol named by that table row.
2. Public documentation contains no archived project codeword or BOM, and no
   obsolete hardware codeword in prose, headings, table headers or captions.
   Fenced code blocks, indented command blocks and inline code/path literals are
   excluded from the prose scan only because they are executable literals such
   as baselines/run_b300_baselines.py that cannot be renamed without breaking
   the launch surface.
3. Every required result figure is referenced by both READMEs, exists, and is a
   structurally valid PNG.
4. The obsolete docs/figures/weaveprofiler_b300_timeline.png file is absent, so
   even the figure filename contains no obsolete hardware codeword.
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "README_zh.md",
    ROOT / "docs" / "TUNABLE_PARAMETERS.md",
]
SCAN_DOCS = DOCS + [
    ROOT / "baselines" / "README.md",
    ROOT / "baselines" / "README_zh.md",
    ROOT / "ablation" / "README.md",
    ROOT / "ablation" / "README_zh.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "README_zh.md",
]
README_DOCS = [ROOT / "README.md", ROOT / "README_zh.md"]
REQUIRED_FIGURES = [
    "docs/figures/weave_design_overview.png",
    "docs/figures/weave_design_rolegraph.png",
    "docs/figures/weave_eval_3hw_speedup.png",
    "docs/figures/weave_bound_utilization_3hw.png",
    "docs/figures/weaveprofiler_sm103_timeline.png",
    "docs/figures/weaveprofiler_h20_timeline.png",
    "docs/figures/weaveprofiler_mi308x_timeline.png",
]
OBSOLETE_FIGURES = [
    "docs/figures/weaveprofiler_b300_timeline.png",
]

CITATION_RE = re.compile(r"\`([^\`\s]+):(\d+)\`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FENCED_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"\`[^\`\n]*\`")

FORBIDDEN = [
    "\ufeff", "v33", "v28", "v20", "r31", "V1864", "D1-D7", "D1–D7",
    "DG_TUNE_D", "DG_ISOLATED_", "DG_MOE_", "AGY_", "kAgy",
    "DELIVERY_20261116", "Phase-2", "Phase-3", "Phase-4", "Phase-5",
    "Phase 2", "Phase 3", "Phase 4", "Phase 5", "3050", "RTX",
]

# Hardware naming and stale-figure rules apply to prose only, case-sensitively,
# so C++ symbols and executable path literals are not false positives.
PROSE_FORBIDDEN = [
    "B300", "L20A", "L20D", "17 作者", "17-author", "dynamicMK", "Residency",
]

REQUIRED = """
WEAVE_DISPATCH_ACTIVE_SMS WEAVE_SCATTER_ACTIVE_SMS WEAVE_SM0_STREAM_SCHED
WEAVE_DISPATCH_WARPS WEAVE_SCATTER_WARPS
WEAVE_CU_WAVE_SPLIT WEAVE_REG_BALANCE
WEAVE_V_HEAD WEAVE_READY_GROUP_SIZE WEAVE_READY_QUEUE
WEAVE_SCATTER_OWNER WEAVE_COMBINE_SYNC_CHUNK
WEAVE_PRECISION_MODE WEAVE_ADAPT_BM
WEAVE_BLOCK_M_L1 WEAVE_BLOCK_M_L2
WEAVE_REG_PULL_BYTES WEAVE_DEDUP_MODE
WEAVE_SCATTER_STAGING WEAVE_LOCAL_SCATTER_DEDUP
""".split()


def knob_tokens(cell: str) -> List[str]:
    return [t for t in IDENT_RE.findall(cell)
            if t.startswith("WEAVE_") or t in {"mma_type", "precision_mode"}]


def prose_only(text: str) -> str:
    """Remove executable literals before the case-sensitive prose scan."""
    text = FENCED_RE.sub("", text)
    kept = []
    for line in text.splitlines():
        if line.startswith("    ") or line.startswith("\t"):
            continue
        kept.append(line)
    text = "\n".join(kept)
    return INLINE_CODE_RE.sub("", text)


def check_citations(path: Path) -> Tuple[Dict[str, int], List[str]]:
    errors: List[str] = []
    rows = citations = 0
    seen = set()
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        knob = cells[0]
        if knob.lower() in {"parameter", "knob", "hardware"}:
            continue
        tokens = knob_tokens(knob)
        if not tokens:
            continue
        seen.update(tokens)
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
                errors.append(
                    f"{path.name}:{lineno}: {rel}:{n} out of range "
                    f"(1..{len(src_lines)})")
                continue
            src = src_lines[n - 1]
            if not any(re.search(rf"{re.escape(t)}(?![A-Za-z0-9_])", src)
                       for t in tokens):
                errors.append(
                    f"{path.name}:{lineno}: {rel}:{n} does not contain "
                    f"any expected symbol {tokens}; line={src.strip()[:160]!r}")
    total_refs = len(CITATION_RE.findall(text))
    if citations != total_refs:
        errors.append(
            f"{path.name}: {total_refs - citations} citation(s) are outside "
            "parameter table rows and were not verified")
    missing = [k for k in REQUIRED if k not in seen]
    for k in missing:
        errors.append(f"{path.name}: required parameter {k} is not documented")
    return {"rows": rows, "citations": citations, "parameters": len(seen)}, errors


def check_forbidden(path: Path) -> List[str]:
    raw = path.read_bytes()
    errors: List[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"{path.name}: UTF-8 BOM at byte 0")
    text = raw.decode("utf-8", errors="replace")
    if "\ufeff" in text:
        errors.append(f"{path.name}: UTF-8 BOM character present")
    folded = text.casefold()
    for token in FORBIDDEN:
        if token == "﻿":
            continue
        if token.casefold() in folded:
            errors.append(f"{path.name}: forbidden token {token!r}")
    prose = prose_only(text)
    for token in PROSE_FORBIDDEN:
        if token in prose:
            errors.append(f"{path.name}: forbidden prose token {token!r}")
    return errors


def png_header(path: Path) -> Tuple[int, int]:
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    length, chunk = struct.unpack(">I4s", raw[8:16])
    if chunk != b"IHDR" or length != 13:
        raise ValueError("missing IHDR")
    width, height = struct.unpack(">II", raw[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("invalid dimensions")
    return width, height


def check_figures() -> List[str]:
    errors: List[str] = []
    for readme in README_DOCS:
        text = readme.read_text(encoding="utf-8")
        refs = set(IMAGE_RE.findall(text))
        for rel in REQUIRED_FIGURES:
            if rel not in refs:
                errors.append(f"{readme.name}: missing figure reference {rel}")
            target = ROOT / rel
            if not target.exists():
                errors.append(f"{readme.name}: missing figure file {rel}")
                continue
            try:
                png_header(target)
            except Exception as exc:
                errors.append(f"{readme.name}: invalid PNG {rel}: {exc}")
    for rel in REQUIRED_FIGURES:
        target = ROOT / rel
        if target.exists():
            try:
                png_header(target)
            except Exception as exc:
                errors.append(f"invalid PNG {rel}: {exc}")
    for rel in OBSOLETE_FIGURES:
        if (ROOT / rel).exists():
            errors.append(f"obsolete figure still present: {rel}")
    return errors


def main() -> None:
    errors: List[str] = []
    for path in SCAN_DOCS:
        if not path.exists():
            errors.append(f"missing documentation file: {path}")
            continue
        errors.extend(check_forbidden(path))

    total_rows = total_citations = 0
    for path in DOCS:
        if not path.exists():
            errors.append(f"missing citation document: {path}")
            continue
        stats, row_errors = check_citations(path)
        total_rows += stats["rows"]
        total_citations += stats["citations"]
        errors.extend(row_errors)
        print(f"[{'PASS' if not row_errors else 'FAIL'}] {path.name}: "
              f"rows={stats['rows']} citations={stats['citations']} "
              f"parameters={stats['parameters']}")

    figure_errors = check_figures()
    errors.extend(figure_errors)
    print(f"[{'PASS' if not figure_errors else 'FAIL'}] figures="
          f"{len(REQUIRED_FIGURES)} referenced by both READMEs; "
          f"obsolete={len(OBSOLETE_FIGURES)}")
    forbidden_hits = sum(1 for e in errors if "forbidden" in e)
    print(f"[{'PASS' if not forbidden_hits else 'FAIL'}] "
          f"forbidden-token/BOM/prose scan: {len(SCAN_DOCS)} documents")
    print(f"TOTAL rows={total_rows} citations={total_citations}")

    if errors:
        for error in errors:
            print("  ERROR:", error)
        print(f"TOTAL ERRORS: {len(errors)}")
        raise SystemExit(1)
    print("ALL TUNABLE-PARAMETER CITATIONS, BLACKLIST AND PNG GATES PASS")


if __name__ == "__main__":
    main()
