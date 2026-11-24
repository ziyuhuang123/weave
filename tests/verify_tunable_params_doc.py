#!/usr/bin/env python3
"""Publication gate for tunable-parameter documentation.

The gate verifies five independent properties:

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
5. The public surface is exactly the 14-knob / 3-category contract
   (Spatial 5, Temporal 3, Implementation 6): the schema in
   deep_gemm/mega/knobs.py has the exact names in the exact order, the six
   retired knobs are absent from it, every generated document lists exactly
   those 14 names as parameters (and none of the retired names or legacy aliases
   as a parameter), and the retired knobs, the two legacy aliases and the accepted-but-not-public
   WEAVE_ADAPT_GEMM_CONFIG / WEAVE_ADAPT_BM are documented in prose.
"""
from __future__ import annotations

import importlib.util
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
# Retired public knob: the scatter executor is now described only through the
# public 14-knob surface (owner + staging), never the internal
# WEAVE_SCATTER_MODE executor pin.
OBSOLETE_PUBLIC_KNOBS = ["WEAVE_SCATTER_MODE"]

# The 14-knob / 3-category public contract, entered here independently of
# deep_gemm/mega/knobs.py (which is cross-checked against it).  Order matters.
EXPECTED_PUBLIC = [
    ("WEAVE_DISPATCH_ACTIVE_SMS", "Spatial"),
    ("WEAVE_SCATTER_ACTIVE_SMS", "Spatial"),
    ("WEAVE_DISPATCH_WARPS", "Spatial"),
    ("WEAVE_SCATTER_WARPS", "Spatial"),
    ("WEAVE_CU_WAVE_SPLIT", "Spatial"),
    ("WEAVE_V_HEAD", "Temporal"),
    ("WEAVE_READY_GROUP_SIZE", "Temporal"),
    ("WEAVE_SCATTER_OWNER", "Temporal"),
    ("WEAVE_PRECISION_MODE", "Implementation"),
    ("WEAVE_GEMM1_CONFIG", "Implementation"),
    ("WEAVE_GEMM2_CONFIG", "Implementation"),
    ("WEAVE_REG_PULL_BYTES", "Implementation"),
    ("WEAVE_DEDUP_MODE", "Implementation"),
    ("WEAVE_SCATTER_STAGING", "Implementation"),
]
EXPECTED_COUNTS = {"Spatial": 5, "Temporal": 3, "Implementation": 6}
# Retired from the public surface: internal compile-time macros / diagnostics only.
RETIRED_INTERNAL = [
    "WEAVE_REDUCE_WARPS", "WEAVE_COMBINE_SYNC_CHUNK", "WEAVE_REG_BALANCE",
    "WEAVE_LOCAL_SCATTER_DEDUP", "WEAVE_SM0_STREAM_SCHED", "WEAVE_READY_QUEUE",
]
# Legacy environment spelling -> public name.
LEGACY_ALIASES = {
    "WEAVE_BLOCK_M_L1": "WEAVE_GEMM1_CONFIG",
    "WEAVE_BLOCK_M_L2": "WEAVE_GEMM2_CONFIG",
}
# Accepted by the host for backward compatibility but NOT public knobs: never a
# parameter row, never a legacy alias of a public knob.
INTERNAL_COMPAT_ENV = ["WEAVE_ADAPT_GEMM_CONFIG", "WEAVE_ADAPT_BM"]
CATEGORY_CELL = {  # category cell text -> canonical category (English and Chinese docs)
    "Spatial": "Spatial", "Temporal": "Temporal", "Implementation": "Implementation",
    "空间": "Spatial", "时间": "Temporal", "实现": "Implementation",
}

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

REQUIRED = [name for name, _ in EXPECTED_PUBLIC]


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


def expected_symbols(tokens: List[str]) -> List[str]:
    """Knob tokens plus the legacy aliases of renamed knobs (the device-side macros
    and the host fallbacks of a renamed knob still carry the legacy spelling)."""
    out = list(tokens)
    for old, new in LEGACY_ALIASES.items():
        if new in tokens:
            out.append(old)
    return out


def check_row_contract(path: Path, param_rows: List[Tuple[int, List[str], List[str]]],
                       text: str) -> List[str]:
    """14-knob / 3-category contract for the parameter rows of one document."""
    errors: List[str] = []
    name = path.name
    names = [tokens[0] for _, tokens, _ in param_rows]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        errors.append(f"{name}: parameter listed more than once: {dup}")
    if set(names) != set(REQUIRED):
        extra = sorted(set(names) - set(REQUIRED))
        missing = sorted(set(REQUIRED) - set(names))
        errors.append(f"{name}: parameter rows differ from the 14-knob contract "
                      f"(extra={extra}, missing={missing})")
    if len(names) != 14:
        errors.append(f"{name}: expected 14 parameter rows, found {len(names)}")
    category_of = dict(EXPECTED_PUBLIC)
    per_category = {c: 0 for c in EXPECTED_COUNTS}
    for lineno, tokens, cells in param_rows:
        knob = tokens[0]
        cell = CATEGORY_CELL.get(cells[1]) if len(cells) > 1 else None
        if cell is None:
            errors.append(f"{name}:{lineno}: {knob} has unknown category cell {cells[1:2]}")
            continue
        per_category[cell] += 1
        if knob in category_of and cell != category_of[knob]:
            errors.append(f"{name}:{lineno}: {knob} is listed as {cell}, "
                          f"the contract says {category_of[knob]}")
    if per_category != EXPECTED_COUNTS:
        errors.append(f"{name}: per-category rows {per_category} != {EXPECTED_COUNTS}")
    # Parameter order inside the document follows the contract order.
    if names and [n for n in names if n in category_of] != [n for n in REQUIRED if n in names]:
        errors.append(f"{name}: parameter rows are not in the contract order")
    # Retired internal knobs and legacy aliases are never parameter rows, and the
    # retired names do not appear anywhere inside a parameter row.
    for lineno, tokens, cells in param_rows:
        row_text = " ".join(cells)
        if (tokens[0] in RETIRED_INTERNAL or tokens[0] in LEGACY_ALIASES
                or tokens[0] in INTERNAL_COMPAT_ENV):
            errors.append(f"{name}:{lineno}: {tokens[0]} must not be a public parameter row")
        for retired in RETIRED_INTERNAL:
            if re.search(rf"{re.escape(retired)}(?![A-Za-z0-9_])", row_text):
                errors.append(f"{name}:{lineno}: parameter row of {tokens[0]} names the "
                              f"retired internal knob {retired}")
    # Every renamed knob documents its legacy alias in its own row.
    for lineno, tokens, cells in param_rows:
        for old, new in LEGACY_ALIASES.items():
            if tokens[0] == new and not re.search(
                    rf"{re.escape(old)}(?![A-Za-z0-9_])", " ".join(cells)):
                errors.append(f"{name}:{lineno}: {new} row does not document its legacy alias {old}")
    # The retired knobs and the alias mapping are documented in prose (outside rows).
    prose_lines = [ln for ln in text.splitlines() if not ln.startswith("|")]
    prose = "\n".join(prose_lines)
    for retired in RETIRED_INTERNAL:
        if not re.search(rf"{re.escape(retired)}(?![A-Za-z0-9_])", prose):
            errors.append(f"{name}: retired knob {retired} is not documented in the "
                          "retired/internal note")
    for compat in INTERNAL_COMPAT_ENV:
        if not re.search(rf"{re.escape(compat)}(?![A-Za-z0-9_])", prose):
            errors.append(f"{name}: accepted-but-not-public {compat} is not documented in the "
                          "retired/internal note")
    for old, new in LEGACY_ALIASES.items():
        if not any(old in ln and new in ln for ln in prose_lines):
            errors.append(f"{name}: legacy alias {old} -> {new} is not documented in the "
                          "alias list")
    return errors


def check_citations(path: Path) -> Tuple[Dict[str, int], List[str]]:
    errors: List[str] = []
    rows = citations = 0
    seen = set()
    param_rows: List[Tuple[int, List[str], List[str]]] = []
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
        param_rows.append((lineno, tokens, cells))
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
            symbols = expected_symbols(tokens)
            if not any(re.search(rf"{re.escape(t)}(?![A-Za-z0-9_])", src)
                       for t in symbols):
                errors.append(
                    f"{path.name}:{lineno}: {rel}:{n} does not contain "
                    f"any expected symbol {symbols}; line={src.strip()[:160]!r}")
    total_refs = len(CITATION_RE.findall(text))
    if citations != total_refs:
        errors.append(
            f"{path.name}: {total_refs - citations} citation(s) are outside "
            "parameter table rows and were not verified")
    missing = [k for k in REQUIRED if k not in seen]
    for k in missing:
        errors.append(f"{path.name}: required parameter {k} is not documented")
    errors.extend(check_row_contract(path, param_rows, text))
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


def check_obsolete_public_knobs(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    errors: List[str] = []
    for name in OBSOLETE_PUBLIC_KNOBS:
        if name in text:
            errors.append(
                f"{path.name}: obsolete public knob {name} present; use "
                "WEAVE_SCATTER_OWNER / WEAVE_SCATTER_STAGING / "
                "WEAVE_LOCAL_SCATTER_DEDUP instead")
    return errors


def check_schema_contract() -> List[str]:
    """deep_gemm/mega/knobs.py must be exactly the 14-knob / 3-category contract."""
    errors: List[str] = []
    path = ROOT / "deep_gemm" / "mega" / "knobs.py"
    try:
        spec = importlib.util.spec_from_file_location("weave_knob_schema_verify", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # runs the import-time 14/5/3/6 assert
    except Exception as exc:  # noqa: BLE001 - report any import-time contract failure
        return [f"knobs.py failed to import: {exc!r}"]
    got = [(k.name, k.category) for k in module.KNOBS]
    if got != EXPECTED_PUBLIC:
        errors.append(f"knobs.py KNOBS differ from the contract: {got}")
    if len(module.KNOBS) != 14:
        errors.append(f"knobs.py has {len(module.KNOBS)} knobs, expected 14")
    counts = {c: sum(1 for k in module.KNOBS if k.category == c) for c in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        errors.append(f"knobs.py per-category counts {counts} != {EXPECTED_COUNTS}")
    if tuple(module.CATEGORIES) != ("Spatial", "Temporal", "Implementation"):
        errors.append(f"knobs.py CATEGORIES is {module.CATEGORIES}")
    if list(module.RETIRED_INTERNAL) != RETIRED_INTERNAL:
        errors.append(f"knobs.py RETIRED_INTERNAL is {list(module.RETIRED_INTERNAL)}")
    if dict(module.LEGACY_ALIASES) != LEGACY_ALIASES:
        errors.append(f"knobs.py LEGACY_ALIASES is {dict(module.LEGACY_ALIASES)}")
    if list(module.INTERNAL_COMPAT_ENV) != INTERNAL_COMPAT_ENV:
        errors.append(f"knobs.py INTERNAL_COMPAT_ENV is {list(module.INTERNAL_COMPAT_ENV)}")
    names = {k.name for k in module.KNOBS}
    for compat in INTERNAL_COMPAT_ENV:
        if compat in names:
            errors.append(f"internal compat env {compat} is listed as a public knob in knobs.py")
    for retired in RETIRED_INTERNAL:
        if retired in names:
            errors.append(f"retired knob {retired} is still in knobs.py KNOBS")
    for old in LEGACY_ALIASES:
        if old in names:
            errors.append(f"legacy alias {old} is listed as a public knob in knobs.py")
    if tuple(module.REQUIRED) != tuple(REQUIRED):
        errors.append("knobs.py REQUIRED differs from the contract order")
    for knob in module.KNOBS:
        text = " ".join([knob.values, knob.default, knob.mechanism])
        for retired in RETIRED_INTERNAL:
            if re.search(rf"{re.escape(retired)}(?![A-Za-z0-9_])", text):
                errors.append(f"{knob.name}: schema text names the retired knob {retired}")
    for old, new in LEGACY_ALIASES.items():
        knob = module.by_name().get(new)
        if knob is None or old not in " ".join([knob.values, knob.default, knob.mechanism]):
            errors.append(f"{new}: schema text does not document its legacy alias {old}")
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

    obsolete_errors: List[str] = []
    for path in DOCS:
        if path.exists():
            obsolete_errors.extend(check_obsolete_public_knobs(path))
    errors.extend(obsolete_errors)
    print(f"[{'PASS' if not obsolete_errors else 'FAIL'}] "
          f"obsolete public-knob scan: {len(DOCS)} documents")

    contract_errors = check_schema_contract()
    errors.extend(contract_errors)
    print(f"[{'PASS' if not contract_errors else 'FAIL'}] 14-knob / 3-category schema "
          "contract (Spatial 5, Temporal 3, Implementation 6; 6 retired; 2 legacy aliases; 2 accepted-but-not-public compat names)")

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
