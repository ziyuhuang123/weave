#!/usr/bin/env python3
"""Check that the Python knob schema and the C++ pybind schema agree.

Both must carry exactly the 14-knob / 3-category public surface (Spatial 5,
Temporal 3, Implementation 6); the six retired knobs, the two legacy aliases
(WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2) and the accepted-but-not-public
INTERNAL_COMPAT_ENV names (WEAVE_ADAPT_GEMM_CONFIG / WEAVE_ADAPT_BM) are not
public entries on either side.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "deep_gemm" / "mega" / "knobs.py"
BINDING_PATH = ROOT / "csrc" / "apis" / "weave_knobs.hpp"


def load_schema():
    spec = importlib.util.spec_from_file_location("weave_knob_schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    schema = load_schema()
    expected = [(k.name, k.category) for k in schema.KNOBS]
    text = BINDING_PATH.read_text(encoding="utf-8")
    found = re.findall(r'\{"(WEAVE_[A-Z0-9_]+)", "(Spatial|Temporal|Implementation)"', text)
    if found != expected:
        print("Python schema and C++ binding diverged")
        for i, (a, b) in enumerate(zip(expected, found)):
            if a != b:
                print(f"  [{i}] python={a} cpp={b}")
        if len(expected) != len(found):
            print(f"  length python={len(expected)} cpp={len(found)}")
        raise SystemExit(1)
    forbidden = {"WEAVE_HOT_COLD_MODE", "WEAVE_COMM_UNITS", "WEAVE_COMP_UNITS"}
    forbidden |= set(schema.RETIRED_INTERNAL) | set(schema.LEGACY_ALIASES) | set(schema.INTERNAL_COMPAT_ENV)
    bad = [name for name, _ in expected + found if name in forbidden]
    if bad:
        print(f"retired / legacy / forbidden names listed as public knobs: {sorted(set(bad))}")
        raise SystemExit(1)
    counts = {c: sum(1 for _, cat in found if cat == c) for c in schema.CATEGORIES}
    if len(found) != 14 or counts != {"Spatial": 5, "Temporal": 3, "Implementation": 6}:
        print(f"C++ binding must list 14 knobs as Spatial 5 / Temporal 3 / Implementation 6, got {counts}")
        raise SystemExit(1)
    # The compiled-defaults exporter must use exactly the public names (and the
    # new spellings of the three renamed knobs), never the retired / legacy ones.
    exported = re.findall(r'out\["(WEAVE_[A-Z0-9_]+)"\]', text)
    if sorted(exported) != sorted(name for name, _ in expected):
        print("weave_knob_defaults() exports differ from the public names: "
              f"extra={sorted(set(exported) - {n for n, _ in expected})} "
              f"missing={sorted({n for n, _ in expected} - set(exported))}")
        raise SystemExit(1)
    print(f"PASS weave knob schema: {len(expected)} entries, categories and names agree "
          f"(Spatial {counts['Spatial']}, Temporal {counts['Temporal']}, "
          f"Implementation {counts['Implementation']}; defaults exporter matches)")


if __name__ == "__main__":
    main()
