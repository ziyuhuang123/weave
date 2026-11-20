#!/usr/bin/env python3
"""Check that the Python knob schema and the C++ pybind schema agree."""
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
    bad = [name for name, _ in expected if name in forbidden]
    if bad:
        print(f"forbidden public knobs in schema: {bad}")
        raise SystemExit(1)
    print(f"PASS weave knob schema: {len(expected)} entries, categories and names agree")


if __name__ == "__main__":
    main()
