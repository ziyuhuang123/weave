"""Standalone Weave cost-model loader.

Importing `deep_gemm` normally requires the compiled `_C` extension and
PyTorch.  This module loads the pure-Python mirror directly from
`deep_gemm/mega/cost_model.py`, so the equations and CTA-0 selection can be
used on a CPU-only machine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "deep_gemm" / "mega" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location("weave_cost_model_impl", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

HARDWARE = _MOD.HARDWARE
CANDIDATES = _MOD.CANDIDATES
HardwareSpec = _MOD.HardwareSpec
WorkloadSpec = _MOD.WorkloadSpec
SpatialCandidate = _MOD.SpatialCandidate
Breakdown = _MOD.Breakdown
encode_r_mask = _MOD.encode_r_mask
choose_v_head = _MOD.choose_v_head
evaluate = _MOD.evaluate
select_policy = _MOD.select_policy

__all__ = [
    "HARDWARE", "CANDIDATES", "HardwareSpec", "WorkloadSpec",
    "SpatialCandidate", "Breakdown", "encode_r_mask", "choose_v_head",
    "evaluate", "select_policy",
]
