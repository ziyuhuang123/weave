#!/usr/bin/env python3
"""Unit tests for the unified WEAVE_PRECISION_MODE resolution (host-only).

Exercises deep_gemm/mega/precision.py without torch or the compiled _C
extension: spellings / aliases, environment precedence, the SM10x / SM90
hardware support matrix, SymmBuffer layout selection (including the legacy
fp8xfp8 FP8-weight buffer), weave_mega_moe backend resolution and the
knob-defaults selector key.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deep_gemm" / "mega" / "precision.py"

spec = importlib.util.spec_from_file_location("weave_precision", MODULE_PATH)
precision = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = precision
spec.loader.exec_module(precision)

SM100 = precision.GpuArch("sm10x", "sm_100")
SM103 = precision.GpuArch("sm10x", "sm_103")
SM90 = precision.GpuArch("sm90", "sm_90")

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exception:  # noqa: BLE001 - report every failure
        FAILURES.append(f"{name}: {type(exception).__name__}: {exception}")
        print(f"FAIL {name}: {type(exception).__name__}: {exception}")
    else:
        print(f"PASS {name}")


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- spellings -------------------------------------------------------------

def test_canonical_spellings():
    expected = {
        "0": "fp8xfp4", "fp8xfp4": "fp8xfp4",
        "1": "fp4xfp4", "fp4xfp4": "fp4xfp4", "nvfp4": "fp4xfp4",
        "2": "fp8xfp8", "fp8xfp8": "fp8xfp8", "fp8": "fp8xfp8",
        "3": "bf16", "bf16": "bf16", "bf16xbf16": "bf16",
    }
    for spelling, canonical in expected.items():
        assert precision.canonical_precision(spelling) == canonical, spelling
        assert precision.canonical_precision(f"  {spelling.upper()} \t") == canonical, spelling
    assert set(precision.ACCEPTED_SPELLINGS) == set(expected)
    # Superset of the historical public tuple keeps old callers working.
    for legacy in ("fp8xfp4", "fp4xfp4", "nvfp4", "fp8", "fp8xfp8", "bf16"):
        assert legacy in precision.ACCEPTED_SPELLINGS, legacy


def test_invalid_spellings():
    for bad in ("fp16", "4", "-1", "int8", "fp8xfp4xfp4", "auto", ""):
        expect_value_error(lambda bad=bad: precision.canonical_precision(bad))


def test_precision_codes():
    for code, name in enumerate(("fp8xfp4", "fp4xfp4", "fp8xfp8", "bf16")):
        assert precision.precision_code(name) == code
        assert precision.precision_code(str(code)) == code
    assert precision.describe("fp4xfp4") == "1=fp4xfp4"


def test_is_auto():
    for value in (None, "", "   ", "auto", "AUTO", " Auto "):
        assert precision.is_auto(value), repr(value)
    for value in ("0", "fp8xfp4", "3", "bf16"):
        assert not precision.is_auto(value), repr(value)


# --- environment -----------------------------------------------------------

def test_env_unset_and_auto():
    assert precision.precision_from_env({}) is None
    assert precision.precision_from_env({"WEAVE_PRECISION_MODE": "auto"}) is None
    assert precision.precision_from_env({"WEAVE_PRECISION_MODE": ""}) is None
    assert precision.precision_from_env({"WEAVE_PRECISION_MODE": "  "}) is None


def test_env_spellings_and_legacy():
    found = precision.precision_from_env({"WEAVE_PRECISION_MODE": " NVFP4 "})
    assert found is not None and found[0] == "fp4xfp4"
    found = precision.precision_from_env({"WEAVE_KNOB_PRECISION_MODE": "3"})
    assert found is not None and found[0] == "bf16"
    # The public name wins over the legacy spelling.
    found = precision.precision_from_env({
        "WEAVE_PRECISION_MODE": "1", "WEAVE_KNOB_PRECISION_MODE": "3"})
    assert found is not None and found[0] == "fp4xfp4"
    # An explicit auto on the public name is not overridden by the legacy one.
    assert precision.precision_from_env({
        "WEAVE_PRECISION_MODE": "auto", "WEAVE_KNOB_PRECISION_MODE": "3"}) is None


def test_env_invalid():
    expect_value_error(lambda: precision.precision_from_env({"WEAVE_PRECISION_MODE": "fp16"}))
    expect_value_error(lambda: precision.precision_from_env({"WEAVE_KNOB_PRECISION_MODE": "9"}))


# --- hardware support matrix -----------------------------------------------

def test_arch_classification():
    assert precision.arch_from_capability((10, 0)) == SM100
    assert precision.arch_from_capability((10, 3)) == SM103
    assert precision.arch_from_capability((9, 0)) == SM90
    assert precision.arch_from_capability((8, 0)) is None
    assert precision.arch_from_capability(None) is None
    assert precision.arch_from_capability((10, 0), is_hip=True) is None


def test_sm10x_support():
    for name in ("fp8xfp4", "fp4xfp4", "bf16"):
        precision.check_precision_supported(name, SM103, "test")
    expect_value_error(lambda: precision.check_precision_supported("fp8xfp8", SM103, "test"))


def test_sm90_support():
    precision.check_precision_supported("fp8xfp8", SM90, "test")
    for name in ("fp8xfp4", "fp4xfp4", "bf16"):
        expect_value_error(lambda name=name: precision.check_precision_supported(name, SM90, "test"))


def test_unknown_arch_is_not_validated():
    for name in precision.PRECISION_NAMES:
        precision.check_precision_supported(name, None, "test")


# --- SymmBuffer resolution --------------------------------------------------

def test_symm_buffer_default_backward_compat():
    # Unset env + omitted mma_type must stay 'fp8xfp4' on every host.
    assert precision.resolve_symm_buffer_mma_type(None, None, {}) == "fp8xfp4"
    assert precision.resolve_symm_buffer_mma_type(None, SM103, {}) == "fp8xfp4"
    assert precision.resolve_symm_buffer_mma_type("auto", SM103, {}) == "fp8xfp4"


def test_symm_buffer_env_selection():
    env = {"WEAVE_PRECISION_MODE": "fp4xfp4"}
    assert precision.resolve_symm_buffer_mma_type(None, SM103, env) == "fp4xfp4"
    assert precision.resolve_symm_buffer_mma_type("auto", SM103, env) == "fp4xfp4"
    env = {"WEAVE_PRECISION_MODE": "3"}
    assert precision.resolve_symm_buffer_mma_type(None, SM103, env) == "bf16xbf16"
    # Explicit mma_type wins over the environment.
    assert precision.resolve_symm_buffer_mma_type("fp8xfp4", SM103, env) == "fp8xfp4"


def test_symm_buffer_legacy_fp8_weight_buffer():
    # The historical SM10x fp8xfp8 spelling is the fp8xfp4-layout buffer.
    assert precision.resolve_symm_buffer_mma_type("fp8xfp8", SM103, {}) == "fp8xfp8"
    assert precision.resolve_symm_buffer_mma_type("fp8xfp8", None, {}) == "fp8xfp8"
    assert precision.symm_buffer_precision("fp8xfp8") == "fp8xfp4"
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("fp8xfp8", SM90, {}))
    # Code 2 is the SM90 backend, never an SM10x SymmBuffer layout.
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("2", SM103, {}))
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("2", None, {}))


def test_symm_buffer_hardware_rejection():
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("bf16", SM90, {}))
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("nvfp4", SM90, {}))
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type(None, SM90,
                                                                      {"WEAVE_PRECISION_MODE": "1"}))
    expect_value_error(lambda: precision.resolve_symm_buffer_mma_type("fp16", SM103, {}))


def test_symm_buffer_precision_mapping():
    assert precision.symm_buffer_precision("fp8xfp4") == "fp8xfp4"
    assert precision.symm_buffer_precision("fp4xfp4") == "fp4xfp4"
    assert precision.symm_buffer_precision("bf16xbf16") == "bf16"


# --- weave_mega_moe resolution ----------------------------------------------

def test_weave_explicit_and_alias():
    assert precision.resolve_weave_precision("fp4xfp4", None, None, SM103, {}) == "fp4xfp4"
    assert precision.resolve_weave_precision(None, "1", None, SM103, {}) == "fp4xfp4"
    assert precision.resolve_weave_precision("nvfp4", "fp4xfp4", None, SM103, {}) == "fp4xfp4"
    assert precision.resolve_weave_precision("auto", "auto", None, SM103,
                                             {"WEAVE_PRECISION_MODE": "1"}) == "fp4xfp4"
    expect_value_error(lambda: precision.resolve_weave_precision("fp4xfp4", "bf16", None, SM103, {}))


def test_weave_precedence():
    # explicit > env > buffer precision (the buffer layout must still agree)
    assert precision.resolve_weave_precision(
        "fp8xfp4", None, "fp8xfp4", SM103, {"WEAVE_PRECISION_MODE": "1"}) == "fp8xfp4"
    assert precision.resolve_weave_precision(
        None, None, "bf16", SM103, {}) == "bf16"
    assert precision.resolve_weave_precision(
        None, None, None, SM103, {"WEAVE_PRECISION_MODE": "3"}) == "bf16"
    expect_value_error(lambda: precision.resolve_weave_precision(None, None, None, None, {}))


def test_weave_buffer_layout_mismatch():
    # The env may not silently relaunch a buffer built for another precision.
    expect_value_error(lambda: precision.resolve_weave_precision(
        None, None, "fp8xfp4", SM103, {"WEAVE_PRECISION_MODE": "1"}))
    expect_value_error(lambda: precision.resolve_weave_precision(
        "bf16", None, "fp4xfp4", SM103, {}))
    # Matching spellings resolve cleanly, including the legacy buffer.
    assert precision.resolve_weave_precision(
        None, None, "fp4xfp4", SM103, {"WEAVE_PRECISION_MODE": "nvfp4"}) == "fp4xfp4"
    assert precision.resolve_weave_precision(
        "fp8xfp4", None, "fp8xfp4", SM103, {}) == "fp8xfp4"


def test_weave_hardware_rejection():
    expect_value_error(lambda: precision.resolve_weave_precision("fp8", None, None, SM103, {}))
    expect_value_error(lambda: precision.resolve_weave_precision("fp8xfp4", None, None, SM90, {}))
    expect_value_error(lambda: precision.resolve_weave_precision(
        None, None, None, SM90, {"WEAVE_PRECISION_MODE": "bf16"}))
    assert precision.resolve_weave_precision("fp8xfp8", None, None, SM90, {}) == "fp8xfp8"


# --- knob defaults selector --------------------------------------------------

def test_knob_defaults_key():
    assert precision.knob_defaults_key("mi308x") == "mi308x"
    assert precision.knob_defaults_key("MI308X") == "mi308x"
    assert precision.knob_defaults_key("nvfp4") == "fp4xfp4"
    assert precision.knob_defaults_key("bf16xbf16") == "bf16"
    assert precision.knob_defaults_key("fp8") == "fp8xfp8"
    assert precision.knob_defaults_key("fp8xfp4") == "fp8xfp4"
    expect_value_error(lambda: precision.knob_defaults_key("fp16"))


def test_knob_defaults_key_env():
    import os
    saved = {name: os.environ.get(name) for name in precision.ENV_NAMES}
    try:
        for name in precision.ENV_NAMES:
            os.environ.pop(name, None)
        assert precision.knob_defaults_key(None) == "fp8xfp4"
        assert precision.knob_defaults_key("auto") == "fp8xfp4"
        os.environ["WEAVE_PRECISION_MODE"] = "fp4xfp4"
        assert precision.knob_defaults_key(None) == "fp4xfp4"
        assert precision.knob_defaults_key("bf16") == "bf16"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    for name, fn in tests:
        check(name, fn)
    if FAILURES:
        print(f"OVERALL=FAIL ({len(FAILURES)} of {len(tests)} failed)")
        raise SystemExit(1)
    print(f"OVERALL=PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()