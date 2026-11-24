"""Unified ``WEAVE_PRECISION_MODE`` resolution for the MegaMoE Python API.

This module is pure Python: it imports neither ``torch`` nor the compiled
``_C`` extension at module load, so the selection rules can be unit-tested on
hosts without CUDA (see ``tests/check_precision_mode_wiring.py``).

Public selector (``README.md`` / ``README_zh.md``)::

    export WEAVE_PRECISION_MODE=fp4xfp4

Accepted spellings (case-insensitive, surrounding blanks ignored):

====  ===========  =============  ==================================  ===========
code  canonical    aliases        backend kernel                      hardware
====  ===========  =============  ==================================  ===========
0     fp8xfp4      --             sm100_fp8_fp4_mega_moe.cuh          SM10x
1     fp4xfp4      nvfp4          sm100_fp4_fp4_mega_moe.cuh          SM10x
2     fp8xfp8      fp8            sm90_fp8_mega_moe.cuh               SM90 (H20)
3     bf16         bf16xbf16      sm100_bf16_mega_moe.cuh             SM10x
====  ===========  =============  ==================================  ===========

``SM10x`` is the compute-capability-10 family (SM100 and SM103 / B300).
Hardware support matrix, checked before any symmetric buffer is allocated:

* SM10x accepts ``0``, ``1`` and ``3`` and rejects ``2`` with ``ValueError``;
* SM90 accepts only ``2`` and rejects ``0``, ``1`` and ``3`` with ``ValueError``;
* any other device (or no visible CUDA device) is not validated here; the C++
  dispatchers keep their own architecture assertions.

Resolution precedence: an explicit argument wins over ``WEAVE_PRECISION_MODE``
(then the legacy spelling ``WEAVE_KNOB_PRECISION_MODE``), which wins over the
symmetric buffer's own precision and finally the default ``fp8xfp4``.  ``None``
and ``"auto"`` both mean "not specified".

One legacy spelling is kept on purpose: ``SymmBuffer(mma_type='fp8xfp8')`` on
SM10x builds the fp8xfp4-layout buffer that the fp8xfp4 backend
(``fp8_fp4_mega_moe``) uses for FP8-E4M3 weights.  It is backend ``0``, not the
SM90 backend ``2``; only the explicit ``SymmBuffer`` argument selects it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

# Index == WEAVE_PRECISION_MODE code (mirrors weave::PrecisionMode in weave_knobs.cuh).
PRECISION_NAMES: Tuple[str, ...] = ("fp8xfp4", "fp4xfp4", "fp8xfp8", "bf16")
DEFAULT_PRECISION = "fp8xfp4"
ENV_NAMES: Tuple[str, ...] = ("WEAVE_PRECISION_MODE", "WEAVE_KNOB_PRECISION_MODE")

# Every accepted spelling -> canonical name.  The C++ parser
# (weave_knobs::precision_mode_env in weave_knob_config.hpp) accepts the same set.
PRECISION_ALIASES: Mapping[str, str] = {
    "0": "fp8xfp4", "fp8xfp4": "fp8xfp4",
    "1": "fp4xfp4", "fp4xfp4": "fp4xfp4", "nvfp4": "fp4xfp4",
    "2": "fp8xfp8", "fp8xfp8": "fp8xfp8", "fp8": "fp8xfp8",
    "3": "bf16", "bf16": "bf16", "bf16xbf16": "bf16",
}
ACCEPTED_SPELLINGS: Tuple[str, ...] = tuple(PRECISION_ALIASES)

# Canonical precision -> SymmBuffer.mma_type string understood by the C++
# layout helpers (csrc/apis/mega_moe.hpp: 'fp4xfp4' selects the NVFP4 layout,
# 'bf16xbf16' the BF16 layout, every other string the MXFP8 layout).
SYMM_BUFFER_MMA_TYPES: Mapping[str, str] = {
    "fp8xfp4": "fp8xfp4", "fp4xfp4": "fp4xfp4", "bf16": "bf16xbf16",
}
LEGACY_SM10X_FP8_WEIGHT_BUFFER = "fp8xfp8"

SUPPORTED_PRECISIONS: Mapping[str, Tuple[str, ...]] = {
    "sm10x": ("fp8xfp4", "fp4xfp4", "bf16"),
    "sm90": ("fp8xfp8",),
}
_BACKEND_KERNELS: Mapping[str, str] = {
    "fp8xfp4": "sm100_fp8_fp4_mega_moe.cuh",
    "fp4xfp4": "sm100_fp4_fp4_mega_moe.cuh",
    "fp8xfp8": "sm90_fp8_mega_moe.cuh",
    "bf16": "sm100_bf16_mega_moe.cuh",
}
_SPELLING_HELP = "0/fp8xfp4, 1/fp4xfp4 (nvfp4), 2/fp8xfp8 (fp8), 3/bf16 (bf16xbf16), or auto"


@dataclass(frozen=True)
class GpuArch:
    """Coarse device family used by the support matrix."""
    family: str  # "sm10x" or "sm90"
    name: str    # e.g. "sm_103"


def is_auto(value) -> bool:
    """``None``, an empty string and ``"auto"`` mean "not specified"."""
    return value is None or str(value).strip().lower() in ("", "auto")


def canonical_precision(value, source: str = "mma_type") -> str:
    """Map any accepted spelling to its canonical name, else raise ``ValueError``."""
    key = str(value).strip().lower()
    if key not in PRECISION_ALIASES:
        raise ValueError(f"{source}={value!r} is not a MegaMoE precision; expected {_SPELLING_HELP}")
    return PRECISION_ALIASES[key]


def precision_code(value) -> int:
    """Numeric ``WEAVE_PRECISION_MODE`` code of a spelling."""
    return PRECISION_NAMES.index(canonical_precision(value))


def describe(precision: str) -> str:
    return f"{PRECISION_NAMES.index(precision)}={precision}"


def precision_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[Tuple[str, str]]:
    """Return ``(canonical, "NAME='raw'")`` from the environment, or ``None`` if unset/auto."""
    env = os.environ if environ is None else environ
    for name in ENV_NAMES:
        raw = env.get(name)
        if raw is None:
            continue
        if is_auto(raw):
            return None
        return canonical_precision(raw, source=name), f"{name}={raw!r}"
    return None


def arch_from_capability(capability, is_hip: bool = False) -> Optional[GpuArch]:
    """Classify a ``(major, minor)`` compute capability; ROCm devices are not classified."""
    if is_hip or capability is None:
        return None
    major, minor = int(capability[0]), int(capability[1])
    name = f"sm_{major}{minor}"
    if major == 10:
        return GpuArch("sm10x", name)
    if major == 9:
        return GpuArch("sm90", name)
    return None


def current_gpu_arch(device=None) -> Optional[GpuArch]:
    """Family of ``device`` (default: the current CUDA device); ``None`` if unknown."""
    try:
        import torch
    except Exception:
        return None
    if getattr(torch.version, "hip", None):
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if device is not None and getattr(device, "type", "cuda") != "cuda":
            return None
        return arch_from_capability(torch.cuda.get_device_capability(device))
    except Exception:
        return None


def check_precision_supported(precision: str, arch: Optional[GpuArch], source: str) -> None:
    """Raise ``ValueError`` if ``precision`` cannot run on ``arch`` (no-op when ``arch`` is None)."""
    if arch is None or precision in SUPPORTED_PRECISIONS[arch.family]:
        return
    supported = ", ".join(describe(p) for p in SUPPORTED_PRECISIONS[arch.family])
    if arch.family == "sm10x":
        hint = (f"{describe(precision)} selects the SM90/H20 backend {_BACKEND_KERNELS[precision]}; "
                f"SM10x supports {supported} (FP8-E4M3 weights run through 0=fp8xfp4 with a "
                f"SymmBuffer(mma_type='fp8xfp8') buffer)")
    else:
        hint = (f"{describe(precision)} selects the SM10x backend {_BACKEND_KERNELS[precision]}; "
                f"SM90/H20 supports only {supported} via SM90SymmBuffer / fp8_mega_moe")
    raise ValueError(f"{source} is not supported on this GPU ({arch.name}, {arch.family} family): {hint}")


def resolve_symm_buffer_mma_type(mma_type=None, arch: Optional[GpuArch] = None,
                                 environ: Optional[Mapping[str, str]] = None) -> str:
    """Resolve and validate the ``mma_type`` of an SM10x ``SymmBuffer``.

    Returns the C++ layout string (``fp8xfp4``, ``fp4xfp4``, ``bf16xbf16`` or the
    legacy explicit ``fp8xfp8``).  With ``mma_type`` omitted and the environment
    unset this is ``'fp8xfp4'``, the historical default.
    """
    if is_auto(mma_type):
        found = precision_from_env(environ)
        precision, source = found if found is not None else (DEFAULT_PRECISION, "default mma_type='fp8xfp4'")
    else:
        if str(mma_type).strip().lower() == LEGACY_SM10X_FP8_WEIGHT_BUFFER:
            if arch is not None and arch.family != "sm10x":
                raise ValueError(
                    f"SymmBuffer(mma_type={mma_type!r}) is the SM10x fp8xfp4-layout buffer for FP8 "
                    f"weights and is not supported on {arch.name}; on SM90/H20 use SM90SymmBuffer")
            return LEGACY_SM10X_FP8_WEIGHT_BUFFER
        precision, source = canonical_precision(mma_type), f"mma_type={mma_type!r}"
    if precision == "fp8xfp8" and (arch is None or arch.family == "sm90"):
        raise ValueError(
            f"{source} ({describe(precision)}) selects the SM90 backend; SymmBuffer is the SM10x MegaMoE "
            "buffer, use SM90SymmBuffer / get_symm_buffer_for_sm90_mega_moe instead")
    check_precision_supported(precision, arch, source)
    return SYMM_BUFFER_MMA_TYPES[precision]


def symm_buffer_precision(mma_type: str) -> str:
    """Backend precision served by an SM10x ``SymmBuffer`` with the given ``mma_type``."""
    if str(mma_type).strip().lower() == LEGACY_SM10X_FP8_WEIGHT_BUFFER:
        return "fp8xfp4"
    return canonical_precision(mma_type, source="SymmBuffer.mma_type")


def resolve_weave_precision(mma_type=None, precision_mode=None,
                            buffer_precision: Optional[str] = None,
                            arch: Optional[GpuArch] = None,
                            environ: Optional[Mapping[str, str]] = None) -> str:
    """Resolve the backend precision of one ``weave_mega_moe`` call.

    ``buffer_precision`` is the backend precision the symmetric buffer was built
    for (``None`` if unknown).  A resolved precision that disagrees with it is a
    layout mismatch and raises ``ValueError`` instead of launching the wrong
    kernel on the buffer.
    """
    explicit = []
    if not is_auto(mma_type):
        explicit.append((canonical_precision(mma_type), f"mma_type={mma_type!r}"))
    if not is_auto(precision_mode):
        explicit.append((canonical_precision(precision_mode, "precision_mode"),
                         f"precision_mode={precision_mode!r}"))
    if len(explicit) == 2 and explicit[0][0] != explicit[1][0]:
        raise ValueError(f"{explicit[0][1]} conflicts with {explicit[1][1]}")
    if explicit:
        precision, source = explicit[0]
    else:
        found = precision_from_env(environ)
        if found is not None:
            precision, source = found
        elif buffer_precision is not None:
            precision, source = buffer_precision, f"the symmetric buffer precision {buffer_precision!r}"
        else:
            raise ValueError(
                "weave_mega_moe requires mma_type, precision_mode, WEAVE_PRECISION_MODE or a "
                "symmetric buffer with a known precision")
    check_precision_supported(precision, arch, source)
    if buffer_precision is not None and buffer_precision != precision:
        raise ValueError(
            f"{source} resolves to {describe(precision)} but the symmetric buffer was built for "
            f"{describe(buffer_precision)}; build the buffer with the same precision (for SymmBuffer, "
            f"pass mma_type={SYMM_BUFFER_MMA_TYPES.get(precision, precision)!r} or omit mma_type so it "
            "follows WEAVE_PRECISION_MODE)")
    return precision


def knob_defaults_key(mma_type) -> str:
    """Spelling passed to ``_C.weave_knobs.defaults`` (keeps the ``'mi308x'`` selector)."""
    if str(mma_type).strip().lower() == "mi308x":
        return "mi308x"
    if is_auto(mma_type):
        found = precision_from_env()
        return found[0] if found is not None else DEFAULT_PRECISION
    return canonical_precision(mma_type)
