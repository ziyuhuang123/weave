#!/usr/bin/env python3
"""Run the four-arm progressive ablation on SM103, H20, or MI308X.

Preset mapping (every arm is a complete preset over the 15 public ``WEAVE_*``
parameters, plus the internal pins of the six knobs retired from the public
surface that the recorded tuned presets were measured with; no other internal
or legacy environment variable is set):

  0  Weave   full tuned preset (spatial search + temporal overlap + dedup)
  1  -S      spatial allocation pinned to the compiled fixed default ratio
  2  -S-T    -S plus temporal overlap disabled (serialized Head / Tail)
  3  -S-T-C  -S-T plus remote-pull dedup disabled

Arm 0 copies the tuned per-platform presets from ``examples/*.sh``.  Each
later arm only overrides the knobs that encode the removed feature:

* ``-S`` (spatial search off): SM103 drops the searched scatter warp
  carve-out and the tuned blockM / register-pull profile
  (``WEAVE_SCATTER_WARPS=0``, ``WEAVE_GEMM1_CONFIG=128``,
  ``WEAVE_GEMM2_CONFIG=128``, ``WEAVE_REG_PULL_BYTES=0``, and the internal
  ``WEAVE_REDUCE_WARPS=0`` pin); H20 drops the tuned blockM profile
  (``WEAVE_GEMM1_CONFIG=0``, ``WEAVE_GEMM2_CONFIG=0``); MI308X pins the wave
  split to the fixed default ``WEAVE_CU_WAVE_SPLIT=0-8``.
* ``-T`` (temporal overlap off): ``WEAVE_V_HEAD=65536`` (helping warps
  complete the entire Head pull before entering GEMM),
  ``WEAVE_READY_GROUP_SIZE=0`` (full-expert readiness) and the internal pins
  ``WEAVE_READY_QUEUE=formula`` (static formula, no dynamic queue) and
  ``WEAVE_COMBINE_SYNC_CHUNK=0`` (global-barrier Combine).
* ``-C`` (dedup off): ``WEAVE_DEDUP_MODE=0`` and the internal
  ``WEAVE_LOCAL_SCATTER_DEDUP=0`` pin.

Spelling of the three renamed knobs: the NVIDIA hosts read both the public
names (``WEAVE_ADAPT_GEMM_CONFIG`` / ``WEAVE_GEMM1_CONFIG`` /
``WEAVE_GEMM2_CONFIG``) and their legacy aliases, so the NVIDIA arms are
exported under the public names.  The MI308X build scripts read only the legacy
spellings (``WEAVE_ADAPT_BM`` / ``WEAVE_BLOCK_M_L1`` / ``WEAVE_BLOCK_M_L2``), so
the MI308X arms are exported under those names (an unknown name would be
silently ignored there and change the measured configuration).

``WEAVE_ABLATION_PRESET`` is additionally exported so harnesses that consume
the preset directly can label the arm; the configuration itself is expressed
through the 15 public parameters listed in ``docs/TUNABLE_PARAMETERS.md`` and the
internal pins above.

An SM103-only ``--reduce-warps`` step sweeps the dedicated streaming Reduce
warp allocation (the internal ``WEAVE_REDUCE_WARPS`` pin) on top of the full
Weave arm:

    python ablation/run_progressive_ablation.py --backend b300 --presets 0 \
        --reduce-warps 0,1,2,4,8 --dry-run

Each step receives a distinct JIT cache directory so a cached object from a
previous arm cannot silently supply the next arm. With --dry-run the script is
CPU-only and prints every environment/command pair.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PRESET_NAMES = {0: "Weave", 1: "-S", 2: "-S-T", 3: "-S-T-C"}
SM103_BACKENDS = ("b300", "sm103")
REDUCE_WARP_CHOICES = (0, 1, 2, 4, 8)

def _load_surface():
    """Load the pure-Python knob schema (deep_gemm/mega/knobs.py)."""
    path = ROOT / "deep_gemm" / "mega" / "knobs.py"
    spec = importlib.util.spec_from_file_location("weave_knob_schema_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SURFACE = _load_surface()
PUBLIC_NAMES = frozenset(SURFACE.REQUIRED)               # the 14 public knobs
# Accepted-but-not-public compat names (WEAVE_ADAPT_GEMM_CONFIG / WEAVE_ADAPT_BM): the
# arm presets still pin them, so the public-table check admits them as well.
COMPAT_NAMES = frozenset(SURFACE.INTERNAL_COMPAT_ENV)
RETIRED_NAMES = frozenset(SURFACE.RETIRED_INTERNAL)      # internal pins only
# Public name -> legacy spelling.  The MI308X build scripts read only the legacy
# spelling; the NVIDIA hosts read both.
LEGACY_SPELLING = {new: old for old, new in SURFACE.LEGACY_ALIASES.items()}
LEGACY_SPELLING["WEAVE_ADAPT_GEMM_CONFIG"] = "WEAVE_ADAPT_BM"
LEGACY_ENV_BACKENDS = ("mi308x",)

# Arm-0 (Weave) presets, identical to the tuned presets in examples/*.sh and
# expressed with the 15 public WEAVE_* parameters.  A platform only lists the
# knobs it supports.
PUBLIC_KNOBS: Dict[str, Dict[str, str]] = {
    "b300": {
        # Spatial
        "WEAVE_DISPATCH_ACTIVE_SMS": "0",
        "WEAVE_SCATTER_ACTIVE_SMS": "0",
        "WEAVE_DISPATCH_WARPS": "4",
        "WEAVE_SCATTER_WARPS": "4",
        # Temporal
        "WEAVE_V_HEAD": "0",
        "WEAVE_READY_GROUP_SIZE": "2",
        "WEAVE_SCATTER_OWNER": "comm_only",
        # Implementation
        "WEAVE_PRECISION_MODE": "0",
        "WEAVE_ADAPT_GEMM_CONFIG": "0",
        "WEAVE_GEMM1_CONFIG": "64",
        "WEAVE_GEMM2_CONFIG": "64",
        "WEAVE_REG_PULL_BYTES": "4096",
        "WEAVE_DEDUP_MODE": "1",
        "WEAVE_SCATTER_STAGING": "smem",
    },
    "h20": {
        # Spatial
        "WEAVE_DISPATCH_ACTIVE_SMS": "0",
        "WEAVE_SCATTER_ACTIVE_SMS": "0",
        "WEAVE_DISPATCH_WARPS": "0",
        # Temporal
        "WEAVE_V_HEAD": "0",
        "WEAVE_READY_GROUP_SIZE": "0",
        "WEAVE_SCATTER_OWNER": "comm_only",
        # Implementation
        "WEAVE_PRECISION_MODE": "2",
        "WEAVE_ADAPT_GEMM_CONFIG": "0",
        "WEAVE_GEMM1_CONFIG": "32",
        "WEAVE_GEMM2_CONFIG": "32",
        "WEAVE_DEDUP_MODE": "1",
        "WEAVE_SCATTER_STAGING": "smem",
    },
    "mi308x": {
        # Spatial
        "WEAVE_DISPATCH_ACTIVE_SMS": "0",
        "WEAVE_SCATTER_ACTIVE_SMS": "0",
        "WEAVE_CU_WAVE_SPLIT": "4-4",
        # Temporal
        "WEAVE_V_HEAD": "0",
        "WEAVE_SCATTER_OWNER": "comm_only",
        # Implementation
        "WEAVE_PRECISION_MODE": "2",
        "WEAVE_ADAPT_GEMM_CONFIG": "1",
        "WEAVE_GEMM1_CONFIG": "0",
        "WEAVE_GEMM2_CONFIG": "0",
        "WEAVE_DEDUP_MODE": "1",
    },
}

# Internal pins of the six knobs retired from the public surface.  The recorded
# tuned presets were measured with these values, so arm 0 keeps setting them.
INTERNAL_PINS: Dict[str, Dict[str, str]] = {
    "b300": {
        "WEAVE_SM0_STREAM_SCHED": "0",
        "WEAVE_REDUCE_WARPS": "4",
        "WEAVE_REG_BALANCE": "0",
        "WEAVE_READY_QUEUE": "formula",
        "WEAVE_COMBINE_SYNC_CHUNK": "32",
        "WEAVE_LOCAL_SCATTER_DEDUP": "0",
    },
    "h20": {
        "WEAVE_SM0_STREAM_SCHED": "0",
        "WEAVE_REG_BALANCE": "0",
        "WEAVE_READY_QUEUE": "sm0_fifo",
        "WEAVE_COMBINE_SYNC_CHUNK": "0",
        "WEAVE_LOCAL_SCATTER_DEDUP": "0",
    },
    "mi308x": {
        "WEAVE_SM0_STREAM_SCHED": "0",
        "WEAVE_LOCAL_SCATTER_DEDUP": "1",
    },
}

# Feature-removal overlays.  ``-S`` pins the spatial allocation to the compiled
# fixed default ratio; ``-T`` disables the Head/Middle/Tail overlap; ``-C``
# disables the remote-pull dedup.  Each overlay has a public part and an
# internal-pin part.
MINUS_S: Dict[str, Dict[str, str]] = {
    "b300": {
        "WEAVE_SCATTER_WARPS": "0",
        "WEAVE_GEMM1_CONFIG": "128",
        "WEAVE_GEMM2_CONFIG": "128",
        "WEAVE_REG_PULL_BYTES": "0",
    },
    "h20": {
        "WEAVE_GEMM1_CONFIG": "0",
        "WEAVE_GEMM2_CONFIG": "0",
    },
    "mi308x": {
        "WEAVE_CU_WAVE_SPLIT": "0-8",
    },
}
INTERNAL_MINUS_S: Dict[str, Dict[str, str]] = {
    "b300": {"WEAVE_REDUCE_WARPS": "0"},
    "h20": {},
    "mi308x": {},
}

MINUS_T: Dict[str, Dict[str, str]] = {
    "b300": {
        "WEAVE_V_HEAD": "65536",
        "WEAVE_READY_GROUP_SIZE": "0",
    },
    "h20": {
        "WEAVE_V_HEAD": "65536",
    },
    "mi308x": {
        "WEAVE_V_HEAD": "65536",
    },
}
INTERNAL_MINUS_T: Dict[str, Dict[str, str]] = {
    "b300": {"WEAVE_READY_QUEUE": "formula", "WEAVE_COMBINE_SYNC_CHUNK": "0"},
    "h20": {"WEAVE_READY_QUEUE": "formula", "WEAVE_COMBINE_SYNC_CHUNK": "0"},
    "mi308x": {},
}

MINUS_C: Dict[str, Dict[str, str]] = {
    backend: {"WEAVE_DEDUP_MODE": "0"}
    for backend in ("b300", "h20", "mi308x")
}
INTERNAL_MINUS_C: Dict[str, Dict[str, str]] = {
    backend: {"WEAVE_LOCAL_SCATTER_DEDUP": "0"}
    for backend in ("b300", "h20", "mi308x")
}


def _validate_tables() -> None:
    """The public tables hold only public names, the internal tables only retired ones."""
    for label, table, allowed in (
            ("PUBLIC_KNOBS", PUBLIC_KNOBS, PUBLIC_NAMES | COMPAT_NAMES),
            ("MINUS_S", MINUS_S, PUBLIC_NAMES | COMPAT_NAMES),
            ("MINUS_T", MINUS_T, PUBLIC_NAMES | COMPAT_NAMES),
            ("MINUS_C", MINUS_C, PUBLIC_NAMES | COMPAT_NAMES),
            ("INTERNAL_PINS", INTERNAL_PINS, RETIRED_NAMES),
            ("INTERNAL_MINUS_S", INTERNAL_MINUS_S, RETIRED_NAMES),
            ("INTERNAL_MINUS_T", INTERNAL_MINUS_T, RETIRED_NAMES),
            ("INTERNAL_MINUS_C", INTERNAL_MINUS_C, RETIRED_NAMES)):
        for backend, env in table.items():
            bad = sorted(set(env) - allowed)
            if bad:
                raise SystemExit(f"{label}[{backend}] holds names outside its surface: {bad}")


_validate_tables()


def _canonical_backend(backend: str) -> str:
    return "b300" if backend in SM103_BACKENDS else backend


def _runner(args: argparse.Namespace, backend: str) -> List[str]:
    if args.runner:
        return shlex.split(args.runner)
    if backend == "b300":
        return [sys.executable, str(ROOT / "tests" / "bench_b300_ep8.py"),
                "--model", args.model, "--seqlen", str(args.seqlen),
                "--num-processes", str(args.num_processes)]
    if backend == "h20":
        return [sys.executable, str(ROOT / "tests" / "bench_h20_ep8.py"),
                "--num-processes", str(args.num_processes)]
    runner = args.mi308x_runner or os.environ.get("WEAVE_MI308X_BENCH_RUNNER", "")
    if not runner:
        return ["<MI308X_RUNNER>", "--ep", str(args.ep)]
    return [sys.executable, str(ROOT / "tests" / "bench_mi308x_ep8_ep16.py"),
            "--ep", str(args.ep), "--runner", runner]


def _arm_knobs(backend: str, preset: int,
               reduce_warps: int | None = None) -> Dict[str, str]:
    """Env of one arm: public knobs + the internal pins, in canonical (public) names."""
    env = dict(PUBLIC_KNOBS[backend])
    internal = dict(INTERNAL_PINS[backend])
    if preset >= 1:
        env.update(MINUS_S[backend])
        internal.update(INTERNAL_MINUS_S[backend])
    if preset >= 2:
        env.update(MINUS_T[backend])
        internal.update(INTERNAL_MINUS_T[backend])
    if preset >= 3:
        env.update(MINUS_C[backend])
        internal.update(INTERNAL_MINUS_C[backend])
    if reduce_warps is not None:
        internal["WEAVE_REDUCE_WARPS"] = str(reduce_warps)
    env.update(internal)
    return env


def _export_names(backend: str, env: Dict[str, str]) -> Dict[str, str]:
    """Spell the env the way the backend reads it (legacy names for MI308X)."""
    if backend not in LEGACY_ENV_BACKENDS:
        return env
    return {LEGACY_SPELLING.get(name, name): value for name, value in env.items()}


def _parse_reduce_warps(raw: str, backend: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            raise SystemExit(f"--reduce-warps values must be integers, got {item!r}")
        if value not in REDUCE_WARP_CHOICES:
            raise SystemExit(
                f"--reduce-warps must be a comma-separated subset of "
                f"{','.join(str(v) for v in REDUCE_WARP_CHOICES)}, got {value}")
        if value not in values:
            values.append(value)
    if values and backend not in SM103_BACKENDS:
        raise SystemExit(
            "--reduce-warps is an SM103-only ablation step; "
            "use --backend b300 (or --backend sm103)")
    return values


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True,
                    choices=("b300", "sm103", "h20", "mi308x"))
    ap.add_argument("--presets", default="0,1,2,3")
    ap.add_argument("--model", default="qwen38max")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--ep", type=int, choices=(8, 16), default=16)
    ap.add_argument("--runner", default="",
                    help="optional full runner command; overrides the default")
    ap.add_argument("--mi308x-runner", default="")
    ap.add_argument("--reduce-warps", default="",
                    help="SM103-only comma-separated sweep of the internal "
                         "WEAVE_REDUCE_WARPS pin run on top of the full Weave "
                         "arm (subset of 0,1,2,4,8)")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "ablation")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _args()
    backend = _canonical_backend(args.backend)
    presets = [int(x) for x in args.presets.split(",") if x.strip()]
    if any(p not in PRESET_NAMES for p in presets):
        raise SystemExit("--presets must be a comma-separated subset of 0,1,2,3")
    reduce_warps = _parse_reduce_warps(args.reduce_warps, backend)

    plan = []
    for preset in presets:
        cmd = _runner(args, backend)
        env = _export_names(backend, _arm_knobs(backend, preset))
        env.update({
            "WEAVE_ABLATION_PRESET": str(preset),
            "WEAVE_PROFILER_LEVEL": "0",
            "DG_JIT_CACHE_DIR": str(args.output_dir / backend /
                                    f"jit_p{preset}"),
        })
        plan.append({
            "preset": preset,
            "name": PRESET_NAMES[preset],
            "env": env,
            "cmd": cmd,
        })
    for value in reduce_warps:
        cmd = _runner(args, backend)
        env = _export_names(backend, _arm_knobs(backend, 0, reduce_warps=value))
        env.update({
            "WEAVE_ABLATION_PRESET": "0",
            "WEAVE_PROFILER_LEVEL": "0",
            "DG_JIT_CACHE_DIR": str(args.output_dir / backend /
                                    f"jit_rw{value}"),
        })
        plan.append({
            "preset": 0,
            "name": f"Weave+RW{value}",
            "reduce_warps": value,
            "env": env,
            "cmd": cmd,
        })

    print(json.dumps({"backend": args.backend, "plan": plan}, indent=2))
    if args.dry_run:
        return 0
    if any(row["cmd"] and row["cmd"][0].startswith("<") for row in plan):
        raise SystemExit("no MI308X runner configured; pass --mi308x-runner")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in plan:
        run_env = dict(os.environ)
        run_env.update(row["env"])
        print(json.dumps({"event": "run", **row}))
        proc = subprocess.run(row["cmd"], cwd=ROOT, env=run_env)
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
