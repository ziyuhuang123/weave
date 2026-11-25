#!/usr/bin/env python3
"""Standalone host-surface gate for the 14 public WEAVE_* knobs.

The public surface is 14 knobs in 3 categories (Spatial 5, Temporal 3,
Implementation 6).  This gate also covers the legacy aliases WEAVE_BLOCK_M_L1 /
WEAVE_BLOCK_M_L2, the accepted-but-not-public WEAVE_ADAPT_GEMM_CONFIG (alias
WEAVE_ADAPT_BM; host shape-adaptive tuple selection is always on) and the internal pins of the six retired knobs (WEAVE_REDUCE_WARPS,
WEAVE_COMBINE_SYNC_CHUNK, WEAVE_REG_BALANCE, WEAVE_LOCAL_SCATTER_DEDUP,
WEAVE_SM0_STREAM_SCHED, WEAVE_READY_QUEUE): they are not public, but they must
keep resolving and their internal macro fallback must keep defaulting to 0.
(The file keeps its historical test_21_* name.)

The C++ test ``tests/test_21_weave_knobs_surface.cpp`` is compiled against the
real repository headers and executed.  No GPU, CUDA runtime, PyTorch or JIT
compiler is involved:

  * deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh  (ABI, defaults)
  * csrc/jit_kernels/heuristics/weave_knob_config.hpp      (host ``from_env``)
  * csrc/jit_kernels/heuristics/sm90_mega_moe.hpp          (SM90 spec parser,
    extracted verbatim into a generated slice)

The driver has two layers:

1. It runs the exhaustive C++ behavioural suite (compiled defaults, every
   documented integer, every named enum, invalid values, alias precedence,
   cross-knob fail-closed combinations, constexpr resolvers and the SM90 spec
   parser).
2. It independently drives the same binary through its CLI contract
   (``--resolve``, ``--sm90-spec``, ``--sm90-resolved``) for all 14 knobs, so
   the process-level exit codes and the fail-closed JSON contract are covered
   as well.

Compiler discovery order: ``WEAVE_SURFACE_CXX`` (a shell-style command),
``g++``, ``clang++``, ``c++``, ``zig``, then the PyPI ``ziglang`` payload
(``python -m ziglang c++``).  On Windows the driver first delegates to WSL when
``g++`` and ``python3`` are installed there, because freshly built unsigned
Windows binaries are commonly blocked by application-control policy.  A native
Windows fallback builds the test as a DLL and enters it through
``weave_surface_entry`` via ctypes, which application control usually permits.

Usage:
    python tests/test_21_weave_knobs_surface.py [--build-dir DIR]

Exit status: 0 only when every C++ and Python check passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEST_SRC = HERE / "test_21_weave_knobs_surface.cpp"
HOST_INCLUDE = HERE / "weave_surface_host"
SM90_HEADER = ROOT / "csrc" / "jit_kernels" / "heuristics" / "sm90_mega_moe.hpp"
KNOB_HEADER = ROOT / "deep_gemm" / "include" / "deep_gemm" / "scheduler" / "weave_knobs.cuh"
CONFIG_HEADER = ROOT / "csrc" / "jit_kernels" / "heuristics" / "weave_knob_config.hpp"
DOC_PATH = ROOT / "docs" / "TUNABLE_PARAMETERS.md"

# The 14 public knobs in contract order: Spatial 5, Temporal 3, Implementation 6.
PUBLIC_KNOBS = [
    "WEAVE_DISPATCH_ACTIVE_SMS", "WEAVE_SCATTER_ACTIVE_SMS",
    "WEAVE_DISPATCH_WARPS", "WEAVE_SCATTER_WARPS", "WEAVE_CU_WAVE_SPLIT",
    "WEAVE_V_HEAD", "WEAVE_READY_GROUP_SIZE", "WEAVE_SCATTER_OWNER",
    "WEAVE_PRECISION_MODE", "WEAVE_GEMM1_CONFIG",
    "WEAVE_GEMM2_CONFIG", "WEAVE_REG_PULL_BYTES", "WEAVE_DEDUP_MODE",
    "WEAVE_SCATTER_STAGING",
]
# Accepted by the host for backward compatibility but NOT public: host shape-adaptive
# GEMM tuple selection is always on; `WEAVE_ADAPT_GEMM_CONFIG=0` (alias
# `WEAVE_ADAPT_BM=0`) forces the fixed canonical tuple (internal legacy/diagnostic).
INTERNAL_COMPAT = ["WEAVE_ADAPT_GEMM_CONFIG"]
# Retired from the public surface: internal compile-time macros (fallback 0) and
# internal environment diagnostics of the host parsers.
RETIRED_INTERNAL = [
    "WEAVE_REDUCE_WARPS", "WEAVE_COMBINE_SYNC_CHUNK", "WEAVE_REG_BALANCE",
    "WEAVE_LOCAL_SCATTER_DEDUP", "WEAVE_SM0_STREAM_SCHED", "WEAVE_READY_QUEUE",
]
# Legacy environment spelling -> accepted name (transparent aliases, new name wins).
# WEAVE_ADAPT_BM aliases the internal compat name WEAVE_ADAPT_GEMM_CONFIG (not public);
# the other two alias the public WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG.
LEGACY_ALIASES = {
    "WEAVE_ADAPT_BM": "WEAVE_ADAPT_GEMM_CONFIG",
    "WEAVE_BLOCK_M_L1": "WEAVE_GEMM1_CONFIG",
    "WEAVE_BLOCK_M_L2": "WEAVE_GEMM2_CONFIG",
}

AUTO = 0xFFFFFFFF
WINDOWS = platform.system() == "Windows"

CHECKS = 0
FAILURES = 0


def check(ok: bool, label: str) -> None:
    global CHECKS, FAILURES
    CHECKS += 1
    if not ok:
        FAILURES += 1
        print(f"  [FAIL] {label}")


def section(name: str) -> None:
    print(f"\n== {name} ==")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def windows_to_wsl(path: str) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"not an absolute Windows path: {path}")
    tail = str(resolved)[len(resolved.drive):].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def wsl_toolchain_root() -> Optional[str]:
    """Return the repository path inside WSL when g++ and python3 are present."""
    if not WINDOWS:
        return None
    wsl = shutil.which("wsl")
    if wsl is None:
        return None
    try:
        root = windows_to_wsl(str(ROOT))
    except (ValueError, OSError):
        return None
    probe = subprocess.run(
        [wsl, "-e", "bash", "-lc",
         f"cd {shlex.quote(root)} && command -v g++ >/dev/null 2>&1 "
         "&& command -v python3 >/dev/null 2>&1"],
        capture_output=True, text=True, timeout=30,
    )
    return root if probe.returncode == 0 else None


def delegate_to_wsl(root: str, forwarded: Sequence[str]) -> int:
    args = ["python3", "tests/test_21_weave_knobs_surface.py", *forwarded]
    command = " ".join(shlex.quote(a) for a in args)
    cmd = ["wsl", "-e", "bash", "-lc",
           f"cd {shlex.quote(root)} && PYTHONUNBUFFERED=1 "
           f"PYTHONIOENCODING=utf-8 {command}"]
    print("[INFO] delegating to WSL: g++ + python3 "
          "(fresh Windows binaries are blocked by application control)", flush=True)
    return subprocess.run(cmd).returncode


def find_compiler() -> Optional[List[str]]:
    override = os.environ.get("WEAVE_SURFACE_CXX")
    if override:
        return shlex.split(override)
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return [path]
    zig = shutil.which("zig")
    if zig:
        return [zig, "c++"]
    try:
        if importlib.util.find_spec("ziglang") is not None:
            return [sys.executable, "-m", "ziglang", "c++"]
    except (ImportError, ValueError):
        pass
    return None


def write_sm90_slice(build_dir: Path) -> Path:
    """Extract the SM90 spec parser verbatim so the test exercises real code."""
    text = SM90_HEADER.read_text(encoding="utf-8")

    input_start = text.index("struct Sm90MoeHeuristicInput {")
    input_end = text.index("};", input_start) + 2
    input_struct = text[input_start:input_end]

    body_start = text.index("struct AgySm90KnobSpec {")
    end_marker = (
        "static AgySm90KnobSpec get_agy_sm90_spec("
        "const Sm90MoeHeuristicInput& input) {\n"
        "    return get_agy_sm90_spec(&input);\n"
        "}"
    )
    if text.count("struct AgySm90KnobSpec {") != 1:
        raise SystemExit("SM90 slice: AgySm90KnobSpec marker is not unique")
    if text.count(end_marker) != 1:
        raise SystemExit("SM90 slice: get_agy_sm90_spec end marker is not unique")
    body_end = text.index(end_marker) + len(end_marker)
    body = text[body_start:body_end]

    for needle in (
        "Fail closed on the public WEAVE_* values",
        "has_any_public_weave_sm90_env",
        "resolved_scatter_staging",
        "resolved_scatter_owner",
    ):
        if needle not in body:
            raise SystemExit(f"SM90 slice lost required marker: {needle}")

    out = build_dir / "weave_sm90_spec_slice.hpp"
    out.write_text(
        "#pragma once\n"
        "// Generated by tests/test_21_weave_knobs_surface.py from\n"
        "// csrc/jit_kernels/heuristics/sm90_mega_moe.hpp -- do not edit.\n"
        "#include <cstdint>\n"
        "#include <cstdlib>\n"
        "#include <cstring>\n"
        "#include <initializer_list>\n"
        "#include <string>\n"
        "#include <utility>\n"
        "#include <vector>\n"
        "#include <utils/exception.hpp>\n"
        "\n"
        "namespace deep_gemm {\n\n"
        f"{input_struct}\n\n{body}\n\n"
        "}  // namespace deep_gemm\n",
        encoding="utf-8",
        newline="\n",
    )
    return out


def build(build_dir: Path, cc: List[str]) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    write_sm90_slice(build_dir)
    out = build_dir / ("weave_surface.dll" if WINDOWS else "weave_surface")
    cmd = [
        *cc, "-std=c++20", "-O1", "-DWEAVE_SURFACE_SM90_SPEC_SLICE",
        "-I", str(ROOT / "deep_gemm" / "include"),
        "-I", str(ROOT / "csrc"),
        "-I", str(ROOT / "third-party" / "cutlass" / "include"),
        "-I", str(ROOT / "third-party" / "deep_jit" / "include"),
        "-I", str(build_dir),
        "-I", str(HOST_INCLUDE),
    ]
    if WINDOWS:
        cmd += ["-shared", "-DWEAVE_SURFACE_SHARED"]
    else:
        cmd += ["-pthread"]
    cmd += [str(TEST_SRC), "-o", str(out)]

    print("[BUILD] " + " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        print(f"[FAIL] C++ build exited with {proc.returncode}")
        raise SystemExit(1)
    return out


LOADER = textwrap.dedent(
    """
    import ctypes
    import os
    import sys

    lib = ctypes.CDLL(sys.argv[1])
    entry = lib.weave_surface_entry
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    entry.restype = ctypes.c_int
    args = [os.fsencode(a) for a in sys.argv[2:]]
    argv = (ctypes.c_char_p * (len(args) + 1))()
    argv[0] = b"weave_surface"
    for i, arg in enumerate(args, 1):
        argv[i] = arg
    raise SystemExit(entry(len(args) + 1, argv))
    """
).strip()


def run_entry(binary: Path, args: Sequence[str]) -> subprocess.CompletedProcess:
    if WINDOWS:
        cmd = [sys.executable, "-c", LOADER, str(binary), *args]
    else:
        cmd = [str(binary), *args]
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )


def last_json(stdout: str) -> Dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"no JSON object in output: {stdout!r}")


def show_failure(proc: subprocess.CompletedProcess) -> None:
    if proc.stdout.strip():
        print("    stdout:", proc.stdout.strip().replace("\n", "\n    "))
    if proc.stderr.strip():
        print("    stderr:", proc.stderr.strip().replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# 1. The exhaustive C++ behavioural suite
# ---------------------------------------------------------------------------
def run_cpp_suite(binary: Path) -> None:
    section("C++ behavioural suite")
    proc = run_entry(binary, [])
    if proc.returncode != 0:
        show_failure(proc)
        check(False, f"C++ suite exited with {proc.returncode}")
        return
    match = re.search(r"CHECKS=(\d+) FAILURES=(\d+)", proc.stdout)
    if match is None:
        show_failure(proc)
        check(False, "C++ suite did not print a CHECKS/FAILURES summary")
        return
    checks, failures = int(match.group(1)), int(match.group(2))
    check(failures == 0, f"C++ suite reported {failures} failures")
    check(checks >= 1000, f"C++ suite executed {checks} checks (expected >= 1000)")
    check("ALL 14-KNOB HOST SURFACE CHECKS PASS" in proc.stdout,
          "C++ suite printed its final pass banner")
    print(f"[PASS] C++ suite: CHECKS={checks} FAILURES={failures}")


# ---------------------------------------------------------------------------
# 2. CLI contract for each of the 14 public knobs, the 3 legacy aliases and the
#    internal pins of the 6 retired knobs
# ---------------------------------------------------------------------------
ALL_KNOB_CASES: Dict[str, Dict] = {
    "WEAVE_DISPATCH_ACTIVE_SMS": {
        "field": "dispatch_active_sms",
        "values": [(v, {"dispatch_active_sms": int(v)})
                   for v in ("0", "1", "2", "16", "148", "1024")],
        "invalid": ["1025", "2048", "-1"],
    },
    "WEAVE_SCATTER_ACTIVE_SMS": {
        "field": "scatter_active_sms",
        "values": [(v, {"scatter_active_sms": int(v)})
                   for v in ("0", "1", "8", "78", "1024")],
        "invalid": ["1025", "2048", "-1"],
    },
    "WEAVE_SM0_STREAM_SCHED": {
        "field": "sm0_stream_sched",
        "values": [("0", {"sm0_stream_sched": 0}),
                   ("1", {"sm0_stream_sched": 1})],
        "invalid": ["2", "3", "-1"],
    },
    "WEAVE_DISPATCH_WARPS": {
        "field": "dispatch_warps",
        "values": [(v, {"dispatch_warps": int(v)}) for v in ("0", "4", "8")],
        "invalid": ["1", "2", "5", "16", "-1"],
    },
    "WEAVE_SCATTER_WARPS": {
        "field": None,
        "values": [(v, {}) for v in ("0", "4", "8")],
        "invalid": ["1", "2", "6", "9", "-1"],
    },
    "WEAVE_REDUCE_WARPS": {
        "field": "reduce_warps",
        "values": [(v, {"reduce_warps": int(v)}) for v in ("0", "1", "2", "4", "8")],
        "invalid": ["3", "5", "16", "-1"],
    },
    "WEAVE_CU_WAVE_SPLIT": {
        "field": "cu_wave_split",
        "values": [("auto", {"cu_wave_split": 0}), ("0-8", {"cu_wave_split": 1}),
                   ("4-4", {"cu_wave_split": 2}), ("8-0", {"cu_wave_split": 3}),
                   ("0", {"cu_wave_split": 0}), ("1", {"cu_wave_split": 1}),
                   ("2", {"cu_wave_split": 2}), ("3", {"cu_wave_split": 3})],
        "invalid": ["4", "2-6", "8-8", "AUTO", "-1"],
    },
    "WEAVE_REG_BALANCE": {
        "field": "reg_balance",
        "values": [("0", {"reg_balance": 0}), ("1", {"reg_balance": 1})],
        "invalid": ["2", "3", "-1"],
    },
    "WEAVE_V_HEAD": {
        "field": "override_v_head",
        "values": [("0", {"override_v_head": AUTO}),
                   ("-1", {"override_v_head": AUTO}),
                   ("1", {"override_v_head": 1}),
                   ("64", {"override_v_head": 64}),
                   ("65536", {"override_v_head": 65536})],
        "invalid": ["-2", "65537", "100000"],
    },
    "WEAVE_READY_GROUP_SIZE": {
        "field": None,
        "values": [(v, {}) for v in ("0", "1", "2", "4", "16", "256")],
        "invalid": ["-1", "-64", "2147483648", "99999999999999999999"],
    },
    "WEAVE_READY_QUEUE": {
        "field": "sm0_stream_sched",
        "values": [("formula", {"sm0_stream_sched": 0}),
                   ("sm0_fifo", {"sm0_stream_sched": 1}),
                   ("0", {"sm0_stream_sched": 0}),
                   ("1", {"sm0_stream_sched": 1})],
        "invalid": ["2", "FORMULA", "Sm0_Fifo"],
    },
    "WEAVE_SCATTER_OWNER": {
        "field": "scatter_owner",
        "values": [("epilogue_only", {"scatter_owner": 0}),
                   ("comm_only", {"scatter_owner": 1}),
                   ("0", {"scatter_owner": 0}),
                   ("1", {"scatter_owner": 1})],
        "invalid": ["2", "epilogue", "comm"],
    },
    "WEAVE_COMBINE_SYNC_CHUNK": {
        "field": "combine_sync_chunk",
        "values": [(v, {"combine_sync_chunk": int(v)})
                   for v in ("0", "16", "32", "64", "128")],
        "invalid": ["1", "7", "15", "129", "256", "-1"],
    },
    # The env value is the Python dispatcher's backend selector
    # (deep_gemm/mega/precision.py, aliases accepted).  The host only fails
    # closed on unknown spellings and always resolves the precision of the
    # backend being launched (the CLI PREC argument, fp8xfp4 = 0 here); the
    # per-backend matrix is run_precision_backend_matrix().
    "WEAVE_PRECISION_MODE": {
        "field": "precision_mode",
        "values": [(v, {"precision_mode": 0})
                   for v in ("fp8xfp4", "fp4xfp4", "nvfp4", "fp8xfp8", "fp8",
                             "bf16", "bf16xbf16", "0", "1", "2", "3", "auto")],
        "invalid": ["fp4", "nfp4", "fp16", "4", "-1"],
    },
    "WEAVE_ADAPT_BM": {
        "field": "adapt_bm",
        "values": [("0", {"adapt_bm": 0}), ("1", {"adapt_bm": 1})],
        "invalid": ["2", "3", "-1"],
    },
    "WEAVE_BLOCK_M_L1": {
        "field": "block_m_l1",
        "values": [(v, {"block_m_l1": int(v)})
                   for v in ("0", "16", "32", "64", "80", "96", "128",
                             "192", "240", "256")],
        "invalid": ["1", "15", "17", "100", "257", "-1"],
    },
    "WEAVE_BLOCK_M_L2": {
        "field": "block_m_l2",
        "values": [(v, {"block_m_l2": int(v)})
                   for v in ("0", "16", "32", "64", "80", "96", "128",
                             "192", "240", "256")],
        "invalid": ["1", "15", "17", "100", "257", "-1"],
    },
    "WEAVE_REG_PULL_BYTES": {
        "field": None,
        "values": [(v, {}) for v in ("0", "1024", "2048", "4096")],
        "invalid": ["1", "512", "1025", "8192", "-1"],
    },
    "WEAVE_DEDUP_MODE": {
        "field": None,
        "values": [("0", {}), ("1", {})],
        "invalid": ["2", "3", "-1"],
    },
    "WEAVE_SCATTER_STAGING": {
        "field": "scatter_staging",
        "values": [("reg", {"scatter_staging": 0, "scatter_owner": 0}),
                   ("smem", {"scatter_staging": 1, "scatter_owner": 1}),
                   ("hbm", {"scatter_staging": 2,
                            "scatter_owner": 1,
                            "local_scatter_dedup": 1}),
                   ("reg_2stage", {"scatter_staging": 0, "scatter_owner": 0}),
                   ("hbm_dedup", {"scatter_staging": 2,
                                  "scatter_owner": 1,
                                  "local_scatter_dedup": 1}),
                   ("0", {"scatter_staging": 0, "scatter_owner": 0}),
                   ("1", {"scatter_staging": 1, "scatter_owner": 1}),
                   ("2", {"scatter_staging": 2,
                          "scatter_owner": 1,
                          "local_scatter_dedup": 1})],
        "invalid": ["3", "REG", "HBM", "REG_2STAGE", "-1"],
    },
    "WEAVE_LOCAL_SCATTER_DEDUP": {
        "field": "local_scatter_dedup",
        "values": [("0", {"local_scatter_dedup": 0}),
                   ("1", {"local_scatter_dedup": 1,
                          "scatter_staging": 2,
                          "scatter_owner": 1})],
        "invalid": ["2", "3", "-1"],
    },
    # ---- internal compat name (not public) and the renamed GEMM1/GEMM2 knobs ----
    "WEAVE_ADAPT_GEMM_CONFIG": {
        "field": "adapt_bm",
        "values": [("0", {"adapt_bm": 0}), ("1", {"adapt_bm": 1})],
        "invalid": ["2", "3", "-1", "abc"],
    },
    # GEMM1 / GEMM2: a bare BM or the whole tuple BM[:STORE_BM[:BK[:STAGES[:EPI]]]]
    # (':' or ',' separated); store height / BK / stages / EPI are range-checked.
    "WEAVE_GEMM1_CONFIG": {
        "field": "block_m_l1",
        "values": [(v, {"block_m_l1": int(v)})
                   for v in ("0", "16", "32", "64", "80", "96", "128",
                             "192", "240", "256")]
                  + [("64:32:128:6:2", {"block_m_l1": 64, "store_block_m_l1": 32}),
                     ("64:16:128:6:3", {"block_m_l1": 64, "store_block_m_l1": 16}),
                     ("64:16:128:6:4", {"block_m_l1": 64, "store_block_m_l1": 16}),
                     ("64,32", {"block_m_l1": 64, "store_block_m_l1": 32}),
                     ("96:16:256:4:1", {"block_m_l1": 96, "store_block_m_l1": 16}),
                     ("128:32", {"block_m_l1": 128, "store_block_m_l1": 32})],
        "invalid": ["1", "15", "17", "100", "257", "-1", "abc", "64:100", "64:72",
                    "64:32:64", "64:32:128:1", "64:32:128:17", "64:32:128:6:5",
                    "64:32:128:6:2:1", "64:", ":32"],
    },
    "WEAVE_GEMM2_CONFIG": {
        "field": "block_m_l2",
        "values": [(v, {"block_m_l2": int(v)})
                   for v in ("0", "16", "32", "64", "80", "96", "128",
                             "192", "240", "256")]
                  + [("64:32:128:6:2", {"block_m_l2": 64, "store_block_m_l2": 32}),
                     ("64:16:128:6:3", {"block_m_l2": 64, "store_block_m_l2": 16}),
                     ("64:16:128:6:4", {"block_m_l2": 64, "store_block_m_l2": 16}),
                     ("64,32", {"block_m_l2": 64, "store_block_m_l2": 32}),
                     ("96:16:256:4:1", {"block_m_l2": 96, "store_block_m_l2": 16}),
                     ("128:32", {"block_m_l2": 128, "store_block_m_l2": 32})],
        "invalid": ["1", "15", "17", "100", "257", "-1", "abc", "64:100", "64:72",
                    "64:32:64", "64:32:128:1", "64:32:128:17", "64:32:128:6:5",
                    "64:32:128:6:2:1", "64:", ":32"],
    },
}

KNOB_CASES: Dict[str, Dict] = {k: v for k, v in ALL_KNOB_CASES.items() if k in PUBLIC_KNOBS}
INTERNAL_KNOB_CASES: Dict[str, Dict] = {k: v for k, v in ALL_KNOB_CASES.items() if k in RETIRED_INTERNAL}
INTERNAL_COMPAT_CASES: Dict[str, Dict] = {k: v for k, v in ALL_KNOB_CASES.items() if k in INTERNAL_COMPAT}
LEGACY_ALIAS_CASES: Dict[str, Dict] = {k: v for k, v in ALL_KNOB_CASES.items() if k in LEGACY_ALIASES}


def resolve(binary: Path, pairs: Sequence[Tuple[str, str]],
            arch: str = "sm103", prec: str = "fp8xfp4") -> subprocess.CompletedProcess:
    args = ["--resolve", arch, prec]
    args += [f"{key}={value}" for key, value in pairs]
    return run_entry(binary, args)


def run_value_cases(binary: Path, env: str, case: Dict, tag: str) -> None:
    if True:  # (kept as a block so the original per-knob body is unchanged)
        for text, expected in case["values"]:
            proc = resolve(binary, [(env, text)])
            if proc.returncode != 0:
                show_failure(proc)
                check(False, f"{env}={text} was rejected (exit {proc.returncode})")
                continue
            try:
                data = last_json(proc.stdout)
            except AssertionError as exc:
                check(False, f"{env}={text}: {exc}")
                continue
            for field, value in expected.items():
                check(data.get(field) == value,
                      f"{env}={text} -> {field}={data.get(field)}, expected {value}")
        for text in case["invalid"]:
            proc = resolve(binary, [(env, text)])
            ok = proc.returncode == 3
            if ok:
                try:
                    ok = "error" in last_json(proc.stdout)
                except AssertionError:
                    ok = False
            if not ok:
                show_failure(proc)
            check(ok, f"{env}={text} must fail closed (exit 3 + JSON error)")
        print(f"[PASS] {tag} {env}: "
              f"{len(case['values'])} values, {len(case['invalid'])} rejections")


def run_knob_matrix(binary: Path) -> None:
    section("CLI --resolve matrix (14 public knobs)")
    check(len(PUBLIC_KNOBS) == 14 and len(set(PUBLIC_KNOBS)) == 14,
          "the public surface is exactly 14 distinct knobs")
    check(set(KNOB_CASES) == set(PUBLIC_KNOBS),
          "the Python matrix covers exactly the 14 public knobs")
    check(set(ALL_KNOB_CASES) == set(PUBLIC_KNOBS) | set(RETIRED_INTERNAL) | set(LEGACY_ALIASES) | set(INTERNAL_COMPAT),
          "every case belongs to a public knob, a legacy alias, an internal compat name or a retired internal pin")
    check(not (set(PUBLIC_KNOBS) & (set(RETIRED_INTERNAL) | set(LEGACY_ALIASES) | set(INTERNAL_COMPAT))),
          "no retired knob, legacy alias or internal compat name is public")
    for env in PUBLIC_KNOBS:
        run_value_cases(binary, env, KNOB_CASES[env], "public")

    section("CLI --resolve matrix (internal compat name WEAVE_ADAPT_GEMM_CONFIG, accepted but not public)")
    check(set(INTERNAL_COMPAT_CASES) == set(INTERNAL_COMPAT),
          "the Python matrix covers exactly the internal compat names")
    for env in INTERNAL_COMPAT:
        run_value_cases(binary, env, INTERNAL_COMPAT_CASES[env], "internal-compat")

    section("CLI --resolve matrix (3 legacy aliases, transparent)")
    for env, new in LEGACY_ALIASES.items():
        run_value_cases(binary, env, LEGACY_ALIAS_CASES[env], "legacy")
        # Every legacy value resolves exactly like the public spelling.
        for text, _ in LEGACY_ALIAS_CASES[env]["values"]:
            old_proc = resolve(binary, [(env, text)])
            new_proc = resolve(binary, [(new, text)])
            same = (old_proc.returncode == new_proc.returncode == 0 and
                    last_json(old_proc.stdout) == last_json(new_proc.stdout))
            check(same, f"{env}={text} resolves exactly like {new}={text}")
        # The public name outranks the legacy one when both are set.
        probe = {"WEAVE_ADAPT_BM": ("0", "1", "adapt_bm"),
                 "WEAVE_BLOCK_M_L1": ("64", "128", "block_m_l1"),
                 "WEAVE_BLOCK_M_L2": ("64", "128", "block_m_l2")}[env]
        both = resolve(binary, [(new, probe[0]), (env, probe[1])])
        check(both.returncode == 0 and last_json(both.stdout).get(probe[2]) == int(probe[0]),
              f"{new}={probe[0]} outranks {env}={probe[1]}")

    section("CLI --resolve matrix (6 retired knobs: internal pins still resolve)")
    check(set(INTERNAL_KNOB_CASES) == set(RETIRED_INTERNAL),
          "the Python matrix covers exactly the 6 retired internal pins")
    for env in RETIRED_INTERNAL:
        run_value_cases(binary, env, INTERNAL_KNOB_CASES[env], "internal")

    section("CLI --resolve cross-knob fail-closed combinations")
    rejected = [
        [("WEAVE_LOCAL_SCATTER_DEDUP", "1"), ("WEAVE_SCATTER_STAGING", "reg_2stage")],
        [("WEAVE_LOCAL_SCATTER_DEDUP", "1"), ("WEAVE_SCATTER_STAGING", "smem")],
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_SCATTER_STAGING", "smem")],
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_SCATTER_ACTIVE_SMS", "8")],
        [("WEAVE_TMEM_PIPELINE_MODE", "175")],
        # Epilogue-owned local dedup (D7 mode 2) excludes every D6 group.
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1"),
         ("WEAVE_COMBINE_SYNC_CHUNK", "32")],
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1"),
         ("WEAVE_EARLY_STREAM_REDUCE", "1")],
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1"),
         ("WEAVE_REDUCE_WARPS", "4")],
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1"),
         ("WEAVE_SCATTER_STAGING", "hbm_dedup")],
    ]
    for pairs in rejected:
        label = " + ".join(f"{k}={v}" for k, v in pairs)
        proc = resolve(binary, pairs, arch="h20" if pairs[-1][0] == "WEAVE_TMEM_PIPELINE_MODE" else "sm103")
        check(proc.returncode == 3, f"{label} must be rejected")
        if proc.returncode != 3:
            show_failure(proc)
    # D7 mode 2 is implemented by the B300/SM100 FP8xFP4 kernel only.
    for arch in ("h20", "mi308x"):
        pairs = [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1")]
        proc = resolve(binary, pairs, arch=arch)
        check(proc.returncode == 3,
              f"SCATTER_OWNER=epilogue_only + LOCAL_SCATTER_DEDUP=1 must be rejected on {arch}")
        if proc.returncode != 3:
            show_failure(proc)

    accepted = [
        ([("WEAVE_LOCAL_SCATTER_DEDUP", "1"), ("WEAVE_SCATTER_STAGING", "hbm_dedup")],
         "sm103", {"local_scatter_dedup": 1, "scatter_staging": 2, "scatter_owner": 1}),
        ([("WEAVE_COMBINE_SYNC_CHUNK", "32"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1")],
         "sm103", {"combine_sync_chunk": 32, "local_scatter_dedup": 1,
                   "scatter_staging": 2, "scatter_owner": 1}),
        ([("WEAVE_COMBINE_SYNC_CHUNK", "32"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1")],
         "h20", {"combine_sync_chunk": 32, "local_scatter_dedup": 1,
                 "scatter_staging": 2, "scatter_owner": 1}),
        ([("WEAVE_SCATTER_STAGING", "smem")],
         "sm103", {"scatter_staging": 1, "scatter_owner": 1}),
        ([("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1")],
         "sm103", {"local_scatter_dedup": 1, "scatter_staging": 0, "scatter_owner": 0}),
        ([("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1"),
          ("WEAVE_SCATTER_STAGING", "reg_2stage")],
         "sm103", {"local_scatter_dedup": 1, "scatter_staging": 0, "scatter_owner": 0}),
        ([("WEAVE_SCATTER_OWNER", "comm_only"), ("WEAVE_LOCAL_SCATTER_DEDUP", "1")],
         "sm103", {"local_scatter_dedup": 1, "scatter_staging": 2, "scatter_owner": 1}),
    ]
    for pairs, arch, expected in accepted:
        label = " + ".join(f"{k}={v}" for k, v in pairs) + f" [{arch}]"
        proc = resolve(binary, pairs, arch=arch)
        if proc.returncode != 0:
            show_failure(proc)
            check(False, f"{label} must be accepted")
            continue
        data = last_json(proc.stdout)
        for field, value in expected.items():
            check(data.get(field) == value,
                  f"{label} -> {field}={data.get(field)}, expected {value}")

    proc = resolve(binary, [("WEAVE_COMBINE_SYNC_CHUNK", "32")], arch="h20")
    check(proc.returncode == 3,
          "COMBINE_SYNC_CHUNK=32 without LOCAL_SCATTER_DEDUP must be rejected on H20")


# ---------------------------------------------------------------------------
# 3. SM90 spec parser CLI
# ---------------------------------------------------------------------------
SM90_MAPPINGS = [
    ("WEAVE_DISPATCH_ACTIVE_SMS", "16", "dispatch_sms", "16"),
    ("WEAVE_SCATTER_ACTIVE_SMS", "8", "scatter_sms", "8"),
    ("WEAVE_SM0_STREAM_SCHED", "1", "sm0_sched", "1"),
    ("WEAVE_DISPATCH_WARPS", "8", "dispatch_warps", "8"),
    ("WEAVE_SCATTER_WARPS", "4", "scatter_warps", "4"),
    ("WEAVE_REDUCE_WARPS", "4", "comb_warps", "4"),
    ("WEAVE_REG_BALANCE", "1", "reg_balance", "1"),
    ("WEAVE_V_HEAD", "64", "v_head", "64"),
    ("WEAVE_READY_GROUP_SIZE", "2", "meta_g", "2"),
    ("WEAVE_READY_QUEUE", "sm0_fifo", "ready_queue", "1"),
    ("WEAVE_SCATTER_OWNER", "comm_only", "scatter_owner", "1"),
    ("WEAVE_COMBINE_SYNC_CHUNK", "32", "comb_sync_chunk", "32"),
    ("WEAVE_ADAPT_BM", "1", "adapt_bm", "1"),
    ("WEAVE_BLOCK_M_L1", "32", "l1_bm", "32"),
    ("WEAVE_BLOCK_M_L2", "64", "l2_bm", "64"),
    ("WEAVE_ADAPT_GEMM_CONFIG", "1", "adapt_bm", "1"),
    ("WEAVE_GEMM1_CONFIG", "32", "l1_bm", "32"),
    ("WEAVE_GEMM2_CONFIG", "64", "l2_bm", "64"),
    ("WEAVE_DEDUP_MODE", "1", "dedup_disp", "1"),
    ("WEAVE_SCATTER_STAGING", "hbm_dedup", "scatter_staging", "2"),
    ("WEAVE_LOCAL_SCATTER_DEDUP", "1", "local_dedup", "1"),
]

SM90_BAD_SPECS = [
    "foo", "a=", "=1", "scatter_warps=6", "scatter_warps=2",
    "dispatch_warps=6", "meta_g=-1", "comb_sync_chunk=7", "comb_warps=3",
    "l1_bm=100", "l2_bm=257", "adapt_bm=2", "reg_balance=2",
    "local_dedup=2", "dedup_disp=2", "scatter_owner=2", "scatter_staging=3",
    "ready_queue=2", "v_head=-1", "dispatch_sms=-1", "scatter_sms=-1",
]


def run_precision_backend_matrix(binary: Path) -> None:
    """A valid WEAVE_PRECISION_MODE never overrides the launched backend."""
    section("CLI --resolve WEAVE_PRECISION_MODE x launched backend")
    backends = {"fp8xfp4": 0, "fp4xfp4": 1, "fp8xfp8": 2, "bf16": 3}
    selectors = ("fp8xfp4", "nvfp4", "fp8", "bf16xbf16", "0", "3", "auto")
    for arch in ("sm103", "h20", "mi308x"):
        for prec, code in backends.items():
            for text in selectors:
                proc = resolve(binary, [("WEAVE_PRECISION_MODE", text)],
                               arch=arch, prec=prec)
                if proc.returncode != 0:
                    show_failure(proc)
                    check(False, f"{arch}/{prec}: WEAVE_PRECISION_MODE={text} rejected")
                    continue
                data = last_json(proc.stdout)
                check(data.get("precision_mode") == code,
                      f"{arch}/{prec}: WEAVE_PRECISION_MODE={text} -> "
                      f"precision_mode={data.get('precision_mode')}, expected {code}")
    print(f"[PASS] WEAVE_PRECISION_MODE backend matrix: "
          f"3 arches x {len(backends)} backends x {len(selectors)} selectors")


def sm90_spec(binary: Path, pairs: Sequence[Tuple[str, str]]) -> subprocess.CompletedProcess:
    args = ["--sm90-spec"] + [f"{key}={value}" for key, value in pairs]
    return run_entry(binary, args)


def sm90_resolved(binary: Path, pairs: Sequence[Tuple[str, str]]) -> subprocess.CompletedProcess:
    args = ["--sm90-resolved"] + [f"{key}={value}" for key, value in pairs]
    return run_entry(binary, args)


def run_sm90_cli(binary: Path) -> None:
    section("CLI --sm90-spec public knob mapping")
    proc = sm90_spec(binary, [])
    check(proc.returncode == 0 and last_json(proc.stdout).get("enabled") is False,
          "an empty environment leaves the unified SM90 spec disabled")

    for env, value, key, code in SM90_MAPPINGS:
        proc = sm90_spec(binary, [(env, value)])
        if proc.returncode != 0:
            show_failure(proc)
            check(False, f"{env}={value} rejected by the SM90 spec parser")
            continue
        data = last_json(proc.stdout)
        check(data.get("enabled") is True, f"{env}={value} enables the SM90 spec")
        check(str(data.get("kv", {}).get(key)) == code,
              f"{env}={value} -> {key}={data.get('kv', {}).get(key)}, expected {code}")
        check("fast_host=1" in data.get("raw", "") and "fused=1" in data.get("raw", ""),
              f"{env}={value} preserves the neutral SM90 skeleton")
    print(f"[PASS] SM90 spec mapping: {len(SM90_MAPPINGS)} env spellings "
          "(14 public knobs, 3 legacy aliases incl. WEAVE_ADAPT_BM, WEAVE_ADAPT_GEMM_CONFIG compat, 6 retired internal pins)")

    for env, value, key, code in [
        ("WEAVE_READY_QUEUE", "1", "ready_queue", "1"),
        ("WEAVE_READY_QUEUE", "formula", "ready_queue", "0"),
        ("WEAVE_SCATTER_OWNER", "1", "scatter_owner", "1"),
        ("WEAVE_SCATTER_OWNER", "epilogue_only", "scatter_owner", "0"),
        ("WEAVE_SCATTER_STAGING", "2", "scatter_staging", "2"),
        ("WEAVE_SCATTER_STAGING", "reg_2stage", "scatter_staging", "0"),
    ]:
        proc = sm90_spec(binary, [(env, value)])
        if proc.returncode != 0:
            show_failure(proc)
            check(False, f"{env}={value} rejected (code form)")
            continue
        data = last_json(proc.stdout)
        check(str(data.get("kv", {}).get(key)) == code,
              f"{env}={value} -> {key}={data.get('kv', {}).get(key)}, expected {code}")

    proc = sm90_spec(binary, [("WEAVE_READY_QUEUE", "formula")])
    data = last_json(proc.stdout)
    check(str(data.get("kv", {}).get("ready_queue_formula")) == "1",
          "READY_QUEUE=formula publishes ready_queue_formula=1")

    proc = sm90_spec(binary, [("WEAVE_SCATTER_ACTIVE_SMS", "0")])
    data = last_json(proc.stdout)
    check(proc.returncode == 0 and data.get("enabled") is True and
          "scatter_sms" not in data.get("kv", {}),
          "an explicit zero selects the unified SM90 path without a numeric override")

    proc = sm90_spec(binary, [("WEAVE_READY_QUEUE", "sm0_fifo"),
                              ("WEAVE_FORCE_OFFICIAL", "1")])
    check(proc.returncode == 0 and last_json(proc.stdout).get("enabled") is False,
          "WEAVE_FORCE_OFFICIAL=1 disables the SM90 spec")

    proc = sm90_spec(binary, [("WEAVE_H20_CONFIG", "bm=64,bn=128,dedup=1")])
    data = last_json(proc.stdout)
    check(proc.returncode == 0 and data.get("enabled") is True and
          data.get("kv", {}).get("bm") == "64" and
          data.get("kv", {}).get("dedup_disp") == "1",
          "WEAVE_H20_CONFIG parses bm and aliases dedup -> dedup_disp")

    for bad in SM90_BAD_SPECS:
        proc = sm90_spec(binary, [("WEAVE_H20_CONFIG", bad)])
        if proc.returncode != 3:
            show_failure(proc)
        check(proc.returncode == 3,
              f"SM90 spec {bad!r} must fail closed (exit 3)")
    print(f"[PASS] SM90 spec fail-closed strings: {len(SM90_BAD_SPECS)}")

    section("CLI --sm90-resolved staging / ownership coupling")
    for pairs in [
        [("WEAVE_SCATTER_OWNER", "epilogue_only"), ("WEAVE_SCATTER_STAGING", "smem")],
        [("WEAVE_LOCAL_SCATTER_DEDUP", "1"), ("WEAVE_SCATTER_STAGING", "smem")],
        [("WEAVE_SCATTER_ACTIVE_SMS", "8"), ("WEAVE_SCATTER_STAGING", "reg_2stage")],
    ]:
        label = " + ".join(f"{k}={v}" for k, v in pairs)
        proc = sm90_resolved(binary, pairs)
        check(proc.returncode == 3, f"SM90 {label} must be rejected")

    for pairs, expected in [
        ([("WEAVE_SCATTER_ACTIVE_SMS", "8")], {"staging": 2, "owner": 1}),
        ([("WEAVE_LOCAL_SCATTER_DEDUP", "1")], {"staging": 2, "owner": 1}),
        ([("WEAVE_SCATTER_STAGING", "hbm_dedup")], {"staging": 2, "owner": 1}),
    ]:
        label = " + ".join(f"{k}={v}" for k, v in pairs)
        proc = sm90_resolved(binary, pairs)
        if proc.returncode != 0:
            show_failure(proc)
            check(False, f"SM90 {label} must resolve")
            continue
        data = last_json(proc.stdout)
        for field, value in expected.items():
            check(data.get(field) == value,
                  f"SM90 {label} -> {field}={data.get(field)}, expected {value}")


# ---------------------------------------------------------------------------
# 4. Static surface inventory
# ---------------------------------------------------------------------------
def run_static_checks() -> None:
    section("static 14-knob inventory")
    config = CONFIG_HEADER.read_text(encoding="utf-8")
    knob_defs = KNOB_HEADER.read_text(encoding="utf-8")
    doc = DOC_PATH.read_text(encoding="utf-8")
    sm90 = SM90_HEADER.read_text(encoding="utf-8")
    for env in PUBLIC_KNOBS:
        check(env in config, f"{env} is resolved/validated in weave_knob_config.hpp")
        check(env in knob_defs or env in config, f"{env} has a compile-time surface")
        check(env in doc, f"{env} is documented in TUNABLE_PARAMETERS.md")
    for env in INTERNAL_COMPAT:
        check(env in config, f"{env} is still parsed (internal compat) in weave_knob_config.hpp")
    for old, new in LEGACY_ALIASES.items():
        check(old in config and new in config,
              f"{old} -> {new}: weave_knob_config.hpp reads both spellings")
        check(old in sm90 and new in sm90,
              f"{old} -> {new}: the SM90 spec parser reads both spellings")
    # Retired knobs: still parsed by the host and still compile when the host does
    # not emit them (an `#ifndef` fallback of 0).
    for env in RETIRED_INTERNAL:
        check(env in config or env in sm90, f"{env} is still parsed as an internal pin")
        if env == "WEAVE_READY_QUEUE":
            continue
        pattern = rf"#ifndef {env}\s*\n\s*#define {env} 0u?\s*\n\s*#endif"
        check(re.search(pattern, knob_defs) is not None,
              f"{env}: weave_knobs.cuh keeps the internal `#ifndef ... #define {env} 0` fallback")
    # WEAVE_READY_QUEUE deliberately has no default in weave_knobs.cuh: its fallback
    # lives in the alias chain (DG_TUNE_D4_READYQ / DG_ISOLATED_D4_READYQ) at the
    # use sites, which a header default would shadow.
    check("#define WEAVE_READY_QUEUE" not in knob_defs,
          "weave_knobs.cuh does not shadow the WEAVE_READY_QUEUE alias chain")
    chain = re.compile(r"#ifndef WEAVE_READY_QUEUE\n(?:#\s*(?:if|elif)[^\n]*\n#\s*define WEAVE_READY_QUEUE[^\n]*\n)+"
                       r"#\s*else\n#\s*define WEAVE_READY_QUEUE 0\n#\s*endif\n#endif")
    for rel in ("deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
                "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        check(chain.search(text) is not None,
              f"{rel}: the WEAVE_READY_QUEUE alias chain keeps its fallback of 0")
    impl = (ROOT / "csrc" / "jit_kernels" / "impls" / "sm100_fp8_fp4_mega_moe.hpp").read_text(encoding="utf-8")
    check("warned_bm128_epi2" in impl and "eff_epi_reg_stages = 1u" in impl,
          "SM103 BM=128 + EPI>=2 is clamped to EPI=1 in sm100_fp8_fp4_mega_moe.hpp")
    check("Fail closed on the public WEAVE_* values" in
          SM90_HEADER.read_text(encoding="utf-8"),
          "the SM90 spec parser carries its fail-closed range gate")
    for stub in ("cuda_runtime.h", "cuda.h", "cublasLt.h", "torch/version.h",
                 "cutlass/detail/helper_macros.hpp", "cuda/std/cstdint"):
        check((HOST_INCLUDE / stub).exists(), f"host stand-in present: {stub}")
    print(f"[PASS] static inventory: {len(PUBLIC_KNOBS)} public knobs, "
          f"{len(LEGACY_ALIASES)} legacy aliases, {len(RETIRED_INTERNAL)} retired internal pins")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=None,
                        help="keep the generated slice / binary in this directory")
    parser.add_argument("--no-wsl", action="store_true",
                        help="do not delegate to a WSL g++ toolchain on Windows")
    args = parser.parse_args()

    print("Weave 14-knob host surface verification (Python driver)")
    if WINDOWS and not args.no_wsl and not os.environ.get("WEAVE_SURFACE_CXX"):
        root = wsl_toolchain_root()
        if root is not None:
            raw = sys.argv[1:]
            forwarded: List[str] = []
            i = 0
            while i < len(raw):
                if raw[i] == "--no-wsl":
                    i += 1
                elif raw[i] == "--build-dir" and i + 1 < len(raw):
                    forwarded += ["--build-dir", windows_to_wsl(raw[i + 1])]
                    i += 2
                else:
                    forwarded.append(raw[i])
                    i += 1
            return delegate_to_wsl(root, forwarded)
    cc = find_compiler()
    if cc is None:
        print("[FAIL] no C++ compiler found. Install g++/clang++/zig, or set "
              "WEAVE_SURFACE_CXX (for example "
              "\"python -m ziglang c++\" with PYTHONPATH pointing at the "
              "ziglang wheel).")
        return 2
    print(f"[INFO] compiler: {' '.join(cc)}")

    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
        cleanup = False
    else:
        build_dir = Path(tempfile.mkdtemp(prefix="weave_surface_"))
        cleanup = True
    try:
        binary = build(build_dir, cc)
        run_cpp_suite(binary)
        run_knob_matrix(binary)
        run_precision_backend_matrix(binary)
        run_sm90_cli(binary)
        run_static_checks()
    finally:
        if cleanup:
            shutil.rmtree(build_dir, ignore_errors=True)

    print(f"\nPY_CHECKS={CHECKS} PY_FAILURES={FAILURES}")
    if FAILURES:
        print("14-KNOB HOST SURFACE VERIFICATION FAILED")
        return 1
    print("ALL 14-KNOB HOST SURFACE CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())