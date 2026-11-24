#!/usr/bin/env python3
"""EP16 num_shared_experts ABI gate (CPU-only).

Compiles tests/test_ep16_shared_experts.cpp against the host headers in
deep_gemm/include/deep_gemm/impls/amd/ep16/host/ and asserts the
lossless-by-construction contract of the v59 shared-experts wiring:

  * frozen ABI sizes (the shared fields live in reserved words);
  * num_shared_experts == 0 keeps the legacy behavior byte-identical
    (ck_flags 0, legacy route_rows formula, no pointer requirements);
  * 1..8 shared experts: validation, row accounting, recommended
    route_rows and the ck_flags [3:0] nibble;
  * fail-closed codes for every malformed configuration;
  * device-side TU isolation of the relaxed ck_flags check in the shared
    amd/include/sub6_6models_generalize.hpp: the EP16 TU (amd/megamoe_amd.hip)
    defines MI308X_EP16_CK_SHARED_EXPERTS and sees
    `(ck.ck_flags & ~0xfu) != 0u`; the EP8 TU (ep8/megamoe_amd_ep8.hip) never
    defines it, sees the original `ck.ck_flags != 0u`, and its view of the
    header is byte-identical to the a0c19cb baseline.

On Windows the driver delegates to WSL when g++ + python3 are available
(the same pattern as tests/test_two_level_adaptive_policy.py).
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_DIR = ROOT / "deep_gemm/include/deep_gemm/impls/amd/ep16/host"
AMD_DIR = ROOT / "deep_gemm/include/deep_gemm/impls/amd"
SUB6 = AMD_DIR / "include/sub6_6models_generalize.hpp"
EP16_TU = AMD_DIR / "megamoe_amd.hip"
EP8_TU = AMD_DIR / "ep8/megamoe_amd_ep8.hip"
EP16_MACRO = "MI308X_EP16_CK_SHARED_EXPERTS"
EP8_PRED = "if (ck.ck_flags != 0u) return false;"
EP16_PRED = "if ((ck.ck_flags & ~0xfu) != 0u) return false;"
# sha256 of amd/include/sub6_6models_generalize.hpp at a0c19cb (LF), i.e. the
# header the silicon-measured EP8 .co was built from.
SUB6_A0C19CB_SHA256 = "b321a3768878c0cfd6f6d5e8359650664e499fefbff5e8c459110dd350b7f107"
WINDOWS = os.name == "nt"


def windows_to_wsl(path: str) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[len(resolved.drive):].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def maybe_delegate_to_wsl():
    if not WINDOWS:
        return None
    wsl = shutil.which("wsl")
    if wsl is None:
        return None
    root = windows_to_wsl(str(ROOT))
    probe = subprocess.run(
        [wsl, "-e", "bash", "-lc",
         f"cd {shlex.quote(root)} && command -v g++ >/dev/null 2>&1 "
         "&& command -v python3 >/dev/null 2>&1"],
        capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        return None
    args = ["python3", "tests/test_ep16_shared_experts.py", *sys.argv[1:]]
    command = " ".join(shlex.quote(a) for a in args)
    print("[INFO] delegating to WSL: g++ + python3", flush=True)
    return subprocess.run(
        [wsl, "-e", "bash", "-lc",
         f"cd {shlex.quote(root)} && PYTHONUNBUFFERED=1 "
         f"PYTHONIOENCODING=utf-8 {command}"]).returncode


def find_compiler():
    override = os.environ.get("WEAVE_SURFACE_CXX")
    if override:
        return shlex.split(override)
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return [path]
    return None


def read_lf(path: Path) -> str:
    return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def resolve_ep8_view(text: str) -> str:
    """Drop the `#if defined(EP16_MACRO)` branch, keep the `#else` branch."""
    out, state = [], None
    for line in text.splitlines(keepends=True):
        if line.strip() == f"#if defined({EP16_MACRO})":
            if state is not None:
                raise SystemExit("[FAIL] nested EP16 guard in sub6 header")
            state = "ep16"
            continue
        if state == "ep16" and line.strip() == "#else":
            state = "ep8"
            continue
        if state in ("ep16", "ep8") and line.strip() == "#endif":
            state = None
            continue
        if state != "ep16":
            out.append(line)
    if state is not None:
        raise SystemExit("[FAIL] unterminated EP16 guard in sub6 header")
    return "".join(out)


def preprocess(cc, text: str, defines):
    # The header's own #include lines are stripped so the preprocessor needs
    # no HIP headers; only the conditional logic is evaluated.
    body = re.sub(r"(?m)^\s*#\s*include\b.*$", "", text)
    cmd = [*cc, "-E", "-P", "-x", "c++", *[f"-D{d}" for d in defines], "-"]
    proc = subprocess.run(cmd, input=body, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"[FAIL] preprocessor exited with {proc.returncode}")
    return proc.stdout


def tu_isolation_checks(cc) -> int:
    n_pass = 0

    def check(ok: bool, msg: str):
        nonlocal n_pass
        if not ok:
            raise SystemExit(f"[FAIL] {msg}")
        n_pass += 1
        print(f"[PASS] {msg}")

    sub6 = read_lf(SUB6)
    ep8_view = resolve_ep8_view(sub6)
    check(hashlib.sha256(ep8_view.encode("utf-8")).hexdigest() == SUB6_A0C19CB_SHA256,
          "EP8 view of sub6 header (EP16 guard resolved to #else) is byte-identical to a0c19cb")
    ep8_pp = preprocess(cc, sub6, [])
    ep16_pp = preprocess(cc, sub6, [f"{EP16_MACRO}=1"])
    check(ep8_pp.count(EP8_PRED) == 1 and EP16_PRED not in ep8_pp,
          "EP8 (macro undefined): preprocessor emits exactly `ck.ck_flags != 0u`")
    check(ep16_pp.count(EP16_PRED) == 1 and EP8_PRED not in ep16_pp,
          "EP16 (macro defined): preprocessor emits exactly `(ck.ck_flags & ~0xfu) != 0u`")
    check(ep8_pp == preprocess(cc, ep8_view, []),
          "EP8 preprocessed sub6 token stream == a0c19cb header token stream")

    ep16_tu = read_lf(EP16_TU)
    inc = ep16_tu.index('#include "sub6_6models_generalize.hpp"')
    m = re.search(rf"(?m)^#define {EP16_MACRO} 1$", ep16_tu)
    check(m is not None and m.start() < inc and ep16_tu.count('sub6_6models_generalize') == 1,
          "EP16 TU defines the macro before its only sub6 include")

    ep8_tu = read_lf(EP8_TU)
    check('#include "sub6_6models_generalize.hpp"' in ep8_tu,
          "EP8 TU includes the shared sub6 header")
    ep8_hits = [f for f in (AMD_DIR / "ep8").rglob("*")
                if f.is_file() and EP16_MACRO.encode() in f.read_bytes()]
    check(not ep8_hits, "no file under amd/ep8 (TU, v20_include, build scripts) mentions the EP16 macro")
    users = sorted(f.relative_to(AMD_DIR).as_posix() for f in AMD_DIR.rglob("*")
                   if f.is_file() and EP16_MACRO.encode() in f.read_bytes())
    check(users == ["include/sub6_6models_generalize.hpp", "megamoe_amd.hip"],
          f"EP16 macro is confined to the sub6 guard + EP16 TU ({users})")
    return n_pass


def main() -> int:
    cc = find_compiler()
    if cc is None:
        raise SystemExit("[FAIL] no C++ compiler found (set WEAVE_SURFACE_CXX)")
    n_iso = tu_isolation_checks(cc)
    with tempfile.TemporaryDirectory(prefix="weave_ep16_shared_") as td:
        binary = Path(td) / "test_ep16_shared_experts"
        cmd = [*cc, "-std=c++17", "-O2", "-I", str(HOST_DIR),
               str(ROOT / "tests/test_ep16_shared_experts.cpp"),
               "-o", str(binary)]
        print("[BUILD] " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f"[FAIL] C++ build exited with {proc.returncode}")
        run = subprocess.run([str(binary)], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=120)
        print(run.stdout, end="")
        if run.returncode != 0:
            print(run.stderr[-2000:], end="")
            raise SystemExit(f"[FAIL] C++ test exited with {run.returncode}")
    n_pass = run.stdout.count("[PASS]")
    print(f"EP16 shared-experts ABI gate: {n_pass} ABI checks + {n_iso} TU-isolation checks passed")
    return 0


if __name__ == "__main__":
    if WINDOWS:
        rc = maybe_delegate_to_wsl()
        if rc is not None:
            raise SystemExit(rc)
    raise SystemExit(main())
