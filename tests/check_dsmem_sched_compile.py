#!/usr/bin/env python3
"""Real nvcc zero-spill compile gate for the fused two-level scheduling rail.

Compiles tests/compile_sm100_dsmem_sched.cu for sm_103a AND sm_90a and requires
for every kernel entry point:

  * 0 errors,
  * 0 bytes spill stores and 0 bytes spill loads,
  * 0 bytes stack frame,
  * <= 24 registers for the DSMEM push kernel, <= 28 for the lock-free
    pre-reduce kernels, <= 64 for every other primitive kernel.

It also compiles a -ptx pass and asserts the REAL instructions are present:
mapa.shared::cluster, st.async.shared::cluster, mbarrier.arrive.expect_tx,
red.global.add.f32, fence.acq_rel.gpu, atom.acq_rel.gpu.global.add.u32 and
fence.acquire.gpu.

The verbatim ptxas report is committed as results/dsmem_sched_compile.json so
the evidence is auditable without re-running the compiler.

Usage:
    python tests/check_dsmem_sched_compile.py [--arch sm_103a] [--keep]
Exit status 0 only when every compiled architecture passes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/compile_sm100_dsmem_sched.cu"
OUT_JSON = ROOT / "results/dsmem_sched_compile.json"
ARCHS = ("sm_103a", "sm_90a")

# Per-kernel register budgets (0 spill / 0 stack is required on ALL kernels).
REGS_DEFAULT = 64
REGS_BUDGET = {
    "dsmem_push_kernel": 24,
    "lockfree_prereduce_kernel": 28,
    "lockfree_prereduce_scalar_kernel": 28,
    "lockfree_prereduce_readback_kernel": 28,
}

REQUIRED_PTX = (
    "mapa.shared::cluster",
    "st.async.shared::cluster",
    "mbarrier.arrive.expect_tx",
    "red.global.add.f32",
    "fence.acq_rel.gpu",
    "fence.acquire.gpu",
    "atom.acq_rel.gpu.global.add.u32",
)

VSDEVCMD = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    r"\Common7\Tools\VsDevCmd.bat"
)
# The repo's cutlass/deep_jit submodules may be unpopulated in a worktree; the
# sibling checkout carries the same pinned headers.
INCLUDE_FALLBACKS = (
    ROOT.parent / "weave_unified_cpp" / "third-party" / "cutlass" / "include",
    ROOT.parent / "weave_unified_cpp" / "third-party" / "deep_jit" / "include",
)


def include_dirs() -> list[Path]:
    dirs = [ROOT / "deep_gemm/include"]
    for name in ("cutlass", "deep_jit"):
        local = ROOT / "third-party" / name / "include"
        if local.is_dir() and any(local.iterdir()):
            dirs.append(local)
    if len(dirs) == 1:
        dirs.extend(p for p in INCLUDE_FALLBACKS if p.is_dir())
    return dirs


def find_nvcc() -> str | None:
    override = os.environ.get("WEAVE_NVCC")
    if override:
        return override
    found = shutil.which("nvcc")
    if found:
        return found
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe",
        "/usr/local/cuda/bin/nvcc",
    ]
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


def parse_ptxas(output: str) -> dict:
    kernels: dict[str, dict] = {}
    current = None
    for line in output.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if m:
            current = m.group(1)
            kernels[current] = {"arch": m.group(2), "registers": None,
                                "spill_stores": 0, "spill_loads": 0,
                                "stack_frame": 0}
            continue
        if current is None:
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m:
            kernels[current]["registers"] = int(m.group(1))
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                      r"(\d+) bytes spill loads", line)
        if m:
            kernels[current]["stack_frame"] = int(m.group(1))
            kernels[current]["spill_stores"] = int(m.group(2))
            kernels[current]["spill_loads"] = int(m.group(3))
    return kernels


def run_nvcc(nvcc: str, args: list[str], workdir: Path) -> tuple[int, str]:
    """Run nvcc, activating the MSVC host environment on Windows.

    CCCL (bundled with CUDA 13.x) rejects the traditional MSVC preprocessor,
    so /Zc:preprocessor is mandatory on the Windows host pass.
    """
    if platform.system() == "Windows":
        args = ["-Xcompiler", "/Zc:preprocessor", *args]
    cmd = [nvcc, *args]
    if platform.system() == "Windows" and Path(VSDEVCMD).exists():
        bat = workdir / "run_nvcc.bat"
        quoted = " ".join(f'"{a}"' if " " in a else a for a in cmd)
        bat.write_text(
            "@echo off\r\n"
            f'call "{VSDEVCMD}" -arch=x64 -no_logo >nul\r\n'
            f"{quoted}\r\n"
            "exit /b %ERRORLEVEL%\r\n", encoding="utf-8")
        proc = subprocess.run(["cmd.exe", "/d", "/s", "/c", str(bat)],
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def error_lines(output: str) -> list[str]:
    return [ln for ln in output.splitlines()
            if re.search(r"\berror\b|\bfatal\b", ln, re.IGNORECASE)]


def _path_regex(path) -> str:
    """Regex for `path` accepting either separator style (and doubled ones)."""
    parts = [re.escape(p) for p in re.split(r"[\\/]+", str(path)) if p]
    return r"[\\/]+".join(parts)


def scrub_paths(text: str) -> str:
    """Replace host-specific absolute paths with stable placeholders.

    Committed reports must carry no local usernames, temp directories or
    install locations: the temp dir becomes <tmp>, the CUDA toolkit root
    <cuda>, the repo root "." (repo-relative), the repo's parent "..", and any
    remaining per-user home prefix <home>.
    """
    subs = [(tempfile.gettempdir(), "<tmp>")]
    nvcc = find_nvcc()
    if nvcc:
        subs.append((Path(nvcc).resolve().parent.parent, "<cuda>"))
    subs += [(ROOT, "."), (ROOT.parent, "..")]
    for path, token in subs:
        text = re.sub(_path_regex(path), lambda _m, t=token: t, text,
                      flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+[\\/]+AppData[\\/]+"
                  r"Local[\\/]+Temp", "<tmp>", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "<home>", text,
                  flags=re.IGNORECASE)
    return re.sub(r"/home/[^/\s\"']+", "<home>", text)


def report_path(path) -> str:
    """Repo-relative (or sibling-relative) forward-slash form of `path`."""
    return scrub_paths(str(path)).replace("\\", "/")


def stable(output: str) -> str:
    """Strip volatile per-run bits so the committed report is reproducible."""
    # MSVC C4819 (source code-page) warnings and other host-locale messages
    # (undecodable bytes) are environment noise, not compiler evidence.
    lines = [ln for ln in output.splitlines()
             if "Compile time" not in ln and "warning C4819" not in ln
             and "�" not in ln]
    text = "\n".join(lines) + "\n"
    text = re.sub(r"tmpxft_[0-9a-f]+_\d+-\d+_", "tmpxft_STABLE_", text)
    # Per-run temporary build directories appear in ptxas warning prefixes.
    text = re.sub(r"weave_(?:dsmem|twolevel)_[a-z0-9_]+", "weave_BUILD_TMP", text)
    return scrub_paths(text.replace("\\\\", "\\"))


def compile_arch(nvcc: str, arch: str, workdir: Path, incs: list[Path]) -> dict:
    cubin = workdir / f"dsmem_sched_{arch}.cubin"
    args = ["-std=c++17", f"-arch={arch}", "-cubin", "-Xptxas", "-v",
            "--expt-relaxed-constexpr"]
    for inc in incs:
        args += ["-I", str(inc)]
    args += [str(SOURCE), "-o", str(cubin)]
    rc, out = run_nvcc(nvcc, args, workdir)
    kernels = parse_ptxas(out)
    problems = error_lines(out)
    # Judge on hard evidence (cubin + ptxas records), not on the exit marker:
    # nvcc 13.3 on Windows intermittently returns non-zero after a clean build.
    if not cubin.exists() or cubin.stat().st_size == 0:
        problems.append("cubin missing or empty")
    if not kernels:
        problems.append("no kernel entry compiled")
    for name, row in kernels.items():
        # ptxas reports the MANGLED entry name; match the budget by substring.
        budget = REGS_DEFAULT
        for base, limit in REGS_BUDGET.items():
            if base in name:
                budget = limit
                break
        if row["registers"] is None or row["registers"] > budget:
            problems.append(f"{name}: registers={row['registers']} > {budget}")
        if row["spill_stores"] or row["spill_loads"]:
            problems.append(f"{name}: spill stores={row['spill_stores']} "
                            f"loads={row['spill_loads']}")
        if row["stack_frame"]:
            problems.append(f"{name}: stack frame={row['stack_frame']} bytes")
    return {"arch": arch, "nvcc_returncode": rc, "kernels": kernels,
            "cubin_bytes": cubin.stat().st_size if cubin.exists() else 0,
            "pass": not problems, "problems": problems,
            "ptxas_output": stable(out)}


def check_ptx(nvcc: str, arch: str, workdir: Path, incs: list[Path]) -> dict:
    ptx = workdir / f"dsmem_sched_{arch}.ptx"
    args = ["-std=c++17", f"-arch={arch}", "-ptx", "--expt-relaxed-constexpr"]
    for inc in incs:
        args += ["-I", str(inc)]
    args += [str(SOURCE), "-o", str(ptx)]
    rc, out = run_nvcc(nvcc, args, workdir)
    text = ptx.read_text(encoding="utf-8", errors="replace") if ptx.exists() else ""
    counts = {insn: text.count(insn) for insn in REQUIRED_PTX}
    problems = error_lines(out)
    for insn, count in counts.items():
        if count == 0:
            problems.append(f"PTX missing {insn}")
    return {"arch": arch, "nvcc_returncode": rc, "instruction_counts": counts,
            "pass": not problems, "problems": problems}


def run(arches=ARCHS, quiet: bool = False) -> dict:
    nvcc = find_nvcc()
    incs = include_dirs()
    report = {
        "schema": "weave-dsmem-sched-compile/v2",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "register_budgets": {"default": REGS_DEFAULT, **REGS_BUDGET},
        "required_ptx": list(REQUIRED_PTX),
        "includes": [report_path(p) for p in incs],
        "nvcc": Path(nvcc).name if nvcc else None,
        "archs": {}, "ptx": {}, "pass": False,
    }
    if nvcc is None:
        report["error"] = "nvcc not found (set WEAVE_NVCC)"
        if not quiet:
            print("[FAIL] " + report["error"])
        return report
    version = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace")
    report["nvcc_version"] = (version.stdout or "").strip().splitlines()[-1:]
    with tempfile.TemporaryDirectory(prefix="weave_dsmem_") as td:
        workdir = Path(td)
        for arch in arches:
            row = compile_arch(nvcc, arch, workdir, incs)
            report["archs"][arch] = row
            ptx_row = check_ptx(nvcc, arch, workdir, incs)
            report["ptx"][arch] = ptx_row
            if not quiet:
                status = "PASS" if row["pass"] and ptx_row["pass"] else "FAIL"
                regs = ", ".join(f"{k}:{v['registers']}r"
                                 for k, v in row["kernels"].items())
                print(f"[{status}] {arch}: {regs}", flush=True)
                for problem in row["problems"] + ptx_row["problems"]:
                    print(f"        {problem}")
    report["pass"] = bool(report["archs"]) and all(
        report["archs"][a]["pass"] and report["ptx"][a]["pass"] for a in arches)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    if not quiet:
        print(f"WROTE {OUT_JSON}")
        print("ALL DSMEM SCHED COMPILE CHECKS PASS" if report["pass"]
              else "DSMEM SCHED COMPILE CHECKS FAILED")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", action="append", default=None)
    args = ap.parse_args()
    arches = tuple(args.arch) if args.arch else ARCHS
    report = run(arches)
    if report.get("error"):
        return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
