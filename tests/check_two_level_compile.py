#!/usr/bin/env python3
"""Real nvcc compile gate for the FULL production two-level kernels.

  * tests/compile_sm100_two_level.cu on sm_100a AND sm_103a, three arms of the
    complete sm100_fp8_fp4_mega_moe_impl instantiation:
      - default: WEAVE_REDUCE_WARPS=4 + WEAVE_ADAPT_BM=1 (fine-grained D6 +
        dedicated Middle Reduce warps).  WEAVE_RUNTIME_POLICY is left at its
        kernel default 0, i.e. the shipping JIT build when
        knobs.sm0_stream_sched == 0: the CTA-0 online policy layer (3-Tier
        LUT publication / PolicyDecision handshake) is compiled OUT.
      - runtime_policy: the same instantiation with -DWEAVE_RUNTIME_POLICY=1,
        which is what the JIT emits when knobs.sm0_stream_sched != 0
        (WEAVE_SM0_STREAM_SCHED=1 or WEAVE_READY_QUEUE=sm0_fifo); this lowers
        the CTA-0 online policy layer into the device pass.
      - fused_shared_int7: -DWEAVE_FUSED_SHARED_INT7=1 with the shape the
        fused SharedLinear2 + INT7 path requires (one shared expert, top-k 8,
        8-row GEMM2 stores, no D6 group).  A control compile of the same shape
        with the switch at 0 must produce different PTX or fail to compile,
        proving the fused path is really instantiated rather than
        constant-folded away.
    Budget for every arm: <= 128 registers, 0 bytes spill stores / loads.
  * tests/compile_sm90_two_level.cu on sm_90a: the repo-convention include-only
    syntax/type-check gate over the whole sm90_fp8_mega_moe.cuh chain.
    Budget: 0 errors.

Usage:
    python tests/check_two_level_compile.py [--arch sm_100a] [--skip-sm90]
Writes results/phase_nk_compile_gates.json (paths repo-relative, temp and
toolkit locations scrubbed).  Exit 0 only when all gates pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import check_dsmem_sched_compile as dsmem

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results/phase_nk_compile_gates.json"
SM100_SOURCE = ROOT / "tests/compile_sm100_two_level.cu"
SM90_SOURCE = ROOT / "tests/compile_sm90_two_level.cu"
SM100_ARCHS = ("sm_100a", "sm_103a")
REGS_LIMIT = 128

# Shape that satisfies the kernel's kFusedSharedInt7 predicate (see
# sm100_fp8_fp4_mega_moe.cuh): kNumSharedExperts == 1, kNumTopk <= 8,
# STORE_BLOCK_M_L2 == 8, BLOCK_N == 128, no D6 group / Middle Reduce warps.
FUSED_SHAPE = ("-DT_NUM_SHARED=1", "-DT_TOPK=8", "-DT_STORE_L2=8",
               "-DWEAVE_REDUCE_WARPS=0", "-DWEAVE_COMBINE_GROUP_TILES=0u",
               "-DWEAVE_ADAPT_BM=0", "-DWEAVE_EXPERT_STEP_MODE=0")
SM100_ARMS = {
    "": (),
    "runtime_policy": ("-DWEAVE_RUNTIME_POLICY=1",),
    "fused_shared_int7": (*FUSED_SHAPE, "-DWEAVE_FUSED_SHARED_INT7=1"),
}
FUSED_CONTROL = (*FUSED_SHAPE, "-DWEAVE_FUSED_SHARED_INT7=0")


def compile_ptx(nvcc: str, source: Path, arch: str, workdir: Path,
                incs: list[Path], defines: tuple[str, ...], tag: str):
    ptx = workdir / f"{source.stem}_{tag}_{arch}.ptx"
    base = ["-std=c++20", "-O3", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", *defines]
    for inc in incs:
        base += ["-I", str(inc)]
    rc, out = dsmem.run_nvcc(
        nvcc, [*base, f"-arch={arch}", "-ptx", str(source), "-o", str(ptx)],
        workdir)
    return ptx, rc, out


def gate(nvcc: str, source: Path, arch: str, workdir: Path,
         incs: list[Path], regs_limit: int | None,
         defines: tuple[str, ...] = (), tag: str = "default") -> dict:
    ptx, rc_ptx, out_ptx = compile_ptx(nvcc, source, arch, workdir, incs,
                                       defines, tag)
    cubin = ptx.with_suffix(".cubin")
    problems = dsmem.error_lines(out_ptx)
    if not ptx.exists() or ptx.stat().st_size == 0:
        problems.append("PTX missing or empty")
    row = {"source": str(source.relative_to(ROOT)).replace("\\", "/"),
           "arch": arch, "defines": list(defines),
           "nvcc_ptx_returncode": rc_ptx,
           "nvcc_errors": len(dsmem.error_lines(out_ptx)),
           "ptx_bytes": ptx.stat().st_size if ptx.exists() else 0}
    if problems:
        row.update({"pass": False, "problems": problems,
                    "nvcc_output": dsmem.stable(out_ptx)})
        return row
    # ptxas is invoked directly so its -v record is unambiguous.
    ptxas = str(Path(nvcc).parent / ("ptxas.exe" if dsmem.platform.system()
                                     == "Windows" else "ptxas"))
    proc = subprocess.run([ptxas, f"-arch={arch}", "-v", str(ptx),
                           "-o", str(cubin)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    out_as = (proc.stdout or "") + (proc.stderr or "")
    rc_as = proc.returncode
    kernels = dsmem.parse_ptxas(out_as)
    problems = dsmem.error_lines(out_as)
    if not cubin.exists() or cubin.stat().st_size == 0:
        problems.append("cubin missing or empty")
    if regs_limit is not None:
        if not kernels:
            problems.append("no kernel entry compiled")
        for name, k in kernels.items():
            if k["registers"] is None or k["registers"] > regs_limit:
                problems.append(f"{name}: registers={k['registers']}")
            if k["spill_stores"] or k["spill_loads"]:
                problems.append(f"{name}: spill stores={k['spill_stores']} "
                                f"loads={k['spill_loads']}")
    row.update({"ptxas_returncode": rc_as, "kernels": kernels,
                "cubin_bytes": cubin.stat().st_size if cubin.exists() else 0,
                "ptxas_output": dsmem.stable(out_as),
                "pass": not problems, "problems": problems})
    return row


def fused_instantiation_check(nvcc: str, arch: str, workdir: Path,
                              incs: list[Path], row: dict) -> None:
    """Prove the fused arm really instantiates the fused path.

    WEAVE_FUSED_SHARED_INT7 feeds only the kernel's kFusedSharedInt7
    predicate, so if that predicate were false the switch-on and switch-off
    builds of the same shape would be token-identical.  The control therefore
    must either produce different PTX or fail to compile (at this shape the
    non-fused final combine does not fit the reusable SMEM), while the fused
    arm itself compiled cleanly.
    """
    if not row["pass"]:
        return
    fused_ptx = workdir / f"{SM100_SOURCE.stem}_fused_shared_int7_{arch}.ptx"
    ctrl_ptx, rc, out = compile_ptx(nvcc, SM100_SOURCE, arch, workdir, incs,
                                    FUSED_CONTROL, "fused_control")
    ctrl_errors = dsmem.error_lines(out)
    ctrl_ok = (not ctrl_errors and ctrl_ptx.exists()
               and ctrl_ptx.stat().st_size > 0)
    if ctrl_ok:
        differs = ctrl_ptx.read_bytes() != fused_ptx.read_bytes()
        outcome = "compiled, PTX differs" if differs else "compiled, PTX identical"
    else:
        differs = True
        outcome = "control failed to compile (non-fused path diverges)"
    row["fused_control"] = {
        "defines": list(FUSED_CONTROL), "nvcc_ptx_returncode": rc,
        "outcome": outcome,
        "control_errors": [dsmem.scrub_paths(e) for e in ctrl_errors],
        "fused_path_instantiated": differs}
    if not differs:
        row["problems"] = row["problems"] + [
            "WEAVE_FUSED_SHARED_INT7=1 PTX is identical to the switch-off "
            "control: fused path not instantiated"]
        row["pass"] = False


def run(sm100_archs=SM100_ARCHS, skip_sm90: bool = False) -> dict:
    nvcc = dsmem.find_nvcc()
    incs = dsmem.include_dirs()
    report = {
        "schema": "weave-phase-nk-compile-gates/v2",
        "toolchain": "nvcc + ptxas (see nvcc_version); MSVC host env via "
                     "VsDevCmd -arch=x64 with /Zc:preprocessor (CCCL)",
        "reality_boundary": "Compilation evidence only. No GPU execution on "
                            "this host: the local GPU is not a target device "
                            "and application control blocks freshly built "
                            "unsigned executables. The device rails run on a "
                            "target GPU via tests/test_cost_model_eq123.cpp.",
        "includes": [dsmem.report_path(p) for p in incs],
        "nvcc": Path(nvcc).name if nvcc else None, "gates": {}, "pass": False,
    }
    if nvcc is None:
        report["error"] = "nvcc not found (set WEAVE_NVCC)"
        print("[FAIL] " + report["error"])
        return report
    version = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace")
    report["nvcc_version"] = (version.stdout or "").strip().splitlines()[-1:]
    with tempfile.TemporaryDirectory(prefix="weave_twolevel_") as td:
        workdir = Path(td)
        for arch in sm100_archs:
            for arm, defines in SM100_ARMS.items():
                key = (f"sm100_two_level_{arm}_{arch}" if arm
                       else f"sm100_two_level_{arch}")
                row = gate(nvcc, SM100_SOURCE, arch, workdir, incs,
                           REGS_LIMIT, defines, arm or "default")
                if arm == "fused_shared_int7":
                    fused_instantiation_check(nvcc, arch, workdir, incs, row)
                report["gates"][key] = row
                regs = ", ".join(
                    f"{k}:{v['registers']}r/{v['spill_stores']}+"
                    f"{v['spill_loads']}B spill/{v['stack_frame']}B stack"
                    for k, v in row.get("kernels", {}).items())
                print(f"[{'PASS' if row['pass'] else 'FAIL'}] {key}: {regs}",
                      flush=True)
                for problem in row["problems"]:
                    print(f"        {problem}")
        if not skip_sm90:
            row = gate(nvcc, SM90_SOURCE, "sm_90a", workdir, incs, None)
            report["gates"]["sm90_two_level_sm_90a"] = row
            print(f"[{'PASS' if row['pass'] else 'FAIL'}] sm90_two_level_sm_90a"
                  " (include-only syntax gate)", flush=True)
            for problem in row["problems"]:
                print(f"        {problem}")
    report["pass"] = all(g["pass"] for g in report["gates"].values())
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"WROTE {OUT_JSON.relative_to(ROOT)}")
    print("ALL TWO-LEVEL PRODUCTION COMPILE GATES PASS" if report["pass"]
          else "TWO-LEVEL PRODUCTION COMPILE GATES FAILED")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", action="append", default=None)
    ap.add_argument("--skip-sm90", action="store_true")
    args = ap.parse_args()
    report = run(tuple(args.arch) if args.arch else SM100_ARCHS, args.skip_sm90)
    if report.get("error"):
        return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
