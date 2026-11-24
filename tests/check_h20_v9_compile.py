#!/usr/bin/env python3
"""Real nvcc/ptxas gate for the H20 V9 fused Swap-AB 4-WG engine (sm_90a).

Unlike the include-only SM90 gate in check_two_level_compile.py, this driver
INSTANTIATES the production kernels at the measured EP4 Qwen3.8-Max work points,
so ptxas emits a real register / stack / spill record.

Nothing about the instantiation is hand-written.  For every arm the LIVE SM90
host heuristics (select_mega_moe_sm90 + get_agy_sm90_spec +
get_agy_sm90_jit_defines, csrc/jit_kernels/heuristics/sm90_mega_moe.hpp) run
under the arm's WEAVE_* environment through tests/h20_v9_host_emitter.cpp, which
writes the exact JIT source SM90FP8MegaMoERuntime::compile_only would hand to
nvcc -- the #define block plus the template instantiation with the host-chosen
BLOCK_M / BLOCK_N / BLOCK_K, pipeline stages, thread layout, experts-per-wave,
swap-AB and N-major flags, and the fused vs. split L1/L2 kernel choice.  A
decoupled WEAVE_BLOCK_M_L1 != WEAVE_BLOCK_M_L2 therefore compiles the real
sm90_fp8_mega_moe_l1_impl + sm90_fp8_mega_moe_l2_impl pair.

Every arm uses WEAVE_* environment variables only (no WEAVE_H20_CONFIG): the 15 public
knobs, their legacy aliases and the internal pins of the retired knobs.

Budget per 4-WG kernel: kNumEpilogueThreads == 512 (4 compute warpgroups),
<= 96 registers, 0 bytes stack frame, 0 spill stores, 0 spill loads.
Informational arms (BLOCK_M=128 phases, which lock a non-swap BN=128 2-WG tile
by construction) are compiled and recorded but do not gate.

Usage:
    python tests/check_h20_v9_compile.py [--matrix] [--jobs N] [--work-dir DIR]
Writes results/h20_v9_compile_gates.json.  Exit 0 only when every gating arm passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_dsmem_sched_compile as dsmem  # noqa: E402
import h20_v9_host_emit as emit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results/h20_v9_compile_gates.json"
ARCH = "sm_90a"
REGS_LIMIT = 96
EPI_THREADS_4WG = 512
# v12: the non-swap BLOCK_M=128 tile (BN=128, 2 compute WGs, 128+128+256 = 512
# threads) has its own budget: <= 65536/512 = 128 registers, 256 epilogue threads,
# and the same 0 stack / 0 spill contract.
REGS_LIMIT_2WG = 128
EPI_THREADS_2WG = 256

WINNER_ENV = {
    "WEAVE_ADAPT_BM": "1",
    "WEAVE_SCATTER_OWNER": "comm_only",
    "WEAVE_SCATTER_STAGING": "smem",
}
SCATTER_ENV = {"WEAVE_SCATTER_OWNER": "comm_only", "WEAVE_SCATTER_STAGING": "smem"}

# The sender publishes per (scatter warp, tile) with one fence.acq_rel.sys
# followed by red.relaxed.sys (fence + relaxed-RMW release pattern) on the SAME
# address the receiver polls with ld.acquire.sys; the claim is a device-scope
# atomicOr on a rank-local word.
STREAM_PROBES = ("fence.acq_rel.sys", "red.relaxed.sys.global.add.u32",
                 "ld.acquire.sys.global.u32", "atom.global.or.b32")
ALL_PROBES = (*STREAM_PROBES, "red.release.sys.global.add.u32",
              "atom.acq_rel.gpu.global.add.u32", "fence.acq_rel.gpu")


def arm_list(matrix: bool, tokens_list: list[int]) -> list[dict]:
    arms = [{"name": f"h20_v9_s{t}", "tokens": t, "env": dict(WINNER_ENV)} for t in tokens_list]
    if not matrix:
        return arms
    # V8 attribution baseline expressed in public knobs only: no interleave and
    # one expert wave (epw = E_local = 128), Reduce off, global-barrier combine.
    arms.append({"name": "h20_v8_baseline_p21_s1024", "tokens": 1024,
                 "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": "0-128",
                         "WEAVE_REDUCE_WARPS": "0", "WEAVE_COMBINE_SYNC_CHUNK": "0"}})
    # WEAVE_REDUCE_WARPS across its legal values (8 fails closed, below).
    for rw in (0, 1, 2, 4):
        arms.append({"name": f"h20_v9_rw{rw}_s4096", "tokens": 4096,
                     "env": {**WINNER_ENV, "WEAVE_REDUCE_WARPS": str(rw)}})
    # Reuse-W0 lane: R=1 keeps the 2-warp dispatch group (64 threads) and the
    # 4-stage pipeline at both seqlens.
    for t in (1024, 4096):
        arms.append({"name": f"h20_rw1_s{t}", "tokens": t,
                     "env": {**WINNER_ENV, "WEAVE_REDUCE_WARPS": "1", "WEAVE_COMBINE_SYNC_CHUNK": "64"},
                     "expect_layout": {"DISP": "64", "STAGES": "4"}})
    # Env-only L1/L2 wave split x chunked combine (interleave 5).
    for ws in (8, 16, 32, 64):
        for ch in (16, 32, 64, 128):
            arms.append({"name": f"h20_v9_ws{ws}_c{ch}_s1024", "tokens": 1024,
                         "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": str(ws),
                                 "WEAVE_COMBINE_SYNC_CHUNK": str(ch)}})
    for ws, ch in (("8", 0), ("64", 0), ("0-8", 64), ("0-64", 0)):
        arms.append({"name": f"h20_v9_ws{ws}_c{ch}_s1024", "tokens": 1024,
                     "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": ws,
                             "WEAVE_COMBINE_SYNC_CHUNK": str(ch)}})
    # The env-only default is now R=0 / interleave off, so the streaming Reduce
    # x wave-split cross terms that the sweep launches are pinned explicitly.
    for rw in (1, 2, 4):
        for ws in ("8", "32"):
            arms.append({"name": f"h20_v9_rw{rw}_ws{ws}_c64_s4096", "tokens": 4096,
                         "env": {**WINNER_ENV, "WEAVE_REDUCE_WARPS": str(rw),
                                 "WEAVE_CU_WAVE_SPLIT": ws, "WEAVE_COMBINE_SYNC_CHUNK": "64"}})
    arms.append({"name": "h20_v9_adapt0_s1024", "tokens": 1024,
                 "env": {**SCATTER_ENV, "WEAVE_ADAPT_BM": "0"}})
    # Explicit WEAVE_CU_WAVE_SPLIT=0 is the default (schema: 0 = auto).
    arms.append({"name": "h20_v9_ws0_s4096", "tokens": 4096,
                 "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": "0"}})
    # Other interleave modes through the M-E spelling.
    for ilv in (1, 2, 3, 4, 6):
        arms.append({"name": f"h20_v9_ws{ilv}-8_s1024", "tokens": 1024,
                     "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": f"{ilv}-8"}})
    # v11 lag-W expert task queue (WEAVE_CU_WAVE_SPLIT=7-W) with the all-warp head,
    # and the W0/W1 gap-Reduce (R=1/2 keep the 2-warp comm group and 4 stages).
    for lag in (2, 4, 6, 8, 12):
        for t in (1024, 4096):
            arms.append({"name": f"h20_v11_ws7-{lag}_vh780_s{t}", "tokens": t,
                         "env": {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": f"7-{lag}", "WEAVE_V_HEAD": "780"},
                         "expect_layout": {"DISP": "64", "STAGES": "4"}})
    for rw in (1, 2):
        for ws in (None, "7-8"):
            for t in (1024, 4096):
                env = {**WINNER_ENV, "WEAVE_REDUCE_WARPS": str(rw), "WEAVE_V_HEAD": "780"}
                if ws:
                    env["WEAVE_CU_WAVE_SPLIT"] = ws
                arms.append({"name": f"h20_v11_rw{rw}_ws{ws or 0}_s{t}", "tokens": t, "env": env,
                             "expect_layout": {"DISP": "64", "STAGES": "4"}})
    # GEMM1/GEMM2 BLOCK_M: shared (fused) and decoupled (split L1/L2 launch).
    for l1, l2 in (("32", "32"), ("16", "16")):
        arms.append({"name": f"h20_v9_bm{l1}_fused_s1024", "tokens": 1024,
                     "env": {**SCATTER_ENV, "WEAVE_BLOCK_M_L1": l1, "WEAVE_BLOCK_M_L2": l2}})
    # BLOCK_M_L1=64 split L1 spills at the 4-WG cap and fails closed (below).
    for l1, l2 in (("32", "64"), ("16", "64"), ("32", "16")):
        for t in (1024, 4096):
            arms.append({"name": f"h20_v9_bm{l1}_{l2}_split_s{t}", "tokens": t,
                         "env": {**SCATTER_ENV, "WEAVE_BLOCK_M_L1": l1, "WEAVE_BLOCK_M_L2": l2}})
    # GEMM1/GEMM2 decoupled BN: BLOCK_M=128 locks GEMM1 to BN=128 (non-swap,
    # 2 WGs by construction) while GEMM2 keeps the BN=256 Swap-AB 4-WG tile.
    arms.append({"name": "h20_v9_bm128_64_split_s1024", "tokens": 1024, "informational_l1": True,
                 "env": {**SCATTER_ENV, "WEAVE_BLOCK_M_L1": "128", "WEAVE_BLOCK_M_L2": "64"}})
    # v12 comm-regime winner family: fused BLOCK_M=128 (non-swap BN=128 2-WG tile),
    # incl. the BLOCK_N=128 streaming Combine (two scatter tiles per 256-column item).
    b128 = {**SCATTER_ENV, "WEAVE_ADAPT_BM": "0", "WEAVE_BLOCK_M_L1": "128", "WEAVE_BLOCK_M_L2": "128"}
    for nm, extra in (("bm128", {}), ("bm128_dw8", {"WEAVE_DISPATCH_WARPS": "8"}),
                      ("bm128_rw2", {"WEAVE_REDUCE_WARPS": "2"}),
                      ("bm128_rw2_cs16", {"WEAVE_REDUCE_WARPS": "2", "WEAVE_COMBINE_SYNC_CHUNK": "16"}),
                      ("bm128_dw8_rw2", {"WEAVE_DISPATCH_WARPS": "8", "WEAVE_REDUCE_WARPS": "2"}),
                      ("bm128_ws7-32_rw2", {"WEAVE_CU_WAVE_SPLIT": "7-32", "WEAVE_REDUCE_WARPS": "2"})):
        arms.append({"name": f"h20_v12_{nm}_s4096", "tokens": 4096, "two_wg": True, "env": {**b128, **extra}})
    # v13 WEAVE_ADAPT_GEMM_CONFIG host whole-tuple selection (Qwen3.8-Max shape):
    # tokens/rank=256 -> M_expected=20 -> decoupled GEMM1 BM32 / GEMM2 BM64 split
    # launch; with a fused-only knob (CU_WAVE_SPLIT) -> the fused canonical BM64
    # tuple.  Explicit GEMM1/GEMM2 pins win over an explicit ADAPT=1.
    arms.append({"name": "h20_v13_adapt_auto_m20_s256", "tokens": 256,
                 "env": {**SCATTER_ENV, "WEAVE_ADAPT_GEMM_CONFIG": "1"}})
    arms.append({"name": "h20_v13_adapt_auto_m20_ws7-32_s256", "tokens": 256,
                 "env": {**SCATTER_ENV, "WEAVE_ADAPT_GEMM_CONFIG": "1", "WEAVE_CU_WAVE_SPLIT": "7-32"}})
    arms.append({"name": "h20_v13_adapt0_s256", "tokens": 256,
                 "env": {**SCATTER_ENV, "WEAVE_ADAPT_GEMM_CONFIG": "0"}})
    arms.append({"name": "h20_v13_split_adapt1_pins_32_16_s1024", "tokens": 1024,
                 "env": {**WINNER_ENV, "WEAVE_GEMM1_CONFIG": "32", "WEAVE_GEMM2_CONFIG": "16"}})
    # v13 zero-Reduce weight-bound timeline arm (V11 h20.png setup with R=0).
    for t in (256, 1024):
        arms.append({"name": f"h20_v13_ws7-32_vh780_rw0_s{t}", "tokens": t,
                     "env": {**SCATTER_ENV, "WEAVE_CU_WAVE_SPLIT": "7-32", "WEAVE_V_HEAD": "780",
                             "WEAVE_REDUCE_WARPS": "0"},
                     "expect_layout": {"DISP": "64", "STAGES": "4"}})
    # v13 WEAVE_DISPATCH_ACTIVE_SMS < kNumSMs keeps the all-warp head pull, incl.
    # D != WEAVE_SCATTER_ACTIVE_SMS.
    for d, k in ((52, 0), (66, 39), (60, 78), (78, 39), (52, 52)):
        env = {**SCATTER_ENV, "WEAVE_ADAPT_GEMM_CONFIG": "0", "WEAVE_DISPATCH_ACTIVE_SMS": str(d)}
        if k:
            env["WEAVE_SCATTER_ACTIVE_SMS"] = str(k)
        arms.append({"name": f"h20_v13_D{d}K{k}_s1024", "tokens": 1024, "env": env})
    # v13 compact-regime timeline arm on the 2-WG tile (pinned GEMM1/2=128, R=0).
    g128 = {**SCATTER_ENV, "WEAVE_GEMM1_CONFIG": "128", "WEAVE_GEMM2_CONFIG": "128"}
    for nm, extra in (("g128_dw8_ws7-32_rw0", {"WEAVE_DISPATCH_WARPS": "8", "WEAVE_CU_WAVE_SPLIT": "7-32",
                                               "WEAVE_REDUCE_WARPS": "0"}),
                      ("g128_dw8_D66K39", {"WEAVE_DISPATCH_WARPS": "8", "WEAVE_DISPATCH_ACTIVE_SMS": "66",
                                           "WEAVE_SCATTER_ACTIVE_SMS": "39"})):
        arms.append({"name": f"h20_v13_{nm}_s4096", "tokens": 4096, "two_wg": True, "env": {**g128, **extra}})
    return arms


def compile_kernel(nvcc: str, incs: list[Path], workdir: Path, src: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    ptx = workdir / (src.stem + ".ptx")
    cubin = workdir / (src.stem + ".cubin")
    base = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    for inc in incs:
        base += ["-I", str(inc)]
    rc_ptx, out_ptx = dsmem.run_nvcc(
        nvcc, [*base, f"-arch={ARCH}", "-ptx", str(src), "-o", str(ptx)], workdir)
    row = {"source": src.name, "nvcc_ptx_returncode": rc_ptx,
           "nvcc_errors": len(dsmem.error_lines(out_ptx))}
    problems = dsmem.error_lines(out_ptx)
    if not ptx.exists() or ptx.stat().st_size == 0:
        problems.append("PTX missing or empty")
    if problems:
        row.update({"problems": problems, "nvcc_output": dsmem.stable(out_ptx)[-6000:]})
        return row
    ptxas = str(Path(nvcc).parent / ("ptxas.exe" if platform.system() == "Windows" else "ptxas"))
    proc = subprocess.run([ptxas, f"-arch={ARCH}", "-v", str(ptx), "-o", str(cubin)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=3600)
    out_as = (proc.stdout or "") + (proc.stderr or "")
    kernels = dsmem.parse_ptxas(out_as)
    problems = dsmem.error_lines(out_as)
    if not cubin.exists() or cubin.stat().st_size == 0:
        problems.append("cubin missing or empty")
    if not kernels:
        problems.append("no kernel entry compiled (instantiation did not lower)")
    text = ptx.read_text(encoding="utf-8", errors="replace")
    row.update({"ptxas_returncode": proc.returncode, "kernels": kernels,
                "ptx_bytes": ptx.stat().st_size, "cubin_bytes": cubin.stat().st_size if cubin.exists() else 0,
                "ptx_probes": {n: (n in text) for n in ALL_PROBES},
                "ptxas_output": dsmem.stable(out_as), "problems": problems})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--tokens", default="1024,4096")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--matrix", action="store_true",
                    help="also compile the V8 baseline, every legal WEAVE_REDUCE_WARPS, the "
                         "WEAVE_CU_WAVE_SPLIT x WEAVE_COMBINE_SYNC_CHUNK grid, the interleave "
                         "modes and the shared / decoupled GEMM1/GEMM2 BLOCK_M arms")
    args = ap.parse_args()

    nvcc = dsmem.find_nvcc()
    if nvcc is None:
        print("[FAIL] nvcc not found")
        return 1
    incs = dsmem.include_dirs()
    build_dir = (Path(args.work_dir) if args.work_dir else ROOT / "scratch/h20_v9_compile").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = emit.build_emitter(build_dir / "host")
    print(f"[INFO] nvcc={nvcc}")

    arms = arm_list(args.matrix, [int(t) for t in args.tokens.split(",")])
    host_rows = []
    unique_src: dict[str, Path] = {}
    (build_dir / "src").mkdir(exist_ok=True)
    for arm in arms:
        assert "WEAVE_H20_CONFIG" not in arm["env"]
        res = emit.run_emitter(binary, arm["tokens"], arm["env"],
                               build_dir / "src" / arm["name"], check=False)
        arm["host"] = res
        if res["returncode"] != 0:
            continue
        arm["kernel_hash"] = {}
        for kname, path in res["sources"].items():
            h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            arm["kernel_hash"][kname] = h
            unique_src.setdefault(h, path)

    # Fail-closed spellings must be rejected by the host before any compile.
    host_rejects = []
    for name, env in (("h20_v9_rw8_host_fail_closed", {**WINNER_ENV, "WEAVE_REDUCE_WARPS": "8"}),
                      ("h20_v9_ws_amd_code_fail_closed", {**WINNER_ENV, "WEAVE_CU_WAVE_SPLIT": "2"}),
                      ("h20_v9_c256_fail_closed", {**WINNER_ENV, "WEAVE_COMBINE_SYNC_CHUNK": "256"}),
                      ("h20_v9_split_bm64_16_fail_closed",
                       {**SCATTER_ENV, "WEAVE_BLOCK_M_L1": "64", "WEAVE_BLOCK_M_L2": "16"}),
                      ("h20_v9_split_bm64_32_fail_closed",
                       {**SCATTER_ENV, "WEAVE_BLOCK_M_L1": "64", "WEAVE_BLOCK_M_L2": "32"})):
        r = emit.run_emitter(binary, 1024, env, build_dir / "src" / name, check=False)
        # A rejection only counts when the host itself asserted (DG_HOST_ASSERT
        # text on stderr), never on a bare non-zero exit.
        rejected = r["returncode"] != 0 and "Assertion error" in r.get("stderr", "")
        host_rejects.append({"arm": name, "env": env, "pass": rejected,
                             "host_error": r.get("stderr", "")[-300:]})
        print(f"[{'PASS' if rejected else 'FAIL'}] {name}")

    print(f"[INFO] {len(arms)} arms -> {len(unique_src)} unique kernel instantiations")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {h: pool.submit(compile_kernel, nvcc, incs, build_dir / "nvcc" / h, p)
                for h, p in unique_src.items()}
        compiled = {h: f.result() for h, f in futs.items()}

    rows = []
    for arm in arms:
        res = arm["host"]
        row = {"arm": arm["name"], "tokens": arm["tokens"], "env": arm["env"],
               "informational_l1": bool(arm.get("informational_l1"))}
        problems = []
        if res["returncode"] != 0:
            problems.append("host rejected: " + res.get("stderr", "")[-300:])
            row.update({"pass": False, "problems": problems})
            rows.append(row)
            print(f"[FAIL] {arm['name']}: host rejected")
            continue
        row["host"] = {k: res[k] for k in res if k.isupper() and k != "DEFINES"}
        row["jit_defines"] = res["defines"]
        is_stream = "#define WEAVE_REDUCE_WARPS" in res["defines"] and res["FUSED"] == "1"
        kernels_out = {}
        for kname, h in arm["kernel_hash"].items():
            c = compiled[h]
            phase = "L1" if kname == "l1" else "L2"
            epi = int(res[f"{phase}_EPI"])
            gating = not (arm.get("informational_l1") and kname == "l1")
            kprob = list(c.get("problems", []))
            for ent, k in c.get("kernels", {}).items():
                rlim = REGS_LIMIT_2WG if arm.get("two_wg") else REGS_LIMIT
                if k["registers"] is None or k["registers"] > rlim:
                    kprob.append(f"{kname}: registers={k['registers']} > {rlim}")
                if k["spill_stores"] or k["spill_loads"]:
                    kprob.append(f"{kname}: spill stores={k['spill_stores']} loads={k['spill_loads']}")
                if k["stack_frame"]:
                    kprob.append(f"{kname}: stack frame={k['stack_frame']} bytes")
            want_epi = EPI_THREADS_2WG if arm.get("two_wg") else EPI_THREADS_4WG
            if epi != want_epi:
                kprob.append(f"{kname}: {epi} epilogue threads != {want_epi} "
                             f"({'2' if arm.get('two_wg') else '4'} compute WGs)")
            if is_stream:
                for n in STREAM_PROBES:
                    if not c.get("ptx_probes", {}).get(n, False):
                        kprob.append(f"{kname}: PTX missing required instruction {n}")
            for key, want in arm.get("expect_layout", {}).items():
                if res.get(f"{phase}_{key}") != want:
                    kprob.append(f"{kname}: {phase}_{key}={res.get(f'{phase}_{key}')} != expected {want}")
            regs = [k["registers"] for k in c.get("kernels", {}).values()]
            kernels_out[kname] = {"kernel_hash": h, "gating": gating, "epilogue_threads": epi,
                                  "bm_bn_bk": [res[f"{phase}_BM"], res[f"{phase}_BN"], res[f"{phase}_BK"]],
                                  "stages": res[f"{phase}_STAGES"],
                                  "threads": [res[f"{phase}_DISP"], res[f"{phase}_NONEPI"], epi],
                                  "registers": regs,
                                  "spill": [(k["spill_stores"], k["spill_loads"], k["stack_frame"])
                                            for k in c.get("kernels", {}).values()],
                                  "ptx_probes": c.get("ptx_probes", {}), "problems": kprob}
            if gating:
                problems += kprob
            print(f"[{'PASS' if not kprob else ('FAIL' if gating else 'INFO')}] {arm['name']}:{kname} "
                  f"bm/bn/bk={kernels_out[kname]['bm_bn_bk']} st={kernels_out[kname]['stages']} "
                  f"thr={kernels_out[kname]['threads']} regs={regs} "
                  f"spill={kernels_out[kname]['spill']}")
            for p in kprob[:4]:
                print("    " + str(p)[:300])
        row.update({"fused": res["FUSED"] == "1", "kernels": kernels_out,
                    "pass": not problems, "problems": problems})
        rows.append(row)

    report = {"schema": "weave-h20-v9-compile-gates/v2",
              "host": platform.node(), "nvcc": nvcc, "arch": ARCH,
              "regs_limit": REGS_LIMIT, "epilogue_threads_4wg": EPI_THREADS_4WG,
              "winner_env": WINNER_ENV, "arms": rows, "host_fail_closed": host_rejects,
              "unique_kernels": len(unique_src),
              "compiled": {h: {k: v for k, v in c.items() if k != "ptxas_output"}
                           for h, c in compiled.items()},
              "pass": bool(rows) and all(r["pass"] for r in rows) and all(h["pass"] for h in host_rejects)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {OUT_JSON}")
    print("ALL H20 V9 SM90A COMPILE GATES PASS" if report["pass"]
          else "H20 V9 SM90A COMPILE GATES FAILED")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
