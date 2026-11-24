#!/usr/bin/env python3
"""Build the V62 2-Node 16xMI308X EP16 champion (arm ``v62_champion``) into a gfx942 P.co.

Provenance (see v62_champion_preset.json, the single source of truth):
  * Benchmark: Nebula job 01a204b684c8479453de03934774988b (2 nodes x 8 MI308X,
    world 16), 6 models x 5 arms, 30/30 16-rank full-element FP64 checks PASS;
    v62_champion wins every cell, geomean 1.3267x vs the V58 C_COAL_68 control,
    2.2772x vs RCCL+AITER on qwen38max (2.4T) S16384
    (results/ep16_mi308x_v62_16gpu_champion.json).
  * Production code object: FI_LEVEL=0, sha256
    e915ce1f51f53cbba0b4739b1e6497f578a5126590a2f0e4bd238fd59386e8b2,
    375960 bytes, 230 VGPR / 0 spill / 0 scratch / 1 s_barrier / 64 KiB LDS.
  * Source: the V62 build-host tree <v62 build tree>/tree/v62_o2/v4, vendored
    verbatim under tree/ (sha256 manifest in the preset; three headers that are
    byte-identical to amd/include/ are shared instead of duplicated).
  * Pipeline: byte-for-byte the recorded V62 builder (<v62 builder dir>/
    src/build_v62.py): clang -S -> __hip_cuid normalisation -> cc1as ->
    relax_amdgpu_inline_branches_v1 -> assemble -> ld.lld -> audit_isa.

This TU is the V46-SP lineage (3-mode CU scheduler, Atrex claims, V44..V62
macro-gated bodies).  It is selected only through the internal -D macros of
the preset; it reads ZERO public WEAVE_* tunables (the 21-knob surface of the
policy-wired port amd/megamoe_amd.hip / ep16/build_port.py is untouched).
The only environment variables consulted are the toolchain/path overrides
already used by ep16/build_port.py (WEAVE_CLANG, WEAVE_LLD, WEAVE_PY,
WEAVE_SNAP_DIR, WEAVE_V3225_DIR, WEAVE_AITER_SITE, WEAVE_PY_SITE).

  python build_v62.py --dry-run        # resolved command + include closure, no ROCm
  python build_v62.py                  # real build (any gfx942 ROCm host, CPU only)
"""
from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

V62 = pathlib.Path(__file__).resolve().parent          # .../amd/ep16/v62
EP16 = V62.parent                                      # .../amd/ep16
AMD = EP16.parent                                      # .../amd
TREE = V62 / "tree"
SRC = TREE / "megamoe_amd.hip"
PRESET_PATH = V62 / "v62_champion_preset.json"


def load_preset() -> Dict:
    return json.loads(PRESET_PATH.read_text(encoding="utf-8"))


def tool_paths() -> Dict[str, pathlib.Path]:
    return {
        "clang": pathlib.Path(os.environ.get("WEAVE_CLANG", "/opt/rocm-7.2.0/lib/llvm/bin/clang")),
        "lld": pathlib.Path(os.environ.get("WEAVE_LLD", "/opt/rocm-7.2.0/lib/llvm/bin/ld.lld")),
        "py": pathlib.Path(os.environ.get("WEAVE_PY", "/opt/conda/envs/python3.10.13/bin/python")),
        "snap": pathlib.Path(os.environ.get("WEAVE_SNAP_DIR", str(AMD / "third_party_overlay"))),
        "v3225": pathlib.Path(os.environ.get("WEAVE_V3225_DIR", str(AMD / "third_party_overlay"))),
    }


def include_dirs() -> List[str]:
    """-I order of the recorded V62 builder (tv, tv/include, R/common, SNAP/overlay_gemm2,
    SNAP, V3225, aiter/torch/rocm), with amd/include appended LAST for the three
    shared byte-identical headers (gda_mesh, v1262_subwave_barrier, weave_ck_512_configs)."""
    t = tool_paths()
    site = os.environ.get("WEAVE_AITER_SITE",
                          "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages/aiter_meta")
    ps = os.environ.get("WEAVE_PY_SITE", "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages")
    return [
        f"-I{TREE}", f"-I{TREE / 'include'}", f"-I{V62 / 'common'}",
        f"-I{t['snap'] / 'overlay_gemm2'}", f"-I{t['snap']}", f"-I{t['v3225']}",
        f"-I{site}/3rdparty/ck_helper",
        f"-I{site}/3rdparty/composable_kernel/include",
        f"-I{site}/3rdparty/composable_kernel/library/include",
        f"-I{site}/csrc/include", f"-I{site}/csrc/ck_gemm_moe_2stages_codegen",
        f"-I{site}/csrc/include/torch", f"-I{ps}/pybind11/include",
        "-isystem", f"{ps}/torch/include",
        "-isystem", f"{ps}/torch/include/torch/csrc/api/include",
        "-isystem", f"{ps}/torch/include/THH",
        "-isystem", "/opt/rocm/include",
        "-isystem", "/opt/conda/envs/python3.10.13/include/python3.10",
        "-I/opt/rocm/include/hip",
        f"-I{AMD / 'include'}",
    ]


# Base flags: verbatim clang_cmd() of the recorded V62 builder at FI_LEVEL=0.
BASE_FLAGS = [
    "-x", "hip", "--offload-arch=gfx942", "--cuda-device-only", "-O3", "-S", "-std=c++20",
    "-DKITTENS_CDNA3", "-DV1098_FORCE_M=0",
    "-DFI_LEVEL=0", "-DFI_TIMER=0", "-DFI_GRAN=0", "-DWEAVE_PARALLEL_ROUTE_PLAN=1", "-DWEAVE_PAR_RP=1",
    "-DWEAVE_CK_G2_LDS_ONLY=0", "-DWEAVE_TILE_TRACE_ONLY=1",
    "-DWEAVE_ADAPTIVE_POST_TIMEOUT=1", "-DWEAVE_S16_PLAN_CAP=0", "-DWEAVE_S16_STATIC_DISPATCH=0",
    "-DWEAVE_OPT_RELAXED_SPIN=1", "-DWEAVE_OPT_LOCAL_SCOPE=1",
    "-DWEAVE_SCATTER_MODE=2", "-DWEAVE_CO_PROFILE=0", "-DWEAVE_TH_BM64=16u", "-DWEAVE_TH_BM128=96u",
    "-DWEAVE_ABLATION_PRESET=0", "-DWEAVE_KNOB_BM_TIER_MODE=2", "-DWEAVE_KNOB_SELECTIVE_DEDUP=1",
]
TAIL_FLAGS = [
    "-mllvm", "-amdgpu-opt-vgpr-liverange=true", "-mllvm", "-sink-insts-to-avoid-spills=true",
    "-w", "-fno-offload-uniform-block", "-fgpu-flush-denormals-to-zero",
    "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false",
]


def clang_cmd(out: pathlib.Path, preset: Dict) -> List[str]:
    return [str(tool_paths()["clang"]), *BASE_FLAGS, *preset["exact_v62_extra_flags"].split(),
            *TAIL_FLAGS, *include_dirs(), str(SRC), "-o", str(out / "stage.s")]


def defines_of(cmd: List[str]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for tok in cmd:
        if tok.startswith("-D"):
            k, _, v = tok[2:].partition("=")
            d[k] = v
    return d


_INC_RX = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def include_closure() -> Tuple[Dict[str, pathlib.Path], List[str]]:
    """Quoted-include closure of the TU, resolved like clang: includer dir first,
    then the -I order of include_dirs() restricted to repository directories.
    Independent of #if state, so it is a superset of what any flag set reads."""
    t = tool_paths()
    dirs = [TREE, TREE / "include", V62 / "common", t["snap"] / "overlay_gemm2", t["snap"],
            t["v3225"], AMD / "include"]
    seen: Dict[pathlib.Path, bool] = {SRC.resolve(): True}
    external = set()
    stack = [SRC]
    while stack:
        f = stack.pop()
        for name in _INC_RX.findall(f.read_text(encoding="utf-8", errors="replace")):
            hit = next((c.resolve() for c in [f.parent / name, *[d / name for d in dirs]] if c.is_file()), None)
            if hit is None:
                external.add(name)
            elif hit not in seen:
                seen[hit] = True
                stack.append(hit)
    rel = {p.relative_to(AMD.resolve()).as_posix(): p for p in seen}
    return dict(sorted(rel.items())), sorted(external)


def sha256_lf(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_manifest(preset: Dict) -> List[str]:
    """Return a list of mismatches between the live include closure and the preset manifest."""
    closure, external = include_closure()
    man = preset["source_manifest"]
    errs = []
    for rel in sorted(set(closure) | set(man)):
        if rel not in closure:
            errs.append(f"manifest file not in live closure: {rel}")
        elif rel not in man:
            errs.append(f"closure file missing from manifest: {rel}")
        elif sha256_lf(closure[rel]) != man[rel]["sha256_lf"]:
            errs.append(f"sha256 drift: {rel}")
    if external != preset["external_includes"]:
        errs.append(f"external include set drift: {external} != {preset['external_includes']}")
    return errs


def run(cmd: List[str], log: pathlib.Path, env: Dict[str, str], timeout: int = 1800) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env=env, timeout=timeout)
    log.write_text(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} rc={p.returncode}; see {log}")
    return p.stdout


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the V62 2-Node 16xMI308X EP16 champion (v62_champion)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved compile line, -D macros and include-closure manifest "
                         "check, then exit without invoking the ROCm toolchain")
    ap.add_argument("--out-dir", default=None, help="build directory (default ep16/v62/build/v62_champion)")
    args, _ = ap.parse_known_args(argv)
    preset = load_preset()
    errs = verify_manifest(preset)
    out = pathlib.Path(args.out_dir) if args.out_dir else V62 / "build" / "v62_champion"
    cmd = clang_cmd(out, preset)
    if args.dry_run:
        closure, external = include_closure()
        print(json.dumps({
            "variant": preset["arm"], "fi": 0, "tu": SRC.relative_to(AMD).as_posix(),
            "exact_v62_extra_flags": preset["exact_v62_extra_flags"],
            "defines": defines_of(cmd),
            "expected_co_sha256": preset["expected_co"]["sha256"],
            "include_closure": {"files": len(closure), "external": external,
                                "manifest_ok": not errs, "manifest_errors": errs},
        }, indent=2))
        print("BUILD_V62_DRY_RUN_DONE" if not errs else "BUILD_V62_DRY_RUN_MANIFEST_FAIL")
        return 0 if not errs else 1
    if errs:
        print("source manifest mismatch -- refusing to build a non-verified TU:\n  " + "\n  ".join(errs))
        return 1
    t = tool_paths()
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("logs", "cache", "cw"):
        (out / sub).mkdir()
    env = dict(os.environ)
    env.update(HIP_CACHE_DIR=str(out / "cache"), AMD_COMGR_CACHE_DIR=str(out / "cache"),
               TMPDIR=str(out / "cw"), TMP=str(out / "cw"), TEMP=str(out / "cw"))
    run(cmd, out / "logs/clang.log", env)
    txt = (out / "stage.s").read_text()
    (out / "stage.norm.s").write_text(re.sub(r"__hip_cuid_[0-9a-f]+", "__hip_cuid_v20_ep16", txt))
    run([str(t["clang"]), "-cc1as", "-triple", "amdgcn-amd-amdhsa", "-target-cpu", "gfx942",
         "-filetype", "asm", str(out / "stage.norm.s"), "-o", str(out / "stage.expanded.s")],
        out / "logs/expand.log", env)
    run([str(t["py"]), str(AMD / "tools/relax_amdgpu_inline_branches_v1.py"),
         str(out / "stage.expanded.s"), str(out / "stage.relaxed.s"),
         "--report", str(out / "logs/relax.json")], out / "logs/relax.log", env)
    run([str(t["clang"]), "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx942", "-c",
         str(out / "stage.relaxed.s"), "-o", str(out / "stage.o")], out / "logs/assemble.log", env)
    run([str(t["lld"]), "-flavor", "gnu", "--no-undefined", "-shared", "-o", str(out / "P.co"),
         str(out / "stage.o")], out / "logs/link.log", env)
    audit_txt = run([str(t["py"]), str(AMD / "tools/audit_isa.py"), str(out / "stage.s"),
                     "--co", str(out / "P.co"), "--advisory"], out / "logs/isa_audit.json", env)
    audit = json.loads(audit_txt[audit_txt.find("{"):])
    pl = audit.get("post_link_elf") or {}
    strict = (int(audit.get("vgpr_spill_count_max", -1)) == 0
              and int(audit.get("private_segment_fixed_size_max", -1)) == 0
              and int(audit.get("s_barrier", -1)) == 1
              and int(audit.get("group_segment_fixed_size_max", 1 << 30)) <= 65536
              and int(audit.get("vgpr_count_max", 9999)) <= 256
              and (not pl or (pl.get("co_passed") is True
                              and int(pl.get("co_vgpr_spill_count_max", -1)) == 0
                              and int(pl.get("co_private_segment_fixed_size_max", -1)) == 0)))
    sha = hashlib.sha256((out / "P.co").read_bytes()).hexdigest()
    match = sha == preset["expected_co"]["sha256"]
    receipt = {"variant": preset["arm"], "co": str(out / "P.co"), "sha256": sha,
               "expected_sha256": preset["expected_co"]["sha256"], "byte_identical": match,
               "strict_isa_gate": strict,
               "gate": {k: audit.get(k) for k in ("vgpr_count_max", "sgpr_count_max", "vgpr_spill_count_max",
                                                   "private_segment_fixed_size_max",
                                                   "group_segment_fixed_size_max", "s_barrier")}}
    (out / "build_receipt_v62_champion.json").write_text(json.dumps(receipt, indent=2))
    print("BUILD_V62_DONE", json.dumps(receipt), flush=True)
    if not strict:
        return 2
    return 0 if match else 3


if __name__ == "__main__":
    sys.exit(main())
