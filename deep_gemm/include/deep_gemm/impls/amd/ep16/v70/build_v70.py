#!/usr/bin/env python3
"""Build the V70 Path-B (160x256, 32 KiB dynamic LDS) 2-Node 16xMI308X EP16 kernel into a gfx942 P.co.

Supports:
  * V70 champion defaults (Path B 4-4 hybrid + e3b4_qht + H9 3-path scatter + H10 5-knob wiring)
  * ALL 17 active MI308X WEAVE_* design-point knobs (zero #error stubs; strict fail-fast validation)
  * ALL 3 CU modes (0-8, 4-4, 8-0, auto)
  * Both FI_LEVEL=0 (production) and FI_LEVEL=2 (WeaveProfiler V4 Full-Duplex)
  * Strict post-compile ISA gate on stage_ep16_turbo_overlap:
      vgpr <= 256, vgpr_spill == 0, private_segment == 0, scratch_ops == 0,
      static_lds == 0 (32 KiB dynamic LDS), zero ds_{read,write} offset >= 32768.

  python build_v70.py --dry-run                 # resolved compile line + manifest check, no ROCm
  python build_v70.py                           # real FI0 build (any gfx942 ROCm host, CPU only)
  python build_v70.py --fi 2 --out-dir ...      # real FI2 WeaveProfiler build
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

V70 = pathlib.Path(__file__).resolve().parent          # .../amd/ep16/v70
EP16 = V70.parent                                      # .../amd/ep16
AMD = EP16.parent                                      # .../amd
TREE = V70 / "tree"
SRC = TREE / "megamoe_amd.hip"
PRESET_PATH = V70 / "v70_champion_preset.json"


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
    t = tool_paths()
    site = os.environ.get("WEAVE_AITER_SITE",
                          "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages/aiter_meta")
    ps = os.environ.get("WEAVE_PY_SITE", "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages")
    return [
        f"-I{TREE}", f"-I{TREE / 'include'}", f"-I{V70 / 'common'}",
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
        "-isystem", "/opt/conda/envs/python3.10/include/python3.10",
        "-isystem", "/opt/conda/envs/python3.10.13/include/python3.10",
        "-I/opt/rocm/include/hip",
        f"-I{AMD / 'include'}",
    ]


BASE_FLAGS = [
    "-x", "hip", "--offload-arch=gfx942", "--cuda-device-only", "-O3", "-S", "-std=c++20",
    "-DKITTENS_CDNA3", "-DV1098_FORCE_M=0",
    "-DFI_TIMER=0", "-DFI_GRAN=0", "-DWEAVE_PARALLEL_ROUTE_PLAN=1", "-DWEAVE_PAR_RP=1",
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


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return None if v is None or v == "" else v


def _decimal(name: str, raw: str, allowed: str) -> int:
    if not re.fullmatch(r"[0-9]{1,9}", raw):
        raise SystemExit(f"Invalid {name}={raw} (allowed: {allowed})")
    return int(raw)


def _choice(name: str, allowed: Tuple[int, ...], default: int, ignore_env: bool) -> int:
    raw = None if ignore_env else _env(name)
    if raw is None:
        return default
    text = " ".join(str(a) for a in allowed)
    val = _decimal(name, raw, text)
    if val not in allowed:
        raise SystemExit(f"Invalid {name}={raw} (allowed: {text})")
    return val


def _uint(name: str, lo: int, hi: int, default: int, ignore_env: bool) -> int:
    raw = None if ignore_env else _env(name)
    if raw is None:
        return default
    val = _decimal(name, raw, f"integer in [{lo},{hi}]")
    if not lo <= val <= hi:
        raise SystemExit(f"Invalid {name}={raw} (expected integer in [{lo},{hi}])")
    return val


def _named(name: str, mapping: Dict[str, int], default: int, ignore_env: bool) -> int:
    raw = None if ignore_env else _env(name)
    if raw is None:
        return default
    if raw in mapping:
        return mapping[raw]
    raise SystemExit(f"Invalid {name}={raw} (allowed: {', '.join(mapping)})")


def resolve_v70_knobs(preset: Dict, *, ignore_env: bool = False) -> Dict[str, str]:
    """Resolve the 17 active MI308X WEAVE_* knobs onto V70 Path-B champion defaults."""
    defs = dict(preset["champion_knob_defaults"])
    if ignore_env:
        return defs

    prec = _env("WEAVE_PRECISION_MODE")
    if prec not in (None, "2", "fp8", "fp8xfp8", "auto"):
        raise SystemExit(f"MI308X V70 supports only WEAVE_PRECISION_MODE=2 (fp8xfp8); got {prec}")

    cu_split = _named(
        "WEAVE_CU_WAVE_SPLIT",
        {"auto": 2, "0": 2, "0-8": 1, "1": 1, "4-4": 2, "2": 2, "8-0": 3, "3": 3},
        int(defs["WEAVE_CU_WAVE_SPLIT"]), ignore_env)
    disp_sms = _uint("WEAVE_DISPATCH_ACTIVE_SMS", 0, 80, int(defs["WEAVE_DISPATCH_ACTIVE_SMS"]), ignore_env)
    scat_sms = _uint("WEAVE_SCATTER_ACTIVE_SMS", 0, 80, int(defs["WEAVE_SCATTER_ACTIVE_SMS"]), ignore_env)
    sm0_sched = _choice("WEAVE_SM0_STREAM_SCHED", (0, 1), int(defs["WEAVE_SM0_STREAM_SCHED"]), ignore_env)
    disp_w = _choice("WEAVE_DISPATCH_WARPS", (0, 4, 8), int(defs["WEAVE_DISPATCH_WARPS"]), ignore_env)
    if disp_w == 0:
        disp_w = int(defs["WEAVE_DISPATCH_WARPS"])
    scat_w = _choice("WEAVE_SCATTER_WARPS", (0, 4, 8), int(defs["WEAVE_SCATTER_WARPS"]), ignore_env)
    if scat_w == 0:
        scat_w = int(defs["WEAVE_SCATTER_WARPS"])
    red_w = _choice("WEAVE_REDUCE_WARPS", (0, 1, 2, 4, 8), int(defs["WEAVE_REDUCE_WARPS"]), ignore_env)

    vh_raw = _env("WEAVE_V_HEAD")
    if vh_raw is None or vh_raw == "auto":
        v_head = int(defs["WEAVE_V_HEAD"])
    else:
        v_head = _decimal("WEAVE_V_HEAD", vh_raw, "auto or non-negative decimal integer")

    rgrp = _choice("WEAVE_READY_GROUP_SIZE", (0, 1, 2, 4, 8, 16), int(defs["WEAVE_READY_GROUP_SIZE"]), ignore_env)
    if rgrp == 0:
        rgrp = int(defs["WEAVE_READY_GROUP_SIZE"])
    owner = _named("WEAVE_SCATTER_OWNER", {"0": 0, "epilogue_only": 0, "1": 1, "comm_only": 1},
                   int(defs["WEAVE_SCATTER_OWNER"]), ignore_env)
    csync = _choice("WEAVE_COMBINE_SYNC_CHUNK", (0, 16, 32, 64, 128, 512, 1024, 2048, 4096),
                    int(defs["WEAVE_COMBINE_SYNC_CHUNK"]), ignore_env)
    if csync == 0:
        csync = int(defs["WEAVE_COMBINE_SYNC_CHUNK"])

    adapt_bm = _choice("WEAVE_ADAPT_BM", (0, 1), int(defs["WEAVE_ADAPT_BM"]), ignore_env)
    bm_l1 = _choice("WEAVE_BLOCK_M_L1", (0, 16, 32, 64, 128), int(defs["WEAVE_BLOCK_M_L1"]), ignore_env)
    bm_l2 = _choice("WEAVE_BLOCK_M_L2", (0, 16, 32, 64, 128), int(defs["WEAVE_BLOCK_M_L2"]), ignore_env)
    # 14-knob surface compatibility: WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG (0=auto -> ADAPT_BM=1; explicit 16/32/64/128 -> fixed BM)
    g1_cfg = _env("WEAVE_GEMM1_CONFIG") if not ignore_env else None
    g2_cfg = _env("WEAVE_GEMM2_CONFIG") if not ignore_env else None
    if g1_cfg is not None or g2_cfg is not None:
        g1_map = {"0": 0, "auto": 0, "16": 16, "32": 32, "64": 64, "128": 128,
                  "BM32_BN128_BK128_4W": 32, "BM64_BN128_BK128_4W": 64, "BM128_BN128_BK64_8W": 128}
        g2_map = {"0": 0, "auto": 0, "16": 16, "32": 32, "64": 64, "128": 128,
                  "BM32_BN256_BK128_4W": 32, "BM64_BN256_BK128_4W": 64, "BM128_BN128_BK64_8W": 128}
        v1 = _named("WEAVE_GEMM1_CONFIG", g1_map, 0, ignore_env)
        v2 = _named("WEAVE_GEMM2_CONFIG", g2_map, 0, ignore_env)
        if v1 == 0 and v2 == 0:
            adapt_bm = 1
        else:
            adapt_bm = 0
            if v1 != 0:
                bm_l1 = v1
            if v2 != 0:
                bm_l2 = v2
    if bm_l1 == 0:
        bm_l1 = int(defs["WEAVE_BLOCK_M_L1"])
    if bm_l2 == 0:
        bm_l2 = int(defs["WEAVE_BLOCK_M_L2"])
    mbset = 6 if (adapt_bm == 1 or bm_l1 == 32 or bm_l2 == 32) else 0

    rpull = _choice("WEAVE_REG_PULL_BYTES", (0, 1024, 2048, 4096, 8192), int(defs["WEAVE_REG_PULL_BYTES"]), ignore_env)
    if rpull == 0:
        rpull = int(defs["WEAVE_REG_PULL_BYTES"])
    dedup = _choice("WEAVE_DEDUP_MODE", (0, 1), int(defs["WEAVE_DEDUP_MODE"]), ignore_env)
    staging = _named(
        "WEAVE_SCATTER_STAGING",
        {"0": 0, "reg": 0, "reg_2stage": 0, "direct": 0, "1": 1, "smem": 1, "2": 2, "hbm": 2, "hbm_dedup": 2},
        int(defs["WEAVE_SCATTER_STAGING"]), ignore_env)
    if staging != 0 and owner == 0:
        raise SystemExit(f"Invalid WEAVE_SCATTER_STAGING={staging} with WEAVE_SCATTER_OWNER=epilogue_only")

    return {
        "WEAVE_CU_WAVE_SPLIT": str(cu_split),
        "WEAVE_DISPATCH_ACTIVE_SMS": str(disp_sms),
        "WEAVE_SCATTER_ACTIVE_SMS": str(scat_sms),
        "WEAVE_SM0_STREAM_SCHED": str(sm0_sched),
        "WEAVE_DISPATCH_WARPS": str(disp_w),
        "WEAVE_SCATTER_WARPS": str(scat_w),
        "WEAVE_REDUCE_WARPS": str(red_w),
        "WEAVE_V_HEAD": str(v_head),
        "WEAVE_READY_GROUP_SIZE": str(rgrp),
        "WEAVE_SCATTER_OWNER": str(owner),
        "WEAVE_COMBINE_SYNC_CHUNK": str(csync),
        "WEAVE_ADAPT_BM": str(adapt_bm),
        "R2X_MBSET": str(mbset),
        "WEAVE_BLOCK_M_L1": str(bm_l1),
        "WEAVE_BLOCK_M_L2": str(bm_l2),
        "WEAVE_REG_PULL_BYTES": str(rpull),
        "WEAVE_DEDUP_MODE": str(dedup),
        "WEAVE_SCATTER_STAGING": str(staging),
    }


def clang_cmd(out: pathlib.Path, preset: Dict, *, fi: int = 0, knobs: Optional[Dict[str, str]] = None) -> List[str]:
    kmap = knobs if knobs is not None else dict(preset["champion_knob_defaults"])
    kflags = [f"-D{k}={v}" for k, v in kmap.items()]
    return [
        str(tool_paths()["clang"]), *BASE_FLAGS,
        f"-DFI_LEVEL={fi}", f"-DWEAVE_PROFILER_LEVEL={fi}",
        *preset["exact_v70_base_flags"].split(),
        *preset["exact_v70_e3_flags"].split(),
        *kflags,
        *TAIL_FLAGS, *include_dirs(), str(SRC), "-o", str(out / "stage.s"),
    ]


def defines_of(cmd: List[str]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for tok in cmd:
        if tok.startswith("-D"):
            k, _, v = tok[2:].partition("=")
            d[k] = v
    return d


_INC_RX = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def include_closure() -> Tuple[Dict[str, pathlib.Path], List[str]]:
    t = tool_paths()
    dirs = [TREE, TREE / "include", V70 / "common",
            t["snap"] / "overlay_gemm2", t["snap"], t["v3225"], AMD / "include"]
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


def audit_stage_ep16(stage_s_text: str) -> Dict[str, int]:
    i_m = stage_s_text.find(".name:           stage_ep16_turbo_overlap")
    if i_m < 0:
        i_m = stage_s_text.find("stage_ep16_turbo_overlap.kd")
    meta = stage_s_text[max(0, i_m - 400):i_m + 800] if i_m >= 0 else ""

    def mget(k: str) -> int:
        mm = re.search(rf"\.{k}:\s+(\d+)", meta)
        return int(mm.group(1)) if mm else -1

    idx_s = stage_s_text.find("stage_ep16_turbo_overlap:")
    idx_e = stage_s_text.find("\n.Lfunc_end", idx_s) if idx_s >= 0 else -1
    if idx_e < 0 and idx_s >= 0:
        idx_e = stage_s_text.find("\t.size\tstage_ep16_turbo_overlap", idx_s)
    body = stage_s_text[idx_s:idx_e] if (idx_s >= 0 and idx_e > idx_s) else ""
    hi_lds = [int(x) for x in re.findall(r"ds_(?:read|write)[^\n]*offset:(\d+)", body) if int(x) >= 32768]
    return {
        "vgpr_count": mget("vgpr_count"),
        "sgpr_count": mget("sgpr_count"),
        "vgpr_spill_count": mget("vgpr_spill_count"),
        "private_segment_fixed_size": mget("private_segment_fixed_size"),
        "group_segment_fixed_size": mget("group_segment_fixed_size"),
        "scratch_ops": len(re.findall(r"\bscratch_(?:load|store)", body)),
        "hi_lds_offsets_ge_32768": len(hi_lds),
        "s_barrier": len(re.findall(r"\bs_barrier\b", body)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the V70 Path-B 2-Node 16xMI308X EP16 champion (v70_champion)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved compile line, 17-knob -D macros, and source manifest check")
    ap.add_argument("--fi", type=int, choices=(0, 2), default=None,
                    help="WeaveProfiler level (0=production, 2=V4 Full-Duplex; defaults to WEAVE_PROFILER_LEVEL or 0)")
    ap.add_argument("--ignore-env", action="store_true",
                    help="ignore public WEAVE_* env vars and build the exact V70 champion preset")
    ap.add_argument("--out-dir", default=None, help="build directory (default ep16/v70/build/v70_champion_fi<fi>)")
    args, _ = ap.parse_known_args(argv)

    preset = load_preset()
    errs = verify_manifest(preset)
    if args.fi is not None:
        fi = args.fi
    elif args.ignore_env:
        fi = 0
    else:
        fi = _choice("WEAVE_PROFILER_LEVEL", (0, 2), 0, ignore_env=False)
    knobs = resolve_v70_knobs(preset, ignore_env=args.ignore_env)
    out = pathlib.Path(args.out_dir) if args.out_dir else V70 / "build" / f"v70_champion_fi{fi}"
    cmd = clang_cmd(out, preset, fi=fi, knobs=knobs)
    if args.dry_run:
        closure, external = include_closure()
        exp_key = "fi0" if fi == 0 else "fi2"
        print(json.dumps({
            "variant": preset["arm"], "fi": fi, "tu": SRC.relative_to(AMD).as_posix(),
            "launch": {"grid": 160, "block": 256, "dynamic_lds_bytes": 32768},
            "knobs": knobs,
            "defines": defines_of(cmd),
            "expected_co_sha256": preset["expected_co"][exp_key]["sha256"],
            "include_closure": {"files": len(closure), "external": external,
                                "manifest_ok": not errs, "manifest_errors": errs},
        }, indent=2))
        print("BUILD_V70_DRY_RUN_DONE" if not errs else "BUILD_V70_DRY_RUN_MANIFEST_FAIL")
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
    (out / "stage.norm.s").write_text(re.sub(r"__hip_cuid_[0-9a-f]+", "__hip_cuid_v70_ep16", txt))
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
    gate = audit_stage_ep16(txt)
    strict = (gate["vgpr_spill_count"] == 0
              and gate["private_segment_fixed_size"] == 0
              and gate["scratch_ops"] == 0
              and gate["group_segment_fixed_size"] == 0
              and gate["hi_lds_offsets_ge_32768"] == 0
              and 0 < gate["vgpr_count"] <= 256)
    sha = hashlib.sha256((out / "P.co").read_bytes()).hexdigest()
    receipt = {"variant": preset["arm"], "fi": fi, "co": str(out / "P.co"), "sha256": sha,
               "strict_isa_gate": strict, "gate": gate, "knobs": knobs}
    (out / f"build_receipt_v70_fi{fi}.json").write_text(json.dumps(receipt, indent=2))
    print("BUILD_V70_DONE", json.dumps(receipt), flush=True)
    return 0 if strict else 2


if __name__ == "__main__":
    sys.exit(main())
