#!/usr/bin/env python3
"""Build the unified policy-wired EP16 kernel into a gfx942 P.co.

All repository paths are relative to this file, so cloning weave.git on
an MI308X ROCm host is sufficient.  The only external dependency is the aiter_meta CK
tree (1100+ files), supplied through WEAVE_AITER_SITE; it is deliberately not
vendored so the git repository stays lightweight.
"""
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
import time

_ap = argparse.ArgumentParser(description="Build the unified policy-wired EP16 kernel")
_ap.add_argument("--dry-run", action="store_true",
                 help="print the resolved WEAVE_* -> -D macro mapping and exit "
                      "without invoking the ROCm toolchain")
_ap.add_argument("--champion", choices=("v62_champion", "v70_champion"), default=None,
                 help="build the V62 (ep16/v62/, Nebula job 01a204b684c8479453de03934774988b, "
                      "P.co sha256 e915ce1f...) or V70 Path-B 160x256 (ep16/v70/, 32 KiB dynamic LDS, "
                      "all 17 active MI308X WEAVE_* knobs + 3 CU modes) EP16 champion TU")
_ARGS, _REST = _ap.parse_known_args()
DRY_RUN = _ARGS.dry_run

if _ARGS.champion == "v62_champion":
    # Delegate to the self-contained V62 builder (ep16/v62/build_v62.py, the
    # byte-for-byte recorded V62 pipeline + v62_champion_preset.json flags and
    # sha256 source manifest).  The 21-knob port path below is not touched.
    import sys
    _v62 = pathlib.Path(__file__).resolve().parent / "v62" / "build_v62.py"
    raise SystemExit(subprocess.call(
        [sys.executable, str(_v62), *(["--dry-run"] if DRY_RUN else []), *_REST]))

if _ARGS.champion == "v70_champion":
    # Delegate to the V70 Path-B (160x256, 32 KiB dynamic LDS) builder supporting
    # all 17 active MI308X WEAVE_* knobs, 0-8/4-4/8-0 CU modes, and FI0/FI2.
    import sys
    _v70 = pathlib.Path(__file__).resolve().parent / "v70" / "build_v70.py"
    raise SystemExit(subprocess.call(
        [sys.executable, str(_v70), *(["--dry-run"] if DRY_RUN else []), *_REST]))

R = pathlib.Path(__file__).resolve().parent          # .../amd/ep16
AMD = R.parent                                       # .../amd
REPO = AMD.parents[4]                                # weave repository root
CLANG = os.environ.get("WEAVE_CLANG", "/opt/rocm-7.2.0/lib/llvm/bin/clang")
LLD = os.environ.get("WEAVE_LLD", "/opt/rocm-7.2.0/lib/llvm/bin/ld.lld")
PY = os.environ.get("WEAVE_PY", "/opt/conda/envs/python3.10.13/bin/python")
# SNAP / V3225 are consumed with the pathlib "/" operator below (e.g.
# SNAP / "overlay_gemm2"), so they must be Path objects, not str.  The previous
# str(...) default caused "TypeError: unsupported operand type(s) for /:
# 'str' and 'str'" on the gfx942 EP16 build when WEAVE_SNAP_DIR was unset.
SNAP = pathlib.Path(os.environ.get("WEAVE_SNAP_DIR", str(AMD / "third_party_overlay")))
V3225 = pathlib.Path(os.environ.get("WEAVE_V3225_DIR", str(AMD / "third_party_overlay")))
SITE = os.environ.get(
    "WEAVE_AITER_SITE",
    "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages/aiter_meta",
)
PS = os.environ.get(
    "WEAVE_PY_SITE",
    "/opt/conda/envs/python3.10.13/lib/python3.10/site-packages",
)
SRC = AMD / "megamoe_amd.hip"

if not DRY_RUN:
    (R / "compiler_work").mkdir(exist_ok=True)
os.environ.update(
    TMPDIR=str(R / "compiler_work"),
    TMP=str(R / "compiler_work"),
    TEMP=str(R / "compiler_work"),
)

inc = [
    f"-I{AMD}", f"-I{AMD / 'ep16'}", f"-I{AMD / 'include'}",
    f"-I{REPO / 'deep_gemm' / 'include'}",
    f"-I{SNAP / 'overlay_gemm2'}", f"-I{SNAP}", f"-I{V3225}",
    f"-I{SITE}/3rdparty/ck_helper",
    f"-I{SITE}/3rdparty/composable_kernel/include",
    f"-I{SITE}/3rdparty/composable_kernel/library/include",
    f"-I{SITE}/csrc/include", f"-I{SITE}/csrc/ck_gemm_moe_2stages_codegen",
    f"-I{SITE}/csrc/include/torch", f"-I{PS}/pybind11/include",
    "-isystem", f"{PS}/torch/include",
    "-isystem", f"{PS}/torch/include/torch/csrc/api/include",
    "-isystem", f"{PS}/torch/include/THH",
    "-isystem", "/opt/rocm/include",
    "-isystem", "/opt/conda/envs/python3.10.13/include/python3.10",
    "-I/opt/rocm/include/hip",
]


def run(cmd, log, timeout=900):
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    log.write_text(p.stdout)
    print(log.name, "rc", p.returncode, "seconds", round(time.time() - t0, 2),
          p.stdout[-1200:], flush=True)
    if p.returncode:
        raise RuntimeError(str(log))
    return p.stdout


def weave_env(name, legacy=None, default=None):
    value = os.environ.get(name)
    if (value is None or value == "") and legacy:
        value = os.environ.get(legacy)
    return default if value is None or value == "" else value


def weave_int(name, legacy=None, default=0):
    return int(weave_env(name, legacy, str(default)))


report = {}
ablation_preset = weave_int("WEAVE_ABLATION_PRESET", default=0)
if ablation_preset not in (0, 1, 2, 3):
    raise SystemExit("WEAVE_ABLATION_PRESET must be one of 0,1,2,3")

profiler_level = weave_int("WEAVE_PROFILER_LEVEL", "FI_LEVEL", 0)
fi_variants = [("fi0_mode2", 0)]
if profiler_level == 2 or os.environ.get("BUILD_FI2_PROFILER", "0") == "1":
    fi_variants.append(("fi2_mode2", 2))

def weave_named_int(name, mapping, legacy=None, default=0):
    raw = weave_env(name, legacy, None)
    if raw is None:
        return default
    if raw in mapping:
        return mapping[raw]
    raise SystemExit(f"Invalid {name}={raw} (allowed: {', '.join(mapping)})")


def _decimal(name, raw, allowed):
    """Strict 1..9 digit decimal (no sign/space/underscore); else exit 1."""
    if not re.fullmatch(r"[0-9]{1,9}", raw):
        raise SystemExit(f"Invalid {name}={raw} (allowed: {allowed})")
    return int(raw)


def _uint(name, lo, hi, default):
    """Integer knob in [lo, hi]; unset/empty -> default; anything else exits 1."""
    raw = weave_env(name, None, None)
    if raw is None:
        return default
    allowed = f"an integer in [{lo},{hi}]"
    value = _decimal(name, raw, allowed)
    if not lo <= value <= hi:
        raise SystemExit(f"Invalid {name}={raw} (expected {allowed})")
    return value


def _choice(name, allowed, default, legacy=None):
    """Integer knob restricted to the tuple `allowed`; unset/empty -> default."""
    raw = weave_env(name, legacy, None)
    if raw is None:
        return default
    used = name if os.environ.get(name) not in (None, "") else (legacy or name)
    text = " ".join(str(a) for a in allowed)
    value = _decimal(used, raw, text)
    if value not in allowed:
        raise SystemExit(f"Invalid {used}={raw} (allowed: {text})")
    return value


for name, fi in fi_variants:
    # WEAVE_SCATTER_OWNER: 0|epilogue_only -> 0, 1|comm_only -> 1, else error.
    # Owner 0 derives SCATTER_MODE 0, owner 1 derives 2, unset derives
    # (preset 3 ? 0 : 2); an explicit WEAVE_SCATTER_MODE still wins.
    owner_raw = weave_env("WEAVE_SCATTER_OWNER", None, None)
    owner_map = {"0": 0, "epilogue_only": 0, "1": 1, "comm_only": 1}
    if owner_raw is not None and owner_raw not in owner_map:
        raise SystemExit(f"Invalid WEAVE_SCATTER_OWNER={owner_raw} (allowed: 0|epilogue_only, 1|comm_only)")
    scatter_owner = None if owner_raw is None else owner_map[owner_raw]
    default_scatter = (0 if ablation_preset == 3 else 2) if scatter_owner is None else (2 if scatter_owner else 0)
    scatter_env = weave_int("WEAVE_SCATTER_MODE", "WEAVE_SCATTER_ENV", default_scatter)
    # WEAVE_ADAPT_BM / WEAVE_DEDUP_MODE are the only public spellings (strict
    # 0/1, anything else exits 1); the tier and selective-dedup macros below
    # stay internal compile-time names.
    adapt_bm = _choice("WEAVE_ADAPT_BM", (0, 1), None)
    bm_tier = 0 if adapt_bm == 0 else 2
    selective = _choice("WEAVE_DEDUP_MODE", (0, 1), 1)
    dualgroup = weave_int("WEAVE_DUALGROUP_MIN_TOKENS", default=64)
    amd_8wave = weave_int("WEAVE_AMD_8WAVE_MIN_TOK", default=dualgroup)
    cu_wave_split = weave_named_int(
        "WEAVE_CU_WAVE_SPLIT",
        {"auto": 0, "0": 0, "0-8": 1, "1": 1, "4-4": 2, "2": 2, "8-0": 3, "3": 3},
        default=0)
    scatter_sms = _uint("WEAVE_SCATTER_ACTIVE_SMS", 0, 80, 0)
    dispatch_sms = _uint("WEAVE_DISPATCH_ACTIVE_SMS", 0, 80, 0)
    # WEAVE_V_HEAD: unset/empty/0/auto -> not passed; else a positive decimal integer.
    v_head_env = weave_env("WEAVE_V_HEAD", None, "auto")
    v_head_raw = 0 if v_head_env == "auto" else _decimal(
        "WEAVE_V_HEAD", v_head_env, "auto|0 or a positive decimal integer")
    bm_l1 = _choice("WEAVE_GEMM1_CONFIG", (0, 16, 32, 64, 128), 0, legacy="WEAVE_BLOCK_M_L1")
    bm_l2 = _choice("WEAVE_GEMM2_CONFIG", (0, 16, 32, 64, 128), 0, legacy="WEAVE_BLOCK_M_L2")
    # MI308X (gfx942) only has the fp8xfp8 path.  Deliberately NOT passed to the
    # compiler: -DWEAVE_PRECISION_MODE would change the CTA-0 cost-model spec, and
    # the validated build keeps the default spec.
    precision_raw = weave_env("WEAVE_PRECISION_MODE", None, None)
    if precision_raw not in (None, "2", "fp8", "fp8xfp8"):
        raise SystemExit("MI308X supports only WEAVE_PRECISION_MODE=2 (fp8xfp8); "
                         f"got WEAVE_PRECISION_MODE={precision_raw}")
    locality = weave_int("WEAVE_LOCALITY_DOMAINS", default=4)
    sm0_sched = _choice("WEAVE_SM0_STREAM_SCHED", (0, 1), 0)
    l2kdep = weave_int("WEAVE_L2_KBLOCK_DEP", default=0)
    ring_blocks = weave_int("WEAVE_RING_BLOCKS", default=0)
    early_stream = weave_int("WEAVE_EARLY_STREAM_REDUCE", default=0)
    # Default stays 0 (weave_knobs.cuh already defaults the macro to 0u).  An
    # explicit WEAVE_LOCAL_SCATTER_DEDUP sets _EXPLICIT=1 so the kernel can tell
    # an explicit request from the header default.
    local_dedup = _choice("WEAVE_LOCAL_SCATTER_DEDUP", (0, 1), 0)
    local_dedup_explicit = 0 if weave_env("WEAVE_LOCAL_SCATTER_DEDUP", None, None) is None else 1
    staging_raw = weave_env("WEAVE_SCATTER_STAGING", None, None)
    scatter_staging = weave_named_int(
        "WEAVE_SCATTER_STAGING",
        {"0": 0, "reg": 0, "reg_2stage": 0, "direct": 0, "1": 1, "smem": 1, "2": 2, "hbm": 2, "hbm_dedup": 2},
        default=0)
    if staging_raw is None and local_dedup == 1 and scatter_owner != 0:
        scatter_staging = 2
    if scatter_staging == 2:
        local_dedup = 1
    if scatter_staging != 0:
        if scatter_owner == 0:
            raise SystemExit(
                f"Invalid WEAVE_SCATTER_STAGING={staging_raw} with WEAVE_SCATTER_OWNER=epilogue_only "
                "(smem / hbm_dedup requires WEAVE_SCATTER_OWNER=comm_only)")
        if scatter_owner is None:
            scatter_owner = 1
    dispatch_warps = _choice("WEAVE_DISPATCH_WARPS", (0, 4, 8), 0)
    scatter_warps = _choice("WEAVE_SCATTER_WARPS", (0, 4, 8), 0)
    ready_group_size = _choice("WEAVE_READY_GROUP_SIZE", (0, 1, 2, 4, 8, 16), 0)
    reg_pull_bytes = _choice("WEAVE_REG_PULL_BYTES", (0, 1024, 2048, 4096, 8192), 0)
    combine_sync_chunk = _choice("WEAVE_COMBINE_SYNC_CHUNK", (0, 16, 32, 64, 128, 512, 1024, 2048, 4096), 0)
    reduce_warps = _choice("WEAVE_REDUCE_WARPS", (0, 1, 2, 4, 8), 0)
    overlap = weave_int("WEAVE_OVERLAP_PIPELINE", default=1)
    all_wave = weave_int("WEAVE_ALL_WAVE_COMBINE", default=1)
    bn2 = weave_int("WEAVE_BN2_MODE", default=0)
    s2_group = weave_int("WEAVE_S2_NGROUP_MODE", default=0)
    route_plan = weave_int("WEAVE_PARALLEL_ROUTE_PLAN", default=1)
    th_bm64 = weave_int("WEAVE_TH_BM64", default=16)
    th_bm128 = weave_int("WEAVE_TH_BM128", default=96)
    out = R / "build" / (name + f"_p{ablation_preset}_m{scatter_env}"
                         + os.environ.get("BUILD_ATTEMPT", ""))
    if not DRY_RUN:
        out.mkdir(parents=True, exist_ok=False)
        (out / "logs").mkdir()
        (out / "cache").mkdir()
        os.environ.update(HIP_CACHE_DIR=str(out / "cache"),
                          AMD_COMGR_CACHE_DIR=str(out / "cache"))
    cmd = [
        CLANG, "-x", "hip", "--offload-arch=gfx942", "--cuda-device-only",
        "-O3", "-S", "-std=c++20", "-DKITTENS_CDNA3", "-DV1098_FORCE_M=0",
        f"-DFI_LEVEL={fi}", f"-DWEAVE_PROFILER_LEVEL={fi}",
        "-DFI_TIMER=0", "-DFI_GRAN=0",
        f"-DWEAVE_PARALLEL_ROUTE_PLAN={route_plan}", "-DWEAVE_PAR_RP=1",
        "-DWEAVE_CK_G2_LDS_ONLY=0", "-DWEAVE_TILE_TRACE_ONLY=1",
        "-DWEAVE_ADAPTIVE_POST_TIMEOUT=1", "-DWEAVE_S16_PLAN_CAP=0",
        "-DWEAVE_S16_STATIC_DISPATCH=0", "-DWEAVE_OPT_RELAXED_SPIN=1",
        "-DWEAVE_OPT_LOCAL_SCOPE=1",
        f"-DWEAVE_SCATTER_MODE={scatter_env}",
        f"-DWEAVE_BM_TIER_MODE={bm_tier}",
        f"-DWEAVE_SELECTIVE_DEDUP={selective}",
        f"-DWEAVE_DUALGROUP_MIN_TOKENS={dualgroup}",
        f"-DWEAVE_AMD_8WAVE_MIN_TOK={amd_8wave}",
        f"-DWEAVE_CU_WAVE_SPLIT={cu_wave_split}",
        f"-DWEAVE_SCATTER_ACTIVE_SMS={scatter_sms}",
        f"-DWEAVE_DISPATCH_ACTIVE_SMS={dispatch_sms}",
        f"-DWEAVE_BLOCK_M_L1={bm_l1}u",
        f"-DWEAVE_BLOCK_M_L2={bm_l2}u",
        *([f"-DWEAVE_V_HEAD={v_head_raw}u"] if v_head_raw > 0 else []),
        f"-DWEAVE_LOCALITY_DOMAINS={locality}",
        f"-DWEAVE_SM0_STREAM_SCHED={sm0_sched}",
        f"-DWEAVE_L2_KBLOCK_DEP={l2kdep}",
        f"-DWEAVE_RING_BLOCKS={ring_blocks}",
        f"-DWEAVE_EARLY_STREAM_REDUCE={early_stream}",
        f"-DWEAVE_LOCAL_SCATTER_DEDUP={local_dedup}",
        f"-DWEAVE_LOCAL_SCATTER_DEDUP_EXPLICIT={local_dedup_explicit}",
        f"-DWEAVE_SCATTER_STAGING={scatter_staging}u",
        f"-DWEAVE_DISPATCH_WARPS={dispatch_warps}u",
        f"-DWEAVE_SCATTER_WARPS={scatter_warps}u",
        f"-DWEAVE_READY_GROUP_SIZE={ready_group_size}u",
        f"-DWEAVE_REG_PULL_BYTES={reg_pull_bytes}u",
        f"-DWEAVE_COMBINE_SYNC_CHUNK={combine_sync_chunk}u",
        f"-DWEAVE_REDUCE_WARPS={reduce_warps}u",
        *([f"-DWEAVE_SCATTER_OWNER={scatter_owner}u",
           f"-DWEAVE_SCATTER_OWNER_EXPLICIT={0 if owner_raw is None else 1}"]
          if scatter_owner is not None else ["-DWEAVE_SCATTER_OWNER_EXPLICIT=0"]),
        f"-DWEAVE_OVERLAP_PIPELINE={overlap}",
        f"-DWEAVE_ALL_WAVE_COMBINE={all_wave}",
        f"-DWEAVE_BN2_MODE={bn2}",
        f"-DWEAVE_S2_NGROUP_MODE={s2_group}",
        "-DWEAVE_CO_PROFILE=0",
        f"-DWEAVE_TH_BM64={th_bm64}u", f"-DWEAVE_TH_BM128={th_bm128}u",
        f"-DWEAVE_ABLATION_PRESET={ablation_preset}",
        "-mllvm", "-amdgpu-opt-vgpr-liverange=true",
        "-mllvm", "-sink-insts-to-avoid-spills=true",
        "-w", "-fno-offload-uniform-block", "-fgpu-flush-denormals-to-zero",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        *inc, str(SRC), "-o", str(out / "stage.s"),
    ]
    if DRY_RUN:
        defines = {}
        for tok in cmd:
            if tok.startswith("-D") and "=" in tok:
                key, value = tok[2:].split("=", 1)
                defines[key] = value
        print(json.dumps({"variant": name, "fi": fi, "preset": ablation_preset,
                          "scatter_mode": scatter_env, "defines": defines}, indent=2))
        continue
    run(cmd, out / "logs/clang.log")
    txt = (out / "stage.s").read_text()
    (out / "stage.norm.s").write_text(
        re.sub(r"__hip_cuid_[0-9a-f]+", "__hip_cuid_v20_ep16", txt))
    run([CLANG, "-cc1as", "-triple", "amdgcn-amd-amdhsa", "-target-cpu",
         "gfx942", "-filetype", "asm", str(out / "stage.norm.s"), "-o",
         str(out / "stage.expanded.s")], out / "logs/expand.log")
    run([PY, str(AMD / "tools/relax_amdgpu_inline_branches_v1.py"),
         str(out / "stage.expanded.s"), str(out / "stage.relaxed.s"),
         "--report", str(out / "logs/relax.json")], out / "logs/relax.log")
    run([CLANG, "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx942", "-c",
         str(out / "stage.relaxed.s"), "-o", str(out / "stage.o")],
        out / "logs/assemble.log")
    run([LLD, "-flavor", "gnu", "--no-undefined", "-shared", "-o",
         str(out / "P.co"), str(out / "stage.o")], out / "logs/link.log")
    audit = json.loads(run(
        [PY, str(AMD / "tools/audit_isa.py"), str(out / "stage.s"),
         "--co", str(out / "P.co"), "--advisory",
         "--symbol", "stage_ep16_turbo_overlap"],
        out / "logs/isa_audit.json"))
    if fi == 0:
        if int(audit.get("vgpr_spill_count_max", -1)) != 0:
            raise RuntimeError(f"{name}: production WEAVE_PROFILER_LEVEL=0 requires zero VGPR spills")
        if int(audit.get("private_segment_fixed_size_max", -1)) != 0:
            raise RuntimeError(f"{name}: production WEAVE_PROFILER_LEVEL=0 requires scratch=0")
        if int(audit.get("s_barrier", -1)) != 1:
            raise RuntimeError(f"{name}: production kernel must contain exactly one s_barrier")
    report[name] = {
        "isa": audit,
        "zero_spills_required": fi == 0,
        "sha256": hashlib.sha256((out / "P.co").read_bytes()).hexdigest(),
        "co": str(out / "P.co"),
    }
    (R / "results").mkdir(exist_ok=True)
    (R / f"results/build_receipt_p{ablation_preset}.json").write_text(
        json.dumps(report, indent=2))

if DRY_RUN:
    print("BUILD_DRY_RUN_DONE")
    raise SystemExit(0)

print("BUILD_DONE", json.dumps({
    k: {"passed": v["isa"]["passed"],
        "spill": v["isa"]["vgpr_spill_count_max"],
        "scratch": v["isa"]["private_segment_fixed_size_max"],
        "s_barrier": v["isa"]["s_barrier"]} for k, v in report.items()}), flush=True)
