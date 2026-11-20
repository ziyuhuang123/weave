#!/usr/bin/env python3
"""Build the unified policy-wired EP16 kernel into a gfx942 P.co.

All repository paths are relative to this file, so cloning weave.git on
mail-21 is sufficient.  The only external dependency is the aiter_meta CK
tree (1100+ files), supplied through WEAVE_AITER_SITE; it is deliberately not
vendored so the git repository stays lightweight.
"""
import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import hashlib
import json
import pathlib
import re
import subprocess
import time

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


report = {}
ablation_preset = int(os.environ.get("WEAVE_ABLATION_PRESET", "0"))
if ablation_preset not in (0, 1, 2, 3):
    raise SystemExit("WEAVE_ABLATION_PRESET must be one of 0,1,2,3")

fi_variants = [("fi0_mode2", 0)]
if os.environ.get("BUILD_FI2_PROFILER", "0") == "1":
    fi_variants.append(("fi2_mode2", 2))

for name, fi in fi_variants:
    scatter_env = os.environ.get("WEAVE_SCATTER_ENV", "0" if ablation_preset == 3 else "2")
    out = R / "build" / (name + f"_p{ablation_preset}_m{scatter_env}"
                         + os.environ.get("BUILD_ATTEMPT", ""))
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    (out / "cache").mkdir()
    os.environ.update(HIP_CACHE_DIR=str(out / "cache"),
                      AMD_COMGR_CACHE_DIR=str(out / "cache"))
    cmd = [
        CLANG, "-x", "hip", "--offload-arch=gfx942", "--cuda-device-only",
        "-O3", "-S", "-std=c++20", "-DKITTENS_CDNA3", "-DV1098_FORCE_M=0",
        f"-DFI_LEVEL={fi}", "-DFI_TIMER=0", "-DFI_GRAN=0",
        "-DWEAVE_PARALLEL_ROUTE_PLAN=1", "-DWEAVE_PAR_RP=1",
        "-DWEAVE_CK_G2_LDS_ONLY=0", "-DWEAVE_TILE_TRACE_ONLY=1",
        "-DWEAVE_ADAPTIVE_POST_TIMEOUT=1", "-DWEAVE_S16_PLAN_CAP=0",
        "-DWEAVE_S16_STATIC_DISPATCH=0", "-DWEAVE_OPT_RELAXED_SPIN=1",
        "-DWEAVE_OPT_LOCAL_SCOPE=1",
        f"-DWEAVE_SCATTER_MODE={int(scatter_env)}",
        "-DWEAVE_CO_PROFILE=0", "-DWEAVE_TH_BM64=16u", "-DWEAVE_TH_BM128=96u",
        f"-DWEAVE_ABLATION_PRESET={ablation_preset}",
        "-DWEAVE_KNOB_BM_TIER_MODE=2", "-DWEAVE_KNOB_SELECTIVE_DEDUP=1",
        "-mllvm", "-amdgpu-opt-vgpr-liverange=true",
        "-mllvm", "-sink-insts-to-avoid-spills=true",
        "-w", "-fno-offload-uniform-block", "-fgpu-flush-denormals-to-zero",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        *inc, str(SRC), "-o", str(out / "stage.s"),
    ]
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
            raise RuntimeError(f"{name}: production FI_LEVEL=0 requires zero VGPR spills")
        if int(audit.get("private_segment_fixed_size_max", -1)) != 0:
            raise RuntimeError(f"{name}: production FI_LEVEL=0 requires scratch=0")
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

print("BUILD_DONE", json.dumps({
    k: {"passed": v["isa"]["passed"],
        "spill": v["isa"]["vgpr_spill_count_max"],
        "scratch": v["isa"]["private_segment_fixed_size_max"],
        "s_barrier": v["isa"]["s_barrier"]} for k, v in report.items()}), flush=True)
