#!/bin/bash
set -euo pipefail

NAME=${1:?Usage: ./build_one.sh <out_name> [prof=0] [th64=16] [th128=96] [extra_flags] [advisory=0]}
PROF=${2:-0}
TH64=${3:-16}
TH128=${4:-96}
EXTRA=${5:-""}
ADVISORY=${6:-0}

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD="$(cd "$WORK/.." && pwd)"
REPO="$(cd "$AMD/../../../../.." && pwd)"
# Defaults are repository-relative; override only when an external aiter/CK
# installation is required.  The 1100-file aiter_meta CK tree is intentionally
# NOT vendored into git.
SNAP="${WEAVE_SNAP_DIR:-$AMD/third_party_overlay}"
V3225="${WEAVE_V3225_DIR:-$AMD/third_party_overlay}"
SITE="${WEAVE_AITER_SITE:-/opt/conda/envs/python3.10.13/lib/python3.10/site-packages/aiter_meta}"
PY_SITE="${WEAVE_PY_SITE:-/opt/conda/envs/python3.10.13/lib/python3.10/site-packages}"
CLANG="${WEAVE_CLANG:-/opt/rocm-7.2.0/lib/llvm/bin/clang}"
LLD="${WEAVE_LLD:-/opt/rocm-7.2.0/lib/llvm/bin/ld.lld}"
PY="${WEAVE_PY:-/opt/conda/envs/python3.10.13/bin/python}"

OUT="$WORK/$NAME"
rm -rf "$OUT"
mkdir -p "$OUT/scratch" "$OUT/cache" "$OUT/logs" "$WORK/compiler_work"
export TMPDIR="$WORK/compiler_work" TMP="$WORK/compiler_work" TEMP="$WORK/compiler_work"
export HIP_CACHE_DIR="$OUT/cache" AMD_COMGR_CACHE_DIR="$OUT/cache"

SRC_HIP="$WORK/megamoe_amd_ep8.hip"

ADV_FLAG=""
if [ "$ADVISORY" = "1" ]; then
  ADV_FLAG="--advisory"
fi

$CLANG -x hip --offload-arch=gfx942 --cuda-device-only -O3 -S \
  -std=c++20 -DKITTENS_CDNA3 -DV1098_FORCE_M=0 -DFI_LEVEL=0 -DFI_TIMER=0 -DFI_GRAN=0 \
  -DWEAVE_PARALLEL_ROUTE_PLAN=1 -DWEAVE_PAR_RP=1 -DWEAVE_CK_G2_LDS_ONLY=0 -DWEAVE_TILE_TRACE_ONLY=1 \
  -DWEAVE_ADAPTIVE_POST_TIMEOUT=1 \
  -DWEAVE_S16_PLAN_CAP=0 -DWEAVE_S16_STATIC_DISPATCH=0 \
  -DWEAVE_OPT_RELAXED_SPIN=1 -DWEAVE_OPT_LOCAL_SCOPE=1 \
  -mllvm -amdgpu-opt-vgpr-liverange=true -mllvm -sink-insts-to-avoid-spills=true \
  $EXTRA -DWEAVE_CO_PROFILE=$PROF -DWEAVE_TH_BM64=${TH64}u -DWEAVE_TH_BM128=${TH128}u \
  -w -fno-offload-uniform-block -fgpu-flush-denormals-to-zero \
  -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
  # EP8 uses the validated v20 comm/compute headers (v20_include/ takes -I
  # priority over the shared include/).  The Phase 2 CTA-0 policy wiring in the
  # shared include/ headers causes device_error=210 on real 8xMI308X EP8; EP16
  # still uses the shared policy-wired headers (required for its zero-spill gate).
  -I"$WORK/v20_include" -I"$AMD/include" -I"$REPO/deep_gemm/include" -I"$SNAP/overlay_gemm2" -I"$SNAP" -I"$V3225" \
  -I"$SITE/3rdparty/ck_helper" -I"$SITE/3rdparty/composable_kernel/include" \
  -I"$SITE/3rdparty/composable_kernel/library/include" \
  -I"$SITE/csrc/include" -I"$SITE/csrc/ck_gemm_moe_2stages_codegen" \
  -I"$SITE/csrc/include/torch" -I"$PY_SITE/pybind11/include" \
  -isystem "$PY_SITE/torch/include" \
  -isystem "$PY_SITE/torch/include/torch/csrc/api/include" \
  -isystem "$PY_SITE/torch/include/THH" -isystem /opt/rocm/include \
  -isystem /opt/conda/envs/python3.10.13/include/python3.10 \
  -I/opt/rocm/include/hip "$SRC_HIP" -o "$OUT/stage.s" \
  >"$OUT/logs/clang.log" 2>&1 || {
    echo "[$NAME] CLANG FAILED:"
    grep -E "error:" "$OUT/logs/clang.log" | head -20 || cat "$OUT/logs/clang.log" | tail -20
    exit 1
  }

sed -E 's/__hip_cuid_[0-9a-f]+/__hip_cuid_v4p4/g' "$OUT/stage.s" > "$OUT/stage.norm.s"
"$CLANG" -cc1as -triple amdgcn-amd-amdhsa -target-cpu gfx942 -filetype asm "$OUT/stage.norm.s" -o "$OUT/stage.expanded.s" >"$OUT/logs/e.log" 2>&1 || {
  echo "[$NAME] expand failed"; tail -12 "$OUT/logs/e.log"; exit 1;
}
"$PY" "$AMD/tools/relax_amdgpu_inline_branches_v1.py" "$OUT/stage.expanded.s" "$OUT/stage.relaxed.s" --report "$OUT/logs/relax.json" >"$OUT/logs/relax.log" 2>&1 || {
  echo "[$NAME] relax failed"; tail -12 "$OUT/logs/relax.log"; exit 1;
}
"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx942 -c "$OUT/stage.relaxed.s" -o "$OUT/stage.o"
"$LLD" -flavor gnu --no-undefined -shared -o "$OUT/P.co.tmp" "$OUT/stage.o"
mv -f "$OUT/P.co.tmp" "$OUT/P.co"

"$PY" "$AMD/tools/audit_isa.py" "$OUT/stage.s" --co "$OUT/P.co" $ADV_FLAG > "$OUT/logs/isa_audit.json"

echo "=== POST-LINK ELF & ISA AUDIT ($NAME) ==="
cat "$OUT/logs/isa_audit.json"
sha256sum "$OUT/P.co" | tee "$OUT/logs/P.co.sha256"
