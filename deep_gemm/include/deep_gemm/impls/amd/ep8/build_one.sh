#!/bin/bash
set -euo pipefail

NAME=${1:?Usage: ./build_one.sh <out_name> [prof=0] [th64=16] [th128=96] [extra_flags] [advisory=0]}
PROF=${2:-0}
TH64=${3:-16}
TH128=${4:-96}
EXTRA=${5:-""}
ADVISORY=${6:-0}

# ---- public WEAVE_* knob parsing (fail fast, exit 1, message on stderr) ----
die() { echo "[build_one] ERROR: $*" >&2; exit 1; }
# weave_uint OUT_VAR ENV_NAME LO HI DEFAULT
#   Read $ENV_NAME (unset/empty -> DEFAULT); require a decimal integer in
#   [LO,HI]; store its canonical form (no leading zeros, so clang never sees an
#   octal literal) in OUT_VAR.  Results go through printf -v (no subshell) so
#   that die() really terminates the script.
weave_uint() {
  local _o=$1 _n=$2 _lo=$3 _hi=$4 _r _v
  _r=${!_n:-$5}
  case "$_r" in
    ''|*[!0-9]*|??????????*) die "Invalid $_n=$_r (expected an integer in [$_lo,$_hi])" ;;
  esac
  _v=$((10#$_r))
  if [ "$_v" -lt "$_lo" ] || [ "$_v" -gt "$_hi" ]; then
    die "Invalid $_n=$_r (expected an integer in [$_lo,$_hi])"
  fi
  printf -v "$_o" '%s' "$_v"
}
# weave_choice OUT_VAR ENV_NAME "ALLOWED..." DEFAULT : integer restricted to a set.
weave_choice() {
  local _o=$1 _n=$2 _allowed=$3 _r _v
  _r=${!_n:-$4}
  case "$_r" in
    ''|*[!0-9]*|??????????*) die "Invalid $_n=$_r (allowed: $_allowed)" ;;
  esac
  _v=$((10#$_r))
  case " $_allowed " in
    *" $_v "*) ;;
    *) die "Invalid $_n=$_r (allowed: $_allowed)" ;;
  esac
  printf -v "$_o" '%s' "$_v"
}

PRESET=${WEAVE_ABLATION_PRESET:-0}
# WEAVE_SCATTER_OWNER: 0|epilogue_only -> 0, 1|comm_only -> 1, anything else is
# an error.  Owner 0 derives SCATTER_MODE 0, owner 1 derives SCATTER_MODE 2,
# unset derives (preset 3 ? 0 : 2); an explicit WEAVE_SCATTER_MODE still wins.
case "${WEAVE_SCATTER_OWNER:-}" in
  '')
    DEFAULT_SCATTER=$([ "$PRESET" = "3" ] && echo 0 || echo 2)
    OWNER_DEF="-DWEAVE_SCATTER_OWNER_EXPLICIT=0" ;;
  0|epilogue_only)
    DEFAULT_SCATTER=0
    OWNER_DEF="-DWEAVE_SCATTER_OWNER=0u -DWEAVE_SCATTER_OWNER_EXPLICIT=1" ;;
  1|comm_only)
    DEFAULT_SCATTER=2
    OWNER_DEF="-DWEAVE_SCATTER_OWNER=1u -DWEAVE_SCATTER_OWNER_EXPLICIT=1" ;;
  *) die "Invalid WEAVE_SCATTER_OWNER=${WEAVE_SCATTER_OWNER} (allowed: 0|epilogue_only, 1|comm_only)" ;;
esac
SCATTER=${WEAVE_SCATTER_MODE:-$DEFAULT_SCATTER}
if [ -n "${WEAVE_ADAPT_BM:-}" ]; then
  weave_choice ADAPT_BM WEAVE_ADAPT_BM "0 1" 0
  BM_TIER=$([ "$ADAPT_BM" = "1" ] && echo 2 || echo 0)
else
  # WEAVE_ADAPT_BM is the only public spelling; the tier macro stays internal.
  BM_TIER=2
fi
# WEAVE_DEDUP_MODE is the only public spelling (strict 0/1); the
# selective-dedup macro stays an internal compile-time name.
weave_choice SELECTIVE WEAVE_DEDUP_MODE "0 1" 1
DUALGROUP=${WEAVE_DUALGROUP_MIN_TOKENS:-64}
AMD_8WAVE_MIN_TOK=${WEAVE_AMD_8WAVE_MIN_TOK:-$DUALGROUP}
case "${WEAVE_CU_WAVE_SPLIT:-0}" in
  auto|0) CU_WAVE_SPLIT=0 ;;
  0-8|1)  CU_WAVE_SPLIT=1 ;;
  4-4|2)  CU_WAVE_SPLIT=2 ;;
  8-0|3)  CU_WAVE_SPLIT=3 ;;
  *) die "Invalid WEAVE_CU_WAVE_SPLIT=${WEAVE_CU_WAVE_SPLIT} (allowed: auto|0, 0-8|1, 4-4|2, 8-0|3)" ;;
esac
weave_uint SCATTER_SMS WEAVE_SCATTER_ACTIVE_SMS 0 80 0
weave_uint DISPATCH_SMS WEAVE_DISPATCH_ACTIVE_SMS 0 80 0
if [ -n "${WEAVE_GEMM1_CONFIG:-}" ]; then
  weave_choice BM_L1 WEAVE_GEMM1_CONFIG "0 16 32 64 128" 0
else
  weave_choice BM_L1 WEAVE_BLOCK_M_L1 "0 16 32 64 128" 0
fi
if [ -n "${WEAVE_GEMM2_CONFIG:-}" ]; then
  weave_choice BM_L2 WEAVE_GEMM2_CONFIG "0 16 32 64 128" 0
else
  weave_choice BM_L2 WEAVE_BLOCK_M_L2 "0 16 32 64 128" 0
fi
# WEAVE_V_HEAD: unset/empty/0/auto -> not passed; else a positive decimal integer.
V_HEAD_DEF=""
case "${WEAVE_V_HEAD:-auto}" in
  auto) ;;
  *[!0-9]*|??????????*) die "Invalid WEAVE_V_HEAD=${WEAVE_V_HEAD} (allowed: auto|0 or a positive decimal integer)" ;;
  *)
    V_HEAD=$((10#${WEAVE_V_HEAD}))
    if [ "$V_HEAD" -gt 0 ]; then V_HEAD_DEF="-DWEAVE_V_HEAD=${V_HEAD}u"; fi ;;
esac
# MI308X (gfx942) only has the fp8xfp8 path.  Deliberately NOT passed to the
# compiler: -DWEAVE_PRECISION_MODE would change the CTA-0 cost-model spec, and
# the validated build keeps the default spec.
case "${WEAVE_PRECISION_MODE:-}" in
  ''|2|fp8|fp8xfp8) ;;
  *) die "MI308X supports only WEAVE_PRECISION_MODE=2 (fp8xfp8); got WEAVE_PRECISION_MODE=${WEAVE_PRECISION_MODE}" ;;
esac
LOCALITY=${WEAVE_LOCALITY_DOMAINS:-4}
weave_choice SM0 WEAVE_SM0_STREAM_SCHED "0 1" 0
L2KDEP=${WEAVE_L2_KBLOCK_DEP:-0}
RING_BLOCKS=${WEAVE_RING_BLOCKS:-0}
EARLY=${WEAVE_EARLY_STREAM_REDUCE:-0}
# Default stays 0 (weave_knobs.cuh already defaults the macro to 0u).  An
# explicit WEAVE_LOCAL_SCATTER_DEDUP sets _EXPLICIT=1 so the kernel can tell
# an explicit request from the header default.
weave_choice LOCAL_DEDUP WEAVE_LOCAL_SCATTER_DEDUP "0 1" 0
LOCAL_DEDUP_EXPLICIT=$([ -n "${WEAVE_LOCAL_SCATTER_DEDUP:-}" ] && echo 1 || echo 0)
STAGING_RAW="${WEAVE_SCATTER_STAGING:-}"
case "$STAGING_RAW" in
  ''|0|reg|reg_2stage|direct) SCATTER_STAGING=0 ;;
  1|smem) SCATTER_STAGING=1 ;;
  2|hbm|hbm_dedup) SCATTER_STAGING=2 ;;
  *) die "Invalid WEAVE_SCATTER_STAGING=${STAGING_RAW} (allowed: 0|reg, 1|smem, 2|hbm)" ;;
esac
if [ -z "$STAGING_RAW" ] && [ "$LOCAL_DEDUP" = "1" ] && [ "${WEAVE_SCATTER_OWNER:-}" != "0" ] && [ "${WEAVE_SCATTER_OWNER:-}" != "epilogue_only" ]; then
  SCATTER_STAGING=2
fi
if [ "$SCATTER_STAGING" = "2" ]; then
  LOCAL_DEDUP=1
fi
if [ "$SCATTER_STAGING" != "0" ]; then
  case "${WEAVE_SCATTER_OWNER:-}" in
    0|epilogue_only) die "Invalid WEAVE_SCATTER_STAGING=${STAGING_RAW} with WEAVE_SCATTER_OWNER=epilogue_only (requires comm_only)" ;;
    '') OWNER_DEF="-DWEAVE_SCATTER_OWNER=1u -DWEAVE_SCATTER_OWNER_EXPLICIT=0" ;;
  esac
fi
weave_choice DISPATCH_WARPS WEAVE_DISPATCH_WARPS "0 4 8" 0
weave_choice SCATTER_WARPS WEAVE_SCATTER_WARPS "0 4 8" 0
weave_choice READY_GROUP_SIZE WEAVE_READY_GROUP_SIZE "0 1 2 4 8 16" 0
weave_choice REG_PULL_BYTES WEAVE_REG_PULL_BYTES "0 1024 2048 4096" 0
weave_choice COMBINE_SYNC_CHUNK WEAVE_COMBINE_SYNC_CHUNK "0 16 32 64 128" 0
weave_choice REDUCE_WARPS WEAVE_REDUCE_WARPS "0 1 2 4 8" 0
OVERLAP=${WEAVE_OVERLAP_PIPELINE:-1}
ALL_WAVE=${WEAVE_ALL_WAVE_COMBINE:-1}
BN2=${WEAVE_BN2_MODE:-0}
S2_GROUP=${WEAVE_S2_NGROUP_MODE:-0}

# CPU-only dry run: print the resolved knob -> macro mapping (the same
# knob-derived -D flags the clang command below receives) and exit before any
# ROCm toolchain is invoked.  Exercised by tests/test_amd_build_knob_dry_run.py.
if [ "${WEAVE_DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' \
    "-DWEAVE_ABLATION_PRESET=$PRESET" \
    "-DWEAVE_SCATTER_MODE=$SCATTER" \
    "-DWEAVE_BM_TIER_MODE=$BM_TIER" \
    "-DWEAVE_SELECTIVE_DEDUP=$SELECTIVE" \
    "-DWEAVE_DUALGROUP_MIN_TOKENS=$DUALGROUP" \
    "-DWEAVE_AMD_8WAVE_MIN_TOK=$AMD_8WAVE_MIN_TOK" \
    "-DWEAVE_CU_WAVE_SPLIT=$CU_WAVE_SPLIT" \
    "-DWEAVE_SCATTER_ACTIVE_SMS=$SCATTER_SMS" \
    "-DWEAVE_DISPATCH_ACTIVE_SMS=$DISPATCH_SMS" \
    "-DWEAVE_BLOCK_M_L1=${BM_L1}u" \
    "-DWEAVE_BLOCK_M_L2=${BM_L2}u" \
    "-DWEAVE_LOCALITY_DOMAINS=$LOCALITY" \
    "-DWEAVE_SM0_STREAM_SCHED=$SM0" \
    "-DWEAVE_L2_KBLOCK_DEP=$L2KDEP" \
    "-DWEAVE_RING_BLOCKS=$RING_BLOCKS" \
    "-DWEAVE_EARLY_STREAM_REDUCE=$EARLY" \
    "-DWEAVE_LOCAL_SCATTER_DEDUP=$LOCAL_DEDUP" \
    "-DWEAVE_LOCAL_SCATTER_DEDUP_EXPLICIT=$LOCAL_DEDUP_EXPLICIT" \
    "-DWEAVE_SCATTER_STAGING=${SCATTER_STAGING}u" \
    "-DWEAVE_DISPATCH_WARPS=${DISPATCH_WARPS}u" \
    "-DWEAVE_SCATTER_WARPS=${SCATTER_WARPS}u" \
    "-DWEAVE_READY_GROUP_SIZE=${READY_GROUP_SIZE}u" \
    "-DWEAVE_REG_PULL_BYTES=${REG_PULL_BYTES}u" \
    "-DWEAVE_COMBINE_SYNC_CHUNK=${COMBINE_SYNC_CHUNK}u" \
    "-DWEAVE_REDUCE_WARPS=${REDUCE_WARPS}u" \
    "-DWEAVE_OVERLAP_PIPELINE=$OVERLAP" \
    "-DWEAVE_ALL_WAVE_COMBINE=$ALL_WAVE" \
    "-DWEAVE_BN2_MODE=$BN2" \
    "-DWEAVE_S2_NGROUP_MODE=$S2_GROUP" \
    "-DWEAVE_PROFILER_LEVEL=$PROF"
  # Owner flags are one or two tokens (unset owner -> only the _EXPLICIT=0 flag).
  for _tok in $OWNER_DEF; do printf '%s\n' "$_tok"; done
  if [ -n "$V_HEAD_DEF" ]; then printf '%s\n' "$V_HEAD_DEF"; fi
  exit 0
fi

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

# EP8 keeps its validated communication headers in the local include tree.
$CLANG -x hip --offload-arch=gfx942 --cuda-device-only -O3 -S \
  -std=c++20 -DKITTENS_CDNA3 -DV1098_FORCE_M=0 -DFI_LEVEL=$PROF -DWEAVE_PROFILER_LEVEL=$PROF -DFI_TIMER=0 -DFI_GRAN=0 \
  -DWEAVE_PARALLEL_ROUTE_PLAN=${WEAVE_PARALLEL_ROUTE_PLAN:-1} -DWEAVE_PAR_RP=1 -DWEAVE_CK_G2_LDS_ONLY=0 -DWEAVE_TILE_TRACE_ONLY=1 \
  -DWEAVE_ADAPTIVE_POST_TIMEOUT=1 \
  -DWEAVE_S16_PLAN_CAP=0 -DWEAVE_S16_STATIC_DISPATCH=0 \
  -DWEAVE_OPT_RELAXED_SPIN=1 -DWEAVE_OPT_LOCAL_SCOPE=1 \
  -mllvm -amdgpu-opt-vgpr-liverange=true -mllvm -sink-insts-to-avoid-spills=true \
  $EXTRA -DWEAVE_CO_PROFILE=$PROF -DWEAVE_TH_BM64=${TH64}u -DWEAVE_TH_BM128=${TH128}u \
  -DWEAVE_ABLATION_PRESET=$PRESET -DWEAVE_SCATTER_MODE=$SCATTER \
  -DWEAVE_BM_TIER_MODE=$BM_TIER -DWEAVE_SELECTIVE_DEDUP=$SELECTIVE \
  -DWEAVE_DUALGROUP_MIN_TOKENS=$DUALGROUP -DWEAVE_OVERLAP_PIPELINE=$OVERLAP \
  -DWEAVE_AMD_8WAVE_MIN_TOK=$AMD_8WAVE_MIN_TOK -DWEAVE_CU_WAVE_SPLIT=$CU_WAVE_SPLIT \
  -DWEAVE_SCATTER_ACTIVE_SMS=$SCATTER_SMS -DWEAVE_DISPATCH_ACTIVE_SMS=$DISPATCH_SMS \
  -DWEAVE_BLOCK_M_L1=${BM_L1}u -DWEAVE_BLOCK_M_L2=${BM_L2}u $V_HEAD_DEF \
  -DWEAVE_LOCALITY_DOMAINS=$LOCALITY \
  -DWEAVE_SM0_STREAM_SCHED=$SM0 -DWEAVE_L2_KBLOCK_DEP=$L2KDEP \
  -DWEAVE_RING_BLOCKS=$RING_BLOCKS -DWEAVE_EARLY_STREAM_REDUCE=$EARLY \
  -DWEAVE_LOCAL_SCATTER_DEDUP=$LOCAL_DEDUP -DWEAVE_LOCAL_SCATTER_DEDUP_EXPLICIT=$LOCAL_DEDUP_EXPLICIT \
  -DWEAVE_SCATTER_STAGING=${SCATTER_STAGING}u -DWEAVE_DISPATCH_WARPS=${DISPATCH_WARPS}u \
  -DWEAVE_SCATTER_WARPS=${SCATTER_WARPS}u -DWEAVE_READY_GROUP_SIZE=${READY_GROUP_SIZE}u \
  -DWEAVE_REG_PULL_BYTES=${REG_PULL_BYTES}u \
  -DWEAVE_COMBINE_SYNC_CHUNK=${COMBINE_SYNC_CHUNK}u -DWEAVE_REDUCE_WARPS=${REDUCE_WARPS}u $OWNER_DEF \
  -DWEAVE_ALL_WAVE_COMBINE=$ALL_WAVE -DWEAVE_BN2_MODE=$BN2 -DWEAVE_S2_NGROUP_MODE=$S2_GROUP \
  -w -fno-offload-uniform-block -fgpu-flush-denormals-to-zero \
  -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
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
