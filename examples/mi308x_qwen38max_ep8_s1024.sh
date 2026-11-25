#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# AMD MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP8 / S=1024
# Unified 14-knob WEAVE_* build + run preset (Spatial / Temporal / Implementation).
# -----------------------------------------------------------------------------
set -euo pipefail

SRC_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
RUN_ROOT="${WEAVE_RUN_ROOT:-$HOME/weave_example_runs/mi308x_ep8_s1024}"
PROFILE_LEVEL="${PROFILE_LEVEL:-0}"
BUILD_NAME="${EP8_BUILD_NAME:-P}"

WEAVE_AITER_SITE="${WEAVE_AITER_SITE:-/opt/conda/envs/python3.10.13/lib/python3.10/site-packages/aiter_meta}"
WEAVE_PY_SITE="${WEAVE_PY_SITE:-/opt/conda/envs/python3.10.13/lib/python3.10/site-packages}"
WEAVE_CLANG="${WEAVE_CLANG:-/opt/rocm-7.2.0/lib/llvm/bin/clang}"
WEAVE_LLD="${WEAVE_LLD:-/opt/rocm-7.2.0/lib/llvm/bin/ld.lld}"
WEAVE_PY="${WEAVE_PY:-/opt/conda/envs/python3.10.13/bin/python}"
WEAVE_TORCHRUN="${WEAVE_TORCHRUN:-torchrun}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "$SRC_ROOT" ]] || fail "WEAVE_UNIFIED_ROOT is not a directory: $SRC_ROOT"
[[ -f "$SRC_ROOT/deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh" ]] || fail "missing EP8 build helper under $SRC_ROOT"
[[ -n "$RUN_ROOT" && "$RUN_ROOT" != "/" ]] || fail "WEAVE_RUN_ROOT must be a non-root, non-empty path"
[[ "$PROFILE_LEVEL" == "0" || "$PROFILE_LEVEL" == "2" ]] || fail "PROFILE_LEVEL must be 0 or 2"

: "${WEAVE_EP8_LAUNCHER:?Set WEAVE_EP8_LAUNCHER to the host launcher .so}"
: "${WEAVE_EP8_IPC_LIB:?Set WEAVE_EP8_IPC_LIB to the IPC driver .so}"
: "${WEAVE_EP8_ROUTES:?Set WEAVE_EP8_ROUTES to the route-table directory}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERLAY="$RUN_ROOT/overlay-$STAMP"
mkdir -p "$OVERLAY/deep_gemm" "$RUN_ROOT/tmp" "$RUN_ROOT/cache/hip" "$RUN_ROOT/cache/comgr" "$RUN_ROOT/results"
cp -a "$SRC_ROOT/deep_gemm/include" "$OVERLAY/deep_gemm/include"

EP8_DIR="$OVERLAY/deep_gemm/include/deep_gemm/impls/amd/ep8"

export TMPDIR="$RUN_ROOT/tmp" TMP="$RUN_ROOT/tmp" TEMP="$RUN_ROOT/tmp"
export HIP_CACHE_DIR="$RUN_ROOT/cache/hip" AMD_COMGR_CACHE_DIR="$RUN_ROOT/cache/comgr"
export PYTHONDONTWRITEBYTECODE=1

# ---- Spatial ----------------------------------------------------------------
export WEAVE_DISPATCH_ACTIVE_SMS="${WEAVE_DISPATCH_ACTIVE_SMS:-0}"
export WEAVE_SCATTER_ACTIVE_SMS="${WEAVE_SCATTER_ACTIVE_SMS:-0}"
export WEAVE_SM0_STREAM_SCHED="${WEAVE_SM0_STREAM_SCHED:-0}"
export WEAVE_CU_WAVE_SPLIT="${WEAVE_CU_WAVE_SPLIT:-4-4}"

# ---- Temporal ---------------------------------------------------------------
export WEAVE_V_HEAD="${WEAVE_V_HEAD:-0}"
export WEAVE_SCATTER_OWNER="${WEAVE_SCATTER_OWNER:-comm_only}"

# ---- Implementation ---------------------------------------------------------
export WEAVE_PRECISION_MODE="${WEAVE_PRECISION_MODE:-2}"
export WEAVE_GEMM1_CONFIG="${WEAVE_GEMM1_CONFIG:-0}"
export WEAVE_GEMM2_CONFIG="${WEAVE_GEMM2_CONFIG:-0}"
export WEAVE_DEDUP_MODE="${WEAVE_DEDUP_MODE:-1}"
export WEAVE_SCATTER_STAGING="${WEAVE_SCATTER_STAGING:-hbm}"

echo "==============================================================================="
echo " AMD MI308X / Qwen3.8-Max / EP8 / S=1024 (Unified WEAVE_* Preset)"
echo " Spatial        : SCATTER_SMS=$WEAVE_SCATTER_ACTIVE_SMS CU_WAVE_SPLIT=$WEAVE_CU_WAVE_SPLIT"
echo " Temporal       : V_HEAD=$WEAVE_V_HEAD SCATTER_OWNER=$WEAVE_SCATTER_OWNER"
echo " Implementation : PRECISION_MODE=$WEAVE_PRECISION_MODE GEMM1_CONFIG=$WEAVE_GEMM1_CONFIG GEMM2_CONFIG=$WEAVE_GEMM2_CONFIG DEDUP_MODE=$WEAVE_DEDUP_MODE SCATTER_STAGING=$WEAVE_SCATTER_STAGING"
echo "==============================================================================="

bash "$EP8_DIR/build_one.sh" "$BUILD_NAME" "$PROFILE_LEVEL" 16 96 "" 0
CO="$EP8_DIR/$BUILD_NAME/P.co"
[[ -f "$CO" ]] || fail "build did not produce $CO"

cd "$SRC_ROOT"
"$WEAVE_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    "$SRC_ROOT/deep_gemm/include/deep_gemm/impls/amd/ep8/bench_ep8.py" \
    --models qwen38max \
    --seqlens 1024 \
    --co "$CO" \
    --out-dir "$RUN_ROOT/results" \
    --package-root "$SRC_ROOT/deep_gemm/include/deep_gemm/impls/amd/ep8" \
    --routes-dir "$WEAVE_EP8_ROUTES" \
    --launcher "$WEAVE_EP8_LAUNCHER" \
    --ipc-lib "$WEAVE_EP8_IPC_LIB" \
    --warmup 5 \
    --rounds 3 \
    --weight-device cuda \
    --zero-mode sync \
    --save-trace
