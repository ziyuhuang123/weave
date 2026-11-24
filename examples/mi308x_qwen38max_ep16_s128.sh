#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# AMD MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP16 / S=128 (2 nodes x 8 ranks)
#
# Tuned build + run preset expressed with the unified WEAVE_* parameters (15
# public knobs: Spatial / Temporal / Implementation, plus the internal pins of the
# retired knobs this preset was measured with).  The MI308X build scripts read only
# the legacy spellings WEAVE_ADAPT_BM / WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2 of
# WEAVE_ADAPT_GEMM_CONFIG / WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG, so those
# names are kept here on purpose.
# -----------------------------------------------------------------------------
set -euo pipefail

SRC_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
RUN_ROOT="${WEAVE_RUN_ROOT:-$HOME/weave_example_runs/mi308x_ep16_s128}"
PROFILE_LEVEL="${PROFILE_LEVEL:-0}"

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
[[ -f "$SRC_ROOT/deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py" ]] || fail "missing EP16 build helper under $SRC_ROOT"
: "${WEAVE_EP16_PKG:?Set WEAVE_EP16_PKG to the package root}"
: "${WEAVE_EP16_HOST_LIBS:?Set WEAVE_EP16_HOST_LIBS to the host .so directory}"
: "${WEAVE_EP16_ROUTE_MANIFEST:?Set WEAVE_EP16_ROUTE_MANIFEST to the route manifest}"
: "${WEAVE_EP16_WEIGHT_CACHE:?Set WEAVE_EP16_WEIGHT_CACHE to the weight cache}"
: "${NODE_RANK:?Set NODE_RANK (0 or 1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -f "$WEAVE_EP16_PKG/baseline/ep_common.py" ]]; then
    EP16_PY_BASE="$WEAVE_EP16_PKG/baseline"
else
    EP16_PY_BASE="$WEAVE_EP16_PKG"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERLAY="$RUN_ROOT/overlay-$STAMP"
mkdir -p "$OVERLAY/deep_gemm" "$RUN_ROOT/tmp" "$RUN_ROOT/cache/hip" "$RUN_ROOT/cache/comgr" "$RUN_ROOT/results" "$RUN_ROOT/live_arms"
cp -a "$SRC_ROOT/deep_gemm/include" "$OVERLAY/deep_gemm/include"
EP16_DIR="$OVERLAY/deep_gemm/include/deep_gemm/impls/amd/ep16"

export TMPDIR="$RUN_ROOT/tmp" TMP="$RUN_ROOT/tmp" TEMP="$RUN_ROOT/tmp"
export HIP_CACHE_DIR="$RUN_ROOT/cache/hip" AMD_COMGR_CACHE_DIR="$RUN_ROOT/cache/comgr"
export PYTHONDONTWRITEBYTECODE=1
unset BUILD_ATTEMPT || true

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
export WEAVE_ADAPT_BM="${WEAVE_ADAPT_BM:-1}"
export WEAVE_BLOCK_M_L1="${WEAVE_BLOCK_M_L1:-0}"
export WEAVE_BLOCK_M_L2="${WEAVE_BLOCK_M_L2:-0}"
export WEAVE_DEDUP_MODE="${WEAVE_DEDUP_MODE:-1}"
export WEAVE_LOCAL_SCATTER_DEDUP="${WEAVE_LOCAL_SCATTER_DEDUP:-1}"

WEAVE_PROFILER_LEVEL="$PROFILE_LEVEL" "$WEAVE_PY" "$EP16_DIR/build_port.py"
CO="$EP16_DIR/build/fi0_mode2_p0_m2/P.co"
[[ -f "$CO" ]] || fail "build did not produce $CO"
cp -f "$CO" "$RUN_ROOT/live_arms/P_ep16_turbo_overlap_mode2.co"
cp -f "$WEAVE_EP16_HOST_LIBS/libep16_turbo_host_mode2.so" "$RUN_ROOT/live_arms/"
cp -f "$WEAVE_EP16_HOST_LIBS/gda_turbo_mode2.so" "$RUN_ROOT/live_arms/"

export PYTHONPATH="$EP16_PY_BASE:$WEAVE_EP16_PKG:${PYTHONPATH:-}"
"$WEAVE_TORCHRUN" --nnodes=2 --nproc-per-node=8 \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    "$SRC_ROOT/deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py" \
    --model qwen38max \
    --s-values 128 \
    --route-manifest "$WEAVE_EP16_ROUTE_MANIFEST" \
    --weight-cache "$WEAVE_EP16_WEIGHT_CACHE" \
    --run-dir "$RUN_ROOT/results" \
    --live-log "$RUN_ROOT/live.jsonl" \
    --warmup 20 \
    --iters 50 \
    --live-arms-dir "$RUN_ROOT/live_arms"
