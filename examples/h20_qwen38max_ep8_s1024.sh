#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NVIDIA H20 (SM90, 78 SMs) / Qwen3.8-Max / EP8 / S=1024
#
# Tuned preset expressed strictly with the unified 21 WEAVE_* parameters
# (Spatial / Temporal / Implementation). Selecting WEAVE_BLOCK_M_L1=32 and
# WEAVE_BLOCK_M_L2=32 synchronously locks the paired BN=256, BK=128, swapAB,
# 4-warpgroup epilogue, and runtime pipeline stage profile.
# -----------------------------------------------------------------------------
set -euo pipefail

WEAVE_UNIFIED_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
PYTHON="${PYTHON:-python3}"
H20_FILTER="${H20_FILTER:-L3.h200_pro_m128}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "$WEAVE_UNIFIED_ROOT" ]] || fail "WEAVE_UNIFIED_ROOT is not a directory: $WEAVE_UNIFIED_ROOT"
[[ -f "$WEAVE_UNIFIED_ROOT/tests/bench_h20_ep8.py" ]] || fail "missing $WEAVE_UNIFIED_ROOT/tests/bench_h20_ep8.py"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON"

# ---- Spatial ----------------------------------------------------------------
export WEAVE_DISPATCH_ACTIVE_SMS="${WEAVE_DISPATCH_ACTIVE_SMS:-0}"
export WEAVE_SCATTER_ACTIVE_SMS="${WEAVE_SCATTER_ACTIVE_SMS:-0}"
export WEAVE_SM0_STREAM_SCHED="${WEAVE_SM0_STREAM_SCHED:-0}"
export WEAVE_DISPATCH_WARPS="${WEAVE_DISPATCH_WARPS:-0}"
export WEAVE_REG_BALANCE="${WEAVE_REG_BALANCE:-0}"

# ---- Temporal ---------------------------------------------------------------
export WEAVE_V_HEAD="${WEAVE_V_HEAD:-0}"
export WEAVE_READY_GROUP_SIZE="${WEAVE_READY_GROUP_SIZE:-0}"
export WEAVE_READY_QUEUE="${WEAVE_READY_QUEUE:-sm0_fifo}"
export WEAVE_SCATTER_OWNER="${WEAVE_SCATTER_OWNER:-comm_only}"
export WEAVE_COMBINE_SYNC_CHUNK="${WEAVE_COMBINE_SYNC_CHUNK:-0}"

# ---- Implementation ---------------------------------------------------------
export WEAVE_PRECISION_MODE="${WEAVE_PRECISION_MODE:-2}"
export WEAVE_ADAPT_BM="${WEAVE_ADAPT_BM:-0}"
export WEAVE_BLOCK_M_L1="${WEAVE_BLOCK_M_L1:-32}"
export WEAVE_BLOCK_M_L2="${WEAVE_BLOCK_M_L2:-32}"
export WEAVE_DEDUP_MODE="${WEAVE_DEDUP_MODE:-1}"
export WEAVE_SCATTER_STAGING="${WEAVE_SCATTER_STAGING:-smem}"
export WEAVE_LOCAL_SCATTER_DEDUP="${WEAVE_LOCAL_SCATTER_DEDUP:-0}"

echo "==============================================================================="
echo " NVIDIA H20 / Qwen3.8-Max / EP8 / S=1024 (Unified WEAVE_* Preset)"
echo " Spatial        : DISPATCH_SMS=$WEAVE_DISPATCH_ACTIVE_SMS SCATTER_SMS=$WEAVE_SCATTER_ACTIVE_SMS SM0_SCHED=$WEAVE_SM0_STREAM_SCHED DISPATCH_WARPS=$WEAVE_DISPATCH_WARPS REG_BALANCE=$WEAVE_REG_BALANCE"
echo " Temporal       : V_HEAD=$WEAVE_V_HEAD READY_GROUP=$WEAVE_READY_GROUP_SIZE READY_QUEUE=$WEAVE_READY_QUEUE SCATTER_OWNER=$WEAVE_SCATTER_OWNER COMBINE_SYNC_CHUNK=$WEAVE_COMBINE_SYNC_CHUNK"
echo " Implementation : PRECISION_MODE=$WEAVE_PRECISION_MODE ADAPT_BM=$WEAVE_ADAPT_BM BLOCK_M_L1=$WEAVE_BLOCK_M_L1 BLOCK_M_L2=$WEAVE_BLOCK_M_L2 DEDUP_MODE=$WEAVE_DEDUP_MODE SCATTER_STAGING=$WEAVE_SCATTER_STAGING LOCAL_SCATTER_DEDUP=$WEAVE_LOCAL_SCATTER_DEDUP"
echo "==============================================================================="

EXTRA=()
if [[ "${WEAVE_ALLOW_OTHER_GPU:-0}" == "1" ]]; then
    EXTRA+=(--allow-other-gpu)
fi

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" tests/bench_h20_ep8.py \
    --num-processes 8 \
    --filter "$H20_FILTER" \
    "${EXTRA[@]}"
