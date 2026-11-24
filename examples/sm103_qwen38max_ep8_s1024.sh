#!/usr/bin/env bash
# EP8 example; not measured on the current revision; does not represent the README performance table.
# The measured winners (4x EP4) are in examples/sm103_qwen38max_ep4_s1024.sh.
# -----------------------------------------------------------------------------
# NVIDIA SM103 (148-SM) / Qwen3.8-Max / EP8 / S=1024
#
# Tuned preset expressed with the unified WEAVE_* parameters (15 public knobs:
# Spatial / Temporal / Implementation, plus the internal pins of the retired knobs
# this preset was measured with; WEAVE_ADAPT_BM / WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2
# are the legacy spellings of WEAVE_ADAPT_GEMM_CONFIG / WEAVE_GEMM1_CONFIG /
# WEAVE_GEMM2_CONFIG and are accepted as transparent aliases).
# -----------------------------------------------------------------------------
set -euo pipefail

WEAVE_UNIFIED_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
PYTHON="${PYTHON:-python3}"
PRECISION="${PRECISION:-fp8xfp4}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "$WEAVE_UNIFIED_ROOT" ]] || fail "WEAVE_UNIFIED_ROOT is not a directory: $WEAVE_UNIFIED_ROOT"
[[ -f "$WEAVE_UNIFIED_ROOT/tests/test_mega_moe.py" ]] || fail "missing $WEAVE_UNIFIED_ROOT/tests/test_mega_moe.py"
[[ -f "$WEAVE_UNIFIED_ROOT/tests/test_nvfp4_mega_moe.py" ]] || fail "missing $WEAVE_UNIFIED_ROOT/tests/test_nvfp4_mega_moe.py"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON"

# ---- Spatial ----------------------------------------------------------------
export WEAVE_DISPATCH_ACTIVE_SMS="${WEAVE_DISPATCH_ACTIVE_SMS:-0}"
export WEAVE_SCATTER_ACTIVE_SMS="${WEAVE_SCATTER_ACTIVE_SMS:-0}"
export WEAVE_SM0_STREAM_SCHED="${WEAVE_SM0_STREAM_SCHED:-0}"
export WEAVE_DISPATCH_WARPS="${WEAVE_DISPATCH_WARPS:-4}"
export WEAVE_SCATTER_WARPS="${WEAVE_SCATTER_WARPS:-4}"
export WEAVE_REDUCE_WARPS="${WEAVE_REDUCE_WARPS:-4}"
export WEAVE_REG_BALANCE="${WEAVE_REG_BALANCE:-0}"

# ---- Temporal ---------------------------------------------------------------
export WEAVE_V_HEAD="${WEAVE_V_HEAD:-0}"
export WEAVE_READY_GROUP_SIZE="${WEAVE_READY_GROUP_SIZE:-2}"
export WEAVE_READY_QUEUE="${WEAVE_READY_QUEUE:-formula}"
export WEAVE_SCATTER_OWNER="${WEAVE_SCATTER_OWNER:-comm_only}"
export WEAVE_COMBINE_SYNC_CHUNK="${WEAVE_COMBINE_SYNC_CHUNK:-32}"

# ---- Implementation ---------------------------------------------------------
case "$PRECISION" in
    fp8xfp4) export WEAVE_PRECISION_MODE=0 ;;
    fp4xfp4) export WEAVE_PRECISION_MODE=1 ;;
    *) fail "PRECISION must be fp8xfp4 or fp4xfp4, got: $PRECISION" ;;
esac
export WEAVE_ADAPT_BM="${WEAVE_ADAPT_BM:-0}"
export WEAVE_BLOCK_M_L1="${WEAVE_BLOCK_M_L1:-64}"
export WEAVE_BLOCK_M_L2="${WEAVE_BLOCK_M_L2:-64}"
export WEAVE_REG_PULL_BYTES="${WEAVE_REG_PULL_BYTES:-4096}"
export WEAVE_DEDUP_MODE="${WEAVE_DEDUP_MODE:-1}"
export WEAVE_SCATTER_STAGING="${WEAVE_SCATTER_STAGING:-smem}"
export WEAVE_LOCAL_SCATTER_DEDUP="${WEAVE_LOCAL_SCATTER_DEDUP:-0}"

echo "==============================================================================="
echo " NVIDIA SM103 / Qwen3.8-Max / EP8 / S=1024 (Unified WEAVE_* Preset)"
echo " Spatial        : DISPATCH_SMS=$WEAVE_DISPATCH_ACTIVE_SMS SCATTER_SMS=$WEAVE_SCATTER_ACTIVE_SMS SM0_SCHED=$WEAVE_SM0_STREAM_SCHED DISPATCH_WARPS=$WEAVE_DISPATCH_WARPS SCATTER_WARPS=$WEAVE_SCATTER_WARPS REDUCE_WARPS=$WEAVE_REDUCE_WARPS REG_BALANCE=$WEAVE_REG_BALANCE"
echo " Temporal       : V_HEAD=$WEAVE_V_HEAD READY_GROUP=$WEAVE_READY_GROUP_SIZE READY_QUEUE=$WEAVE_READY_QUEUE SCATTER_OWNER=$WEAVE_SCATTER_OWNER COMBINE_SYNC_CHUNK=$WEAVE_COMBINE_SYNC_CHUNK"
echo " Implementation : PRECISION_MODE=$WEAVE_PRECISION_MODE ADAPT_BM=$WEAVE_ADAPT_BM BLOCK_M_L1=$WEAVE_BLOCK_M_L1 BLOCK_M_L2=$WEAVE_BLOCK_M_L2 REG_PULL_BYTES=$WEAVE_REG_PULL_BYTES DEDUP_MODE=$WEAVE_DEDUP_MODE SCATTER_STAGING=$WEAVE_SCATTER_STAGING LOCAL_SCATTER_DEDUP=$WEAVE_LOCAL_SCATTER_DEDUP"
echo "==============================================================================="

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

if [[ "$PRECISION" == "fp8xfp4" ]]; then
    "$PYTHON" tests/test_mega_moe.py \
        --num-processes 8 \
        --num-max-tokens-per-rank 1024 \
        --num-tokens 1024 \
        --hidden 8192 \
        --intermediate-hidden 2048 \
        --num-experts 512 \
        --num-topk 10 \
        --num-shared-experts 0 \
        --num-correctness-tests 2 \
        --mma-type fp8xfp4
else
    "$PYTHON" tests/test_nvfp4_mega_moe.py \
        --num-processes 8 \
        --num-max-tokens-per-rank 1024 \
        --num-tokens 1024 \
        --hidden 8192 \
        --intermediate-hidden 2048 \
        --num-experts 512 \
        --num-topk 10 \
        --num-shared-experts 0 \
        --fast-math 0 \
        --num-correctness-tests 2
fi
