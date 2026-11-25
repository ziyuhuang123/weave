#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NVIDIA SM103 (148-SM) / 4x EP4 / E=256, top-8, H=7168, I=2048, fp8xfp4
# Preset configurations for S_GLOBAL=1024|4096|8192 (default 1024):
#   S=1024 : WEAVE_DISPATCH_ACTIVE_SMS=40
#   S=4096 : WEAVE_DISPATCH_ACTIVE_SMS=96
#   S=8192 : defaults (auto)
# -----------------------------------------------------------------------------
set -euo pipefail

WEAVE_UNIFIED_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
PYTHON="${PYTHON:-python3}"
S_GLOBAL="${S_GLOBAL:-1024}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "$WEAVE_UNIFIED_ROOT" ]] || fail "WEAVE_UNIFIED_ROOT is not a directory: $WEAVE_UNIFIED_ROOT"
[[ -f "$WEAVE_UNIFIED_ROOT/tests/test_mega_moe.py" ]] || fail "missing $WEAVE_UNIFIED_ROOT/tests/test_mega_moe.py"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON"

# Start from the defaults of all public knobs (and of the internal retired pins).
for knob in WEAVE_DISPATCH_ACTIVE_SMS WEAVE_SCATTER_ACTIVE_SMS WEAVE_SM0_STREAM_SCHED \
            WEAVE_DISPATCH_WARPS WEAVE_SCATTER_WARPS WEAVE_REDUCE_WARPS WEAVE_CU_WAVE_SPLIT \
            WEAVE_REG_BALANCE WEAVE_V_HEAD WEAVE_READY_GROUP_SIZE WEAVE_READY_QUEUE \
            WEAVE_SCATTER_OWNER WEAVE_COMBINE_SYNC_CHUNK WEAVE_PRECISION_MODE WEAVE_ADAPT_BM \
            WEAVE_BLOCK_M_L1 WEAVE_BLOCK_M_L2 WEAVE_REG_PULL_BYTES WEAVE_DEDUP_MODE \
            WEAVE_SCATTER_STAGING WEAVE_LOCAL_SCATTER_DEDUP \
            WEAVE_ADAPT_GEMM_CONFIG WEAVE_GEMM1_CONFIG WEAVE_GEMM2_CONFIG; do
    unset "$knob"
done

case "$S_GLOBAL" in
    1024) TOKENS_PER_RANK=256;  export WEAVE_DISPATCH_ACTIVE_SMS=40 ;;
    4096) TOKENS_PER_RANK=1024; export WEAVE_DISPATCH_ACTIVE_SMS=96 ;;
    8192) TOKENS_PER_RANK=2048 ;;   # measured winner is the default configuration
    *) fail "S_GLOBAL must be 1024, 4096 or 8192, got: $S_GLOBAL" ;;
esac

echo "==============================================================================="
echo " NVIDIA SM103 / 4x EP4 / E=256 top-8 H=7168 I=2048 / S=$S_GLOBAL (measured winner)"
echo " WEAVE_DISPATCH_ACTIVE_SMS=${WEAVE_DISPATCH_ACTIVE_SMS:-<default>}   (all other public knobs: default)"
echo "==============================================================================="

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" tests/test_mega_moe.py \
    --num-processes 4 \
    --num-max-tokens-per-rank "$TOKENS_PER_RANK" \
    --num-tokens "$TOKENS_PER_RANK" \
    --hidden 7168 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 8 \
    --num-shared-experts 0 \
    --num-correctness-tests 2 \
    --mma-type fp8xfp4
