#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SM103 (148-SM) / Qwen3.8-Max / EP8 / S=1024
#
# Exact tuned preset recovered from the archived capture scripts, expressed
# only with unified WEAVE_* names.
#
# 3-axis summary
#   Spatial        : WEAVE_BLOCK_M=64, WEAVE_BLOCK_N=128, WEAVE_BLOCK_K=128,
#                    WEAVE_STORE_BLOCK_M=32; 148 SMs; 4 dispatch + 4 TMA/UMMA
#                    + 8 epilogue warp roles.
#   Temporal       : WEAVE_SCATTER_MODE=1 (dispatch-warp scatter),
#                    WEAVE_READY_GROUP_SIZE=2,
#                    WEAVE_COMBINE_GROUP_TILES=2,
#                    WEAVE_COMBINE_REDUCE_WARPS=16.
#   Implementation : WEAVE_PRECISION_MODE=0 (fp8xfp4), WEAVE_DEDUP_MODE=1,
#                    WEAVE_EXPERTS_PER_WAVE=2, WEAVE_REG_PULL_BYTES=4096,
#                    WEAVE_SCATTER_SMEM_ALIAS=1,
#                    WEAVE_SCHED_MODE=3 (reserved, currently identical to 0).
#
# Evidence boundary
#   * The archived trace has no selected_cfg field. The scalar knobs above come
#     from the archived capture worker script.
#   * The tile 64x128x128 + store-32 is heuristic-derived: for Qwen3.8-Max EP8
#     at S=1024, expected tokens per expert = 1024 * 8 * 10 / 512 = 20.0, which
#     falls in the <= 32.5 branch -> (BM=64, BN=128, BK=128, store_m=32,
#     epilogue warpgroup=1).  It is not a literal trace field.
#   * The archived run enabled the tuned block-shape heuristic. In the unified
#     code the explicit WEAVE_BLOCK_* / WEAVE_STORE_BLOCK_M overrides select
#     the same branch, so no extra master switch is needed.
#   * WEAVE_SCHED_MODE=3 is a reserved value with mode-0 behaviour. It is kept
#     because the archived run set it; do not read extra semantics into it.
#   * WEAVE_SHARED_L2_EARLY_BLOCKS=77 was used ONLY when capturing the timeline.
#     The clean 359.55 us latency measurement used the default (10). Set
#     WEAVE_CAPTURE=1 to reproduce the capture-time value.
#
# Usage
#   bash examples/sm103_qwen38max_ep8_s1024.sh
#   PRECISION=fp4xfp4 bash examples/sm103_qwen38max_ep8_s1024.sh
#   WEAVE_CAPTURE=1 bash examples/sm103_qwen38max_ep8_s1024.sh
#
# Environment
#   WEAVE_UNIFIED_ROOT   path to the unified repository checkout
#   PYTHON               Python interpreter (default: python3)
#   PRECISION            fp8xfp4 (default, archived trace mode) or fp4xfp4
#   WEAVE_CAPTURE        1 = set the capture-only shared-L2 seed to 77
#   WEAVE_EXPECT_GPU_SUBSTR  optional GPU-name guard, e.g. "SM103"
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

case "$PRECISION" in
    fp8xfp4) export WEAVE_PRECISION_MODE=0 ;;
    fp4xfp4) export WEAVE_PRECISION_MODE=1 ;;
    *) fail "PRECISION must be fp8xfp4 or fp4xfp4, got: $PRECISION" ;;
esac

# ---- Spatial axis -----------------------------------------------------------
export WEAVE_BLOCK_M=64
export WEAVE_BLOCK_N=128
export WEAVE_BLOCK_K=128
export WEAVE_STORE_BLOCK_M=32

# ---- Temporal axis ----------------------------------------------------------
export WEAVE_SCATTER_MODE=1
export WEAVE_READY_GROUP_SIZE=2
export WEAVE_COMBINE_GROUP_TILES=2
export WEAVE_COMBINE_REDUCE_WARPS=16

# ---- Implementation axis ----------------------------------------------------
export WEAVE_ABLATION_PRESET=0
export WEAVE_DEDUP_MODE=1
export WEAVE_EXPERTS_PER_WAVE=2
export WEAVE_REG_PULL_BYTES=4096
export WEAVE_SCATTER_SMEM_ALIAS=1
export WEAVE_SCHED_MODE=3
export WEAVE_ENABLE_MULTICAST=1

# Capture-only: the timeline used the 77-block shared-L2 seed; production keeps
# the default 10 and the clean latency number.
if [[ "${WEAVE_CAPTURE:-0}" == "1" ]]; then
    export WEAVE_SHARED_L2_EARLY_BLOCKS=77
else
    unset WEAVE_SHARED_L2_EARLY_BLOCKS || true
fi

echo "==============================================================================="
echo " SM103 / Qwen3.8-Max / EP8 / S=1024"
echo " precision              : $PRECISION"
echo " tile                   : ${WEAVE_BLOCK_M}x${WEAVE_BLOCK_N}x${WEAVE_BLOCK_K} store ${WEAVE_STORE_BLOCK_M}"
echo " scatter / ready / comb : mode=${WEAVE_SCATTER_MODE} ready_group=${WEAVE_READY_GROUP_SIZE} combine_tiles=${WEAVE_COMBINE_GROUP_TILES} reduce_warps=${WEAVE_COMBINE_REDUCE_WARPS}"
echo " dedup / epw / pull     : dedup=${WEAVE_DEDUP_MODE} experts_per_wave=${WEAVE_EXPERTS_PER_WAVE} pull_bytes=${WEAVE_REG_PULL_BYTES}"
echo " capture-only L2 seed   : ${WEAVE_SHARED_L2_EARLY_BLOCKS:-default(10)}"
echo "==============================================================================="

"$PYTHON" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")
name = torch.cuda.get_device_name(0)
print(f"GPU 0: {name}")
expect = os.environ.get("WEAVE_EXPECT_GPU_SUBSTR", "")
if expect and expect.lower() not in name.lower():
    raise SystemExit(f"ERROR: expected GPU name containing {expect!r}, got {name!r}")
PY

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "NOTE: CUDA_VISIBLE_DEVICES is unset; all 8 ranks must be visible."
fi
echo "NOTE: if the GPUs are busy with another process, stop and wait; never kill it."

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

if [[ "$PRECISION" == "fp8xfp4" ]]; then
    echo "[run] fp8xfp4 correctness + benchmark via tests/test_mega_moe.py"
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
    echo "[run] fp4xfp4 correctness via tests/test_nvfp4_mega_moe.py"
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

echo "[done] preset run finished."
echo "[trace] this preset does not by itself emit a paired SW+HW Perfetto trace."
echo "[trace] render the archived timeline with:"
echo "        python examples/profile_and_render_timeline.py --preset sm103 --render"
