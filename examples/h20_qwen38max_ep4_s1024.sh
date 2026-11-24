#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NVIDIA H20 (SM90, 78 SMs) / Qwen3.8-Max (E=512, top-10, H=8192, I=2048) / 4x EP4
# Measured-winner script: S=1024 and S=4096 (global)
#
# Two public configurations were measured against the upstream official kernel
# (palindromic ABBA in the same batch; ledger results/h20_ep4_pure21_live_sweep.json):
#
#   CONFIG=default (ledger arm p21), S=1024: 2230-2237 us, 1.454-1.480x vs official
#                                    S=4096: 5250-5310 us, 1.159-1.163x vs official
#       WEAVE_SCATTER_OWNER=comm_only  WEAVE_SCATTER_STAGING=smem  WEAVE_ADAPT_GEMM_CONFIG=1
#       (legacy spelling of the last knob: WEAVE_ADAPT_BM)
#
#   CONFIG=bm32_64 (tuned),          S=1024: 2173.8-2182.9 us, 1.487-1.514x vs official
#                                    S=4096: 5234.5-5276.1 us, 1.164-1.166x vs official
#       WEAVE_SCATTER_OWNER=comm_only  WEAVE_SCATTER_STAGING=smem
#       WEAVE_GEMM1_CONFIG=32          WEAVE_GEMM2_CONFIG=64
#       (legacy spellings: WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2)
#     (four public knobs are required; bm32_64 runs as a two-kernel split
#      launch, so the internal WEAVE_REDUCE_WARPS pin has no effect there, and
#      the pinned GEMM1/GEMM2 split wins over WEAVE_ADAPT_GEMM_CONFIG)
#
# Select with CONFIG=default|bm32_64 (default: default).  Every other public
# WEAVE_* knob is unset so that it takes its default.
#
# Known property (all Weave H20 configurations): repeated launches differ by
# about 1e-8 (BF16 last-bit level) while the official kernel is bit-stable; the
# result stays within calc_diff ~3.6e-4 of the FP32 reference.  See
# docs/TUNABLE_PARAMETERS.md, section "Known limitations".
#
# Scope of this script: it fixes the configuration and launches the generic
# SM90 harness (tests/bench_h20_ep8.py -> tests/test_mega_moe_sm90.py) on 4
# ranks.  That harness has no Qwen3.8-Max scenario, so the scenario below
# (H20_FILTER) is the harness's own; the README timings came from a separate
# palindromic ABBA driver that is not part of this repository and from the
# pre-integration source-branch kernel revision (the integrated revision has
# only compile-level and CPU-gate verification).  Use this script for the
# environment of the measured winners, not to reproduce the table digit for digit.
# -----------------------------------------------------------------------------
set -euo pipefail

WEAVE_UNIFIED_ROOT="${WEAVE_UNIFIED_ROOT:-$HOME/work/weave_unified_cpp}"
PYTHON="${PYTHON:-python3}"
H20_FILTER="${H20_FILTER:-L3.h200_pro_m128}"
CONFIG="${CONFIG:-default}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "$WEAVE_UNIFIED_ROOT" ]] || fail "WEAVE_UNIFIED_ROOT is not a directory: $WEAVE_UNIFIED_ROOT"
[[ -f "$WEAVE_UNIFIED_ROOT/tests/bench_h20_ep8.py" ]] || fail "missing $WEAVE_UNIFIED_ROOT/tests/bench_h20_ep8.py"
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

export WEAVE_SCATTER_OWNER=comm_only
export WEAVE_SCATTER_STAGING=smem
case "$CONFIG" in
    default) export WEAVE_ADAPT_GEMM_CONFIG=1 ;;
    bm32_64) export WEAVE_GEMM1_CONFIG=32
             export WEAVE_GEMM2_CONFIG=64 ;;
    *) fail "CONFIG must be default or bm32_64, got: $CONFIG" ;;
esac

echo "==============================================================================="
echo " NVIDIA H20 / Qwen3.8-Max / 4x EP4 / measured winner CONFIG=$CONFIG"
echo " WEAVE_SCATTER_OWNER=$WEAVE_SCATTER_OWNER WEAVE_SCATTER_STAGING=$WEAVE_SCATTER_STAGING"
echo " WEAVE_ADAPT_GEMM_CONFIG=${WEAVE_ADAPT_GEMM_CONFIG:-<default>} WEAVE_GEMM1_CONFIG=${WEAVE_GEMM1_CONFIG:-<default>} WEAVE_GEMM2_CONFIG=${WEAVE_GEMM2_CONFIG:-<default>}"
echo " (all other public knobs: default)"
echo "==============================================================================="

EXTRA=()
if [[ "${WEAVE_ALLOW_OTHER_GPU:-0}" == "1" ]]; then
    EXTRA+=(--allow-other-gpu)
fi

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" tests/bench_h20_ep8.py \
    --num-processes 4 \
    --filter "$H20_FILTER" \
    "${EXTRA[@]}"
