#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# H20 (SM90, 78 SMs) / Qwen3.8-Max / EP8 / S=1024
#
# Exact tuned preset. The string below is copied verbatim from the archived
# trace metadata field otherData.meta.selected_cfg. It is passed through the
# public unified entry WEAVE_H20_CONFIG.
#
# 3-axis summary
#   Spatial        : bm=32, bn=256, bk=128, wgs=4, swap=1; 78 SMs;
#                    4 epilogue warp groups.
#   Temporal       : interleave=5, dyn=1, stages=0 (runtime
#                    auto-selected pipeline depth), swap_pipe=1.
#   Implementation : dispatch dedup on, dedup tier on, scatter dedup OFF,
#                    register combine on, idle-warp combine on, fast pull on.
#
# Unified-name mapping of every active key
#   interleave=5      -> WEAVE_INTERLEAVE_L1_L2=5
#   dyn=1             -> WEAVE_DYN_SCHED=1
#   scat_disp=1       -> WEAVE_SCATTER_ROLE_DISP=1
#   d_mask_mode=1     -> WEAVE_HETERO_MASK_MODE=1
#   bk=128            -> WEAVE_BLOCK_K=128
#   swap_pipe=1       -> WEAVE_SWAP_PIPE=1
#   fast_pull=1       -> WEAVE_FAST_PULL=1
#   dedup_disp=1      -> WEAVE_DEDUP_DISPATCH=1
#   reg_comb=1        -> WEAVE_REG_COMBINE=1
#   all_idle_comb=1   -> WEAVE_ALL_IDLE_COMBINE=1
#   fixed spec keys without a standalone public env binding:
#     fast_host, fused, epw, compact, alias, bm, bn, wgs, swap, stages,
#     fast_swap_epi, rank_cache, fast_clean, dedup_tier, fast_comb.
#   They are reproduced exactly by the WEAVE_H20_CONFIG string itself.
#   stages=0 means "runtime auto-selects the pipeline depth"; it does NOT mean
#   a zero-stage pipeline.
#   The archived hot/cold ordering key is internal to the H20 auto regime and is
#   no longer a public knob.
#   The trace has dedup_disp=1 and dedup_tier=1 but no dedup_scat key, so
#   scatter dedup is OFF (the WEAVE_DEDUP_SCATTER default is 0).
#
# Usage
#   bash examples/h20_qwen38max_ep8_s1024.sh
#   H20_FILTER=L3.h200_pro_m128 bash examples/h20_qwen38max_ep8_s1024.sh
#   WEAVE_ALLOW_OTHER_GPU=1 bash examples/h20_qwen38max_ep8_s1024.sh
#
# Environment
#   WEAVE_UNIFIED_ROOT       path to the unified repository checkout
#   PYTHON                   Python interpreter (default: python3)
#   H20_FILTER               scenario filter (default: L3.h200_pro_m128)
#   WEAVE_ALLOW_OTHER_GPU    1 = pass --allow-other-gpu to the harness
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

# Exact selected_cfg from the archived trace metadata. Do not add dedup_scat=1.
export WEAVE_H20_CONFIG='fast_host=1,fused=1,interleave=5,epw=8,compact=1,dyn=1,alias=1,scat_disp=1,d_mask_mode=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,swap_pipe=1,fast_swap_epi=1,rank_cache=1,fast_clean=1,fast_pull=1,dedup_disp=1,dedup_tier=1,reg_comb=1,fast_comb=1,all_idle_comb=1'

echo "==============================================================================="
echo " H20 / Qwen3.8-Max / EP8 / S=1024"
echo " exact WEAVE_H20_CONFIG:"
echo "   $WEAVE_H20_CONFIG"
echo "------------------------------------------------------------------------------"
echo " WEAVE_INTERLEAVE_L1_L2    = 5   (interleave)"
echo " WEAVE_DYN_SCHED           = 1   (dyn)"
echo " WEAVE_SCATTER_ROLE_DISP   = 1   (scat_disp)"
echo " WEAVE_HETERO_MASK_MODE    = 1   (d_mask_mode)"
echo " WEAVE_BLOCK_K             = 128 (bk)"
echo " WEAVE_SWAP_PIPE           = 1   (swap_pipe)"
echo " WEAVE_FAST_PULL           = 1   (fast_pull)"
echo " WEAVE_DEDUP_DISPATCH      = 1   (dedup_disp)"
echo " WEAVE_REG_COMBINE         = 1   (reg_comb)"
echo " WEAVE_ALL_IDLE_COMBINE    = 1   (all_idle_comb)"
echo " WEAVE_DEDUP_SCATTER       = 0   (dedup_scat absent)"
echo " stages=0                  -> runtime auto pipeline depth"
echo "==============================================================================="

echo "NOTE: if the GPUs are busy with another process, stop and wait; never kill it."

EXTRA=()
if [[ "${WEAVE_ALLOW_OTHER_GPU:-0}" == "1" ]]; then
    EXTRA+=(--allow-other-gpu)
fi

cd "$WEAVE_UNIFIED_ROOT"
export PYTHONDONTWRITEBYTECODE=1

echo "[run] H20 correctness harness with filter: $H20_FILTER"
"$PYTHON" tests/bench_h20_ep8.py \
    --num-processes 8 \
    --filter "$H20_FILTER" \
    "${EXTRA[@]}"

echo "[done] preset run finished."
echo "[trace] the exact Qwen3.8-Max S=1024 timeline is archived; replay it with:"
echo "        python examples/profile_and_render_timeline.py --preset h20 --render"
