#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MI308X (gfx942) / Qwen3.8-Max / EP16 / S=128  (2 nodes x 8 ranks)
#
# Exact tuned build preset for the current unified EP16 source tree and the
# mode-2 overlap runner.
#
# 3-axis summary
#   Spatial        : 80 CUs per GPU, 512-thread 4+4 static overlap kernel;
#                    WEAVE_BM_TIER_MODE=2 (M tiers 16/32/64/128/256,
#                    thresholds WEAVE_TH_BM64=16, WEAVE_TH_BM128=96);
#                    fixed BN=128, BK=128.
#   Temporal       : WEAVE_SCATTER_MODE=2 (node-partial pre-reduced scatter),
#                    WEAVE_OVERLAP_PIPELINE=1 (mode-2 overlap),
#                    WEAVE_DUALGROUP_MIN_TOKENS=64. For S=128 the rank-local
#                    token count is 128/16 = 8 < 64, so the dual-group
#                    threshold is NOT reached; EP16 stays on the 4+4 static
#                    overlap path.
#   Implementation : WEAVE_ABLATION_PRESET=0, WEAVE_SELECTIVE_DEDUP=1,
#                    WEAVE_ALL_WAVE_COMBINE=1, WEAVE_BN2_MODE=0,
#                    WEAVE_S2_NGROUP_MODE=0, WEAVE_PARALLEL_ROUTE_PLAN=1,
#                    WEAVE_PROFILER_LEVEL=0 production / 2 profiler,
#                    WEAVE_TURBO_MODES=2 at run time.
#
# Out-of-tree build
#   The EP16 build helper writes compiler_work/ and build/ below its own source
#   directory. This script copies deep_gemm/include into a fresh RUN_ROOT
#   overlay and runs the helper from that copy. The source checkout is not
#   modified.
#
# External artifacts (not vendored in the public repository)
#   WEAVE_EP16_PKG        package root containing baseline/ep_common.py,
#                         baseline/ep_harness.py, baseline/ep_reference.py,
#                         baseline/paper_step0.py, ep16/, and
#                         ep16_turbo_runner.py
#   WEAVE_EP16_HOST_LIBS  directory containing
#                         libep16_turbo_host_mode2.so and gda_turbo_mode2.so
#   WEAVE_EP16_ROUTE_MANIFEST  route manifest for qwen38max
#   WEAVE_EP16_WEIGHT_CACHE    weight cache for qwen38max
#   NODE_RANK, MASTER_ADDR, MASTER_PORT  two-node rendezvous
#
# The public tree contains the kernel and the Python reference/ABI modules, but
# not the full host .so / GDA .so build chain. The script fails fast if any
# required artifact is missing.
#
# Correctness contract (archived verified run)
#   Qwen3.8-Max EP16 S=128: 16 ranks, t_local=8, local_experts=32,
#   1,048,576 elements checked across ranks, FP64 reference on every rank,
#   rel_l2 max 0.01253 <= 0.02, max row rel_l2 0.01961 <= 0.05, PASS.
#
# Usage
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   WEAVE_EP16_PKG=/abs/pkg \
#   WEAVE_EP16_HOST_LIBS=/abs/live_arms \
#   WEAVE_EP16_ROUTE_MANIFEST=/abs/route_manifest.json \
#   WEAVE_EP16_WEIGHT_CACHE=/abs/weights.pt \
#   bash examples/mi308x_qwen38max_ep16_s128.sh
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
[[ -n "$RUN_ROOT" && "$RUN_ROOT" != "/" ]] || fail "WEAVE_RUN_ROOT must be a non-root, non-empty path"
[[ "$PROFILE_LEVEL" == "0" || "$PROFILE_LEVEL" == "2" ]] || fail "PROFILE_LEVEL must be 0 or 2"

[[ -d "$WEAVE_AITER_SITE" ]] || fail "WEAVE_AITER_SITE is not a directory: $WEAVE_AITER_SITE"
[[ -d "$WEAVE_PY_SITE" ]] || fail "WEAVE_PY_SITE is not a directory: $WEAVE_PY_SITE"
[[ -x "$WEAVE_CLANG" ]] || fail "WEAVE_CLANG is not executable: $WEAVE_CLANG"
[[ -x "$WEAVE_LLD" ]] || fail "WEAVE_LLD is not executable: $WEAVE_LLD"
[[ -x "$WEAVE_PY" ]] || fail "WEAVE_PY is not executable: $WEAVE_PY"
command -v "$WEAVE_TORCHRUN" >/dev/null 2>&1 || fail "torchrun not found: $WEAVE_TORCHRUN"

: "${WEAVE_EP16_PKG:?Set WEAVE_EP16_PKG to the archived package root}"
: "${WEAVE_EP16_HOST_LIBS:?Set WEAVE_EP16_HOST_LIBS to the directory with the two mode-2 host .so files}"
: "${WEAVE_EP16_ROUTE_MANIFEST:?Set WEAVE_EP16_ROUTE_MANIFEST to the qwen38max route manifest}"
: "${WEAVE_EP16_WEIGHT_CACHE:?Set WEAVE_EP16_WEIGHT_CACHE to the qwen38max weight cache}"
: "${NODE_RANK:?Set NODE_RANK (0 or 1) for the two-node launch}"
: "${MASTER_ADDR:?Set MASTER_ADDR for the two-node launch}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -f "$WEAVE_EP16_PKG/baseline/ep_common.py" ]]; then
    EP16_PY_BASE="$WEAVE_EP16_PKG/baseline"
elif [[ -f "$WEAVE_EP16_PKG/ep_common.py" ]]; then
    EP16_PY_BASE="$WEAVE_EP16_PKG"
else
    fail "missing ep_common.py under $WEAVE_EP16_PKG/baseline or $WEAVE_EP16_PKG"
fi
[[ -f "$WEAVE_EP16_PKG/ep16/mega_host.py" ]] || fail "missing $WEAVE_EP16_PKG/ep16/mega_host.py"
[[ -f "$WEAVE_EP16_PKG/ep16_turbo_runner.py" ]] || fail "missing $WEAVE_EP16_PKG/ep16_turbo_runner.py"
[[ -f "$WEAVE_EP16_HOST_LIBS/libep16_turbo_host_mode2.so" ]] || fail "missing libep16_turbo_host_mode2.so in WEAVE_EP16_HOST_LIBS"
[[ -f "$WEAVE_EP16_HOST_LIBS/gda_turbo_mode2.so" ]] || fail "missing gda_turbo_mode2.so in WEAVE_EP16_HOST_LIBS"
[[ -f "$WEAVE_EP16_ROUTE_MANIFEST" ]] || fail "route manifest not found: $WEAVE_EP16_ROUTE_MANIFEST"
[[ -f "$WEAVE_EP16_WEIGHT_CACHE" ]] || fail "weight cache not found: $WEAVE_EP16_WEIGHT_CACHE"

# ---- fresh out-of-tree overlay ---------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERLAY="$RUN_ROOT/overlay-$STAMP"
mkdir -p "$OVERLAY/deep_gemm" "$RUN_ROOT/tmp" "$RUN_ROOT/cache/hip" "$RUN_ROOT/cache/comgr" "$RUN_ROOT/results" "$RUN_ROOT/live_arms"
cp -a "$SRC_ROOT/deep_gemm/include" "$OVERLAY/deep_gemm/include"

EP16_DIR="$OVERLAY/deep_gemm/include/deep_gemm/impls/amd/ep16"

export TMPDIR="$RUN_ROOT/tmp" TMP="$RUN_ROOT/tmp" TEMP="$RUN_ROOT/tmp"
export HIP_CACHE_DIR="$RUN_ROOT/cache/hip" AMD_COMGR_CACHE_DIR="$RUN_ROOT/cache/comgr"
export PYTHONDONTWRITEBYTECODE=1
unset BUILD_ATTEMPT || true

# ---- exact tuned build flags ------------------------------------------------
export WEAVE_ABLATION_PRESET=0
export WEAVE_SCATTER_MODE=2
export WEAVE_BM_TIER_MODE=2
export WEAVE_SELECTIVE_DEDUP=1
export WEAVE_DUALGROUP_MIN_TOKENS=64
export WEAVE_OVERLAP_PIPELINE=1
export WEAVE_ALL_WAVE_COMBINE=1
export WEAVE_BN2_MODE=0
export WEAVE_S2_NGROUP_MODE=0
export WEAVE_PARALLEL_ROUTE_PLAN=1
export WEAVE_TH_BM64=16
export WEAVE_TH_BM128=96
export WEAVE_PROFILER_LEVEL="$PROFILE_LEVEL"

echo "==============================================================================="
echo " MI308X / Qwen3.8-Max / EP16 / S=128 (2 nodes x 8 ranks)"
echo " source                 : $SRC_ROOT"
echo " overlay                : $OVERLAY"
echo " profile level          : $PROFILE_LEVEL (0 production, 2 profiler)"
echo " spatial                : BM tiers mode 2 (16/32/64/128/256), thresholds 16/96, BN=128 BK=128"
echo " temporal               : scatter_mode=2 overlap=1 dualgroup_min=64 (local tokens 8 -> threshold NOT reached)"
echo " implementation         : preset=0 selective_dedup=1 all_wave_combine=1 bn2=0 s2_group=0 parallel_route=1"
echo " runtime                : WEAVE_TURBO_MODES=2"
echo "==============================================================================="

echo "NOTE: if the GPUs are busy with another process, stop and wait; never kill it."
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true
fi

echo "[build] EP16 helper from the overlay copy"
"$WEAVE_PY" "$EP16_DIR/build_port.py"

BUILD_DIR="$EP16_DIR/build/fi0_mode2_p0_m2"
CO="$BUILD_DIR/P.co"
[[ -f "$CO" ]] || fail "build did not produce $CO"
ARCHIVED_CO="$WEAVE_EP16_HOST_LIBS/P_ep16_turbo_overlap_mode2.co"
if [[ "${WEAVE_EP16_USE_ARCHIVED_CO:-0}" == "1" ]]; then
    [[ -f "$ARCHIVED_CO" ]] || fail "WEAVE_EP16_USE_ARCHIVED_CO=1 but $ARCHIVED_CO is missing"
    cp -f "$ARCHIVED_CO" "$RUN_ROOT/live_arms/P_ep16_turbo_overlap_mode2.co"
    echo "[build] using archived .co (WEAVE_EP16_USE_ARCHIVED_CO=1)"
else
    cp -f "$CO" "$RUN_ROOT/live_arms/P_ep16_turbo_overlap_mode2.co"
    echo "[build] using fresh .co; the host .so must match the current ABI"
fi
cp -f "$WEAVE_EP16_HOST_LIBS/libep16_turbo_host_mode2.so" "$RUN_ROOT/live_arms/"
cp -f "$WEAVE_EP16_HOST_LIBS/gda_turbo_mode2.so" "$RUN_ROOT/live_arms/"
echo "[build] fresh P.co: $CO ($(stat -c %s "$CO") bytes)"
sha256sum "$CO"

export WEAVE_TURBO_MODES=2
export PYTHONPATH="$EP16_PY_BASE:$WEAVE_EP16_PKG:${PYTHONPATH:-}"

echo "[run] EP16 mode-2 runner (qwen38max, S=128, 2 nodes x 8 ranks)"
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

echo "[done] preset run finished. Results: $RUN_ROOT/results"
echo "[note] no 3-panel EP16 preview is produced; the matched preview set is"
echo "       the three EP8 S=1024 timelines."
