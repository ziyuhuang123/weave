#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MI308X (gfx942, 80 CUs) / Qwen3.8-Max / EP8 / S=1024
#
# Exact tuned build + run preset for the current unified EP8 source tree.
#
# 3-axis summary
#   Spatial        : 80 CUs, 8 waves/CU; WEAVE_BM_TIER_MODE=2 (M tiers
#                    16/32/64/128/256, thresholds WEAVE_TH_BM64=16 and
#                    WEAVE_TH_BM128=96); fixed BN=128 and BK=128.
#   Temporal       : WEAVE_SCATTER_MODE=2, WEAVE_OVERLAP_PIPELINE=1,
#                    WEAVE_DUALGROUP_MIN_TOKENS=64. For S=1024 the rank-local
#                    token count is 1024/8 = 128 >= 64, so the 8-wave
#                    dual-compute (4+4) path is active.
#   Implementation : WEAVE_ABLATION_PRESET=0, WEAVE_SELECTIVE_DEDUP=1,
#                    WEAVE_ALL_WAVE_COMBINE=1, WEAVE_BN2_MODE=0,
#                    WEAVE_S2_NGROUP_MODE=0, WEAVE_PARALLEL_ROUTE_PLAN=1,
#                    WEAVE_PROFILER_LEVEL=2 for a trace / 0 for production.
#
# Out-of-tree build
#   The unified build helper writes compiler_work/ and its output below its own
#   source directory. To leave the source checkout untouched, this script
#   copies deep_gemm/include into a fresh RUN_ROOT overlay and runs the helper
#   from that copy. All temporary files, caches, build products and results
#   stay under RUN_ROOT.
#
# Evidence boundary
#   * The archived MI308X EP8 timeline metadata records gridX=80,
#     waves_per_cu=8, 80 unique physical CUs, S=1024 and rank=2. It does NOT
#     carry the full compile-time knob string.
#   * The archived EP8 timeline was produced by the archived EP8 source tree.
#     This script rebuilds the current unified EP8 kernel with the same tuned
#     preset; it does not claim a byte-identical binary or trace.
#   * The benchmark driver needs three archived host/runtime artifacts that are
#     not vendored in the public repository:
#       WEAVE_EP8_LAUNCHER  frozen host launcher .so
#       WEAVE_EP8_IPC_LIB   IPC driver .so
#       WEAVE_EP8_ROUTES    route-table directory containing the qwen38max table
#     The script fails fast if any of them is missing.
#
# Usage
#   WEAVE_EP8_LAUNCHER=/abs/frozen_host_launcher.so \
#   WEAVE_EP8_IPC_LIB=/abs/libc406_ipc_driver.so \
#   WEAVE_EP8_ROUTES=/abs/routes \
#   bash examples/mi308x_qwen38max_ep8_s1024.sh
#
#   PROFILE_LEVEL=2 ... bash examples/mi308x_qwen38max_ep8_s1024.sh
#
# Environment
#   WEAVE_UNIFIED_ROOT     path to the unified repository checkout
#   WEAVE_RUN_ROOT         writable run root (default: $HOME/weave_example_runs/...)
#   PROFILE_LEVEL          0 = production (default), 2 = instrumented build
#   WEAVE_AITER_SITE       aiter_meta install tree
#   WEAVE_PY_SITE          Python site-packages containing torch/pybind11
#   WEAVE_CLANG/WEAVE_LLD  ROCm LLVM tools
#   WEAVE_PY               Python used by the build helper
#   WEAVE_TORCHRUN         torchrun executable (default: torchrun)
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

[[ -d "$WEAVE_AITER_SITE" ]] || fail "WEAVE_AITER_SITE is not a directory: $WEAVE_AITER_SITE"
[[ -d "$WEAVE_PY_SITE" ]] || fail "WEAVE_PY_SITE is not a directory: $WEAVE_PY_SITE"
[[ -x "$WEAVE_CLANG" ]] || fail "WEAVE_CLANG is not executable: $WEAVE_CLANG"
[[ -x "$WEAVE_LLD" ]] || fail "WEAVE_LLD is not executable: $WEAVE_LLD"
[[ -x "$WEAVE_PY" ]] || fail "WEAVE_PY is not executable: $WEAVE_PY"
command -v "$WEAVE_TORCHRUN" >/dev/null 2>&1 || fail "torchrun not found: $WEAVE_TORCHRUN"

: "${WEAVE_EP8_LAUNCHER:?Set WEAVE_EP8_LAUNCHER to the archived frozen host launcher .so}"
: "${WEAVE_EP8_IPC_LIB:?Set WEAVE_EP8_IPC_LIB to the archived IPC driver .so}"
: "${WEAVE_EP8_ROUTES:?Set WEAVE_EP8_ROUTES to the route-table directory}"
[[ -f "$WEAVE_EP8_LAUNCHER" ]] || fail "WEAVE_EP8_LAUNCHER not found: $WEAVE_EP8_LAUNCHER"
[[ -f "$WEAVE_EP8_IPC_LIB" ]] || fail "WEAVE_EP8_IPC_LIB not found: $WEAVE_EP8_IPC_LIB"
[[ -d "$WEAVE_EP8_ROUTES" ]] || fail "WEAVE_EP8_ROUTES not found: $WEAVE_EP8_ROUTES"
[[ -f "$WEAVE_EP8_ROUTES/qwen38max_S262144.pt" ]] || fail "missing route table: $WEAVE_EP8_ROUTES/qwen38max_S262144.pt"

# ---- fresh out-of-tree overlay ---------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERLAY="$RUN_ROOT/overlay-$STAMP"
mkdir -p "$OVERLAY/deep_gemm" "$RUN_ROOT/tmp" "$RUN_ROOT/cache/hip" "$RUN_ROOT/cache/comgr" "$RUN_ROOT/results"
cp -a "$SRC_ROOT/deep_gemm/include" "$OVERLAY/deep_gemm/include"

EP8_DIR="$OVERLAY/deep_gemm/include/deep_gemm/impls/amd/ep8"

export TMPDIR="$RUN_ROOT/tmp" TMP="$RUN_ROOT/tmp" TEMP="$RUN_ROOT/tmp"
export HIP_CACHE_DIR="$RUN_ROOT/cache/hip" AMD_COMGR_CACHE_DIR="$RUN_ROOT/cache/comgr"
export PYTHONDONTWRITEBYTECODE=1

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
export WEAVE_CO_PROFILE="$PROFILE_LEVEL"

echo "==============================================================================="
echo " MI308X / Qwen3.8-Max / EP8 / S=1024"
echo " source                 : $SRC_ROOT"
echo " overlay                : $OVERLAY"
echo " profile level          : $PROFILE_LEVEL (0 production, 2 trace)"
echo " spatial                : BM tiers mode 2 (16/32/64/128/256), thresholds 16/96, BN=128 BK=128"
echo " temporal               : scatter_mode=2 overlap=1 dualgroup_min=64 (local tokens 128 -> 4+4)"
echo " implementation         : preset=0 selective_dedup=1 all_wave_combine=1 bn2=0 s2_group=0 parallel_route=1"
echo "==============================================================================="

echo "NOTE: if the GPUs are busy with another process, stop and wait; never kill it."
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true
fi

echo "[build] EP8 helper from the overlay copy"
bash "$EP8_DIR/build_one.sh" "$BUILD_NAME" "$PROFILE_LEVEL" 16 96 "-DWEAVE_ABLATION_PRESET=0" 0

CO="$EP8_DIR/$BUILD_NAME/P.co"
[[ -f "$CO" ]] || fail "build did not produce $CO"
echo "[build] P.co: $CO ($(stat -c %s "$CO") bytes)"
sha256sum "$CO"

echo "[run] EP8 benchmark (qwen38max, S=1024, 8 ranks)"
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

echo "[done] preset run finished. Results: $RUN_ROOT/results"
echo "[note] --save-trace writes the harness probe trace, not a paired SW+HW"
echo "       Perfetto timeline. The annotated preview uses the archived trace."
