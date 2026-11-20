#!/usr/bin/env bash
# Reproducible installation helper for the official baselines used by Weave.
#
# The script performs source installs only; it never edits the Weave tree.
# Set BASELINE_ROOT to a persistent directory with enough space.
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-$HOME/weave_baselines}"
PYTHON="${PYTHON:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
JOBS="${JOBS:-8}"

mkdir -p "$BASELINE_ROOT/src" "$BASELINE_ROOT/wheels" "$BASELINE_ROOT/logs"

usage() {
  cat <<'EOF'
Usage: install_baselines.sh [nvidia|amd|all]

  nvidia  DeepGEMM, DeepEP, FlashInfer, SGLang and vLLM (CUDA hosts)
  amd     AITER and ROCm builds of SGLang / vLLM (gfx942 hosts)
  all     Install both stacks (normally on separate hosts)

Environment:
  BASELINE_ROOT  Persistent install root (default: $HOME/weave_baselines)
  PYTHON         Python executable (default: python3)
  CUDA_HOME      CUDA toolkit root
  ROCM_HOME      ROCm root
  JOBS           Parallel build jobs
EOF
}

install_nvidia() {
  command -v "$PYTHON" >/dev/null
  "$PYTHON" -m pip install --upgrade pip setuptools wheel ninja cmake
  test -d "$CUDA_HOME" || { echo "CUDA_HOME not found: $CUDA_HOME" >&2; exit 2; }

  cd "$BASELINE_ROOT/src"
  test -d DeepGEMM || git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git
  test -d DeepEP || git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
  test -d flashinfer || git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git

  "$PYTHON" -m pip install -e ./DeepGEMM
  (cd DeepEP && MAX_JOBS="$JOBS" "$PYTHON" -m pip install -e .)
  (cd flashinfer && MAX_JOBS="$JOBS" "$PYTHON" -m pip install -e . --no-build-isolation)

  # Inference-engine baselines. Pin wheels to the CUDA version used by the host.
  "$PYTHON" -m pip install "sglang[all]"
  "$PYTHON" -m pip install "vllm"
}

install_amd() {
  command -v "$PYTHON" >/dev/null
  test -d "$ROCM_HOME" || { echo "ROCM_HOME not found: $ROCM_HOME" >&2; exit 2; }
  "$PYTHON" -m pip install --upgrade pip setuptools wheel ninja cmake

  cd "$BASELINE_ROOT/src"
  test -d aiter || git clone --depth 1 https://github.com/ROCm/aiter.git
  (cd aiter && MAX_JOBS="$JOBS" "$PYTHON" -m pip install -e .)

  # The ROCm wheels select the local HIP runtime through the installed torch.
  "$PYTHON" -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/rocm6.4
  "$PYTHON" -m pip install "sglang[all]"
  "$PYTHON" -m pip install "vllm"
}

case "${1:-all}" in
  nvidia) install_nvidia ;;
  amd) install_amd ;;
  all) install_nvidia; install_amd ;;
  -h|--help) usage ;;
  *) usage; exit 2 ;;
esac

"$PYTHON" - <<'PY'
import importlib.util
mods = ["torch", "deep_gemm", "sglang", "vllm"]
for name in mods:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY
