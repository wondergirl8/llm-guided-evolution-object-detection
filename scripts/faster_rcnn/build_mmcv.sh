#!/usr/bin/env bash
# Build kernels for H200 (sm_90) using the toolkit matching PyTorch's CUDA ABI.
set -euo pipefail
cd "$(dirname "$0")/../.."
module load cuda/12.1.1
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4
export PATH="$PWD/.venv-faster-rcnn/bin:$PATH"
export PIP_CACHE_DIR="$PWD/.cache-faster-rcnn/pip"
WHEELS="$PWD/.cache-faster-rcnn/h200-wheels"
mkdir -p "$WHEELS"
if [[ ! -f "$WHEELS/build.sha256" ]]; then
    .venv-faster-rcnn/bin/python -m pip wheel mmcv==2.1.0 \
        --no-deps --no-build-isolation --no-binary mmcv --wheel-dir "$WHEELS"
    (cd "$WHEELS" && sha256sum mmcv-2.1.0-cp310-cp310-linux_x86_64.whl > build.sha256)
fi
(cd "$WHEELS" && sha256sum --check build.sha256)
UV_CACHE_DIR="$PWD/.cache-faster-rcnn" uv pip install \
    --python .venv-faster-rcnn/bin/python --reinstall --no-deps \
    "$WHEELS"/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl
