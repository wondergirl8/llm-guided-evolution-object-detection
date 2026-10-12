#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache-faster-rcnn"
if [[ ! -x .venv-faster-rcnn/bin/python ]]; then
    uv venv --python 3.10 .venv-faster-rcnn
fi
.venv-faster-rcnn/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Expected isolated Python 3.10 environment"'
DEPENDENCIES=sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/requirements.lock
uv pip install --python .venv-faster-rcnn/bin/python \
    -r "$DEPENDENCIES" \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    --only-binary mmcv
bash scripts/faster_rcnn/build_mmcv.sh
.venv-faster-rcnn/bin/python scripts/faster_rcnn/setup_checkpoint.py
.venv-faster-rcnn/bin/python -c 'import torch; from mmcv.ops import nms, roi_align; import mmdet; print(torch.__version__, mmdet.__version__)'
