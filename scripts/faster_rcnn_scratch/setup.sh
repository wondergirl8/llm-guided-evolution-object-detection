#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache-faster-rcnn"
[[ -x .venv-faster-rcnn/bin/python ]] || uv venv --python 3.10 .venv-faster-rcnn
uv pip install --python .venv-faster-rcnn/bin/python -r sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/requirements.lock --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html --only-binary mmcv
bash scripts/faster_rcnn/build_mmcv.sh
.venv-faster-rcnn/bin/python -c 'from mmcv.ops import nms,roi_align; import mmdet'
