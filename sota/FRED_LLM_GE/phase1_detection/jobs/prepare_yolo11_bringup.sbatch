#!/bin/bash
#SBATCH --job-name=fred-yolo11-data
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-yolo11-data-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load uv
export UV_CACHE_DIR="$PWD/data/.uv-cache"
PY=data/.venv-yolo11/bin/python
test -x "$PY" || { echo "Run the FRED source inventory job first" >&2; exit 1; }
uv pip install --python "$PY" \
  'huggingface-hub==0.35.1' PyYAML 'Pillow>=10' certifi 'ultralytics>=8.3,<9'

INVENTORY=data/fred_phase0/metadata/source_inventory.json
OFFICIAL=data/fred_phase0/metadata/official_challenging_split.json
PROJECT=sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json
MANIFEST=data/fred_phase0/manifests/yolo11_bringup.sqlite
EXPORT=data/fred_yolo11_bringup
test -f "$INVENTORY" && test -f "$OFFICIAL" && test -f "$PROJECT"
test ! -e "$MANIFEST" && test ! -e "$EXPORT" || {
  echo "Bring-up output already exists; inspect it before rerunning" >&2
  exit 1
}

read -r TRAIN_SEQ VAL_SEQ < <("$PY" -c \
  'import json; s=json.load(open("sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json")); print(s["train"][0], s["validation"][0])')
echo "Preparing FRED train sequence $TRAIN_SEQ and validation sequence $VAL_SEQ"
"$PY" -m sota.FRED_LLM_GE.phase0_data build-manifest \
  --inventory "$INVENTORY" \
  --official-split "$OFFICIAL" \
  --project-split "$PROJECT" \
  --sequence "$TRAIN_SEQ" --sequence "$VAL_SEQ" \
  --output "$MANIFEST"
"$PY" -m sota.FRED_LLM_GE.phase1_detection.adapters.yolo11 \
  --manifest "$MANIFEST" --inventory "$INVENTORY" \
  --output "$EXPORT" --train-limit 32 --val-limit 32

"$PY" -c 'from ultralytics import YOLO; YOLO("yolo11m.pt")'
test -s "$EXPORT/data.yaml" && test -s yolo11m.pt
echo "FRED YOLO11 bounded data export and checkpoint ready"
