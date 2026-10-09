#!/bin/bash
#SBATCH --job-name=fred-integration-prepare
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-integration-prepare-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load uv
export UV_CACHE_DIR="$PWD/data/.uv-cache"
PY=data/.venv-yolo11/bin/python
test -x "$PY" || uv venv --python 3.12 data/.venv-yolo11
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 environment; existing environment was preserved"'
uv pip install --python "$PY" -r sota/FRED_LLM_GE/requirements-bringup.txt

DATA=${FRED_YOLO_DATA:-data/fred_yolo11_bringup/data.yaml}
if [ ! -f "$DATA" ]; then
  test "$DATA" = data/fred_yolo11_bringup/data.yaml || {
    echo "Custom export does not exist: $DATA" >&2
    exit 1
  }
  BACKEND=sota.FRED_LLM_GE.archive.phase0_data_old
  INVENTORY=data/fred_phase0/metadata/source_inventory.json
  OFFICIAL=data/fred_phase0/metadata/official_challenging_split.json
  "$PY" -m "$BACKEND" verify-source
  test -f "$INVENTORY" || "$PY" -m "$BACKEND" inventory
  test -f "$OFFICIAL" || "$PY" -m "$BACKEND" build-official-splits --inventory "$INVENTORY"
  bash sota/FRED_LLM_GE/phase1_detection/jobs/prepare_yolo11_bringup.sbatch
else
  echo "Reusing existing bounded export: $DATA"
fi
"$PY" - "$DATA" <<'PY'
import sys
from pathlib import Path
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11.train_eval import validate_data
source = validate_data(Path(sys.argv[1]))
print("VALIDATED EXPORT:", sys.argv[1])
print("DATASET REVISION:", source["dataset_revision"])
print("MANIFEST SHA256:", source["manifest_sha256"])
print("PURPOSE:", source.get("purpose"))
PY
if [ ! -s yolo11m.pt ]; then
  "$PY" -c 'from ultralytics import YOLO; YOLO("yolo11m.pt")'
fi
test -s yolo11m.pt
uv pip check --python "$PY"
echo "PREPARATION COMPLETE; the dependent GPU check may now start"
