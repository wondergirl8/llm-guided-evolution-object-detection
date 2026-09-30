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

MAX_CANDIDATES=${FRED_BRINGUP_MAX_CANDIDATES:-8}
[[ "$MAX_CANDIDATES" =~ ^[1-9][0-9]*$ ]] || {
  echo "FRED_BRINGUP_MAX_CANDIDATES must be a positive integer" >&2
  exit 1
}
REPORT_DIR="data/fred_phase0/validation/bringup_${SLURM_JOB_ID}"
mkdir -p "$REPORT_DIR"
echo "Sequence inspection reports: $REPORT_DIR"

select_valid_sequence() {
  local split=$1 sequence report log
  shift
  for sequence in "$@"; do
    report="$REPORT_DIR/${split}_${sequence}.json"
    log="$REPORT_DIR/${split}_${sequence}.log"
    echo "Inspecting $split sequence $sequence"
    if "$PY" -m sota.FRED_LLM_GE.phase0_data inspect-sequence "$sequence" \
      --inventory "$INVENTORY" --official-split "$OFFICIAL" \
      --project-split "$PROJECT" --output "$report" >"$log" 2>&1; then
      if "$PY" - "$report" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
sys.exit(0 if report['sequences'][0]['annotation_count'] > 0 else 1)
PY
      then
        SELECTED_SEQUENCE=$sequence
        echo "Selected $split sequence $sequence (validation passed, annotations present)"
        return 0
      fi
      echo "Rejected: no annotations; report: $report"
      continue
    fi
    if [ ! -s "$report" ]; then
      cat "$log" >&2
      echo "Inspection did not produce a report; stopping" >&2
      return 1
    fi
    "$PY" - "$report" <<'PY'
import json, sys
from collections import Counter
report = json.load(open(sys.argv[1]))
counts = Counter(f['code'] for f in report['findings'] if f['severity'] == 'error')
print(f"Rejected: {dict(counts)}; report: {sys.argv[1]}")
PY
  done
  echo "No valid $split sequence among the first $# candidates" >&2
  return 1
}

mapfile -t TRAIN_CANDIDATES < <("$PY" - "$PROJECT" train "$MAX_CANDIDATES" <<'PY'
import json, sys
split = json.load(open(sys.argv[1]))
print(*split[sys.argv[2]][:int(sys.argv[3])], sep='\n')
PY
)
mapfile -t VAL_CANDIDATES < <("$PY" - "$PROJECT" validation "$MAX_CANDIDATES" <<'PY'
import json, sys
split = json.load(open(sys.argv[1]))
print(*split[sys.argv[2]][:int(sys.argv[3])], sep='\n')
PY
)
select_valid_sequence train "${TRAIN_CANDIDATES[@]}"
TRAIN_SEQ=$SELECTED_SEQUENCE
select_valid_sequence validation "${VAL_CANDIDATES[@]}"
VAL_SEQ=$SELECTED_SEQUENCE
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
