#!/bin/bash
#SBATCH --job-name=fred-event-recheck
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-event-recheck-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

for name in SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX; do
  [[ ${!name:-} =~ ^[0-9]+$ ]] || { echo "$name must be set by Slurm" >&2; exit 2; }
done
if (( SLURM_ARRAY_TASK_MIN != 0 ||
      SLURM_ARRAY_TASK_MAX + 1 != SLURM_ARRAY_TASK_COUNT )); then
  echo "Submit a contiguous zero-based array such as --array=0-7%1" >&2
  exit 2
fi

PY=data/.venv-yolo11/bin/python
OLD_DIR=data/fred_phase0/validation/development_audit_6cadd2db302f
OFFICIAL=data/fred_phase0/metadata/official_challenging_split.json
INVENTORY=data/fred_phase0/metadata/source_inventory.json
PROJECT=sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json
CODE_REVISION=$(git rev-parse --short=12 HEAD)
NEW_DIR="data/fred_phase0/validation/development_audit_${CODE_REVISION}"
test -x "$PY" && test -s "$OLD_DIR/summary.json" || {
  echo "The completed original audit summary and Phase 0 Python are required" >&2
  exit 1
}
test "$NEW_DIR" != "$OLD_DIR" || { echo "Recheck requires a new code revision" >&2; exit 1; }

INDEX_TEXT=$("$PY" -m sota.FRED_LLM_GE.phase0_data.development_audit \
  affected-filename-indexes --summary "$OLD_DIR/summary.json" --expected-count 100 \
  --inventory "$INVENTORY" --official "$OFFICIAL" --project "$PROJECT")
mapfile -t INDEXES <<< "$INDEX_TEXT"
echo "Rechecking ${#INDEXES[@]} affected sequences under code $CODE_REVISION"
echo "New reports: $NEW_DIR; original reports retained: $OLD_DIR"
for ((position = SLURM_ARRAY_TASK_ID; position < ${#INDEXES[@]}; position += SLURM_ARRAY_TASK_COUNT)); do
  test "$(git rev-parse --short=12 HEAD)" = "$CODE_REVISION" || {
    echo "Checkout revision changed during the recheck; stopping" >&2
    exit 1
  }
  index=${INDEXES[$position]}
  echo "===== Recheck official development index $index ====="
  FRED_AUDIT_REPORT_DIR=$NEW_DIR SLURM_ARRAY_TASK_ID=$index bash \
    sota/FRED_LLM_GE/phase0_data/jobs/audit_development_sequences.sbatch
done
echo "Event filename recheck task ${SLURM_ARRAY_TASK_ID} complete"
