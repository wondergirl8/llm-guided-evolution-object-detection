#!/bin/bash
#SBATCH --job-name=fred-dev-audit
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-dev-audit-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Submit with --array, starting with --array=0-7%1" >&2
  exit 2
fi

PY=data/.venv-yolo11/bin/python
INVENTORY=data/fred_phase0/metadata/source_inventory.json
OFFICIAL=data/fred_phase0/metadata/official_challenging_split.json
PROJECT=sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json
test -x "$PY" && test -s "$INVENTORY" && test -s "$OFFICIAL" && test -s "$PROJECT" || {
  echo "Pinned FRED source inventory and split files are required" >&2
  exit 1
}

SEQUENCE_ID=$("$PY" -m sota.FRED_LLM_GE.archive.phase0_data_old.development_audit \
  sequence-id --official "$OFFICIAL" --index "$SLURM_ARRAY_TASK_ID")
CODE_REVISION=$(git rev-parse --short=12 HEAD)
REPORT_DIR=${FRED_AUDIT_REPORT_DIR:-"data/fred_phase0/validation/development_audit_${CODE_REVISION}"}
case "$REPORT_DIR" in
  data/fred_phase0/validation/development_audit_*) ;;
  *) echo "FRED_AUDIT_REPORT_DIR must name a development_audit directory under Phase 0 validation" >&2; exit 2 ;;
esac
REPORT="$REPORT_DIR/sequence_${SEQUENCE_ID}.json"
mkdir -p "$REPORT_DIR"
echo "Audit sequence $SEQUENCE_ID (${SLURM_ARRAY_TASK_ID}); code: $CODE_REVISION; report: $REPORT"

verify_report() {
  "$PY" -m sota.FRED_LLM_GE.archive.phase0_data_old.development_audit verify-report \
    --report "$REPORT" --sequence "$SEQUENCE_ID" \
    --inventory "$INVENTORY" --official "$OFFICIAL" --project "$PROJECT"
}

if [ -e "$REPORT" ]; then
  verify_report
  echo "Existing complete inspection retained"
  exit 0
fi

# An inspection can return nonzero for source annotation findings and still
# publish a complete report. Preserve those findings as audit evidence.
set +e
"$PY" -m sota.FRED_LLM_GE.archive.phase0_data_old inspect-sequence "$SEQUENCE_ID" \
  --inventory "$INVENTORY" --official-split "$OFFICIAL" \
  --project-split "$PROJECT" --output "$REPORT"
inspect_status=$?
set -e
if [ ! -s "$REPORT" ]; then
  echo "Inspection exited $inspect_status without a report" >&2
  exit 1
fi
verify_report
echo "Inspection completed; data findings remain in $REPORT"
