#!/bin/bash
#SBATCH --job-name=fred-policy-review
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-policy-review-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$(git branch --show-current)" = fred-yolo11-infrastructure || {
  echo "Wrong branch; no policy review performed" >&2; exit 1;
}
PY=data/.venv-yolo11/bin/python
SUMMARY=data/fred_phase0/validation/development_audit_fa6815b20050/summary.json
test -x "$PY" && test -s "$SUMMARY" || {
  echo "Existing ICE Python environment and completed development audit are required" >&2; exit 1;
}
"$PY" -m sota.FRED_LLM_GE.archive.phase0_data_old.policy_review \
  --summary "$SUMMARY" --output "data/fred_policy_review/${SLURM_JOB_ID}"
echo "DG-P0-04 numerical verification complete; visual review and split freeze remain."
