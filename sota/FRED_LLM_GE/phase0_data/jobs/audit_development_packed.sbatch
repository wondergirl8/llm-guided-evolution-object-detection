#!/bin/bash
#SBATCH --job-name=fred-dev-pack
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-dev-pack-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Each queued array task processes a stride of official challenging-train
# indexes. This keeps the submitted-job count small under the ICE QOS limit.
# A retry verifies and skips already published sequence reports.
for name in SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX; do
  value=${!name:-}
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "$name must be set by a Slurm job array" >&2
    exit 2
  fi
done

task_id=$SLURM_ARRAY_TASK_ID
task_count=$SLURM_ARRAY_TASK_COUNT
task_min=$SLURM_ARRAY_TASK_MIN
task_max=$SLURM_ARRAY_TASK_MAX
if (( task_count < 1 || task_min < 0 || task_max > 171 ||
      task_max - task_min + 1 != task_count ||
      task_id < task_min || task_id > task_max )); then
  echo "Submit a contiguous array of official development indexes within 0-171" >&2
  exit 2
fi

echo "Packed audit task $task_id: indexes $task_id,$((task_id + task_count)),... through 171"
for ((index = task_id; index <= 171; index += task_count)); do
  echo "===== Official development index $index ====="
  SLURM_ARRAY_TASK_ID=$index bash \
    sota/FRED_LLM_GE/phase0_data/jobs/audit_development_sequences.sbatch
done
echo "Packed audit task $task_id complete"
