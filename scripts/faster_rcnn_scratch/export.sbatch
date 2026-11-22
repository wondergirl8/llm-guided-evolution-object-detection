#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_export
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=run_job_outputs/export/frcnn-scratch-export-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load uv
for stage in search final; do
  uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set "$stage" --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8
done
