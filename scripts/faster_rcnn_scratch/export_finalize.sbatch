#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_exports_ready
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=run_job_outputs/export/frcnn-scratch-finalize-export-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load uv
for stage in search final; do
  uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set "$stage" --finalize
done
