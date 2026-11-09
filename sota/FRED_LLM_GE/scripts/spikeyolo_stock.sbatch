#!/usr/bin/env bash
#SBATCH --job-name=fred-spikeyolo-stock
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --constraint=H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=run_job_outputs/baseline/spikeyolo-%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
test -f sota/FRED_LLM_GE/phase1_detection/spikeyolo_baseline.py
: "${SLURM_ARRAY_TASK_ID:?This baseline requires array indices 0, 1, 2}"
module load uv
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline \
  run-seed --seed "$SLURM_ARRAY_TASK_ID" --device cuda
