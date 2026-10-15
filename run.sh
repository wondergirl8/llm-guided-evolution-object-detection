#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 7:45:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
echo "launching LLM Guided Evolution"
hostname
module load uv

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
# Slurm does not create --output directories for the jobs this run submits.
mkdir -p run_job_outputs/{islands,evolution,evaluation,server}
# Keep D-FINE's state separate while preserving existing YOLO run paths.
run_tag="fred_yolo11"
if [[ "${FRED_SEED_FAMILY:-yolo11}" == "dfine" ]]; then
  run_tag="fred_dfine"
fi
uv run python run_improved.py --checkpoints "${run_tag}_checkpoints" --global_path "${run_tag}_checkpoints"
