#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
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
# Uses the project selected by the src/cfg/constants.py symlink (FRED YOLO11).
uv run python run_improved.py --checkpoints fred_yolo11_checkpoints --global_path fred_yolo11_checkpoints
