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
# Keep each full-size seed's state separate while preserving existing YOLO11 paths.
run_tag="fred_yolo11"
if [[ "${FRED_SEED_FAMILY:-yolo11}" == "dfine" ]]; then
  run_tag="fred_dfine"
elif [[ "${FRED_SEED_FAMILY:-yolo11}" == "yolo26_p2" ]]; then
  run_tag="fred_yolo26_p2_map5095_coverage_v1"
elif [[ "${FRED_SEED_FAMILY:-yolo11}" == "tiny_yolo11" ]]; then
  run_tag="fred_tiny_yolo11"
fi
# FRED_EVENTS (e.g. raw_w10000_polarity) trains on other event frames; keep
# that run's checkpoints apart too (src/cfg/constants_fred_yolo11.py).
if [[ -n "${FRED_EVENTS:-}" ]]; then
  run_tag="${run_tag}_${FRED_EVENTS}"
fi
uv run python run_improved.py --checkpoints "${run_tag}_checkpoints" --global_path "${run_tag}_checkpoints"
