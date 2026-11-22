#!/usr/bin/env bash
#SBATCH --job-name=frcnn_epoch8_test
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=run_job_outputs/evaluation/frcnn-epoch8-test-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
: "${FRED_RUN_TAG:?Set the existing run tag}"
exec .venv-faster-rcnn/bin/python \
  sota/FRED_LLM_GE/phase2/faster_rcnn_scratch/interim_eval.py \
  --run-root "fred_runs/$FRED_RUN_TAG" \
  --output "fred_runs/$FRED_RUN_TAG/interim_test/epoch8_xXx4LVf0TsysH0uUoegVF3Yu1cg"
