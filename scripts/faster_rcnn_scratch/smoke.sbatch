#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_smoke
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
: "${FRED_RUN_TAG:?Set a fresh Faster R-CNN run tag}"
.venv-faster-rcnn/bin/python scripts/faster_rcnn_scratch/smoke.py --output "fred_runs/$FRED_RUN_TAG/smoke.json" --resume
