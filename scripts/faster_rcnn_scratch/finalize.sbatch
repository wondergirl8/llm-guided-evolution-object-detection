#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_final
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-final-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; module load cuda
: "${FRED_RUN_TAG:?Set the Faster R-CNN run tag}"
.venv-faster-rcnn/bin/python -m sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.finalize --run-root "fred_runs/$FRED_RUN_TAG"
