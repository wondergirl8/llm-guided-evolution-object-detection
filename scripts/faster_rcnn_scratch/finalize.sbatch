#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_final
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-final-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; module load cuda
.venv-faster-rcnn/bin/python sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/finalize.py --checkpoints fred_faster_rcnn_scratch_checkpoints
