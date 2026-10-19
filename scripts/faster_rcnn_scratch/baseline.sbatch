#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_seed
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-seed-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; module load cuda
PYTHON=.venv-faster-rcnn/bin/python; TRAINER=sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/train_eval.py
"$PYTHON" "$TRAINER" --smoke --device cuda
"$PYTHON" "$TRAINER" --model network --device cuda
