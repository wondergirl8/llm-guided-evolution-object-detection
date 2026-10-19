#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_20ep
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-20epoch-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
PYTHON=.venv-faster-rcnn/bin/python
BASE=sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch
"$PYTHON" "$BASE/train_eval.py" --model network --run_name seed_20epoch --epochs 20 --track_epochs --device cuda
"$PYTHON" analysis/scripts/fred_epoch_pareto.py --history "$BASE/reports/seed_20epoch/epoch_history.json" --output "$BASE/results/seed_20epoch_graphs"
