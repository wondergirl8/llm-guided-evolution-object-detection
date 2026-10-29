#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_20gen
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_job_outputs/islands/frcnn-scratch-20gen-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; module load uv; export LLMGE_DETECTOR=faster_rcnn_scratch
trap '.venv/bin/python scripts/faster_rcnn_scratch/stop_server.py' EXIT
.venv/bin/python scripts/faster_rcnn_scratch/wait_server.py
uv run python run_improved.py --checkpoints fred_faster_rcnn_scratch_checkpoints --global_path fred_faster_rcnn_scratch_checkpoints --prompt_group FRED_FasterRCNNScratch/Normal
