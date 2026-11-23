#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_20gen
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_job_outputs/islands/frcnn-scratch-20gen-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; module load uv; export LLMGE_DETECTOR=faster_rcnn_scratch FRED_SEED_FAMILY=faster_rcnn_scratch
: "${FRED_RUN_TAG:?Set the Faster R-CNN run tag}"
trap '.venv/bin/python scripts/faster_rcnn_scratch/stop_server.py' EXIT
.venv-faster-rcnn/bin/python -m sota.FRED_LLM_GE.phase1_detection.faster_rcnn_baseline summarize --run-root "fred_runs/$FRED_RUN_TAG"
.venv/bin/python scripts/faster_rcnn_scratch/wait_server.py
uv run python run_improved.py --prompt_group FRED_FasterRCNNScratch/Normal
