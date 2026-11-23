#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_report
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-report-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
: "${FRED_RUN_TAG:?Set the Faster R-CNN run tag}"
uv run python analysis/scripts/fred_pareto_graphs.py --family faster_rcnn_scratch --checkpoints "fred_runs/$FRED_RUN_TAG/checkpoints" --models_dir "fred_runs/$FRED_RUN_TAG/candidates/models" --seed_file sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py --title "FRED Faster R-CNN Scratch LLM-GE" --output "fred_runs/$FRED_RUN_TAG/plots"
