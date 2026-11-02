#!/usr/bin/env bash
#SBATCH --job-name=frcnn_scratch_report
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_job_outputs/evaluation/frcnn-scratch-report-%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"
uv run python analysis/scripts/fred_pareto_graphs.py --checkpoints fred_faster_rcnn_scratch_checkpoints --seed_results sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/results/seed_results.csv --models_dir sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/models --seed_file sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py --winner_manifest sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/final_results/winner_selection.json --title "FRED Faster R-CNN Scratch LLM-GE" --output analysis/results/analysis/fred_faster_rcnn_scratch_20gen
