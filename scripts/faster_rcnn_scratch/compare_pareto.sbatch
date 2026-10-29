#!/usr/bin/env bash
#SBATCH --job-name=frcnn_pareto_test
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=run_job_outputs/evaluation/frcnn-pareto-test-%A_%a.out
#SBATCH --array=0-2
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
IDS=(xXx005EzIza8AimNURDwEWdbye4 xXxT2dgPIjQdOfJ2sLEswEb2hTV xXx7KjH1KlxilhpYvHIVxTndgfe)
.venv-faster-rcnn/bin/python sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/compare_pareto.py --gene_id "${IDS[$SLURM_ARRAY_TASK_ID]}"
