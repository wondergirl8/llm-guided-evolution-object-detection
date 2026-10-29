#!/usr/bin/env bash
#SBATCH --job-name=frcnn_pareto_table
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=run_job_outputs/evaluation/frcnn-pareto-table-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python analysis/scripts/pareto_held_out_table.py
