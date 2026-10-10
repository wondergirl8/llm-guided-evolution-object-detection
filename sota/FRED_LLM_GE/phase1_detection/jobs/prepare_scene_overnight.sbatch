#!/bin/bash
#SBATCH --job-name=fred-night-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=data/logs/fred-night-data-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test $# -eq 2
test "$(git branch --show-current)" = fred-yolo11-infrastructure
test "$(git rev-parse HEAD)" = "$2"
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
export OMP_NUM_THREADS=4
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase0_data.prepare_scene_diagnostic "$1"
