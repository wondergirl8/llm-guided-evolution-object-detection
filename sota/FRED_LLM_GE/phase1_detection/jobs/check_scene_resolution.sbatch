#!/bin/bash
#SBATCH --job-name=fred-resolution
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-resolution-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test $# -eq 2
test "$(git branch --show-current)" = fred-yolo11-infrastructure
test "$(git rev-parse HEAD)" = "$2"
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
module load cuda
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase1_detection.scene_resolution_check \
  --execute "$1"
