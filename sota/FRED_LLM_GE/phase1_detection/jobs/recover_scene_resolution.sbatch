#!/bin/bash
#SBATCH --job-name=fred-resolution-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-resolution-recovery-%j.out
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
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase1_detection.recover_scene_resolution \
  --execute "$1"
