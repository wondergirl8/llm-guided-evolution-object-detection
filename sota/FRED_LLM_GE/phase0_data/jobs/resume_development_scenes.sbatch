#!/bin/bash
#SBATCH --job-name=fred-scene-resume
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=data/logs/fred-scene-resume-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
test $# -eq 1 || { echo "Usage: sbatch resume_development_scenes.sbatch SOURCE_AUDIT" >&2; exit 2; }
PY=data/.venv-yolo11/bin/python
SOURCE=$1
OUTPUT="data/fred_scene_audit/resume_${SLURM_JOB_ID}"
test -x "$PY" && test -d "$SOURCE"
test ! -e "$OUTPUT"
echo "Scene resume source: $SOURCE; output: $OUTPUT; memory: 48G; serial collection"
"$PY" -m sota.FRED_LLM_GE.archive.phase0_data_old.resume_scene_audit \
  --source "$SOURCE" --output "$OUTPUT"
