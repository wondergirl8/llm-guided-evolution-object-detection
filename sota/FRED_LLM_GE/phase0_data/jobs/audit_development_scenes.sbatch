#!/bin/bash
#SBATCH --job-name=fred-scene-audit
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=data/logs/fred-scene-audit-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$(git branch --show-current)" = fred-yolo11-infrastructure || {
  echo "Wrong branch; no scene audit performed" >&2; exit 1;
}
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-7]$ ]] || \
   [[ ${SLURM_ARRAY_TASK_COUNT:-} != 8 ]] || \
   [[ ${SLURM_ARRAY_TASK_MIN:-} != 0 ]] || [[ ${SLURM_ARRAY_TASK_MAX:-} != 7 ]]; then
  echo "Submit exactly --array=0-7%1 to keep queued jobs bounded and cache access serial" >&2
  exit 2
fi
PY=data/.venv-yolo11/bin/python
REV=$(git rev-parse --short=12 HEAD)
OUTPUT="data/fred_scene_audit/v1_${REV}"
test -x "$PY" || { echo "Existing FRED ICE Python environment is required" >&2; exit 1; }
echo "Scene audit shard $SLURM_ARRAY_TASK_ID of 8; output $OUTPUT"
"$PY" -m sota.FRED_LLM_GE.phase0_data.scene_audit --output "$OUTPUT" \
  collect --shard "$SLURM_ARRAY_TASK_ID" --shards 8 --summarize-when-complete
echo "Scene audit shard complete; the final successful shard publishes summary.json."
