#!/bin/bash
# Validate existing ICE artifacts; no source collection, GPU or Slurm job.
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
PY=data/.venv-yolo11/bin/python
test -x "$PY" || { echo "Missing existing YOLO11 environment: $PY" >&2; exit 1; }
"$PY" -m sota.FRED_LLM_GE.phase0_data.check_scene_split "$@"
