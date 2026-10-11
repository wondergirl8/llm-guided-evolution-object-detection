#!/usr/bin/env bash
# Reuse fred-run-setup's CPU export before spending GPU time on training.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sbatch >/dev/null || { echo 'Run this on ICE.' >&2; exit 1; }
test "$(git branch --show-current)" = yolo11-scratch-evolution
test -x .venv/bin/python || { echo 'Run uv sync first.' >&2; exit 1; }
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'sota/FRED_LLM_GE/phase1_detection/seeds/yolo11')
from option_b import load_manifest
m = load_manifest()
print(f"CPU export: {len(m['train_sequences'])} training sequences at step {m['frame_step']}; "
      f"{len(m['val_sequences'])} validation sequences at step {m['val_frame_step']}")
PY
prep="$PWD/fred_ultralytics_yolo11_output/preparation-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$prep"
array=$(sbatch --parsable --job-name=yolo11-b-export --array=0-7%4 \
  --cpus-per-task=4 --mem=8G --time=04:00:00 \
  --output="$prep/export-%A_%a.log" scripts/yolo11_scratch_job.sh prepare)
array=${array%%;*}
finalize=$(sbatch --parsable --job-name=yolo11-b-ready --dependency="afterok:$array" \
  --cpus-per-task=2 --mem=4G --time=01:00:00 \
  --output="$prep/finalize-%j.log" scripts/yolo11_scratch_job.sh finalize)
finalize=${finalize%%;*}
printf '%s\n' "$prep" > fred_ultralytics_yolo11_output/latest-preparation.txt
printf 'Export array: %s\nFinalize job: %s\nLogs: %s\n' "$array" "$finalize" "$prep"
printf 'Check: sacct -j %s,%s --format=JobID,State,ExitCode\n' "$array" "$finalize"
printf 'When both succeed, check the server and run: bash scripts/run_yolo11_option_b.sh\n'
