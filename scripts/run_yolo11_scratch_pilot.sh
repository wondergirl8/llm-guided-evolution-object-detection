#!/usr/bin/env bash
# ICE only: seed baseline first, followed by the existing LLM-GE controller.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sbatch >/dev/null || { echo "Run this on ICE, where sbatch is available." >&2; exit 1; }
test -x .venv/bin/python || { echo "Run uv sync in this fresh checkout first." >&2; exit 1; }
test -s hostname.log || { echo "Put the team's active LLM server hostname in hostname.log first." >&2; exit 1; }
.venv/bin/python - <<'PY'
import importlib.metadata
from src.cfg import constants
assert importlib.metadata.version('ultralytics') == '8.3.203'
assert constants.SEED_NETWORK.endswith('seed_ultralytics_yolo11.py')
assert constants.TRAIN_FILE.endswith('train_ultralytics.py')
PY

# Fixed low-cost engineering workload, identical for the seed and candidates.
export FRED_DATA_PROFILE=a
export FRED_YOLO_SCALE=m FRED_TRAIN_SEQUENCES=5 FRED_VAL_SEQUENCES=1
export FRED_EPOCHS=1 FRED_FRAME_STEP=200 FRED_IMG_WIDTH=320 FRED_BATCH=2 FRED_DEVICE=cuda
export LLMGE_GENERATIONS=3 LLMGE_POPULATION=4
export YOLO11_PILOT_ROOT="$PWD/fred_ultralytics_yolo11_output/pilot-$(date +%Y%m%d-%H%M%S)-$$"
export LLMGE_OUTPUT_DIR="$YOLO11_PILOT_ROOT/candidates"
mkdir -p "$YOLO11_PILOT_ROOT" run_job_outputs/{evaluation,evolution,islands}
.venv/bin/python - <<'PY'
import json, os, pathlib, urllib.request
from src.cfg.constants import PORT
host = pathlib.Path('hostname.log').read_text().strip()
with urllib.request.urlopen(f'http://{host}:{PORT}/', timeout=10) as response:
    assert response.status == 200
keys = ('FRED_YOLO_SCALE', 'FRED_TRAIN_SEQUENCES', 'FRED_VAL_SEQUENCES', 'FRED_EPOCHS',
        'FRED_FRAME_STEP', 'FRED_IMG_WIDTH', 'FRED_BATCH', 'FRED_DEVICE',
        'LLMGE_GENERATIONS', 'LLMGE_POPULATION', 'LLMGE_OUTPUT_DIR')
config = {key: os.environ[key] for key in keys}
config.update(llm_server_hostname=host, llm_server_port=PORT)
pathlib.Path(os.environ['YOLO11_PILOT_ROOT'], 'pilot_config.json').write_text(json.dumps(config, indent=2))
PY
baseline=$(sbatch --parsable --job-name=yolo11-scratch-base --gres=gpu:1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --output="$YOLO11_PILOT_ROOT/baseline-%j.log" scripts/yolo11_scratch_job.sh baseline)
baseline=${baseline%%;*}
evolution=$(sbatch --parsable --job-name=yolo11-scratch-ge --dependency="afterok:$baseline" \
  --cpus-per-task=2 --mem=4G --time=04:00:00 \
  --output="$YOLO11_PILOT_ROOT/evolution-%j.log" scripts/yolo11_scratch_job.sh evolve)
evolution=${evolution%%;*}
printf 'Baseline job: %s\nEvolution job: %s\nArtifacts: %s\n' "$baseline" "$evolution" "$YOLO11_PILOT_ROOT"
printf 'Status: squeue -j %s,%s\n' "$baseline" "$evolution"
printf 'Completion: sacct -j %s,%s --format=JobID,State,ExitCode\n' "$baseline" "$evolution"
