#!/usr/bin/env bash
# One full option-B seed evaluation warms the sparse cache before evolution.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sbatch >/dev/null || { echo "Run this on ICE." >&2; exit 1; }
test "$(git branch --show-current)" = yolo11-scratch-evolution
test -x .venv/bin/python || { echo "Run uv sync first." >&2; exit 1; }
test -s hostname.log || { echo "The LLM server must write hostname.log first." >&2; exit 1; }
export FRED_DATA_PROFILE=b FRED_YOLO_SCALE=m FRED_DEVICE=cuda
unset FRED_TRAIN_SEQUENCES FRED_VAL_SEQUENCES
export FRED_FRAME_STEP=50 FRED_IMG_WIDTH=640 FRED_BATCH=4
export FRED_EPOCHS=${FRED_EPOCHS:-3}
export LLMGE_GENERATIONS=${LLMGE_GENERATIONS:-3} LLMGE_POPULATION=${LLMGE_POPULATION:-4}
# Keep the existing job wrapper; its directory variable also supports normal runs.
export YOLO11_PILOT_ROOT="$PWD/fred_ultralytics_yolo11_output/option-b-$(date +%Y%m%d-%H%M%S)-$$"
export LLMGE_OUTPUT_DIR="$YOLO11_PILOT_ROOT/candidates"
mkdir -p "$YOLO11_PILOT_ROOT" run_job_outputs/{evaluation,evolution,islands}
.venv/bin/python - <<'PY'
import importlib.metadata, json, os, pathlib, sys, urllib.request
from src.cfg import constants
sys.path.insert(0, str(pathlib.Path(constants.TRAIN_FILE).parent))
from train_ultralytics import parse_args
from option_b import MANIFEST_PATH, load_manifest
assert importlib.metadata.version('ultralytics') == '8.3.203'
assert constants.SEED_NETWORK.endswith('seed_ultralytics_yolo11.py')
assert constants.TRAIN_FILE.endswith('train_ultralytics.py')
args = parse_args()
manifest = load_manifest()
host = pathlib.Path('hostname.log').read_text().strip()
with urllib.request.urlopen(f'http://{host}:{constants.PORT}/', timeout=10) as response:
    assert response.status == 200
root = pathlib.Path(os.environ['YOLO11_PILOT_ROOT'])
(root / 'split_manifest.json').write_bytes(MANIFEST_PATH.read_bytes())
config = {key: os.environ[key] for key in ('FRED_DATA_PROFILE', 'FRED_YOLO_SCALE', 'FRED_FRAME_STEP',
          'FRED_IMG_WIDTH', 'FRED_BATCH', 'FRED_EPOCHS', 'LLMGE_GENERATIONS', 'LLMGE_POPULATION')}
config.update(manifest_sha256=manifest['sha256'], llm_server_hostname=host, llm_server_port=constants.PORT)
(root / 'run_config.json').write_text(json.dumps(config, indent=2))
print(f"Option B: {len(args.train_sequences)} train / {len(args.val_sequences)} validation sequences; "
      f"every {args.frame_step}th frame; {args.epochs} epochs; population {constants.population_size}; "
      f"{constants.num_generations - 1} evolution generations", flush=True)
PY
baseline=$(sbatch --parsable --job-name=yolo11-b-base --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --output="$YOLO11_PILOT_ROOT/baseline-%j.log" scripts/yolo11_scratch_job.sh baseline)
baseline=${baseline%%;*}
evolution=$(sbatch --parsable --job-name=yolo11-b-ge --dependency="afterok:$baseline" \
  --cpus-per-task=2 --mem=4G --time=06:00:00 \
  --output="$YOLO11_PILOT_ROOT/evolution-%j.log" scripts/yolo11_scratch_job.sh evolve)
evolution=${evolution%%;*}
printf 'Baseline job: %s\nEvolution job: %s\nRun folder: %s\n' "$baseline" "$evolution" "$YOLO11_PILOT_ROOT"
printf 'Status: squeue -j %s,%s\nCompletion: sacct -j %s,%s --format=JobID,State,ExitCode\n' \
  "$baseline" "$evolution" "$baseline" "$evolution"
