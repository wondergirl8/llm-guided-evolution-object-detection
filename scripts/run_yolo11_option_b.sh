#!/usr/bin/env bash
# Long, resumable Python architecture evolution using the team option-B export.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sbatch >/dev/null || { echo 'Run this on ICE.' >&2; exit 1; }
test "$(git branch --show-current)" = yolo11-scratch-evolution
test -x .venv/bin/python || { echo 'Run uv sync first.' >&2; exit 1; }
test -s hostname.log || { echo 'The LLM server must write hostname.log first.' >&2; exit 1; }
resume=0
export LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-4096}
export LLMGE_INITIALIZATION_BATCHES=${LLMGE_INITIALIZATION_BATCHES:-3}
export LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS=${LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS:-60}
if [[ $# -eq 2 && $1 = --resume ]]; then
  resume=1
  export YOLO11_PILOT_ROOT="$(cd "$2" && pwd)"
  test -s "$YOLO11_PILOT_ROOT/baseline/seed_results.csv"
  if test -s "$YOLO11_PILOT_ROOT/controller-job.txt"; then
    previous=$(cat "$YOLO11_PILOT_ROOT/controller-job.txt")
    queue_ids=$(squeue -h -u "$USER" -o '%i')
    if printf '%s\n' "$queue_ids" | awk -v wanted="$previous" '$1 == wanted {found=1} END {exit !found}'; then
      echo "Controller $previous is still queued or running; wait before resuming." >&2
      exit 1
    fi
  fi
elif [[ $# -eq 0 ]]; then
  export FRED_DATA_PROFILE=b FRED_YOLO_SCALE=n FRED_DEVICE=cuda
  unset FRED_TRAIN_SEQUENCES FRED_VAL_SEQUENCES
  export FRED_FRAME_STEP=50 FRED_VAL_FRAME_STEP=10 FRED_IMG_WIDTH=1280 FRED_BATCH=16 FRED_EPOCHS=20 FRED_LR=0.001
  export LLMGE_GENERATIONS=${LLMGE_GENERATIONS:-1001} LLMGE_POPULATION=${LLMGE_POPULATION:-4}
  export LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS=12600 LLM_SERVER_READY_TIMEOUT=60
  export YOLO11_PILOT_ROOT="$PWD/fred_ultralytics_yolo11_output/option-b-team-$(date +%Y%m%d-%H%M%S)-$$"
else
  echo 'Usage: bash scripts/run_yolo11_option_b.sh [--resume RUN_FOLDER]' >&2
  exit 2
fi
export YOLO11_RESUME=$resume
mkdir -p "$YOLO11_PILOT_ROOT" run_job_outputs/{evaluation,evolution,islands}
.venv/bin/python - <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, shlex, sys, urllib.request
connection_path = pathlib.Path('llm-server-connection.json')
connection = json.loads(connection_path.read_text()) if connection_path.is_file() else None
if connection:
    os.environ['LLMGE_PORT'] = str(connection['port'])
from src.cfg import constants
sys.path.insert(0, str(pathlib.Path(constants.TRAIN_FILE).parent))
from option_b import MANIFEST_PATH, PROTOCOL_PATH, load_manifest, load_exported_data
assert importlib.metadata.version('ultralytics') == '8.3.203'
assert constants.SEED_NETWORK.endswith('seed_ultralytics_yolo11.py')
assert constants.TRAIN_FILE.endswith('train_ultralytics.py')
m = load_manifest()
# Validate both complete exports before submitting any GPU job; no image decoding.
train, val = load_exported_data(m, 1280)
host = connection['hostname'] if connection else pathlib.Path('hostname.log').read_text().strip()
with urllib.request.urlopen(f'http://{host}:{constants.PORT}/', timeout=10) as response:
    assert response.status == 200
    if connection:
        if str(json.load(response).get('slurm_job_id')) != str(connection['slurm_job_id']):
            raise ValueError('Ready endpoint does not belong to the selected Slurm server job')
root = pathlib.Path(os.environ['YOLO11_PILOT_ROOT'])
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
identity = dict(manifest_sha256=m['sha256'], trainer_sha256=sha(constants.TRAIN_FILE),
                seed_sha256=sha(constants.SEED_NETWORK), protocol_sha256=sha(PROTOCOL_PATH))
keys = ('FRED_DATA_PROFILE', 'FRED_YOLO_SCALE', 'FRED_DEVICE', 'FRED_FRAME_STEP', 'FRED_VAL_FRAME_STEP',
        'FRED_IMG_WIDTH', 'FRED_BATCH', 'FRED_EPOCHS', 'FRED_LR', 'LLMGE_GENERATIONS', 'LLMGE_POPULATION',
        'LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS', 'LLM_SERVER_READY_TIMEOUT')
if os.environ['YOLO11_RESUME'] == '1':
    config = json.loads((root / 'run_config.json').read_text())
    if any(config.get(k) != v for k, v in identity.items()):
        raise ValueError('Run identity changed; start a fresh run instead of mixing protocols')
    if not list((root / 'checkpoints').glob('checkpoint_gen_*.pkl')):
        print('Initialization has no completed checkpoint yet; restarting it with the saved baseline.', flush=True)
else:
    config = {k: os.environ[k] for k in keys}
    if int(config['LLMGE_GENERATIONS']) < 2 or int(config['LLMGE_POPULATION']) < 4:
        raise ValueError('Need generations >= 2 and population >= 4')
    config.update(identity, training_frames=len(train['images']), validation_frames=len(val['images']),
                  llm_server_hostname=host, llm_server_port=constants.PORT,
                  protocol_source_commit=m['protocol_source_commit'])
    (root / 'split_manifest.json').write_bytes(MANIFEST_PATH.read_bytes())
    (root / 'protocol.yaml').write_bytes(PROTOCOL_PATH.read_bytes())
    (root / 'run_config.json').write_text(json.dumps(config, indent=2))
exports = {k: config[k] for k in keys}
generation = {k: os.environ[k] for k in ('LLM_MAX_NEW_TOKENS', 'LLMGE_INITIALIZATION_BATCHES',
              'LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS')}
if any(int(value) <= 0 for value in generation.values()):
    raise ValueError('Generation limits must be positive')
exports.update(generation)
exports['LLMGE_PORT'] = str(constants.PORT)
exports.update(LLMGE_OUTPUT_DIR=str(root / 'candidates'), YOLO11_PILOT_ROOT=str(root))
import time
(root / f'generation-session-{time.time_ns()}.json').write_text(json.dumps({
    **generation, 'llm_server_hostname': host, 'llm_server_port': constants.PORT,
    'generation_source_sha256': {p: sha(p) for p in ('src/llm_mutation.py', 'src/llm_crossover.py',
        'src/llm_utils.py', 'src/generated_class.py', 'src/candidate_preflight.py',
        'templates/FRED/UltralyticsRules.txt')}}, indent=2))
(root / 'environment.sh').write_text('\n'.join(f'export {k}={shlex.quote(v)}' for k, v in exports.items()) + '\n')
print(f"YOLO11n scratch: {len(train['images'])} training / {len(val['images'])} validation frames; "
      f"20 epochs; population {config['LLMGE_POPULATION']}; target {int(config['LLMGE_GENERATIONS']) - 1} generations", flush=True)
PY
source "$YOLO11_PILOT_ROOT/environment.sh"
unset FRED_TRAIN_SEQUENCES FRED_VAL_SEQUENCES
baseline='existing'
evolution_args=(--parsable --job-name=yolo11-b-ge)
if [[ $resume -eq 0 ]]; then
  baseline=$(sbatch --parsable --job-name=yolo11-b-base --gres=gpu:1 \
    --cpus-per-task=12 --mem=32G --time=04:00:00 \
    --output="$YOLO11_PILOT_ROOT/baseline-%j.log" scripts/yolo11_scratch_job.sh baseline)
  baseline=${baseline%%;*}
  evolution_args+=(--dependency="afterok:$baseline")
fi
evolution=$(sbatch "${evolution_args[@]}" \
  --cpus-per-task=4 --mem=16G --time=07:45:00 \
  --output="$YOLO11_PILOT_ROOT/evolution-%j.log" scripts/yolo11_scratch_job.sh evolve)
evolution=${evolution%%;*}
printf '%s\n' "$evolution" > "$YOLO11_PILOT_ROOT/controller-job.txt"
printf '%s\n' "$YOLO11_PILOT_ROOT" > fred_ultralytics_yolo11_output/latest-option-b.txt
printf 'Baseline job: %s\nController job: %s\nRun folder: %s\n' "$baseline" "$evolution" "$YOLO11_PILOT_ROOT"
printf 'Status: squeue -j %s\nResume after this job ends with a ready server:\n' "$evolution"
printf 'bash scripts/run_yolo11_option_b.sh --resume %q\n' "$YOLO11_PILOT_ROOT"
