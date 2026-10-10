#!/bin/bash
#SBATCH --job-name=fred-stability-check
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-stability-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
module load cuda
PY=data/.venv-yolo11/bin/python
RUN_ID="stability_check_${SLURM_JOB_ID}"
DATA=${FRED_YOLO_DATA:-data/fred_yolo11_bringup/data.yaml}
test -x "$PY" && test -s "$DATA" && test -s yolo11m.pt || {
  echo "Environment, existing bounded export, and yolo11m.pt are required" >&2
  exit 1
}
"$PY" - "$DATA" <<'PY'
import importlib.metadata, json, sys, torch
from pathlib import Path
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as t
assert importlib.metadata.version('ultralytics') == '8.4.165', 'Diagnostic requires the existing pinned Ultralytics version'
assert torch.cuda.is_available(), 'CUDA GPU unavailable'
data = Path(sys.argv[1])
source = t.validate_data(data)
previous = json.loads((t.SCRIPT_DIR / 'results' / 'learning_check_6133680_evaluation.json').read_text())['metadata']
assert source['manifest_sha256'] == previous['manifest_identity'], 'Dataset manifest differs from the previous run'
assert t.sha256_file(data.parent / 'source.json') == previous['source_record_sha256'], 'Dataset provenance differs from the previous run'
assert t.sha256_file(t.SCRIPT_DIR / 'network.py') == previous['model_identity'], 'Seed differs from the previous run'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)
PY
"$PY" sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py \
  --model network --data "$DATA" --weights "$PWD/yolo11m.pt" \
  --epochs 50 --fraction 1 --batch 2 --imgsz 640 --device 0 --no-amp --run-id "$RUN_ID"
"$PY" - "$RUN_ID" "$DATA" <<'PY'
import json, sys
from pathlib import Path
from ultralytics import YOLO
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as t
from sota.FRED_LLM_GE.phase1_detection.fitness import evaluate_frames
run, data = sys.argv[1], Path(sys.argv[2])
report = json.loads((t.SCRIPT_DIR / 'results' / f'{run}_evaluation.json').read_text())
history = t.validate_training_history(t.SCRIPT_DIR / 'runs' / run / 'results.csv')
assert history['epochs_completed'] == 50, 'Incomplete learning check'
assert history == report['metadata']['training_history'], 'Training history changed after evaluation'
assert report['metadata']['training_config']['amp'] is False, 'AMP was not disabled'
config, root = t._data_config(data)
image_dir = root / config['train']
images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in t.IMAGE_SUFFIXES)
model = YOLO(str(t.SCRIPT_DIR / 'trained_models' / f'{run}.pt'))
frames = t.predict_frames(model, images, root / 'labels' / image_dir.name, 640, 2, '0')
result = evaluate_frames(frames, params=report['metrics']['params'],
                         expected_sample_ids=[p.stem for p in images])
record = {
    'purpose': 'Full-precision small-data stability diagnostic; training scores are not selection fitness',
    'comparison_run_id': 'learning_check_6133680',
    'run_id': run, 'epochs': 50, 'metadata': report['metadata'],
    'train_images': len(images), 'validation_images': report['num_frames'],
    'train': {'mAP50': result.map50, 'mAP50_95': result.map50_95},
    'validation': report['metrics'],
}
directory = Path('data/fred_diagnostics')
directory.mkdir(parents=True, exist_ok=True)
output = directory / f'{run}.json'
with output.open('x') as handle:
    handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
print('TRAINING HISTORY:', history)
print('TRAIN METRICS:', record['train'])
print('VALIDATION METRICS:', record['validation'])
print('DIAGNOSTIC REPORT:', output)
print('STABILITY CHECK COMPLETE; not a formal baseline or evolution run')
PY
