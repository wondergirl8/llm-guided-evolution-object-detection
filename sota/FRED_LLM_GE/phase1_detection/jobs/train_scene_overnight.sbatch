#!/bin/bash
#SBATCH --job-name=fred-night-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-night-gpu-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test $# -eq 2
test "$(git branch --show-current)" = fred-yolo11-infrastructure
test "$(git rev-parse HEAD)" = "$2"
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
module load cuda
PY=data/.venv-yolo11/bin/python
DIRECTORY=$1
DATA="$DIRECTORY/export/data.yaml"
RUN_ID="scene_check_${SLURM_JOB_ID}"
test -s yolo11m.pt && test -s "$DATA"
"$PY" -m sota.FRED_LLM_GE.data.scene_diagnostic_export "$DIRECTORY" --verify-only
"$PY" - "$DATA" <<'PY'
import importlib.metadata, json, sys, torch
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as t
assert importlib.metadata.version('ultralytics') == '8.4.165', 'Pinned environment changed'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one allocated GPU required'
previous = json.loads((t.SCRIPT_DIR / 'results/learning_rate_check_6135264_evaluation.json').read_text())
assert previous['metadata']['model_identity'] == t.sha256_file(t.SCRIPT_DIR / 'network.py'), 'Seed changed'
assert previous['metadata']['training_config']['amp'] is False
assert previous['metadata']['training_config']['lr0'] == 0.0002
t.validate_data(sys.argv[1])
print('Allocated GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)
PY
"$PY" sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py \
  --model network --data "$DATA" --weights "$PWD/yolo11m.pt" \
  --epochs 50 --fraction 1 --batch 2 --imgsz 640 --device 0 \
  --run-id "$RUN_ID" --no-amp --adamw-lr 0.0002
"$PY" - "$DIRECTORY" "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as t
from sota.FRED_LLM_GE.data.scene_diagnostic_export import verify_export
directory, run = Path(sys.argv[1]), sys.argv[2]
report_path = t.SCRIPT_DIR / 'results' / f'{run}_evaluation.json'
report = json.loads(report_path.read_text())
history = t.validate_training_history(t.SCRIPT_DIR / 'runs' / run / 'results.csv')
assert history['epochs_completed'] == 50 and history == report['metadata']['training_history']
assert report['metadata']['training_config']['amp'] is False
assert report['metadata']['training_config']['lr0'] == 0.0002
source = verify_export(directory)
assert report['num_frames'] == source['counts']['validation']
summary = {'status': 'completed', 'run_id': run, 'data': source, 'history': history,
           'validation': report['metrics'], 'evaluation_report': str(report_path),
           'best_checkpoint': str(t.SCRIPT_DIR / 'trained_models' / f'{run}.pt')}
with (directory / 'summary.json').open('x') as handle:
    handle.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(summary, indent=2))
print('OVERNIGHT SCENE CHECK COMPLETE; engineering diagnostic only')
PY
