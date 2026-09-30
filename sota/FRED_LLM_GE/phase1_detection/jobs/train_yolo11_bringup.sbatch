#!/bin/bash
#SBATCH --job-name=fred-yolo11-smoke
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-yolo11-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda
PY=data/.venv-yolo11/bin/python
TRAINER=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train.py
DATA=data/fred_yolo11_bringup/data.yaml
RUN_ID=seed_mscale_v1
RESULT="sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/${RUN_ID}_results.csv"
test -x "$PY" && test -s "$DATA" && test -s yolo11m.pt
test ! -e "$RESULT" || {
  echo "Seed results already exist; inspect them before rerunning" >&2
  exit 1
}

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; torch.cuda.set_device(0); print(torch.cuda.get_device_name(0))'
"$PY" "$TRAINER" --model network --data "$DATA" \
  --weights "$PWD/yolo11m.pt" --run-id "$RUN_ID" --preflight-only
"$PY" "$TRAINER" --model network --data "$DATA" \
  --weights "$PWD/yolo11m.pt" --epochs 1 --batch 2 --imgsz 640 --device 0 \
  --run-id "$RUN_ID"
test -s "$RESULT"
echo "FRED YOLO11 bounded seed smoke run complete"
