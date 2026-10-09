#!/bin/bash
#SBATCH --job-name=fred-yolo11-candidate-smoke
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=data/logs/fred-yolo11-candidate-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch train_proposed_yolo11_smoke.sbatch PROPOSAL_JOB_ID" >&2
  exit 2
fi

module load cuda
PY=data/.venv-yolo11/bin/python
DATA=data/fred_yolo11_bringup/data.yaml
TRAINER=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py
PROPOSAL_JOB_ID=$1
CANDIDATE_ID="P${PROPOSAL_JOB_ID}"
PROPOSAL_DIR="data/fred_generation_smoke/${PROPOSAL_JOB_ID}"
PROPOSAL="$PROPOSAL_DIR/proposal.json"
SOURCE="$PROPOSAL_DIR/network_${CANDIDATE_ID}.py"
RUN_ID="proposal_smoke_${CANDIDATE_ID}"
RESULT="sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/${RUN_ID}_results.csv"

test -x "$PY" && test -s "$DATA" && test -s yolo11m.pt && \
  test -s "$PROPOSAL" && test -s "$SOURCE" && \
  test -s "$PROPOSAL_DIR/preflight_passed.txt" || {
  echo "The successful candidate proposal and bounded YOLO11 data are required" >&2
  exit 1
}
test ! -e "$RESULT" || {
  echo "Candidate smoke result already exists; inspect it before rerunning" >&2
  exit 1
}

"$PY" - "$PROPOSAL" "$SOURCE" "$CANDIDATE_ID" <<'PY'
import json
import sys
from pathlib import Path

from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11.validator import validate_source

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("kind") != "fred_yolo11_unscored_candidate_proposal_v1":
    raise ValueError("unexpected candidate proposal kind")
if record.get("status") != "candidate_generated_unscored":
    raise ValueError("candidate proposal did not succeed")
if record.get("candidate_id") != sys.argv[3]:
    raise ValueError("candidate ID does not match proposal record")
if validate_source(Path(sys.argv[2])) != record.get("genome"):
    raise ValueError("candidate genome differs from proposal record")
print(f"proposal verified: {sys.argv[3]} {record['genome']}", flush=True)
PY

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; torch.cuda.set_device(0); print(torch.cuda.get_device_name(0))'
"$PY" "$TRAINER" --model "network_${CANDIDATE_ID}" \
  --variant_dir "$PROPOSAL_DIR" --data "$DATA" \
  --weights "$PWD/yolo11m.pt" --preflight-only
"$PY" "$TRAINER" --model "network_${CANDIDATE_ID}" \
  --variant_dir "$PROPOSAL_DIR" --data "$DATA" \
  --weights "$PWD/yolo11m.pt" --epochs 1 --fraction 1 --batch 2 --imgsz 640 --device 0 \
  --run-id "$RUN_ID"
test -s "$RESULT"
echo "Candidate training/evaluation smoke complete; result is not evolution fitness: $RESULT"
