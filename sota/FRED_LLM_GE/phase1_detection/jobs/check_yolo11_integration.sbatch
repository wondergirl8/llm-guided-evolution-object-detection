#!/bin/bash
#SBATCH --job-name=fred-integration-check
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=data/logs/fred-integration-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
PY=data/.venv-yolo11/bin/python
DATA=${FRED_YOLO_DATA:-data/fred_yolo11_bringup/data.yaml}
TRAINER=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py
RUN_ID="integration_seed_${SLURM_JOB_ID}"
OUTPUT=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11

test -x "$PY" && test -s "$DATA" && test -s yolo11m.pt || {
  echo "Environment, bounded labeled export, and yolo11m.pt are required" >&2
  exit 1
}
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("Allocated GPU:", torch.cuda.get_device_name(0))'
"$PY" "$TRAINER" --model network --data "$DATA" --weights "$PWD/yolo11m.pt" \
  --epochs 1 --fraction 1 --batch 2 --imgsz 640 --device 0 --run-id "$RUN_ID" --preflight-only
"$PY" "$TRAINER" --model network --data "$DATA" --weights "$PWD/yolo11m.pt" \
  --epochs 1 --fraction 1 --batch 2 --imgsz 640 --device 0 --run-id "$RUN_ID"
"$PY" - "$OUTPUT" "$RUN_ID" <<'PY'
import csv, json, math, sys
from pathlib import Path
root, run = Path(sys.argv[1]), sys.argv[2]
checkpoint = root / "trained_models" / f"{run}.pt"
report = root / "results" / f"{run}_evaluation.json"
result = root / "results" / f"{run}_results.csv"
assert checkpoint.is_file() and checkpoint.stat().st_size > 0, "MISSING CHECKPOINT"
record = json.loads(report.read_text())
rows = list(csv.reader(result.open()))
assert len(rows) == 1 and len(rows[0]) == 3, "INVALID RESULTS CSV"
a, b, n = map(float, rows[0])
assert all(math.isfinite(v) for v in (a, b, n)) and 0 <= a <= 1 and 0 <= b <= 1 and n > 0
assert record["gene_id"] == run and record["metadata"]["candidate_gene_id"] == "seed"
assert abs(a-record["metrics"]["mAP50"]) < 1e-6
assert abs(b-record["metrics"]["mAP50_95"]) < 1e-6
print("CHECKPOINT:", checkpoint)
print("EVALUATION REPORT:", report)
print("CSV:", result)
print("ENGINEERING CHECK COMPLETE; this is not a formal research baseline or generation")
PY
