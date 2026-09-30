#!/bin/bash
#SBATCH --job-name=fred-yolo11-propose
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=data/logs/fred-yolo11-propose-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda
module load uv
PY=data/.venv-yolo11/bin/python
DATA=data/fred_yolo11_bringup/data.yaml
test -x "$PY" && test -s "$DATA" && test -s yolo11m.pt || {
  echo "The successful YOLO11m smoke environment, export, and checkpoint are required" >&2
  exit 1
}
export UV_CACHE_DIR="$PWD/data/.uv-cache"
export HF_HOME="$PWD/data/.hf-cache"
uv pip install --python "$PY" \
  'transformers==4.56.2' 'accelerate==1.10.1' 'huggingface-hub==0.35.1'

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; torch.cuda.set_device(0); print(torch.cuda.get_device_name(0))'
CANDIDATE_ID="P${SLURM_JOB_ID}"
OUTPUT="data/fred_generation_smoke/${SLURM_JOB_ID}"
"$PY" -m sota.FRED_LLM_GE.phase1_detection.seeds.yolo11.propose \
  --candidate-id "$CANDIDATE_ID" --output-dir "$OUTPUT"
"$PY" sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train.py \
  --model "network_${CANDIDATE_ID}" --variant_dir "$OUTPUT" \
  --data "$DATA" --weights "$PWD/yolo11m.pt" --preflight-only
printf 'Candidate source and YOLO11m construction passed preflight. No fitness was assigned.\n' \
  > "$OUTPUT/preflight_passed.txt"
echo "FRED unscored LLM candidate ready: $OUTPUT/network_${CANDIDATE_ID}.py"
