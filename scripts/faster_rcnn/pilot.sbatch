#!/usr/bin/env bash
#SBATCH --job-name=frcnn_pilot
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_job_outputs/islands/frcnn-pilot-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load uv
export LLMGE_DETECTOR=faster_rcnn
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"
# Stop only the server owned by this launch when orchestration ends.
cleanup() {
    .venv/bin/python scripts/faster_rcnn/stop_server.py
}
trap cleanup EXIT
# Start only after the dependency baseline succeeded, and after server readiness.
.venv/bin/python scripts/faster_rcnn/wait_server.py
uv run python run_improved.py --checkpoints fred_faster_rcnn_checkpoints \
    --global_path fred_faster_rcnn_checkpoints --prompt_group FRED_FasterRCNN/Normal
