#!/usr/bin/env bash
#SBATCH --job-name=frcnn_llm_server
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C H200
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --output=run_job_outputs/server/frcnn-server-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
module load uv
export LLMGE_DETECTOR=faster_rcnn
export CUDA_VISIBLE_DEVICES=0,1
export SERVER_HOSTNAME="$(hostname)"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"
printf '%s\n' "$SERVER_HOSTNAME" > fred_faster_rcnn_hostname.log
exec uv run python -m uvicorn server:app --host "$SERVER_HOSTNAME" --port "${LLMGE_PORT:-8294}" --workers 1
