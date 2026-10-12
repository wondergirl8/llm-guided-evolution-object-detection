#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
set -euo pipefail
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}

hostname

module load cuda
module load uv

# Slurm selects the usable devices, which may be IDs or MIG UUIDs.
# Preserve its mask; 0,1 may refer to GPUs outside this allocation.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LLM_SERVER_EXPECTED_GPUS=2
echo "Slurm CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi -L || true
fi
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

# Write hostname to file so tests can find the server
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"
mkdir -p run_job_outputs/server
printf '%s\n' "$SERVER_HOSTNAME" > "run_job_outputs/server/hostname-${SLURM_JOB_ID}.log"

# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER:-<not set>}"

# Default behavior: START island controller unless explicitly disabled
if [ "${SUBMIT_ISLAND_CONTROLLER:-1}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER})"
fi

# Reserve a usable port before model loading; readiness passes it to all clients.
exec uv run python scripts/launch_yolo11_server.py
