#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}

hostname

module load cuda
module load uv

# Use exactly the devices Slurm assigned to this server job.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

# Write hostname to file so tests can find the server
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"

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

# Use the project's configured port so the run's LLM jobs find this server.
PORT=$(uv run python -c "from src.cfg.constants import PORT; print(PORT)")
echo "Serving on port $PORT"
uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port $PORT --workers 1
