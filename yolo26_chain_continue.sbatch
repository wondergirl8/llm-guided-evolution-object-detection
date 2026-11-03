#!/bin/bash
#SBATCH --job-name=yolo26-chain
#SBATCH --partition=pace-cpu
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=run_job_outputs/islands/yolo26-chain-%j.out

# Submit one more YOLO26-P2 server/controller pair only after a clean pause.
# Slurm dependencies are supplied by the caller; this script never runs two
# controllers for the same tag at once.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

previous_controller=${1:?previous controller job ID required}
previous_server=${2:?previous server job ID required}
remaining=${3:?remaining continuation windows required}
[[ $previous_controller =~ ^[0-9]+$ && $previous_server =~ ^[0-9]+$ && $remaining =~ ^[1-9][0-9]*$ ]]
[[ ${FRED_SEED_FAMILY:-} == yolo26_p2 && ${FRED_RUN_TAG:-} == yolo26_p2_* ]]
[[ ${SUBMIT_ISLAND_CONTROLLER:-} == 0 && ${FRED_NUM_GENERATIONS:-} == 21 ]]
[[ -n ${FRED_RUN_TAG:-} && -x .venv/bin/python && -s server.sh && -s run.sh ]]
chain_root="${FRED_CHAIN_ROOT:?chain output folder required}"
[[ $chain_root == "$PWD/fred_runs/$FRED_RUN_TAG/chain" ]]
[[ $(cat run_job_outputs/yolo26_current_tag.txt) == "$FRED_RUN_TAG" ]]
mkdir -p "$chain_root"

state=$(.venv/bin/python - "fred_runs/$FRED_RUN_TAG/status.json" "$previous_server" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    status = json.load(stream)
if status.get("status") == "complete" and status.get("last_completed_generation") == 20:
    print("complete")
elif status.get("status") == "paused_for_server_refresh" and str(status.get("server_job_id")) == sys.argv[2]:
    from pathlib import Path
    checkpoints = Path(sys.argv[1]).parent / "checkpoints"
    pairs = [p for p in checkpoints.glob("checkpoint_gen_*.pkl")
             if p.stat().st_size and (checkpoints / p.name.replace("checkpoint_", "global_")).is_file()
             and (checkpoints / p.name.replace("checkpoint_", "global_")).stat().st_size]
    if not pairs:
        raise SystemExit("Refusing to chain without a saved checkpoint pair")
    print("paused")
else:
    raise SystemExit(f"Refusing to chain from unexpected status: {status}")
PY
)
if [[ $state == paused ]]; then
  mkdir "$chain_root/handoff-$previous_controller" || {
    echo "Refusing duplicate handoff for controller $previous_controller" >&2
    exit 1
  }
fi
if [[ $state == complete ]]; then
  echo "YOLO26-P2 run $FRED_RUN_TAG reached generation 20; no further jobs submitted."
  exit 0
fi

# A server has an eight-hour allocation. The controller starts only after that
# server starts, then waits for its HTTP endpoint in run_improved.py.
server_submission=$(sbatch --parsable --partition=coe-gpu --export=ALL,SUBMIT_ISLAND_CONTROLLER=0 server.sh)
server_job=${server_submission%%;*}
[[ $server_job =~ ^[0-9]+$ ]]
printf '%s\n' "$server_job" > run_job_outputs/yolo26_current_server.id
printf 'server\t%s\t%s\n' "$server_job" "$previous_controller" >> "$chain_root/jobs.tsv"
echo "Continuation server: $server_job"

controller_submission=$(sbatch --parsable --dependency="after:${server_job}+2" \
  --export="ALL,LLMGE_SERVER_JOB_ID=$server_job" run.sh)
controller_job=${controller_submission%%;*}
[[ $controller_job =~ ^[0-9]+$ ]]
printf '%s\n' "$controller_job" > run_job_outputs/yolo26_current_controller.id
printf 'controller\t%s\t%s\n' "$controller_job" "$server_job" >> "$chain_root/jobs.tsv"
echo "Continuation controller: $controller_job"

if (( remaining > 1 )); then
  next_remaining=$((remaining - 1))
  chain_submission=$(sbatch --parsable \
    --dependency="afterok:${controller_job},afterany:${server_job}" \
    --export="ALL,LLMGE_SERVER_JOB_ID=$server_job" \
    --output="$chain_root/transition-%j.out" \
    yolo26_chain_continue.sbatch "$controller_job" "$server_job" "$next_remaining")
  chain_job=${chain_submission%%;*}
  [[ $chain_job =~ ^[0-9]+$ ]]
  printf 'transition\t%s\t%s\n' "$chain_job" "$controller_job" >> "$chain_root/jobs.tsv"
  echo "Next transition: $chain_job ($next_remaining more windows after this one)"
fi
