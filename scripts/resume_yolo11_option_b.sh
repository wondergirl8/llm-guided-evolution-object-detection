#!/usr/bin/env bash
# Start a fresh LLM server and resume the saved option-B search when it is ready.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(git branch --show-current)" = yolo11-scratch-evolution
test -x .venv/bin/python || { echo 'Run uv sync first.' >&2; exit 1; }
auto=0
if [[ ${1:-} = --auto ]]; then
  auto=1
  shift
fi
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash scripts/resume_yolo11_option_b.sh [--auto] [RUN_FOLDER]' >&2; exit 2
fi
root=${1:-$(cat fred_ultralytics_yolo11_output/latest-option-b.txt)}
root=$(cd "$root" && pwd)
test -s "$root/run_config.json"
test -s "$root/baseline/seed_results.csv"
queue_ids=$(squeue -h -u "$USER" -o '%i')
job_active() {
  printf '%s\n' "$queue_ids" | awk -v wanted="$1" '$1 == wanted {found=1} END {exit !found}'
}
schedule_chain() {
  .venv/bin/python - "$root" "$1" "$2" <<'PY'
import sys
from pathlib import Path
from scripts.continue_yolo11_option_b import queue_continuation
queue_continuation(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
PY
}
if test -s "$root/chain-job.txt"; then
  chain=$(cat "$root/chain-job.txt")
  if [[ $chain != ${SLURM_JOB_ID:-} ]] && job_active "$chain"; then
    echo "Automatic continuation $chain is already queued or running; no duplicate submitted."
    exit 0
  fi
fi
if [[ $auto -eq 1 ]]; then
  test ! -e "$root/STOP-AUTO" || { echo 'STOP-AUTO exists; remove it to re-enable renewal.' >&2; exit 1; }
  target=$(.venv/bin/python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["LLMGE_GENERATIONS"])-1)' "$root/run_config.json")
  if test -s "$root/checkpoints/checkpoint_gen_$target.pkl"; then
    echo "Generation target $target already completed; no jobs submitted."
    exit 0
  fi
fi
active_job=''
for marker in controller-job.txt resume-job.txt; do
  if test -s "$root/$marker"; then
    job=$(cat "$root/$marker")
    if job_active "$job"; then
      if [[ $auto -eq 0 ]]; then
        echo "Job $job is still queued or running for this run; do not start a duplicate." >&2
        exit 1
      fi
      active_job=$job
    fi
  fi
done
if test -n "$active_job"; then
  server=$(cat "$root/server-job.txt")
  job_active "$server" || { echo 'Saved server is not active; inspect the current jobs.' >&2; exit 1; }
  touch "$root/AUTO-ENABLED"
  if ! schedule_chain "$server" "$active_job"; then
    rm -f "$root/AUTO-ENABLED"
    exit 1
  fi
  echo "Automatic renewal attached to this run. Current jobs continue."
  exit 0
fi
if test -n "$(squeue -h -u "$USER" -n LLMGE01_Server -o '%i')"; then
  echo 'An LLM server is already queued or running. Once ready, use run_yolo11_option_b.sh --resume with this run folder.' >&2
  exit 1
fi
mkdir -p run_job_outputs/server
export LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-4096}
export LLMGE_INITIALIZATION_BATCHES=${LLMGE_INITIALIZATION_BATCHES:-3}
export LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS=${LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS:-60}
server=$(SUBMIT_ISLAND_CONTROLLER=0 sbatch --parsable server.sh)
server=${server%%;*}
printf '%s\n' "$server" > "$root/server-job.txt"
# 'after' releases this job when the server starts, rather than when it finishes.
if ! resume=$(sbatch --parsable --job-name=yolo11-b-resume --dependency="after:$server" \
  --kill-on-invalid-dep=yes --cpus-per-task=2 --mem=4G --time=01:05:00 \
  --output="$root/resume-%j.log" scripts/yolo11_scratch_job.sh resume-when-ready "$root" "$server"); then
  scancel "$server"
  exit 1
fi
resume=${resume%%;*}
printf '%s\n' "$resume" > "$root/resume-job.txt"
if [[ $auto -eq 1 ]]; then
  touch "$root/AUTO-ENABLED"
  if ! schedule_chain "$server" "$resume"; then
    rm -f "$root/AUTO-ENABLED"
    scancel "$resume" "$server"
    exit 1
  fi
  printf 'Automatic renewal enabled until the saved generation target, a code failure or STOP-AUTO.\n'
fi
printf 'Server job: %s\nResume job: %s\nRun folder: %s\n' "$server" "$resume" "$root"
printf 'The resume job waits for model loading, then submits the controller using the saved baseline.\n'
printf 'Check: squeue -j %s,%s\nResume log: %s/resume-%s.log\n' "$server" "$resume" "$root" "$resume"
