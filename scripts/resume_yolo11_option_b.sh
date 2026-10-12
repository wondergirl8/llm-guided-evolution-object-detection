#!/usr/bin/env bash
# Start a fresh LLM server and resume the saved option-B search when it is ready.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(git branch --show-current)" = yolo11-scratch-evolution
test -x .venv/bin/python || { echo 'Run uv sync first.' >&2; exit 1; }
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash scripts/resume_yolo11_option_b.sh [RUN_FOLDER]' >&2; exit 2
fi
root=${1:-$(cat fred_ultralytics_yolo11_output/latest-option-b.txt)}
root=$(cd "$root" && pwd)
test -s "$root/run_config.json"
test -s "$root/baseline/seed_results.csv"
for marker in controller-job.txt resume-job.txt; do
  if test -s "$root/$marker"; then
    job=$(cat "$root/$marker")
    if test -n "$(squeue -h -j "$job" -o '%i')"; then
      echo "Job $job is still queued or running for this run; do not start a duplicate." >&2
      exit 1
    fi
  fi
done
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
  --kill-on-invalid-dep=yes --cpus-per-task=2 --mem=4G --time=00:45:00 \
  --output="$root/resume-%j.log" scripts/yolo11_scratch_job.sh resume-when-ready "$root" "$server"); then
  scancel "$server"
  exit 1
fi
resume=${resume%%;*}
printf '%s\n' "$resume" > "$root/resume-job.txt"
printf 'Server job: %s\nResume job: %s\nRun folder: %s\n' "$server" "$resume" "$root"
printf 'The resume job waits for model loading, then submits the controller using the saved baseline.\n'
printf 'Check: squeue -j %s,%s\nResume log: %s/resume-%s.log\n' "$server" "$resume" "$root" "$resume"
