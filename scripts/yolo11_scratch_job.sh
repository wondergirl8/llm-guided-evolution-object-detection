#!/usr/bin/env bash
# Submitted by the YOLO11 launchers; uses the team's existing scheduler.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run through a YOLO11 launcher from the repository root}"
module load uv
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
trainer=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_ultralytics.py
case "${1:-}" in
  resume-when-ready)
    .venv/bin/python scripts/wait_yolo11_server.py "${3:?Missing server job}"
    bash scripts/run_yolo11_option_b.sh --resume "${2:?Missing run folder}"
    ;;
  prepare)
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search \
      --shard "${SLURM_ARRAY_TASK_ID:?Missing shard}" --num_shards 8
    ;;
  finalize)
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize
    ;;
  baseline)
    module load cuda
    : "${YOLO11_PILOT_ROOT:?Missing isolated run directory}"
    uv run python "$trainer" --model network
    # Preserve baseline evidence before any later seed rerun.
    mkdir -p "$YOLO11_PILOT_ROOT/baseline"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/seed_evaluation.json "$YOLO11_PILOT_ROOT/baseline/"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/seed_results.csv "$YOLO11_PILOT_ROOT/baseline/"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/trained_models/seed.pt "$YOLO11_PILOT_ROOT/baseline/"
    ;;
  evolve)
    : "${YOLO11_PILOT_ROOT:?Missing isolated run directory}"
    exec 9>"$YOLO11_PILOT_ROOT/controller.lock"
    flock -n 9 || { echo "Another controller owns this run." >&2; exit 1; }
    test -s "$YOLO11_PILOT_ROOT/baseline/seed_results.csv"
    uv run python run_improved.py --checkpoints "$YOLO11_PILOT_ROOT/checkpoints" \
      --global_path "$YOLO11_PILOT_ROOT/global" --prompt_group FRED/Ultralytics
    test -s "$YOLO11_PILOT_ROOT/checkpoints/checkpoint_gen_$((${LLMGE_GENERATIONS:-3} - 1)).pkl"
    echo "Evolution target completed; inspect candidate failures and lineage before assessing results."
    ;;
  *) echo "Expected prepare, finalize, baseline, evolve or resume-when-ready" >&2; exit 2 ;;
esac
