#!/usr/bin/env bash
# Submitted by run_yolo11_scratch_pilot.sh; uses the team's existing scheduler.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run through the pilot launcher from the repository root}"
module load cuda
module load uv
: "${YOLO11_PILOT_ROOT:?Missing isolated pilot directory}"
trainer=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_ultralytics.py
case "${1:-}" in
  baseline)
    uv run python "$trainer" --model network
    # Preserve baseline evidence before any later seed rerun.
    mkdir -p "$YOLO11_PILOT_ROOT/baseline"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/seed_evaluation.json "$YOLO11_PILOT_ROOT/baseline/"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/seed_results.csv "$YOLO11_PILOT_ROOT/baseline/"
    cp sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/trained_models/seed.pt "$YOLO11_PILOT_ROOT/baseline/"
    ;;
  evolve)
    test -s "$YOLO11_PILOT_ROOT/baseline/seed_results.csv"
    uv run python run_improved.py --checkpoints "$YOLO11_PILOT_ROOT/checkpoints" \
      --global_path "$YOLO11_PILOT_ROOT/global" --prompt_group FRED/Ultralytics
    test -s "$YOLO11_PILOT_ROOT/checkpoints/checkpoint_gen_2.pkl"
    echo "Pilot completed two generations; inspect candidate failures and lineage before assessing results."
    ;;
  *) echo "Expected baseline or evolve" >&2; exit 2 ;;
esac
