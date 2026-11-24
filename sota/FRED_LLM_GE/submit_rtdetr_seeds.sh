#!/bin/bash
# Submit RT-DETR final eval for seeds 30 down to 16.
# Each job: H100 or H200, 16h wall, 16G RAM, 20 epochs (from protocol.yaml).
# Logs: logs/rtdetr_s<seed>_<jobid>.{out,err}
# Reports: phase1_detection/seeds/yolo11/data_cache/challenging/held_out_reports/rtdetr_r18_seed_s<seed>_evaluation.json

set -euo pipefail

cd "/home/hice1/rsebek3/scratch/llm-guided-evolution-object-detection/sota/FRED_LLM_GE"
mkdir -p logs

for seed in $(seq 30 -1 16); do
    name="rtdetr_r18_seed_s${seed}"
    report="phase1_detection/seeds/yolo11/data_cache/challenging/held_out_reports/${name}_evaluation.json"

    # Skip seeds whose report already exists (safe to rerun this script)
    if [[ -f "$report" ]]; then
        echo "skip seed ${seed}: report exists"
        continue
    fi

    jobid=$(sbatch --parsable \
        --job-name="rtdetr_s${seed}" \
        --nodes=1 --ntasks=1 --cpus-per-task=12 \
        --gres=gpu:H100:1 -C "H100|H200" \
        --time=16:00:00 --mem=16G \
        --output="logs/rtdetr_s${seed}_%j.out" \
        --error="logs/rtdetr_s${seed}_%j.err" \
        --wrap="uv run python -m phase1_detection.final_eval --family rtdetr --name ${name} --seed ${seed}")

    echo "queued seed ${seed}: job ${jobid}"
done

echo "done. check with: squeue -u \$USER"