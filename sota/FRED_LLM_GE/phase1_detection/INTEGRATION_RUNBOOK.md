# FRED Event-only YOLO11 integration and stability checks

Work on `fred-yolo11-infrastructure` only. The combined integration was
committed by the user after incorporating `MosesTheRedSea-main` at
`3b48a878685fbb020598c3048797bb468e07330c`. The earlier uncommitted-merge patch
workflow is obsolete: use a reviewed commit on your branch and pull that branch
on ICE. Do not apply `fred-integration.patch` again.

## Existing engineering evidence

- Preparation job 6132562 installed the bounded dependencies and reused the export.
- Integration job 6132563 exercised the protected seed, detection forward pass,
  one-epoch training, checkpoint loading, and shared COCO evaluator.
- Learning job 6133680 trained for 50 epochs on 32 training and 32 validation
  images. It fitted the training subset but validation stayed weak. Its CSV
  contains NaN validation losses in epochs 21, 22, and 25–32, so its successful
  process exit is not evidence of stable training.

These are engineering diagnostics. Do not repeat the preparation or one-epoch
integration job just to test the precision change.

## Loss-history and precision controls

The canonical `seeds/yolo11/train_eval.py` supports `--amp` (the previous default)
and `--no-amp` (full precision). The selected option is recorded in the evaluation
JSON's `metadata.training_config` and its configuration hash.

The trainer checks all six training/validation box, classification, and DFL loss
columns after each recorded epoch and again before publishing a checkpoint or
fitness. NaN, infinity, missing history, malformed loss cells, or missing/duplicate
epochs exit nonzero with `FRED_RUN_FAILED [training_history]` and a failure JSON.
Raw history/checkpoints remain in the run directory for diagnosis; no checkpoint
is promoted to `trained_models` and no success report or fitness CSV is written.
A healthy final epoch cannot hide earlier invalid losses. Finite losses alone do
not establish convergence; their maximum values are retained for inspection.

The callback and history format were checked against Ultralytics v8.4.165:
[trainer CSV/callback ordering](https://github.com/ultralytics/ultralytics/blob/v8.4.165/ultralytics/engine/trainer.py)
and [precision configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.165/ultralytics/cfg/default.yaml).

## Full-precision stability diagnostic

`jobs/check_yolo11_stability.sbatch` repeats the learning diagnostic with AMP
disabled, retaining 50 epochs, batch 2, image size 640, full subset, seed 0,
pretraining, augmentations, optimizer defaults, and shared evaluator settings.
This tests a plausible source of instability; AMP is not yet a confirmed cause.

The job reuses `data/.venv-yolo11`, `data/fred_yolo11_bringup/data.yaml`, and
`yolo11m.pt`. It requires the existing `learning_check_6133680_evaluation.json`,
checks the previous manifest/provenance and seed identity, and refuses tracked
uncommitted changes. It does not download new data or upgrade dependencies.
It requests one GPU, eight CPUs, 24 GB RAM, and at most two hours. Record the
allocated GPU printed in the log when comparing runs.

After reviewing, committing, and pushing the changes to your branch yourself,
paste this in the existing ICE checkout:

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
git fetch origin fred-yolo11-infrastructure
git merge --ff-only origin/fred-yolo11-infrastructure
JOB=sota/FRED_LLM_GE/phase1_detection/jobs/check_yolo11_stability.sbatch
test -s "$JOB"
test -x data/.venv-yolo11/bin/python
test -s data/fred_yolo11_bringup/data.yaml
test -s yolo11m.pt
mkdir -p data/logs
JOB_ID=$(sbatch --parsable "$JOB")
JOB_ID=${JOB_ID%%;*}
printf 'Stability-check job: %s
Log: data/logs/fred-stability-%s.out
' "$JOB_ID" "$JOB_ID"
squeue -j "$JOB_ID"
BASH
```

Supply your usual PACE account/partition options to `sbatch` if required. Once
finished, replace `JOB_ID` below with the printed number:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 100 data/logs/fred-stability-JOB_ID.out
```

Success requires `STABILITY CHECK COMPLETE`, 50 finite epochs, and AMP disabled.
The log prints separate training and validation shared COCO scores. Preserve and
compare the loss maxima too: eliminating NaN does not establish healthy learning
or generalization. If this run still fails, inspect its first invalid epoch and
failure JSON before changing another training variable.

Artifacts use `stability_check_JOB_ID` to avoid overwriting prior runs:

- `seeds/yolo11/runs/stability_check_JOB_ID/`: raw history, plots, and checkpoints.
- `seeds/yolo11/trained_models/stability_check_JOB_ID.pt`: accepted checkpoint.
- `seeds/yolo11/results/stability_check_JOB_ID_evaluation.json`: shared validation
  report, precision configuration, loss maxima, and history hash.
- `seeds/yolo11/results/stability_check_JOB_ID_results.csv`: shared metric row.
- `data/fred_diagnostics/stability_check_JOB_ID.json`: separate train/validation
  comparison; training scores are not selection fitness.
- `seeds/yolo11/failures/stability_check_JOB_ID_failure.json`: rejection details
  when the trainer fails.

## Research boundary

The prepared-data exporter is `sota.FRED_LLM_GE.data.yolo_export`, using
`archive.phase0_data_old` as an explicit compatibility backend. The team's
streaming loader and fusion seed remain separate; this diagnostic does not
settle the authoritative loader decision.

The bounded 32/32 export must not supply research selection fitness. The frozen
split/leakage gate, controlled training/fitness protocol, validated baseline, and
complete evolution bridge/pilot still precede a research generation. Do not
start `run_improved.py` with this smoke dataset.
