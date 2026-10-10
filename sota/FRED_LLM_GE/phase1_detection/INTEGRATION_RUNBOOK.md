# FRED Event-only YOLO11 integration and stability checks

Work on `fred-yolo11-infrastructure` only. The combined integration was
committed by the user after incorporating `MosesTheRedSea-main` at
`3b48a878685fbb020598c3048797bb468e07330c`. The earlier uncommitted-merge patch
workflow is obsolete: use a reviewed commit on your branch and pull that branch
on ICE. Do not apply `fred-integration.patch` again.

## Explicit configuration selection

`src/cfg/constants.py` defaults to FRED YOLO11 on this branch, as requested.
Set `LLMGE_CONFIG=mujoco` **before starting Python** to select the team's
unchanged MuJoCo profile. Explicit `LLMGE_CONFIG=fred-yolo11` also works.
The selection applies to both `cfg.constants`
and `src.cfg.constants`, and is inherited by child processes. Unknown or empty
profile names fail immediately. Do not replace the shared file with a FRED
symlink. This selects configuration only; it does not complete the evolution
bridge or authorize a research generation.

The default is encoded in the shared file, not inferred from the Git branch.
Merging this file into another branch would also make FRED its default; review
that choice with the team as part of any future merge.

The standalone YOLO11 seed/trainer and existing diagnostic jobs are independent
of this selector and keep their explicit training arguments.

After manually committing/pushing this change, paste the following in the ICE
checkout. This is a CPU configuration check; it submits no GPU job and does not
repeat completed training.

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
git fetch origin fred-yolo11-infrastructure
git merge --ff-only origin/fred-yolo11-infrastructure
CHECK=sota/FRED_LLM_GE/phase1_detection/jobs/check_yolo11_configuration.sh
test -s "$CHECK"
bash "$CHECK"
BASH
```

Success prints `PROFILE OK: mujoco`, `PROFILE OK: fred-yolo11`,
`DEFAULT OK: fred-yolo11`, and `CONFIGURATION CHECK COMPLETE`.

## Existing engineering evidence

- Preparation job 6132562 installed the bounded dependencies and reused the export.
- Integration job 6132563 exercised the protected seed, detection forward pass,
  one-epoch training, checkpoint loading, and shared COCO evaluator.
- Learning job 6133680 trained for 50 epochs on 32 training and 32 validation
  images. It fitted the training subset but validation stayed weak. Its CSV
  contains NaN validation losses in epochs 21, 22, and 25–32, so its successful
  process exit is not evidence of stable training.
- Full-precision job 6134619 completed all 50 epochs with finite losses. Shared
  validation mAP50 improved to 0.59250 and mAP50:95 to 0.18361, but classification
  loss spiked to 460,444,000 early in training. Finite status alone does not resolve
  the remaining instability. Its actual auto-selected optimizer was AdamW at
  LR 0.002, not the unused `lr0: 0.01` shown in `args.yaml`.
- Explicit-AdamW job 6135264 used LR 0.0002, beta1 0.9, bias warmup LR 0,
  and AMP disabled. All 50 epochs had finite losses; maximum validation
  classification loss was 10.7165. The downloaded CSV's SHA-256 matched the
  evaluation report (`e80f2eb1c085bff2e4a94b5737071f4d9ca28184478750e90b9922f39080a638`).
  Shared validation mAP50 was 0.63880 and mAP50:95 was 0.25940; training scores
  were 1.00000 and 0.66215. This resolved the observed loss blowup in this run.
  Validation metrics peaked before epoch 50 while training losses kept falling,
  so preserve the best checkpoint rather than treating the last epoch as best.

These are engineering diagnostics. Do not repeat the preparation or one-epoch
integration job, or rerun the completed tiny-data checks without a new question.

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

## Controlled learning-rate diagnostic

Supply one positive finite learning rate to the same job to repeat the
full-precision run with explicit AdamW. The initial trial is `0.0002`, ten times
lower than the actual rate used in job 6134619. This is a diagnostic hypothesis,
not a frozen formal training protocol or a guarantee of improved accuracy.

The trainer's optional `--adamw-lr` sets AdamW, its initial learning rate,
`momentum=0.9` (AdamW beta1), and `warmup_bias_lr=0.0`. Those latter settings
preserve the pinned automatic AdamW behavior; explicitly selecting AdamW without
them would also change beta1 and bias warmup. The schedule, other warmup settings,
data, batch size, seed, augmentations, and evaluation stay as in the previous run.
Without the option, the trainer retains `optimizer=auto` behavior.
The [pinned optimizer implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.165/ultralytics/engine/trainer.py#L1080-L1089)
documents why changing `lr0` alone while keeping `auto` would be ignored.

After manually committing/pushing on your branch and pulling it on ICE, submit:

```bash
mkdir -p data/logs
sbatch --parsable --job-name=fred-lr-check \
  sota/FRED_LLM_GE/phase1_detection/jobs/check_yolo11_stability.sbatch 0.0002
```

This mode compares against `stability_check_6134619_evaluation.json` and uses
`learning_rate_check_JOB_ID` for every run artifact. Logs still use
`data/logs/fred-stability-JOB_ID.out`. No argument retains the original
full-precision diagnostic mode and `stability_check_JOB_ID` names.
Each diagnostic report records its comparison run, requested change, optimizer
settings, loss maxima, and separate training/validation scores. Inspect both
early loss behavior and final scores before adopting a larger-data run.

Artifacts below use `stability_check_JOB_ID` for the original no-argument mode;
substitute `learning_rate_check_JOB_ID` for the lower-LR mode:

- `seeds/yolo11/runs/stability_check_JOB_ID/`: raw history, plots, and checkpoints.
- `seeds/yolo11/trained_models/stability_check_JOB_ID.pt`: accepted checkpoint.
- `seeds/yolo11/results/stability_check_JOB_ID_evaluation.json`: shared validation
  report, precision configuration, loss maxima, and history hash.
- `seeds/yolo11/results/stability_check_JOB_ID_results.csv`: shared metric row.
- `data/fred_diagnostics/stability_check_JOB_ID.json`: separate train/validation
  comparison; training scores are not selection fitness.
- `seeds/yolo11/failures/stability_check_JOB_ID_failure.json`: rejection details
  when the trainer fails.

## Resume the incomplete scene audit

ICE array 6092655 left 136 of 172 scene records in
`data/fred_scene_audit/v1_ac33341b2e72`. Six shards failed with
`OUT_OF_MEMORY` at the 16 GB allocation; shards 2 and 3 completed. The current
input/collector context differs from those saved records only in `code_revision`.
The original collector, source configuration, and split membership are unchanged.

`phase0_data/jobs/resume_development_scenes.sbatch` is a serial CPU job with
48 GB memory and a four-hour limit. It imports the archived compatibility backend
explicitly; this does not select a team-wide loader. It verifies all saved record
contexts and thumbnail hashes before staging a new job-specific output directory,
then uses the unchanged collector to skip the staged records and collect only
missing sequences. Source artifacts remain unchanged. Reuse fails if any recorded
input hash, collector source hash, or sampling policy differs; only a Git revision
difference is permitted. More memory addresses the observed allocation failure,
but successful remote completion still needs verification.

Reused records contain the current verification `context`, an original
`producing_context`, and `reused_from` source path/hash/context. The
`resume_receipt.json` lists the reused records and the resume wrapper's source hash.
Fresh records retain the current producing context. Do not rewrite the original
records' Git revisions to make a resume pass.

After manually committing/pushing and pulling this change on ICE, submit:

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
git fetch origin fred-yolo11-infrastructure
git merge --ff-only origin/fred-yolo11-infrastructure
JOB=sota/FRED_LLM_GE/phase0_data/jobs/resume_development_scenes.sbatch
SOURCE=data/fred_scene_audit/v1_ac33341b2e72
test -s "$JOB" && test -d "$SOURCE"
test -x data/.venv-yolo11/bin/python
mkdir -p data/logs
JOB_ID=$(sbatch --parsable --partition=pace-cpu --account=isye "$JOB" "$SOURCE")
JOB_ID=${JOB_ID%%;*}
printf 'Scene-resume job: %s\nLog: data/logs/fred-scene-resume-%s.out\n' "$JOB_ID" "$JOB_ID"
squeue -j "$JOB_ID"
BASH
```

Success requires `SCENE AUDIT RESUME COMPLETE: 172/172`, scheduler exit `0:0`,
and `data/fred_scene_audit/resume_JOB_ID/summary.json` with `complete: true`.
Download that summary, `resume_receipt.json`, and the overview pages for manual
scene-group review. Completion does not freeze a split or establish absence of
leakage. If interrupted, a later job can use its partially completed resume
directory as the source; verified records and their original producing contexts
remain reusable across another unrelated commit.

## Approved working scene split

Bill approved the scene-group proposal for infrastructure work on 2026-10-10.
The decision is recorded in
`docs/decisions/DG-P0-02-scene-groups-infrastructure-v1.md`; exact membership is
`configs/phase0/project_split_scene_groups_infrastructure_v1.json`.
It assigns 129 sequences to training and 43 to validation (the complete skyline
and arched-hall groups), preserving all 59 official held-out test IDs. Every
reviewed group stays wholly on one side. Uncertain outdoor connections are
conservatively merged. Recording-session independence remains unverified.

This is the working split for new infrastructure development. Preserve the
previous split, manifest, 32/32 export and training runs as evidence of their
original conditions. New manifest builds must pass the new JSON explicitly with
`--project-split` and use fresh output/export paths; old validation sequence 8 now
belongs to training, so the old export cannot represent the new split. Archived
audit/resume defaults deliberately retain the original split for reproduction.
The manifest's `BRINGUP` approval reference preserves non-freeze status under
the approved annotation policy. This does not select the team's loader or settle
the formal baseline/evolution protocol.

On the Mac, review the change and manually commit/push these eight files on
`fred-yolo11-infrastructure`:

```bash
bash <<'BASH'
set -euo pipefail
cd /Users/billnguyen/Documents/llm-guided-evolution-object-detection
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git add -- \
  sota/FRED_LLM_GE/archive/phase0_data_old/splits.py \
  sota/FRED_LLM_GE/configs/phase0/project_split_scene_groups_infrastructure_v1.json \
  sota/FRED_LLM_GE/docs/decisions/DG-P0-02-scene-groups-infrastructure-v1.md \
  sota/FRED_LLM_GE/artifacts/phase0/known_data_issues.yaml \
  sota/FRED_LLM_GE/phase0_data/check_scene_split.py \
  sota/FRED_LLM_GE/phase0_data/test_scene_split.py \
  sota/FRED_LLM_GE/phase0_data/jobs/check_scene_split.sh \
  sota/FRED_LLM_GE/phase1_detection/INTEGRATION_RUNBOOK.md
git commit -m "Adopt approved FRED infrastructure scene-group split"
git push origin fred-yolo11-infrastructure
BASH
```

Then paste this in the existing ICE repository. This is a quick read-only check
using the completed audit, not a Slurm/data-collection/training job:

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
git fetch origin fred-yolo11-infrastructure
git merge --ff-only origin/fred-yolo11-infrastructure
bash sota/FRED_LLM_GE/phase0_data/jobs/check_scene_split.sh
BASH
```

Success prints `SCENE SPLIT CHECK COMPLETE: 129 train / 43 validation; 59 held out;
no training submitted`. The checker validates the existing official-manifest
hash/membership, approved groups, saved summary/receipt hashes, every record and
thumbnail hash, and reviewed frame totals. Audit records retain their original
split labels and producing contexts. It never rewrites them to match the new
membership. Missing or changed evidence fails visibly. Keep this result with the
completed audit before preparing a fresh larger-data export.

## Bounded overnight scene diagnostic

The user's 2026-10-10 request authorizes an unattended infrastructure diagnostic,
not formal baseline/evolution execution. The supplied `Object Detection VIP
(5).pdf`, page 3, September 28 notes, states a 32 GPU-hour volume across jobs and
a maximum queue of 50 jobs. Its SHA-256 is
`b460cff1e8cc3e9ae77c690780d06a0b10a7180366337751bf27e43dea591a14`.
The submission helper conservatively sums **full requested time limits times
total GPUs** for all the user's existing queued/running jobs, including dependency
jobs. Generic and model-specific GPU TRES are counted once. CPU jobs count toward
the queue envelope but consume zero GPU-hours. An unknown/unbounded GPU request,
an existing array that cannot be counted safely, or insufficient headroom stops
submission visibly. This is a queue-reservation check, not a historical usage or
team-wide allocation report. Slurm remains authoritative for live QOS/account rules.

The workflow adds exactly two jobs:

- CPU preparation: at most 2 hours, 4 CPUs, 48 GiB, `pace-cpu`; no GPU request.
- GPU training/evaluation: at most 2 hours, 8 CPUs, 24 GiB, one GPU, `coe-gpu`.

Both use the `isye` account and `coe-ice` QOS seen in completed user jobs. Both
requests must pass `sbatch --test-only` before either is submitted. Thus new
GPU reservation is only **2 GPU-hours**, and new requested runtime is at most
4 hours serially; queue waiting can extend elapsed time. This is a maximum,
not a prediction or an instruction to keep GPUs idle until the cap.

GPU execution requires CPU success (`afterok`) and waits for existing GPU jobs
to finish (`afterany`). An invalid dependency cancels the GPU job; a GPU
submission rejection cancels the just-submitted CPU job. No job array, LLM
server, evolutionary population or automatic retry/resubmission is launched.
An already queued/running overnight workflow is rejected to avoid duplicates.
Keep the repository at the submitted revision overnight: each job refuses
different HEAD, tracked changes, another branch or a merge in progress.

Preparation checks the completed scene evidence, then considers at most three
sequences per group in numeric order (courtyard prefers previously validated
sequence 3). It preserves each new inspection report and any rejected source
findings. Transport/incomplete/unexpected failures stop rather than being
silently skipped. Exactly one valid representative is required per group; no
held-out test archive is inspected. This bounded expansion covers the five
reviewed scenes, **not all 172 development sequences**.

Using the existing prepared-data compatibility backend, it builds a separate
manifest and validation report, then exports exactly 512 evenly spaced
annotated frames per representative. This produces 1,536 training images
(three training groups) and 1,024 validation images (two validation groups).
Frame eligibility remains annotation-only and is not a balanced background
sampling protocol. Insufficient eligible frames stop instead of reducing the
budget. Approved DG-P0-04 conversions preserve raw-box lineage. Old manifests,
exports, reports and training runs are preserved. The existing bounded source
cache may download/evict pinned source objects while preparing new sequences;
it is not a full 205 GB dataset download. Shared-filesystem free space does not
establish the user's personal quota; `pace-quota` remains the relevant check.

Training uses the protected YOLO11m seed, existing pretrained weights and pinned
Ultralytics 8.4.165 environment: 50 epochs, batch 2, image size 640, seed 0,
fraction 1, explicit AdamW LR 0.0002, AMP disabled. It verifies the seed against
job 6135264's receipt, checks all losses, preserves best/last checkpoints and
requires the full epoch history before publishing a success summary. The
existing evaluator records validation mAP50/mAP50:95 and parameter count. These
are engineering results on a new dataset, not directly comparable scores to
the old 32/32 run and not evolutionary selection fitness. A Slurm timeout leaves
partial logs/run artifacts for diagnosis; it cannot produce a success summary.

After manually committing/pushing this change on the Mac, paste into ICE:

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
git fetch origin fred-yolo11-infrastructure
git merge --ff-only origin/fred-yolo11-infrastructure
if command -v pace-quota >/dev/null 2>&1; then pace-quota; fi
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase1_detection.submit_scene_overnight
BASH
```

Save the printed two job IDs and output directory. Outputs are in a fresh
`data/fred_overnight/scene_*/` directory, including `submission.json`, `jobs.json`,
new inspection records, `manifest_validation.json`, `manifest.sqlite`, `export/`
and (only after success) `summary.json`. GPU run artifacts remain under
`phase1_detection/seeds/yolo11/runs/scene_check_JOBID`, `results/` and
`trained_models/`. Logs are `data/logs/fred-night-data-CPU_JOBID.out` and
`data/logs/fred-night-gpu-GPU_JOBID.out`. You may disconnect after the submission
receipt; Slurm executes independently of the terminal. In the morning inspect
`sacct -j CPU_JOBID,GPU_JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES`,
the two logs and the summary. Complete success prints
`OVERNIGHT SCENE CHECK COMPLETE; engineering diagnostic only`.

## Research boundary

The prepared-data exporter is `sota.FRED_LLM_GE.data.yolo_export`, using
`archive.phase0_data_old` as an explicit compatibility backend. The team's
streaming loader and fusion seed remain separate; this diagnostic does not
settle the authoritative loader decision.

The bounded 32/32 export must not supply research selection fitness. The frozen
split/leakage gate, controlled training/fitness protocol, validated baseline, and
complete evolution bridge/pilot still precede a research generation. Do not
start `run_improved.py` with this smoke dataset.

The next data expansion uses the approved working scene-group membership above;
the final research freeze and larger-data protocol remain separate decisions.
Treat LR 0.0002 and full precision as the working engineering
settings supported by job 6135264, not a frozen DG-P1-04 decision. DG-P1-04/05/06
still precede formal baseline execution. Configuration-check success alone is
not evidence that the entire branch is ready to merge or that evolution works.
