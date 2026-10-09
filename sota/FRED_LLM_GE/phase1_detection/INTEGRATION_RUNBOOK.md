# Combined FRED Event-only YOLO11 integration check

The local `fred-yolo11-infrastructure` checkout contains a resolved, uncommitted
merge of `MosesTheRedSea-main` at `3b48a878685fbb020598c3048797bb468e07330c`.
No commit, push, or PR was made. The original HEAD is
`ac33341b2e72346f176bb3b730d0a691e7a4f93a`.

The attached binary patch applies this exact combined source to an ICE checkout
at that original HEAD without creating a commit. Do not pull or reset the local
checkout while its merge is pending. A later PR requires your reviewed commit;
resolving local conflicts does not establish a published PR's mergeability.

## What this check runs

1. A CPU preparation job installs the bounded-run dependencies into
   `data/.venv-yolo11`. It reuses and validates the existing labeled export and
   `yolo11m.pt`; only missing data/checkpoint artifacts are prepared/downloaded.
2. One GPU job validates the protected Event-only seed, constructs YOLO11m,
   performs a detection forward pass, trains for one epoch on the entire bounded
   export, and evaluates the resulting checkpoint using the team's COCO reporter.
3. Success requires a checkpoint, evaluation JSON, and matching three-column
   result CSV. Failure exits nonzero and publishes no success fitness.

Existing proposal job 6009285 and candidate training job 6009377 do not need to
be repeated. This run checks the newly combined trainer/evaluator path.

The prepared-data exporter now lives at `sota.FRED_LLM_GE.data.yolo_export`,
using `archive.phase0_data_old` as an explicit compatibility backend. The main
branch's streaming loader and fusion seed remain separate. This integration
does not settle the team's authoritative loader decision.

## Upload and submit on ICE

Upload `fred-integration.patch` to the **root of your existing ICE repository**.
Open a terminal in that repository. Paste this complete block:

```bash
bash <<'BASH'
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
test "$(git rev-parse HEAD)" = ac33341b2e72346f176bb3b730d0a691e7a4f93a
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
test -s fred-integration.patch
git apply --check fred-integration.patch
git apply fred-integration.patch
mkdir -p data/logs
PREP_ID=$(sbatch --parsable sota/FRED_LLM_GE/phase1_detection/jobs/prepare_yolo11_integration.sbatch)
PREP_ID=${PREP_ID%%;*}
RUN_ID=$(sbatch --parsable --dependency="afterok:$PREP_ID" sota/FRED_LLM_GE/phase1_detection/jobs/check_yolo11_integration.sbatch)
RUN_ID=${RUN_ID%%;*}
printf 'PREPARATION JOB: %s\nGPU CHECK JOB: %s\n' "$PREP_ID" "$RUN_ID"
printf 'Logs: data/logs/fred-integration-prepare-%s.out and data/logs/fred-integration-%s.out\n' "$PREP_ID" "$RUN_ID"
squeue -j "$PREP_ID,$RUN_ID"
BASH
```

The block stops before modifying source if the branch, HEAD, tracked work, pending
merge, or patch check differs. It does not commit or push. If submission fails
after the patch applied, do not reapply it: inspect the message, then submit the
two jobs from the `mkdir` step. If preparation fails, the dependent GPU job does
not start; inspect its log and cancel that waiting GPU job with `scancel <id>`.

The check requests one generic GPU for at most two hours. Supply your normal
PACE account/partition options to both `sbatch` calls if your checkout's scheduler
configuration requires them. Source is validated locally; ICE scheduling and
actual GPU execution require the submitted-job output.

After completion, replace the IDs below with the printed values:

```bash
sacct -j PREPARATION_ID,GPU_CHECK_ID --format=JobID,State,ExitCode,Elapsed
tail -n 60 data/logs/fred-integration-prepare-PREPARATION_ID.out
tail -n 100 data/logs/fred-integration-GPU_CHECK_ID.out
```

Paste those outputs back for verification. The GPU log must end with the
checkpoint/report/CSV paths and `ENGINEERING CHECK COMPLETE`.

## Research boundary

This is an engineering integration run, not a formal baseline or an evolutionary
generation. Its bounded 32/32 export must not supply selection fitness. The frozen
split/leakage gate, controlled training/fitness protocol, validated non-evolutionary
baseline, and complete evolution bridge/pilot still precede a research generation.
Do not start `run_improved.py` with this smoke dataset.
