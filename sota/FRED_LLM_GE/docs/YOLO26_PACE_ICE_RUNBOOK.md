# YOLO26-P2 LLM-GE on PACE ICE

This is the copy-and-paste guide for a **YOLO26-P2 evolutionary search** and
its results. The search controller evaluates the stock seed as part of
generation 0; these commands do not launch a separate seed-only experiment.
For the shared YOLO26/D-FINE protocol and optional held-out test evaluation,
see [CUSTOM_SEEDS_LLM_GE_RUN.md](CUSTOM_SEEDS_LLM_GE_RUN.md).

Commands in this guide run in the **PACE SSH terminal**, from the repository
root, unless a section explicitly says **Mac terminal**. Slurm allocates the
requested CPUs and GPUs. An OnDemand interactive allocation is not needed.
The server's eight-hour limit is one execution window, not a guarantee that
all 21 generations finish. `tail -F` only displays a log; press Ctrl-C to
leave `tail` without stopping any Slurm job.

**Active-tag warning:** This document was created after trial 2 began. Do not
`git pull` it into the PACE checkout merely to read it while resuming trial 2:
even a docs-only commit changes the Git commit in that run's fingerprint.
Read it on the Mac or copy it to PACE without changing the run checkout's
commit.

## Start here after logging in

Choose **one** path before pasting launch commands:

- **New experiment:** sections 1, 2 (use the common exports, then let section 3
  assign a new tag), 3, and 5. Confirm the dataset, model path, generated
  scripts, and server readiness at the indicated stops. Section 3 clears the
  old trial-2-only time margin. This starts at generation 0, including the
  stock seed.
- **Resume trial 2:** sections 1, 2 (existing trial-2 block), 4, and 5. Use
  the **current PACE** job and checkpoint evidence. The IDs and generation
  recorded in this document are historical examples. Do not launch a second
  controller if one is still active.

These are ordered command blocks with checks between them, not one block to
paste all at once. After section 5, use section 6 to monitor, section 7 to
**generate** search results and Pareto plots, section 8 to copy them, and
section 9 for the separate held-out final evaluation. Section 10 copies that
evaluation to the Mac. The search does **not** automatically run the plotter or
final evaluation. A stopped server does not by itself mean the controller
stopped; inspect both jobs before restarting.

If this file is only on the Mac, read it there while using the PACE terminal.
To put a **reference copy** on PACE without changing the experiment checkout,
run this in a **Mac terminal** with the VPN connected (from this repository's
root):

```bash
scp sota/FRED_LLM_GE/docs/YOLO26_PACE_ICE_RUNBOOK.md \
  agupta3057@login-ice.pace.gatech.edu:~/YOLO26_PACE_ICE_RUNBOOK.md
```

### What happened in the October trial

- Trial 1 failed before completing generation 0 because the stock seed was
  incorrectly checked as an edited model. Commit `863282e61` fixed that; do
  not combine trial 1 with trial 2.
- Trial 2 completed generations 0–3, then 4–6, but its controller crashed
  twice when `squeue` returned `MM:SS` time left. The parser fix and its tests
  are required in the code for a **new** run. The old trial 2 checkpoint is
  tied to its original commit and cannot use that fix for resume.
- Two `retry_exhausted` candidates visible near generation 6 were individual
  proposal failures. The controller recorded and replaced failed candidates;
  they were not the controller crash. The available local trial-2 copy has no
  operator or evaluation logs, so their exact rejection causes are unverified.
- A server submission once waited with `BadConstraints`; explicit
  `--partition=coe-gpu` resolved that submission. The downloaded plot and CSV
  files show results through generation 8. The copied `status.json` says
  `paused_for_server_refresh`, which is a pause rather than proof of a
  completed 21-generation search.

Valid LLM proposals and candidate training are stochastic. A future run can
still have rejected candidates or cluster failures; the controller must record
them and refill rather than treat failed candidates as successful scores.

## 1. Connect and identify the run

On the Mac, connect to the GT VPN and then:

```bash
ssh agupta3057@login-ice.pace.gatech.edu
```

On PACE (change this path only if the checkout has moved):

```bash
cd /storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
module load uv
readlink src/cfg/constants.py
git rev-parse --short HEAD
git status --short
```

`readlink` should show `constants_fred_yolo11.py`. **Before resuming an
existing tag, do not pull code, run `uv sync`, or regenerate the Slurm scripts.**
The checkpoint fingerprint includes the Git commit, controller and adapter
files, prompts, protocol, search manifests, dependency versions, and generated
`slurm-config/slurm_config.yaml`. The exact experiment must remain fixed.

List recent YOLO26 tags and the latest checkpoint **pair** in each. This is a
read-only inventory; a pair at generation 0 can still be provisional.

```bash
uv run python - <<'PY'
from pathlib import Path
import json
import re

roots = sorted((p for p in Path('fred_runs').glob('yolo26_p2_*') if p.is_dir()),
               key=lambda p: p.stat().st_mtime, reverse=True)
for root in roots[:10]:
    status_file = root / 'status.json'
    try:
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
    except (OSError, ValueError):
        status = {'status': 'unreadable'}
    checkpoints = root / 'checkpoints'
    paired = []
    for file in checkpoints.glob('checkpoint_gen_*.pkl'):
        match = re.fullmatch(r'checkpoint_gen_(\d+)\.pkl', file.name)
        if match and (checkpoints / f'global_gen_{match.group(1)}.pkl').is_file():
            paired.append(int(match.group(1)))
    print(root.name, 'status=', status.get('status', 'missing'),
          'last_completed=', status.get('last_completed_generation', '?'),
          'latest_pair=', max(paired) if paired else 'none')
PY
```

The known YOLO26 tag from the October 2026 session is
`yolo26_p2_20261023_eot0_trial2`. At the end of that chat, controller
`6515608` had loaded the generation-3 pair and started generation 4. **That
is a historical observation, not a current job-status check.** The earlier
`yolo26_p2_20261022_eot0_trial1` failed before a valid generation 0; do not
interpret it as a completed search.

## 2. Set the variables for a chosen tag

For the **existing trial 2**, use this block in every new PACE shell:

```bash
export FRED_SEED_FAMILY=yolo26_p2
export FRED_RUN_TAG=yolo26_p2_20261023_eot0_trial2
export FRED_NUM_GENERATIONS=21
export LLMGE_PORT=8479 LLMGE_CLUSTER=pace-ice
export LOCAL=false LOCAL_LLM=true INFERENCE_SUBMISSION=false
export LLMGE_LLM_MODEL=mixtral SUBMIT_ISLAND_CONTROLLER=0
export HF_HUB_DISABLE_XET=1
export LLMGE_MIN_SERVER_REMAINING_SECONDS=7200
unset FRED_EVENTS LLMGE_BOUNDED_SESSION LLMGE_MODEL_PATH
unset LLM_MAX_NEW_TOKENS LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS
```

For a **new experiment**, use the same settings but choose a **new, unused**
`FRED_RUN_TAG` matching `yolo26_p2_YYYYMMDD_eot0_trialN`. Record it, the
commit, port, and every server/controller job ID. Never change the tag while
resuming one experiment. Before a fresh run, verify that both search exports
exist and that their finalized counts were 10,053 training and 6,733
validation frames:

```bash
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_train_step50/data.yaml && echo 'train export OK'
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_val_step10/data.yaml && echo 'validation export OK'
grep -F 'search_train_step50: 152 sequences, 10053 frames' run_job_outputs/export/finalize-search.log
grep -F 'search_val_step10: 20 sequences, 6733 frames' run_job_outputs/export/finalize-search.log
```

If the export is missing, complete **section 1** of
[CUSTOM_SEEDS_LLM_GE_RUN.md](CUSTOM_SEEDS_LLM_GE_RUN.md) before launching.
The search export was already completed for the October 2026 trial; do not
submit it again merely to resume that tag.

## 3. Start a new YOLO26 search

Use this section **only for a new tag**, after the dataset checks. Make sure
no other family is using this checkout: `hostname.log`, `run.sh`, `server.sh`,
and the generated Slurm configuration are shared. The existing independent
baseline evidence should be documented before treating evolution as a formal
research comparison; the generation-0 seed point is only the search reference.

This fresh-run path requires commit `ad3e60f03` or a descendant, which
contains the `MM:SS` parser fix and its regression tests. Pull **only after**
deciding that no older tag needs to resume from this checkout; a changed Git
commit invalidates that tag's checkpoint fingerprint. Then synchronize
dependencies:

```bash
git pull --ff-only origin fred-custom-seeds
unset LLMGE_MIN_SERVER_REMAINING_SECONDS LLMGE_REAL_SQUEUE
uv sync
```

Stop if the pull or dependency sync fails. The next block creates the tag and
generates the Slurm scripts; these are the normal new-run commands. The
following checks are optional diagnostics, not commands that start the run:

```bash
pace-quota
LLMGE_AUTO_START_SERVER=0 uv run python -m pytest tests/operators/test_fred_population_integrity.py -q -k server_window
```

The test covers the `MM:SS` server-time format that crashed trial 2. Use
`python -m pytest` as shown: invoking `uv run pytest` directly can fail to
import the repository-root `run_improved.py` during test collection. The
final-evaluation code is checked separately in section 9, when needed.

```bash
RUN_DATE=$(date +%Y%m%d)
RUN_SUFFIX=$(date +%H%M%S)
export FRED_RUN_TAG="yolo26_p2_${RUN_DATE}_eot0_trial${RUN_SUFFIX}"
echo "new tag: $FRED_RUN_TAG"
test ! -e "fred_runs/$FRED_RUN_TAG" && echo 'new tag OK'
uv run python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.RUN_TAG, c.SEED_NETWORK, c.num_generations, c.RUN_CHECKPOINT_DIR, c.MODEL_PATH)'
test -d "$(uv run python -c 'from src.cfg.constants import MODEL_PATH; print(MODEL_PATH)')" && echo 'LLM model path OK'
mkdir -p run_job_outputs/{server,islands,evolution,evaluation}
uv run python slurm.py
grep -E 'run_improved.py --checkpoints .* --global_path ' run.sh
grep -F 'Skipping island controller submission' server.sh
test -s slurm-config/slurm_config.yaml && echo 'Slurm config OK'
```

The timestamp makes a new tag; record the printed value for every later
shell. Stop if `new tag OK` is absent. Confirm that the
configuration command prints `yolo26_p2`, the chosen tag,
`seed_yolo26n_p2.yaml`, and 21 generations. If any check fails, stop before
submitting. The generated controller has a 7 h 45 m CPU allocation; the LLM
server has an 8 h two-H200 allocation. Candidate evaluations request their
own GPU jobs. Run `slurm.py` once for this new tag; never regenerate it while
the tag is active.

Continue at **section 5** to submit the server and controller.

## 4. Resume the last saved generation of an existing tag

First identify and inspect the **latest controller job for this tag** and the
current run state. The tag is printed near the start of each controller log:

```bash
grep -lF "FRED run tag: $FRED_RUN_TAG;" run_job_outputs/islands/slurm-*.out
squeue -u "$USER" -o '%i %T %j %R %L'
cat "fred_runs/$FRED_RUN_TAG/status.json"
```

From the matching log paths, set the **newest relevant controller ID** and
inspect it. For example, `6515608` was the last known ID in the earlier chat,
but a later resume will have a different ID:

```bash
export PREVIOUS_RUN_JOB_ID=REPLACE_WITH_LATEST_CONTROLLER_JOB_ID
sacct -j "$PREVIOUS_RUN_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
tail -n 80 "run_job_outputs/islands/slurm-$PREVIOUS_RUN_JOB_ID.out"
```

If the controller is still running, **do not submit another one**. If the
status is `complete`, go to section 6. If it stopped early, verify that no
operator or candidate evaluation jobs for this run remain active and read the
traceback or pause reason first. `status.json` may still say `running` after
an unhandled controller crash; `sacct` and the log are decisive. For
`depleted`, `coverage_failed`, a fingerprint mismatch, or another unexplained
error, investigate the child logs in `run_job_outputs/evolution/` and
`run_job_outputs/evaluation/` before resubmitting.

With the **original code, tag, environment, and Slurm configuration** still
in place, this read-only check identifies the latest paired checkpoint and
compares both saved fingerprints with the current experiment. It may take a
minute to import the project.

```bash
uv run python - <<'PY'
from pathlib import Path
import pickle
import re
from src.cfg import constants as c
from run_improved import run_fingerprint

folder = Path(c.RUN_CHECKPOINT_DIR)
generations = []
for file in folder.glob('checkpoint_gen_*.pkl'):
    match = re.fullmatch(r'checkpoint_gen_(\d+)\.pkl', file.name)
    if match and (folder / f'global_gen_{match.group(1)}.pkl').is_file():
        generations.append(int(match.group(1)))
if not generations:
    raise SystemExit('STOP: no paired checkpoint found')
generation = max(generations)
expected = run_fingerprint()
for prefix in ('checkpoint', 'global'):
    path = folder / f'{prefix}_gen_{generation}.pkl'
    with path.open('rb') as stream:
        saved = pickle.load(stream)  # only this user's trusted run files
    if saved.get('protocol_fingerprint') != expected:
        raise SystemExit(f'STOP: fingerprint mismatch in {path}')
print(f'Checkpoint pair for generation {generation} matches this run')
PY
```

If this check fails, **do not resume** until the code and experiment settings
are reconciled. Trial 2 used commit `863282e61`, whose server-time parser
crashed twice when Slurm returned `54:54` or `45:53`. A later `squeue` wrapper
**did not intercept** the second failure. Do not use that wrapper again. The
trial was subsequently run with a six-hour controller limit and a two-hour
server margin so the old parser would not encounter the under-one-hour format.
Keep these settings only for trial 2 on its original commit:

```bash
git rev-parse --short HEAD  # expect 863282e61 for trial 2
export LLMGE_MIN_SERVER_REMAINING_SECONDS=7200
sed -i '/fred_runtime_bin/d' run.sh
grep -n 'fred_runtime_bin' run.sh  # expect no output
```

The newer `ff2eb0eee` commit was used for a separate final-evaluation retry.
If the PACE checkout has advanced to that commit, trial 2's checkpoint
fingerprint will not match. Do not attempt to resume the search from that
checkout without restoring its original code and environment.

Continue at section 5. The controller will choose its saved state; do not
manually edit or copy checkpoints to force a generation number.

## 5. Submit the server, then exactly one controller

For the October trial 2, `--partition=coe-gpu` avoided the earlier
`BadConstraints` submission. Use that partition for this checkout's server:

```bash
SERVER_JOB_ID=$(sbatch --parsable --partition=coe-gpu --export=ALL server.sh)
export LLMGE_SERVER_JOB_ID=${SERVER_JOB_ID%%;*}
echo "server job: $LLMGE_SERVER_JOB_ID"
squeue -j "$LLMGE_SERVER_JOB_ID" -o '%i %T %R %L'
```

`PENDING (Priority)` or `PENDING (Resources)` means the job is queued; its
eight-hour clock has not started and the log may not exist. `BadConstraints`
needs investigation rather than repeated submission. Once the job says
`RUNNING`, follow its log:

```bash
tail -F "run_job_outputs/server/slurm-$LLMGE_SERVER_JOB_ID.out"
```

Wait for `Uvicorn running`, `Serving on port 8479`, and
`Skipping island controller submission`. Press **Ctrl-C once** to leave
`tail`. For trial 2 on the old commit, verify the server still has **more than
seven hours** left before continuing:

```bash
squeue -j "$LLMGE_SERVER_JOB_ID" -o '%i %T %L'
```

Submit **one** controller. The six-hour override is only for trial 2's old
parser; a new tag on code that accepts `MM:SS` uses the generated limit:

```bash
if [[ "$FRED_RUN_TAG" == yolo26_p2_20261023_eot0_trial2 ]]; then
  RUN_JOB_ID=$(sbatch --parsable --time=6:00:00 --export=ALL run.sh)
else
  RUN_JOB_ID=$(sbatch --parsable --export=ALL run.sh)
fi
export RUN_JOB_ID=${RUN_JOB_ID%%;*}
echo "controller job: $RUN_JOB_ID"
squeue -j "$LLMGE_SERVER_JOB_ID,$RUN_JOB_ID" -o '%i %T %j %R %L'
tail -F "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
```

For trial 2, the resumed log should report the checkpoint it loaded and the
next generation (the prior session loaded generation 3 and started 4).
Press Ctrl-C to leave `tail`; you may then close the terminal and VPN. The
submitted Slurm jobs continue independently. **The server alone does not
advance evolution** when `SUBMIT_ISLAND_CONTROLLER=0`.

## 6. Monitor and decide whether the search is complete

In a fresh PACE shell, repeat sections 1 and 2 for the **same tag**. Set
`RUN_JOB_ID` to the latest controller ID you recorded (do not assume the
historical `6515608` is still latest), then:

```bash
export RUN_JOB_ID=REPLACE_WITH_LATEST_CONTROLLER_JOB_ID
squeue -u "$USER" -o '%i %T %j %R %L'
sacct -j "$RUN_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
tail -n 80 "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
grep -E 'Generation 0 failures|Finished Generation|paused_for_server_refresh|depleted|coverage_failed|Traceback|Stock seed failed' \
  "run_job_outputs/islands/slurm-$RUN_JOB_ID.out" | tail -n 35
cat "fred_runs/$FRED_RUN_TAG/status.json"
```

During generation 0, `status.json` may be absent and a checkpoint may be
provisional. A plotting error saying `No valid candidates` at that stage is
not a completed result. If the controller stops short of generation 20,
follow section 4 before another server/controller window. Do not run two
controllers for the same tag.

For the intended 21-generation search, require **all** of these before
calling the result final. The latest controller's `sacct` state above should
also be `COMPLETED` with exit code `0:0`; inspect any other state or traceback
before reporting the run as successful:

```bash
uv run python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s.get("status") == "complete" and s.get("last_completed_generation") == 20, s; print(s)' \
  "fred_runs/$FRED_RUN_TAG/status.json"
test -s "fred_runs/$FRED_RUN_TAG/checkpoints/checkpoint_gen_20.pkl" && echo 'population checkpoint OK'
test -s "fred_runs/$FRED_RUN_TAG/checkpoints/global_gen_20.pkl" && echo 'global checkpoint OK'
```

## 7. Generate data and Pareto plots for one run

Run this **after at least one generation has valid evaluated candidates**.
Set `FRED_RUN_TAG` to the exact tag you want to inspect. The plotter reads
`global_gen_*.pkl` and includes the stock seed as gene `seed`. Its outputs
are search-validation scores (mAP50, mAP50:95, parameter count) from
`challenging_train`, not held-out `challenging_test` scores.

If section 6 passed, use the canonical output folder:

```bash
export FRED_SEED_FAMILY=yolo26_p2
: "${FRED_RUN_TAG:?Set the exact YOLO26 run tag from section 2 first}"
test -d "fred_runs/$FRED_RUN_TAG/checkpoints" && echo "plotting $FRED_RUN_TAG"
PLOT_DIR="analysis/results/analysis/fred_pareto/$FRED_RUN_TAG"
uv run python analysis/scripts/fred_pareto_graphs.py \
  --family yolo26_p2 \
  --checkpoints "fred_runs/$FRED_RUN_TAG/checkpoints" \
  --models_dir "fred_runs/$FRED_RUN_TAG/candidates/models" \
  --output "$PLOT_DIR"
ls -lh "$PLOT_DIR"/{pareto2d_final.png,candidates.csv,pareto_models.csv}
head -n 8 "$PLOT_DIR/candidates.csv"
```

The folder also contains per-generation plots, a best-by-generation plot,
and (when multiple generations exist) an animated GIF. Check that
`candidates.csv` includes non-seed genes before interpreting evolution:

```bash
uv run python -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); assert any(r["gene_id"] != "seed" for r in rows), "No evaluated evolved candidates"; print(len(rows), "valid candidates including seed")' \
  "$PLOT_DIR/candidates.csv"
```

### Partial plots when the search has not reached generation 20

You may plot a **completed saved generation** to inspect progress. Do not
present the plotter's file named `pareto2d_final.png` as the final result of
the 21-generation study. Use a clearly labelled output directory such as:

```bash
export LAST_COMPLETED_GENERATION=3  # replace using status.json and paired checkpoints
PLOT_DIR="analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/through_gen_$LAST_COMPLETED_GENERATION"
uv run python analysis/scripts/fred_pareto_graphs.py \
  --family yolo26_p2 \
  --checkpoints "fred_runs/$FRED_RUN_TAG/checkpoints" \
  --models_dir "fred_runs/$FRED_RUN_TAG/candidates/models" \
  --output "$PLOT_DIR"
```

The plotter uses **every** `global_gen_*.pkl` in that checkpoint folder.
Confirm that its printed final generation matches the directory label. If a
newer generation was saved meanwhile, choose a new label and output folder.
Repeat section 7 after the full search completes.

The underlying per-candidate artifacts are in:

```text
fred_runs/<tag>/status.json
fred_runs/<tag>/checkpoints/checkpoint_gen_<N>.pkl
fred_runs/<tag>/checkpoints/global_gen_<N>.pkl
fred_runs/<tag>/candidates/models/network_<gene_id>.py
fred_runs/<tag>/candidates/results/<gene_id>_results.csv
fred_runs/<tag>/candidates/results/<gene_id>_evaluation.json
fred_runs/<tag>/candidates/trained_models/<gene_id>.pt
fred_runs/<tag>/operator_jobs/
```

### Inspect the last few tags

Use the inventory in section 1 to choose a tag, then repeat section 7 for
**each** tag that has valid evaluated checkpoint data. Each tag gets its own
output folder and CSVs. A failed tag with no valid candidates cannot produce
a meaningful Pareto plot; inspect its controller log instead. The local Mac
checkout also has an archived October 19 YOLO26 handoff in
`sota/FRED_LLM_GE/new-yolo26-dfine-results/fred-yolo26-p2-full-outputs-20261019T160847Z-3122367/`;
it already has `analysis/yolo26_p2/candidate_results.csv` and figures, and
uses a different layout from `fred_runs/<tag>`.

## 8. Copy results to the Mac (optional)

Run these in a **Mac terminal**, with the GT VPN connected, after generating
plots on PACE. Set the tag to the run you want to download. Repeat for each
tag of interest.

```bash
FRED_RUN_TAG=yolo26_p2_20261023_eot0_trial2
PACE_ROOT=/storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
LOCAL_ROOT="$HOME/Desktop/fred-yolo26-results/$FRED_RUN_TAG"
mkdir -p "$LOCAL_ROOT"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/" "$LOCAL_ROOT/pareto/"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/fred_runs/$FRED_RUN_TAG/status.json" "$LOCAL_ROOT/"
```

For a **complete copy of this tag's run data** (potentially much larger), run:

```bash
rsync -avP "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/fred_runs/$FRED_RUN_TAG/" "$LOCAL_ROOT/run/"
```

This includes candidate source, result CSVs and evaluation JSONs, trained
weights, checkpoints, and operator output. Shared Slurm logs under
`run_job_outputs/` are outside this tagged folder; record the job IDs if you
need those logs later. Keep the tag and the PACE commit with downloaded
results. A separate final evaluation is needed for held-out test performance.

## 9. Train and evaluate a selected YOLO26-P2 model on the held-out test set

This is a **separate** workflow after search-validation selection. It trains
the frozen architecture from scratch for the protocol's **50 final epochs** on
the exported `challenging_train` data, then evaluates `challenging_test` once.
The search plot and CSV scores are **not** held-out test scores. Do not use test
results to choose another architecture or change prompts.

For an active search tag, **do not pull code just to run final evaluation**:
the search checkpoint fingerprint includes the Git commit. The first YOLO26
final-evaluation attempt failed with `KeyError: 'training'`; commit
`ff2eb0eee` fixed the call to `adapter.training_overrides()`. Finish or
intentionally freeze the search before updating that checkout, or use a
separate checkout for final evaluation. Before submitting, verify the PACE
copy of `final_eval.py` contains that fix:

```bash
cd /storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
module load uv
git rev-parse --short HEAD
grep -n 'adapter.training_overrides()' sota/FRED_LLM_GE/phase1_detection/final_eval.py
```

If the `grep` has no matching line, stop; update the final-evaluation code
only after preserving the search checkout needed for any future resume. If the
search is finished or you have deliberately frozen it, the published fix can
be pulled with the commands below. Check the resulting code before continuing;
stop if the pull fails or the line is still absent:

```bash
git status --short
git pull --ff-only origin fred-custom-seeds
git rev-parse --short HEAD
grep -n 'adapter.training_overrides()' sota/FRED_LLM_GE/phase1_detection/final_eval.py
```

Trial 2's selected model `xXxfTSwXSazarm8WL5kFr0TRZFG` had a corrected
final-evaluation retry submitted as job `6613887` under the name
`yolo26_p2_xXxfTSwXSazarm8WL5kFr0TRZFG_final_20261026`. That was its
**submission state on October 26, 2026**, not a current result. Check the
current job and files on PACE before submitting another evaluation of that
model:

```bash
sacct -j 6613887 --format=JobID,State,ExitCode,Elapsed -P
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
NAME=yolo26_p2_xXxfTSwXSazarm8WL5kFr0TRZFG_final_20261026
ls -lh "$ART/final_runs/$NAME/model.pt" "$ART/held_out_reports/${NAME}_evaluation.json"
```

### 9.1 Freeze the model and check the final training export

In **every new PACE shell**, set the exact search tag and select a non-seed
`gene_id` from that tag's `pareto_models.csv`. Replace the two example values
below. If the search stopped before generation 20, record that the selection
was based on a partial search.

```bash
export FRED_SEED_FAMILY=yolo26_p2
export FRED_RUN_TAG=yolo26_p2_20261023_eot0_trial2  # replace for another run
export GENE_ID=xXxfTSwXSazarm8WL5kFr0TRZFG  # replace with your frozen choice
export HF_HUB_DISABLE_XET=1
PLOT_DIR="analysis/results/analysis/fred_pareto/$FRED_RUN_TAG"
head -n 12 "$PLOT_DIR/pareto_models.csv"
test -s "fred_runs/$FRED_RUN_TAG/candidates/models/network_${GENE_ID}.py" && echo 'selected model exists'
```

Confirm that the chosen ID is in that run's Pareto CSV:

```bash
uv run python - "$PLOT_DIR/pareto_models.csv" "$GENE_ID" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='') as stream:
    matches = [row for row in csv.DictReader(stream) if row['gene_id'] == sys.argv[2]]
if sys.argv[2] == 'seed' or len(matches) != 1:
    raise SystemExit('STOP: select one evolved gene_id from this run\'s pareto_models.csv')
print('Frozen search candidate:', matches[0])
PY
```

The final training export is shared across YOLO26 runs using the same
protocol. The October 2026 trial exported **113,340 frames**. Reuse it if the
file exists; the export array does not start final evaluation automatically.
Reset `FINAL_DATA` after each login so an empty shell variable cannot make a
valid export appear missing:

```bash
FINAL_DATA=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/final_train_step5/data.yaml
if test -s "$FINAL_DATA"; then echo 'Final training data ready'; else echo 'Final training export missing'; fi
```

**Only if the export is missing**, check quota and Hugging Face access, then
submit the eight CPU export shards:

```bash
pace-quota
uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])'
mkdir -p run_job_outputs/export
FINAL_EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=32G -t 8:00:00 \
  --output=run_job_outputs/export/final-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
FINAL_EXPORT_JOB_ID=${FINAL_EXPORT_JOB_ID%%;*}
echo "final export array: $FINAL_EXPORT_JOB_ID"
```

The array runs after logout. On return, set `FINAL_EXPORT_JOB_ID` to the ID
printed above and check **each** task `_0` through `_7`. Its eight-hour value
is a walltime limit, not a completion estimate. Continue only when all eight
say `COMPLETED|0:0`:

```bash
export FINAL_EXPORT_JOB_ID=REPLACE_WITH_FINAL_EXPORT_ARRAY_JOB_ID
sacct -j "$FINAL_EXPORT_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
```

Then finalize and verify the export. If `FINAL_DATA` was already ready, skip
the array and this finalization step:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --finalize
FINAL_DATA=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/final_train_step5/data.yaml
test -s "$FINAL_DATA" && echo 'Final training data ready'
```

### 9.2 Submit one final GPU job

Use the **same tag and gene ID** frozen above. Check `squeue` and existing
final-run folders before submission so a prior job is not duplicated. The
name below contains the tag and gene ID; keep it when checking and copying
results. If this name already has a partial output, inspect the prior failure
before choosing a distinct retry name.

```bash
export FRED_SEED_FAMILY=yolo26_p2
: "${FRED_RUN_TAG:?Set the chosen search tag in this shell}"
: "${GENE_ID:?Set the frozen gene ID in this shell}"
export HF_HUB_DISABLE_XET=1
export FINAL_NAME="${FRED_RUN_TAG}_${GENE_ID}_final"
FINAL_DATA=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/final_train_step5/data.yaml
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
MODEL_FILE="fred_runs/$FRED_RUN_TAG/candidates/models/network_${GENE_ID}.py"
REPORT="$ART/held_out_reports/${FINAL_NAME}_evaluation.json"
mkdir -p run_job_outputs/final_eval
squeue -u "$USER" -o '%i %T %j %R %L'
```

Once the checks in 9.1 have passed and no final-evaluation job for this model
is active, submit. `STOP` means do not treat the job as submitted:

```bash
if [[ "$GENE_ID" == seed || ! -s "$FINAL_DATA" || ! -s "$MODEL_FILE" ]]; then
  echo 'STOP: final training data or selected evolved model is missing'
elif [[ -e "$ART/final_runs/$FINAL_NAME" || -e "$REPORT" ]]; then
  echo 'STOP: this final-evaluation name already has output; inspect it first'
else
  FINAL_EVAL_JOB_ID=$(sbatch --parsable --partition=coe-gpu --export=ALL \
    --job-name=yolo26_final -G 1 -C H200 -c 12 --mem=64G -t 16:00:00 \
    --output=run_job_outputs/final_eval/slurm-%j.out \
    --wrap='module load cuda; module load uv; unset VIRTUAL_ENV; uv run python -m sota.FRED_LLM_GE.phase1_detection.final_eval --family "$FRED_SEED_FAMILY" --name "$FINAL_NAME" --model "network_${GENE_ID}" --variant_dir "fred_runs/$FRED_RUN_TAG/candidates/models"')
  FINAL_EVAL_JOB_ID=${FINAL_EVAL_JOB_ID%%;*}
  echo "final evaluation job: $FINAL_EVAL_JOB_ID"
fi
```

Slurm runs the job after logout. Sixteen hours is its **maximum running
time**, not a promise of completion. On return, set the job ID and the same
`FINAL_NAME` again; inspect the **main job** state and log:

```bash
export FRED_RUN_TAG=yolo26_p2_20261023_eot0_trial2  # replace for another run
export GENE_ID=xXxfTSwXSazarm8WL5kFr0TRZFG  # replace with the frozen choice
export FINAL_NAME="${FRED_RUN_TAG}_${GENE_ID}_final"  # use the exact recorded retry name if different
export FINAL_EVAL_JOB_ID=REPLACE_WITH_FINAL_EVALUATION_JOB_ID
squeue -j "$FINAL_EVAL_JOB_ID" -o '%i %T %R %M %L'
sacct -j "$FINAL_EVAL_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
tail -n 60 "run_job_outputs/final_eval/slurm-$FINAL_EVAL_JOB_ID.out"
```

Require `COMPLETED|0:0` for the main job **and** both nonempty files. A
`COMPLETED` `.extern` or `.batch` line does not override a failed main job.
The JSON stores metric fractions; the command below prints percentages:

```bash
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
: "${FINAL_NAME:?Set the exact final-evaluation name in this shell}"
REPORT="$ART/held_out_reports/${FINAL_NAME}_evaluation.json"
test -s "$ART/final_runs/$FINAL_NAME/model.pt" && echo 'trained model exists'
test -s "$REPORT" && echo 'held-out report exists'
uv run python - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
if report.get('mode') != 'held-out':
    raise SystemExit('STOP: this is not a held-out evaluation report')
metrics = report['metrics']
print(f"held-out mAP50: {100 * metrics['mAP50']:.2f}%")
print(f"held-out mAP50:95: {100 * metrics['mAP50_95']:.2f}%")
print(f"parameters: {metrics['params']:,}")
print(f"test frames: {report['num_frames']}")
PY
```

If the job failed or timed out, inspect its log before retrying. A partially
created `final_runs/<name>` directory is **not** a final result.

## 10. Copy held-out evaluation results to the Mac

Run in a **Mac terminal** with the VPN connected, after section 9.2 passes.
Set `FRED_RUN_TAG` and `FINAL_NAME` to the exact values for the finished job.
Section 8 separately copies the search plots, CSVs, candidate artifacts, and
checkpoints.

```bash
FRED_RUN_TAG=yolo26_p2_20261023_eot0_trial2  # replace for another run
FINAL_NAME=yolo26_p2_xXxfTSwXSazarm8WL5kFr0TRZFG_final_20261026  # replace
PACE_ROOT=/storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
LOCAL_ROOT="$HOME/Desktop/fred-yolo26-results/$FRED_RUN_TAG/final_eval/$FINAL_NAME"
mkdir -p "$LOCAL_ROOT"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/$ART/held_out_reports/${FINAL_NAME}_evaluation.json" "$LOCAL_ROOT/"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/$ART/final_runs/$FINAL_NAME/" "$LOCAL_ROOT/final_run/"
```
