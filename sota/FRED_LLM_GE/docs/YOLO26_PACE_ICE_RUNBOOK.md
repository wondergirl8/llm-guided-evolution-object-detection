# YOLO26-P2 on PACE ICE

This is the command path for a new YOLO26-P2 search, resuming it, making Pareto plots, copying results, and running the separate held-out final evaluation. Run PACE commands from the repository root. Run copy commands on the Mac with the GT VPN connected.

**Choose one path:** For a new search, use sections 1–3. For a resume, use sections 1, 4, then 3. Use sections 5–8 for results and final evaluation. Commands under “Only if data is missing” are conditional; skip them when the data already exists.

The LLM server and controller are separate Slurm jobs. The server's eight-hour limit is one window, so a 21-generation search may need more than one window. Press Ctrl-C to leave a log displayed with tail; that does not stop the Slurm job. Record the run tag and every job ID.

## 1. Log in and set the YOLO26 environment

On the **Mac**:

~~~bash
ssh agupta3057@login-ice.pace.gatech.edu
~~~

On **PACE**, paste this block in every new shell before using a YOLO26 run tag:

~~~bash
cd /storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
module load uv
export FRED_SEED_FAMILY=yolo26_p2 FRED_NUM_GENERATIONS=21
export LLMGE_PORT=8479 LLMGE_CLUSTER=pace-ice
export LOCAL=false LOCAL_LLM=true INFERENCE_SUBMISSION=false
export LLMGE_LLM_MODEL=mixtral SUBMIT_ISLAND_CONTROLLER=0
export HF_HUB_DISABLE_XET=1
unset FRED_EVENTS LLMGE_BOUNDED_SESSION LLMGE_MODEL_PATH
unset LLM_MAX_NEW_TOKENS LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS
unset LLMGE_MIN_SERVER_REMAINING_SECONDS LLMGE_REAL_SQUEUE
~~~

The family setting is required. Without it, the configuration defaults to YOLO11 and rejects a yolo26_p2 tag.

## 2. Start a new search

Use this section only for a **new tag**. Do not pull or regenerate scripts in a checkout that still needs to resume an older tag: the checkpoint fingerprint includes the Git commit and generated Slurm configuration. Do not run another family in this shared checkout at the same time.

~~~bash
git pull --ff-only origin fred-custom-seeds
uv sync
export FRED_RUN_TAG="yolo26_p2_$(date +%Y%m%d)_eot0_trial$(date +%H%M%S)"
echo "run tag: $FRED_RUN_TAG"
test ! -e "fred_runs/$FRED_RUN_TAG" && echo 'new tag OK'
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_train_step50/data.yaml && echo 'search training data ready'
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_val_step10/data.yaml && echo 'search validation data ready'
uv run python -c 'from pathlib import Path; from src.cfg import constants as c; print(c.SEED_FAMILY, c.RUN_TAG, Path(c.SEED_NETWORK).name, c.num_generations, c.MODEL_PATH); assert c.SEED_FAMILY == "yolo26_p2" and c.num_generations == 21 and Path(c.SEED_NETWORK).name == "seed_yolo26n_p2.yaml" and Path(c.MODEL_PATH).is_dir()'
mkdir -p run_job_outputs/{server,islands,evolution,evaluation}
uv run python slurm.py
~~~

Stop if a command fails. The printed configuration should show yolo26_p2, your new tag, seed_yolo26n_p2.yaml, and 21. The two data files are shared search exports; if either is missing, use the **search export** instructions in section 9 before continuing.
The yolo11 directory name in their paths is the shared Event-frame cache, not
the model selected for this run.

If you **already created a new tag** in this shell, keep it. Run the environment block in section 1, set your existing FRED_RUN_TAG, and continue with the data and configuration checks above. Do not generate a second tag. Run slurm.py once for that tag.

## 3. Submit the LLM server, then one controller

Submit the server:

~~~bash
SERVER_JOB_ID=$(sbatch --parsable --partition=coe-gpu --export=ALL server.sh)
export LLMGE_SERVER_JOB_ID=${SERVER_JOB_ID%%;*}
echo "server job: $LLMGE_SERVER_JOB_ID"
squeue -j "$LLMGE_SERVER_JOB_ID" -o '%i %T %R %L'
~~~

A PENDING job is queued. Wait for RUNNING, then inspect its log:

~~~bash
tail -F "run_job_outputs/server/slurm-$LLMGE_SERVER_JOB_ID.out"
~~~

Wait for “Serving on port 8479” and “Skipping island controller submission”; then press Ctrl-C. If the server exits or the log reports an error, inspect it before submitting the controller. The coe-gpu partition avoided a prior BadConstraints submission.

Submit **one** controller from the same shell, while the server is ready:

~~~bash
RUN_JOB_ID=$(sbatch --parsable --export=ALL run.sh)
RUN_JOB_ID=${RUN_JOB_ID%%;*}
echo "controller job: $RUN_JOB_ID"
tail -F "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
~~~

Record both IDs. Ctrl-C leaves tail; Slurm jobs continue after logout. The controller trains the stock seed as part of generation 0 and then evaluates evolved candidates. The search does not automatically generate Pareto plots or run the held-out final evaluation.
The generation-0 seed score is a search reference; a formal comparison also
needs independently documented baseline evidence.

## 4. Monitor or resume the same search

In a new PACE shell, repeat section 1 and set the **exact saved tag** and latest controller job ID:

~~~bash
export FRED_RUN_TAG=YOUR_SAVED_YOLO26_TAG
RUN_JOB_ID=YOUR_LATEST_CONTROLLER_JOB_ID
squeue -u "$USER" -o '%i %T %j %R %L'
sacct -j "$RUN_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
tail -n 80 "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
cat "fred_runs/$FRED_RUN_TAG/status.json"
~~~

If that controller is still active, **do not submit another**. A finished 21-generation search has status complete, last_completed_generation 20, and a main Slurm job state of COMPLETED with exit code 0:0.

If the controller stopped before generation 20 because its server window ended, keep the same tag, code, environment, and generated scripts. **Do not pull, run uv sync, or rerun slurm.py.** Start a new server and one controller using section 3. The controller automatically loads the latest saved checkpoint pair and checks its fingerprint. If the log shows a traceback, depleted population, coverage failure, or fingerprint mismatch, investigate before restarting; status.json alone can be stale after a crash.

**October 2026 trial 2 exception:** That tag used commit 863282e61 and an older server-time parser. It cannot resume from a checkout pulled to the new branch tip. Resume it only in its original matching checkout. Before starting the server, run this; the printed commit must be 863282e61:

~~~bash
git rev-parse --short HEAD
export LLMGE_MIN_SERVER_REMAINING_SECONDS=7200
sed -i '/fred_runtime_bin/d' run.sh
~~~

Start the server as in section 3 with more than seven hours left. Then submit
the controller with this six-hour override **instead of** the normal
controller submission:

~~~bash
RUN_JOB_ID=$(sbatch --parsable --time=6:00:00 --export=ALL run.sh)
RUN_JOB_ID=${RUN_JOB_ID%%;*}
echo "controller job: $RUN_JOB_ID"
~~~

This exception is not needed for a new tag.

## 5. Generate results, data tables, and Pareto plots

In a PACE shell, use section 1's environment and set the tag to the run you want to inspect. This works after at least one saved generation has valid candidates:

~~~bash
export FRED_RUN_TAG=YOUR_YOLO26_TAG
uv run python analysis/scripts/fred_pareto_graphs.py
~~~

The output is in analysis/results/analysis/fred_pareto/<tag>/:

- candidates.csv: valid evaluated candidates, with search-validation mAP50, mAP50:95, and parameter count.
- pareto_models.csv: candidates on the three-objective Pareto front and their model paths.
- pareto2d_gen_<N>.png: the front through each saved generation.
- pareto2d_final.png, best_by_generation.png, and, when applicable, pareto2d_fronts.gif.

The plotter uses the latest **saved** generation. Its filename pareto2d_final.png means “latest plotted front”; it is a final 21-generation result only when section 4 confirms generation 20 and a successful controller job. These scores are search-validation results, not held-out test results.

Raw candidate data and weights live under fred_runs/<tag>/candidates/, and checkpoints and status.json live under fred_runs/<tag>/. To list recent tags, run:

~~~bash
ls -dt fred_runs/yolo26_p2_* | head -n 10
~~~

For the last few runs, set each tag in turn and rerun the two plot commands above.

## 6. Copy search results to the Mac

In a **Mac terminal** with the VPN connected, replace the tag. The first rsync copies plots and CSV tables; the second copies the full run data, including checkpoints, candidate source, evaluations, and weights. The full copy can be large.

~~~bash
FRED_RUN_TAG=YOUR_YOLO26_TAG
PACE_ROOT=/storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
LOCAL_ROOT="$HOME/Desktop/fred-yolo26-results/$FRED_RUN_TAG"
mkdir -p "$LOCAL_ROOT"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/" "$LOCAL_ROOT/pareto/"
rsync -avP "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/fred_runs/$FRED_RUN_TAG/" "$LOCAL_ROOT/run/"
~~~

Repeat with another tag to copy another run. Slurm logs are in the shared run_job_outputs/ directory rather than the tagged run folder; keep the job IDs if you need to retrieve those logs.

## 7. Train the chosen model and run the held-out final evaluation

This is a **separate** job after choosing a model from pareto_models.csv. It trains that frozen architecture from scratch for the protocol's 50 final epochs, then evaluates challenging_test. Do not use held-out test scores to select or revise the architecture.

On PACE, use section 1's environment, then set the search tag and one **non-seed** gene ID from that run's pareto_models.csv:

~~~bash
export FRED_RUN_TAG=YOUR_YOLO26_TAG
cat "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/pareto_models.csv"
export GENE_ID=YOUR_SELECTED_GENE_ID
export FINAL_NAME="${FRED_RUN_TAG}_${GENE_ID}_final"
FINAL_DATA=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/final_train_step5/data.yaml
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
MODEL_FILE="fred_runs/$FRED_RUN_TAG/candidates/models/network_${GENE_ID}.py"
test "$GENE_ID" != seed && test -s "$MODEL_FILE" && echo 'selected model ready'
test -s "$FINAL_DATA" && echo 'final training data ready'
~~~

If the final training data test fails, run the **final export** in section 9 first. A separate checkout with the final_eval.py fix is required if the search must still resume from the old October trial-2 commit. For new tags on the current fred-custom-seeds branch, that fix is already present.

Check for an existing final-evaluation job or output before submitting another:

~~~bash
squeue -u "$USER" -o '%i %T %j %R %L'
mkdir -p run_job_outputs/final_eval
if test "$GENE_ID" = seed || ! test -s "$MODEL_FILE" || ! test -s "$FINAL_DATA"; then
  echo 'STOP: selected model or final training data is missing'
elif test -e "$ART/final_runs/$FINAL_NAME" || test -e "$ART/held_out_reports/${FINAL_NAME}_evaluation.json"; then
  echo 'STOP: this final name already has output; inspect it first'
else
  FINAL_EVAL_JOB_ID=$(sbatch --parsable --partition=coe-gpu --export=ALL \
    --job-name=yolo26_final -G 1 -C H200 -c 12 --mem=64G -t 16:00:00 \
    --output=run_job_outputs/final_eval/slurm-%j.out \
    --wrap='module load cuda; module load uv; unset VIRTUAL_ENV; uv run python -m sota.FRED_LLM_GE.phase1_detection.final_eval --family "$FRED_SEED_FAMILY" --name "$FINAL_NAME" --model "network_${GENE_ID}" --variant_dir "fred_runs/$FRED_RUN_TAG/candidates/models"')
  FINAL_EVAL_JOB_ID=${FINAL_EVAL_JOB_ID%%;*}
  echo "final evaluation job: $FINAL_EVAL_JOB_ID"
fi
~~~

Submit only when no prior final-evaluation job for this model is active. The 16-hour value is a walltime limit, not an expected runtime. Record the job ID and FINAL_NAME.

Later, in a new PACE shell, set FINAL_NAME and FINAL_EVAL_JOB_ID to the recorded values:

~~~bash
export FINAL_NAME=YOUR_RECORDED_FINAL_NAME
FINAL_EVAL_JOB_ID=YOUR_FINAL_EVAL_JOB_ID
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
sacct -j "$FINAL_EVAL_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P
tail -n 80 "run_job_outputs/final_eval/slurm-$FINAL_EVAL_JOB_ID.out"
test -s "$ART/final_runs/$FINAL_NAME/model.pt" && echo 'trained model ready'
test -s "$ART/held_out_reports/${FINAL_NAME}_evaluation.json" && echo 'held-out report ready'
cat "$ART/held_out_reports/${FINAL_NAME}_evaluation.json"
~~~

Require the **main** Slurm job to say COMPLETED with exit code 0:0 and both files to exist. The report metrics are fractions: multiply mAP values by 100 for percentages. A partial final_runs directory is not a completed result.

## 8. Copy final evaluation to the Mac

In a **Mac terminal** with the VPN connected:

~~~bash
FRED_RUN_TAG=YOUR_YOLO26_TAG
FINAL_NAME=YOUR_RECORDED_FINAL_NAME
PACE_ROOT=/storage/ice1/6/9/agupta3057/llm-guided-evolution-object-detection
ART=sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/yolo26_p2
LOCAL_ROOT="$HOME/Desktop/fred-yolo26-results/$FRED_RUN_TAG/final_eval/$FINAL_NAME"
mkdir -p "$LOCAL_ROOT"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/$ART/held_out_reports/${FINAL_NAME}_evaluation.json" "$LOCAL_ROOT/"
rsync -av "agupta3057@login-ice.pace.gatech.edu:$PACE_ROOT/$ART/final_runs/$FINAL_NAME/" "$LOCAL_ROOT/final_run/"
~~~

## 9. Only if data is missing: one-time exports

The October 2026 search data was already exported. Use these commands only when section 2's search data checks fail. First make the export log folder, submit eight CPU shards, and record the job ID:

~~~bash
mkdir -p run_job_outputs/export
EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=16G -t 4:00:00 \
  --output=run_job_outputs/export/search-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
EXPORT_JOB_ID=${EXPORT_JOB_ID%%;*}
echo "search export array: $EXPORT_JOB_ID"
~~~

Check the array and wait until tasks 0–7 each report COMPLETED and 0:0:

~~~bash
sacct -j "$EXPORT_JOB_ID" --format=JobID,State,ExitCode -P
~~~

Then finalize once:

~~~bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize
~~~

The expected search exports contain 10,053 training frames and 6,733 validation frames; inspect the finalizer output. If shards failed, fix and rerun them before finalizing. Do not export challenging_test for search.

Use this separate branch only when section 7's final training data check fails:

~~~bash
mkdir -p run_job_outputs/export
FINAL_EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=32G -t 8:00:00 \
  --output=run_job_outputs/export/final-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
FINAL_EXPORT_JOB_ID=${FINAL_EXPORT_JOB_ID%%;*}
echo "final export array: $FINAL_EXPORT_JOB_ID"
~~~

Check the array and wait until tasks 0–7 each report COMPLETED and 0:0:

~~~bash
sacct -j "$FINAL_EXPORT_JOB_ID" --format=JobID,State,ExitCode -P
~~~

Then finalize once:

~~~bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --finalize
~~~

The October 2026 final export had 113,340 frames. For the shared protocol and more detail about exports, see [CUSTOM_SEEDS_LLM_GE_RUN.md](CUSTOM_SEEDS_LLM_GE_RUN.md).
