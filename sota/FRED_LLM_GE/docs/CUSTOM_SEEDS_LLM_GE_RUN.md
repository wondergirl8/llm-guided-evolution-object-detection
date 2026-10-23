# YOLO26-P2 and D-FINE: complete FRED LLM-GE run on PACE ICE

Run every command from the repository root on PACE. This procedure uses the
released 33.3 ms Event frames, `challenging_train` for search training and
validation, and a separate `challenging_test` evaluation after model selection.
It does **not** submit a standalone seed-only run: the controller trains and
scores the stock seed once as generation-zero member `seed`.

These two families need separate run tags and should run **sequentially in one
checkout**. `hostname.log`, generated Slurm scripts, and `slurm_config.yaml`
are shared across the checkout. An eight-hour server is one execution window;
a 21-generation search may need several server/controller windows.

## 0. Verify the checkout and resources

Sync the branch containing the custom-seed launch and proposal fixes to PACE
before starting. Keep its code, protocol, seed, and generated Slurm
configuration fixed for each tagged run and its resumes. Record `git rev-parse
HEAD` and any local `git diff`; the tagged checkpoint fingerprint includes the
protocol, search manifests, family code, prompts, software versions, and
generated Slurm configuration.

The fixes in this workspace are currently local edits. Checking out the remote
`fred-custom-seeds` branch alone will omit them until the intended code changes
are committed and pushed or transferred as an exact patch. Exclude archived
result folders and generated outputs when transferring the code. Verify the
PACE checkout against the intended commit and patch before submitting jobs.

Before starting evolution, verify that the chosen stock seed has already been
trained and evaluated independently under the intended FRED protocol, with its
baseline evidence documented. The controller's single generation-zero seed
score does not satisfy the project's independent baseline requirement. The
existing baseline runner is YOLO11-specific, so do not run it for either
custom family unchanged.

```bash
module load uv
uv sync
readlink src/cfg/constants.py               # constants_fred_yolo11.py
git rev-parse HEAD
git status --short
test -f sota/FRED_LLM_GE/configs/phase1/protocol.yaml
test -f sota/FRED_LLM_GE/seeds/seed_yolo26n_p2.yaml
test -f sota/FRED_LLM_GE/seeds/seed_dfine.py
test -f templates/FRED_YOLO26_P2/ConstantRules.txt
test -f templates/FRED_DFINE/ConstantRules.txt
test -f slurm-config/pace-ice.txt
pace-quota                                  # check scratch file and byte quotas
export HF_HUB_DISABLE_XET=1
unset FRED_EVENTS                           # released Event frames
unset LLMGE_BOUNDED_SESSION                 # use the ordinary eight-hour scripts below
mkdir -p run_job_outputs/{export,server,islands,evolution,evaluation,final_eval}
```

Confirm Hugging Face access without displaying a token:

```bash
uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])'
```

If access is missing, log in with
`uv run python -c 'from huggingface_hub import login; login()'`.
For a formal baseline comparison, retain independent baseline evidence for
each seed before interpreting the evolutionary improvement. The generation-zero
seed point is a search reference, not a multi-seed baseline estimate.

## 1. Export the shared search data once

If both exported search folders already have complete `data.yaml` files,
run the finalize/check command below and skip the array submission. Otherwise:

```bash
EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=16G -t 4:00:00 \
  --output=run_job_outputs/export/search-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
EXPORT_JOB_ID=${EXPORT_JOB_ID%%;*}
echo "$EXPORT_JOB_ID"
```

Wait for tasks 0–7 and verify each has `COMPLETED` / `0:0` before finalizing:

```bash
squeue -j "$EXPORT_JOB_ID"
sacct -j "$EXPORT_JOB_ID" --format=JobID,State,ExitCode -P
```

Only after all eight tasks succeed, finalize and check the expected counts:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize \
  | tee run_job_outputs/export/finalize-search.log
grep -F 'search_train_step50: 152 sequences, 10053 frames' run_job_outputs/export/finalize-search.log
grep -F 'search_val_step10: 20 sequences, 6733 frames' run_job_outputs/export/finalize-search.log
```

Both counts must appear. The reusable training and validation exports are under
`sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/`.
Repair and resubmit failed array shards before finalizing. Do not export
`challenging_test` for search.

```bash
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_train_step50/data.yaml
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/search_val_step10/data.yaml
```

## 2. Configure one family and generate its Slurm scripts

Use **one** of these family keys. Give each new experiment a new tag. On resume,
reuse its exact tag, generation count, code/protocol, and Slurm configuration.
The example port must be free; use a different port for a concurrent server in
another checkout.

```bash
export FRED_SEED_FAMILY=yolo26_p2           # for the second run: dfine
export FRED_RUN_TAG="${FRED_SEED_FAMILY}_20261022_eot0_trial1"
export FRED_NUM_GENERATIONS=21              # generation 0 plus 1–20
export LLMGE_PORT=8479
export LLMGE_CLUSTER=pace-ice
export LOCAL=false LOCAL_LLM=true INFERENCE_SUBMISSION=false
export LLMGE_LLM_MODEL=mixtral
export SUBMIT_ISLAND_CONTROLLER=0
export HF_HUB_DISABLE_XET=1
unset FRED_EVENTS
unset LLMGE_BOUNDED_SESSION
unset LLMGE_MODEL_PATH LLM_MAX_NEW_TOKENS LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS
uv run python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.RUN_TAG, c.SEED_NETWORK, c.num_generations, c.RUN_CHECKPOINT_DIR, c.MODEL_PATH)'
test -d "$(uv run python -c 'from src.cfg.constants import MODEL_PATH; print(MODEL_PATH)')"
uv run python slurm.py
grep -E 'run_improved.py --checkpoints .* --global_path ' run.sh
grep -E 'Serving on port|CUDA_VISIBLE_DEVICES' server.sh
test -s slurm-config/slurm_config.yaml
```

The printed family, seed, tag, and generation count must match your plan.
`run.sh` must use named controller arguments, and `server.sh` must **not** set
`CUDA_VISIBLE_DEVICES=0,1`. The generator writes a 7 h 45 m CPU controller,
an 8 h two-H200 LLM server, and the candidate/operator job templates. The
candidate jobs request one H200, 12 CPUs, 32 GB, and four hours. Slurm assigns
all CPUs and GPUs; no OnDemand session is needed for these batch allocations.

Do not run `slurm.py` again while this tag is active. Finish or resume it before
switching the shared generated scripts to the second family.

## 3. Start the server, then the evolution controller

```bash
SERVER_JOB_ID=$(sbatch --parsable --export=ALL server.sh)
export LLMGE_SERVER_JOB_ID=${SERVER_JOB_ID%%;*}
echo "$LLMGE_SERVER_JOB_ID"
squeue -j "$LLMGE_SERVER_JOB_ID" -o '%i %T %R %L'
tail -f "run_job_outputs/server/slurm-$LLMGE_SERVER_JOB_ID.out"
```

Wait for `Uvicorn running`, `Serving on port 8479` (or your port), and
`Skipping island controller submission`. Press Ctrl-C to stop **tail**, not the
server. Then:

```bash
RUN_JOB_ID=$(sbatch --parsable --export=ALL run.sh)
RUN_JOB_ID=${RUN_JOB_ID%%;*}
echo "$RUN_JOB_ID"
tail -f "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
```

The controller submits LLM operator jobs and candidate GPU jobs. Generation 0
contains the stock seed and seven LLM candidates. YOLO26's first four
candidates target distinct P2–P5 channel reductions; three slots are open.
The controller now requires eight distinct successful architectures or records
`depleted`/`coverage_failed` instead of treating a short population as a
complete generation. D-FINE operator attempts and rejection feedback are
preserved beside each candidate model.

## 4. Monitor and resume until complete

```bash
squeue -u "$USER" -o '%i %T %j %R %L'
sacct -j "$RUN_JOB_ID" --format=JobID,State,ExitCode -P
cat "fred_runs/$FRED_RUN_TAG/status.json"
grep -En 'Finished Generation|distinct .* architectures|failures|depleted|coverage_failed|paused_for_server_refresh|Traceback' \
  "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
```

Inspect `run_job_outputs/evolution/` for proposal failures and
`run_job_outputs/evaluation/` for training failures. If the controller stops
before generation 20, first verify that it has exited and inspect the status
and the latest **paired** checkpoint files. For a normal server-window pause,
submit a new `server.sh`, export its new job ID, wait for readiness, and submit
`run.sh` once with the **same** family, tag, port, generation count, code,
protocol, and generated Slurm configuration. Do not run two controllers for
the same tag. A pending generation-zero checkpoint retains submitted
evaluation job IDs and resumes generation zero before generation one. For
`depleted`, `coverage_failed`, or another error, investigate
the child logs before attempting a resume; do not change the experiment under
the same tag.

Completion checks for the example 21-generation run:

```bash
uv run python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s.get("status") == "complete" and s.get("last_completed_generation") == 20, s; print(s)' \
  "fred_runs/$FRED_RUN_TAG/status.json"
test -f "fred_runs/$FRED_RUN_TAG/checkpoints/checkpoint_gen_20.pkl"
test -f "fred_runs/$FRED_RUN_TAG/checkpoints/global_gen_20.pkl"
```

The run's candidate sources, fitness CSVs, trained weights, validation
reports, controller checkpoints, and operator records are in:

```text
fred_runs/<tag>/candidates/models/network_<gene_id>.py
fred_runs/<tag>/candidates/results/<gene_id>_results.csv
fred_runs/<tag>/candidates/trained_models/<gene_id>.pt
fred_runs/<tag>/candidates/results/<gene_id>_evaluation.json
fred_runs/<tag>/checkpoints/
fred_runs/<tag>/operator_jobs/
fred_runs/<tag>/status.json
```

YOLO26 also writes
`fred_runs/<tag>/operator_jobs/initial_yolo26_coverage.json`.

## 5. Create the search Pareto data and plots

Run after any completed generation checkpoint; run again after generation 20
for final search results:

```bash
uv run python analysis/scripts/fred_pareto_graphs.py \
  --family "$FRED_SEED_FAMILY" \
  --checkpoints "fred_runs/$FRED_RUN_TAG/checkpoints" \
  --models_dir "fred_runs/$FRED_RUN_TAG/candidates/models" \
  --output "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG"
test -s "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/pareto2d_final.png"
uv run python -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); assert any(r["gene_id"] != "seed" for r in rows), "No evaluated evolved candidates"; print(len(rows), "valid search candidates including seed")' \
  "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/candidates.csv"
```

The output includes `pareto2d_final.png` (mAP50 and mAP50:95 versus parameter
count), per-generation plots, `candidates.csv`, and `pareto_models.csv`.
These scores are from `challenging_train` **search validation**. Review and
freeze the chosen evolved gene ID using these results before seeing test scores.

## 6. Optional held-out final evaluation of a frozen evolved model

This is the separate workflow shown in the team screenshots. It retrains one
chosen architecture on all `challenging_train` and scores `challenging_test`
once. It is **not** required to generate search Pareto plots; never feed test
scores back into architecture or prompt selection. Check scratch quota again:
the final export is roughly 113k training frames, about 230k files and 7 GB,
and the test frames are read from downloads rather than exported.

```bash
pace-quota
FINAL_EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=32G -t 8:00:00 \
  --output=run_job_outputs/export/final-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
FINAL_EXPORT_JOB_ID=${FINAL_EXPORT_JOB_ID%%;*}
echo "$FINAL_EXPORT_JOB_ID"
```

Wait for all eight tasks to show `COMPLETED` / `0:0` before finalizing:

```bash
squeue -j "$FINAL_EXPORT_JOB_ID"
sacct -j "$FINAL_EXPORT_JOB_ID" --format=JobID,State,ExitCode -P
```

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --finalize
test -s sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/final_train_step5/data.yaml
```

Choose a non-seed `gene_id` from `pareto_models.csv`, then submit one final GPU job for
each frozen family/model. Set `GENE_ID` and `FINAL_NAME` to distinct values:

```bash
export GENE_ID=replace_with_gene_id  # Replace with a non-seed ID from pareto_models.csv.
export FINAL_NAME="${FRED_SEED_FAMILY}_${GENE_ID}_final"
if [[ "$GENE_ID" == seed || ! -f "fred_runs/$FRED_RUN_TAG/candidates/models/network_${GENE_ID}.py" ]]; then
  echo "Select an existing non-seed gene ID from pareto_models.csv"
else
  FINAL_EVAL_JOB_ID=$(sbatch --parsable --export=ALL -G 1 -C H200 -c 12 --mem=64G -t 16:00:00 \
    --output=run_job_outputs/final_eval/slurm-%j.out \
    --wrap='module load cuda; module load uv; unset VIRTUAL_ENV; uv run python -m sota.FRED_LLM_GE.phase1_detection.final_eval --family "$FRED_SEED_FAMILY" --name "$FINAL_NAME" --model "network_${GENE_ID}" --variant_dir "fred_runs/$FRED_RUN_TAG/candidates/models"')
  FINAL_EVAL_JOB_ID=${FINAL_EVAL_JOB_ID%%;*}
  echo "$FINAL_EVAL_JOB_ID"
fi
```

The default final recipe is 50 epochs. A one-epoch smoke run can use
`--epochs 1` with a different `--name`; it is a pipeline check, not a final
score. D-FINE accepts `--dfine-reg-max` only if that value was frozen from
search-validation experiments before test evaluation. Final outputs are under
`sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/<family>/final_runs/<name>/model.pt`
and the sibling `<family>/held_out_reports/<name>_evaluation.json`.

After the job ends, require `COMPLETED` / `0:0` and verify both artifacts:

```bash
sacct -j "$FINAL_EVAL_JOB_ID" --format=JobID,State,ExitCode -P
test -s "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/$FRED_SEED_FAMILY/final_runs/$FINAL_NAME/model.pt"
test -s "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/$FRED_SEED_FAMILY/held_out_reports/${FINAL_NAME}_evaluation.json"
```

After finishing the first family, return to step 2 with `FRED_SEED_FAMILY=dfine`,
a fresh `dfine_...` tag, and new generated Slurm scripts. Reuse the verified
search and final data exports.
