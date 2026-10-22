# FRED detector evolution: run checklist

Run from the repository root on PACE. This checklist covers the shared FRED pipeline and gives working commands for the **currently configured YOLO11n Event-frame run**. Commands for a different detector family become valid only after that family's setup in [“Add another model family”](#add-another-model-family) is complete.

| Family key | Current state | Can use the YOLO11 commands unchanged? |
| --- | --- | --- |
| `yolo11` | YOLO11n seed, adapter, search-space checks, and three-seed baseline runner are wired. | Yes. |
| `dfine` | A project-owned D-FINE-inspired prototype is registered. It lacks the YOLO11-specific baseline runner and search-space policy. | No. Finish its family setup first. |
| `tiny_yolo11` | Registered, but its protocol uses a smaller proxy dataset rather than the exported challenging-search split. | No, if the goal is a directly comparable FRED search. |
| `rtdetr`, `faster_rcnn` | Seed-area files exist, but their adapter files are empty and they are not registered. | No. Finish the family setup first. |

**Current defaults:** 21 total generations (generation 0 plus 1–20), eight initial YOLO11 built graphs, 20 training epochs per candidate, and one search training seed. The exported **search** train/validation sequences both come from `challenging_train`. Do not use `challenging_test` to select architectures. The current YOLO11 run trains from scratch; this differs from the seed and initialization policy in the [formal Phase 1 plan](PHASE1_DETECTION_PLAN.md), so document that distinction when reporting results.

## 1. Set up the checkout and choose a run

Do this from the same checkout that will submit the SLURM jobs. Replace the example tag and port. A **new** experiment needs a new tag; a **resume** needs the original tag and the same generation count, code commit, protocol, and seed.

**Change "yolo11" to the appropriate model being evolved**

```bash
module load uv
uv sync
readlink src/cfg/constants.py
export FRED_SEED_FAMILY=yolo11
unset FRED_EVENTS
export FRED_RUN_TAG=yolo11_20261022_eot0_trial1
export FRED_NUM_GENERATIONS=21
export LLMGE_PORT=8479
export SUBMIT_ISLAND_CONTROLLER=0
mkdir -p run_job_outputs/{export,baseline,server,islands,evolution,evaluation}
```

**Check:** `readlink` must print `constants_fred_yolo11.py`. The tag must start with the family key and follow `<family>_YYYYMMDD_eot0_<name>` for the current EoT-off configuration. The port must be unused. `server.sh` otherwise defaults to submitting another island controller, which is why `SUBMIT_ISLAND_CONTROLLER=0` is explicit.

```bash
uv run python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.RUN_TAG, c.num_generations, c.SEED_NETWORK)'
git rev-parse HEAD
git status --short
```

**Check:** the printed family, tag, `21`, and seed file are the intended ones. Record the commit and review local code changes before a comparable run. The baseline and resume checks include the Git commit, so finish version-history changes **before** training baselines or starting a new run. Existing generated outputs may remain untracked.

Run these focused local regression tests once for each frozen code version. They do **not** submit jobs, train a full model, or start an LLM server because auto-start is disabled:

```bash
LLMGE_AUTO_START_SERVER=0 uv run python -m pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_challenging_split.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_protocol_fingerprint.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_result_contract.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_yolo11_search_space.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_yolo11_training_protocol.py \
  tests/operators/test_fred_population_integrity.py \
  tests/operators/test_fred_seed_generation.py
```

**Check:** all tests pass. For a different family, add and run its adapter, training-protocol, and search-space tests instead of treating YOLO11 tests as proof that the new model works.

## 2. Prepare the search data (once per frozen data protocol)

Authenticate to Hugging Face if needed. The `whoami` command checks access without printing your token.

```bash
uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])'
```

If that fails because you are not logged in, run this and then retry `whoami`:

```bash
uv run python -c 'from huggingface_hub import login; login()'
```

If this exact search export is already complete, skip the array job and run the **finalize/check** commands below. Otherwise submit eight CPU shards:

```bash
EXPORT_JOB_ID=$(sbatch --parsable --export=ALL --array=0-7 -c 4 --mem=16G -t 4:00:00 \
  --output=run_job_outputs/export/export-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8')
EXPORT_JOB_ID=${EXPORT_JOB_ID%%;*}
echo "$EXPORT_JOB_ID"
```

Wait for every array task, then check its exit code. **An empty `squeue` is not proof of success.**

```bash
squeue -j "$EXPORT_JOB_ID"
sacct -j "$EXPORT_JOB_ID" --format=JobID,State,ExitCode -P
rg -n 'Traceback|ERROR|Exception' run_job_outputs/export/export-"$EXPORT_JOB_ID"_*.out
```

**Check:** tasks `0`–`7` are `COMPLETED` with `0:0`. The final `rg` may return no matches; that is good. Repair and rerun only failed shards with the same shard count.

Finalize after all shards pass:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize \
  | tee run_job_outputs/export/finalize-search.log
rg 'search_train_step50: 152 sequences, 10053 frames' run_job_outputs/export/finalize-search.log
rg 'search_val_step10: 20 sequences, 6733 frames' run_job_outputs/export/finalize-search.log
```

**Check:** both `rg` commands print a line. These counts are for the current released Event frames and split; a changed protocol requires its own expected counts. If finalization reports missing sequences, fix those shards first. Review timestamp reports in `sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging/*/reports/` before treating the export as verified.

## 3. Train the stock baseline (once per family, code, and protocol)

A baseline is the **unmodified seed**, trained independently with seeds 0, 1, and 2 through the same [candidate runner](../phase1_detection/candidate_runner.py) and protocol as candidates. This concrete command is for `yolo11` only: the existing [baseline runner](../phase1_detection/baseline_runner.py) hardcodes YOLO11. Do not submit it for another family. If the three `seed_N` directories already exist, skip submission and try `summarize` below; `run-seed` refuses an existing directory.

```bash
BASELINE_JOB_ID=$(sbatch --parsable --export=ALL --array=0-2 -G 1 -C H200 \
  -c 12 --mem=32G -t 4:00:00 \
  --output=run_job_outputs/baseline/seed-%A_%a.out \
  --wrap='module load uv; uv run python sota/FRED_LLM_GE/phase1_detection/baseline_runner.py run-seed --seed "$SLURM_ARRAY_TASK_ID"')
BASELINE_JOB_ID=${BASELINE_JOB_ID%%;*}
echo "$BASELINE_JOB_ID"
sacct -j "$BASELINE_JOB_ID" --format=JobID,State,ExitCode -P
```

**Check:** all three tasks finish `COMPLETED`/`0:0`, with `seed_N/run.json` and `seed_N/training_args/seed/args.yaml`. The YOLO11 arguments must show the frozen optimizer and `pretrained: false`. Review the logs for AMP self-check/download messages; a separate Ultralytics AMP reference model is not candidate initialization.

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/baseline_runner.py summarize
uv run python -m json.tool sota/FRED_LLM_GE/baseline_validation/phase1_detection/yolo11_stock/summary.json | head -60
```

**Check:** `summarize` succeeds and reports three seeds, mAP50, mAP50:95, parameters, and wall time. It rejects stale or incomplete results. If a seed directory exists but failed, inspect it before retrying; do not overwrite evidence blindly. The stock seed is also evaluated once as `seed` in evolution's generation 0; that is separate from this three-seed noise estimate.

## 4. Start the LLM server, then the evolution controller

Submit **one** server. Keep the same family/tag/port environment in this shell. The server requests two H200 GPUs; `run.sh` launches the controller, which submits operator and candidate-evaluation jobs.

```bash
SERVER_JOB_ID=$(sbatch --parsable --export=ALL server.sh)
export LLMGE_SERVER_JOB_ID=${SERVER_JOB_ID%%;*}
echo "$LLMGE_SERVER_JOB_ID"
squeue -j "$LLMGE_SERVER_JOB_ID" -o '%i %T %R %L'
tail -f "run_job_outputs/server/slurm-$LLMGE_SERVER_JOB_ID.out"
```

**Check:** wait for `Uvicorn running`, `Serving on port 8479` (or your chosen port), and `Skipping island controller submission` in **this job's log**. Press Ctrl-C to leave `tail -f`; it does not stop the server. If pending, use the `squeue` reason. If the server exits, inspect the traceback before proceeding. The default server model path points to Llama-3.3-70B-Instruct; `mixtral` in operator logs is a client label.

Only after the server is ready:

```bash
RUN_JOB_ID=$(sbatch --parsable --export=ALL run.sh)
RUN_JOB_ID=${RUN_JOB_ID%%;*}
echo "$RUN_JOB_ID"
tail -f "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
```

**Check:** the log prints your run tag, checkpoint directory, generation limit, and the server job ID. Candidate jobs appear in `run_job_outputs/evaluation/`; operator jobs appear in `run_job_outputs/evolution/`. The server has an eight-hour allocation and the controller has 7 h 45 m, so a long run may need a fresh server/controller pair to continue.

## 5. Monitor, resume, and plot

While running:

```bash
squeue -u "$USER" -o '%i %T %j %R %L'
rg -n 'Generation [0-9]+ (distinct graphs|failures)|Finished Generation|Best Fitness|depleted|paused_for_server_refresh|Traceback' \
  "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
cat "fred_runs/$FRED_RUN_TAG/status.json"
```

**Check:** generation 0 contains the stock seed and eight distinct successful built graphs. Later populations can be smaller. Inspect failure reasons in the controller and child-job logs. A pending evaluation job may simply be waiting for a GPU; check its SLURM reason.

A 21-generation run has succeeded when `status.json` says `"status": "complete"` and `"last_completed_generation": 20`, the log says `Finished Generation 20`, and both generation-20 checkpoint files exist:

```bash
test -f "fred_runs/$FRED_RUN_TAG/checkpoints/checkpoint_gen_20.pkl"
test -f "fred_runs/$FRED_RUN_TAG/checkpoints/global_gen_20.pkl"
```

If the run pauses for server refresh, wait for the old controller to exit. Keep the **same** tag, 21-generation setting, code commit, seed, and protocol. Start a new server as in step 4, wait for `Uvicorn running`, set its new `LLMGE_SERVER_JOB_ID`, and submit `run.sh` once. It resumes from the latest matching checkpoint pair. Changing the generation count or protocol under the same tag is rejected by the checkpoint fingerprint. Do not run two controllers against one tag.

Create plots after any checkpoint exists:

```bash
uv run python analysis/scripts/fred_pareto_graphs.py \
  --family "$FRED_SEED_FAMILY" \
  --checkpoints "fred_runs/$FRED_RUN_TAG/checkpoints" \
  --models_dir "fred_runs/$FRED_RUN_TAG/candidates/models" \
  --output "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG"
test -s "analysis/results/analysis/fred_pareto/$FRED_RUN_TAG/pareto2d_final.png"
```

**Check:** the output folder contains the final two-panel plot, per-generation plots, `candidates.csv`, and `pareto_models.csv`. The plots include every valid candidate recorded through the last checkpoint, including the generation-0 stock seed. The independent three-seed baseline is **not** added as three more candidate points. See [analysis notes](../../../analysis/README.md).

## Add another model family

These are the three main code decisions:

| Question | Answer |
| --- | --- |
| Copy `yolo11_search_space.py`? | **No.** Create a new policy for the new model's genome. Reuse its interface and test ideas, not its YAML layer rules. |
| Edit `candidate_runner.py`? | **Yes, minimally.** Import/register the new adapter in both import paths and `ADAPTERS`. Keep shared data loading, scoring, and failure handling there. |
| Edit `baseline_runner.py`? | **The YOLO11 version cannot run another family as-is.** Add a separate family baseline runner or generalize it carefully. Keep baselines independent of LLM-GE. |

Then complete this file checklist **before** changing `FRED_SEED_FAMILY` and submitting a run:

| Action | File(s) | Ready when… |
| --- | --- | --- |
| **Create the unmodified seed and define mutable parts.** | [`seeds/`](../seeds/) | The family can rebuild both the stock seed and a candidate saved as `network_<id>.py`. The `.py` name is a controller convention; YOLO11 stores YAML inside it. |
| **Create the training adapter.** | New file under [`adapters/`](../phase1_detection/adapters/), implementing [`FamilyAdapter`](../phase1_detection/adapters/base.py) | `build`, `train`, `predict`, and `postprocessing` work; predictions are pixel `xyxy` boxes and scores. Parameter counting and saving are verified. |
| **Register the adapter.** | [`candidate_runner.py`](../phase1_detection/candidate_runner.py) | Both direct-script and package imports work; `ADAPTERS` includes the family. Run `uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py --help` and confirm the family appears. Add a small branch only if its training artifacts require one. |
| **Freeze the training recipe.** | [`protocol.yaml`](../configs/phase1/protocol.yaml) | `families.<key>` names the seed, uses `data: challenging_search`, and fixes the family's training settings. Seed and candidates use the same settings. |
| **Connect FRED run settings.** | `_FAMILIES` and `CANDIDATE_POLICY_MODULE` in [`constants_fred_yolo11.py`](../../../src/cfg/constants_fred_yolo11.py) | The family key selects the right seed, prompts, rules, timeout, and policy. This historical filename is the current shared FRED config. The run tag starts with that family key. |
| **Write prompts for this genome.** | New family folder under [`templates/`](../../../templates/) | Mutation rules and crossover templates describe what can change and the exact response syntax; one response from each operator can be parsed. |
| **Create architecture validation and reuse rules.** | New family policy under [`phase1_detection/`](../phase1_detection/), modeled on the *interface* of [`yolo11_search_space.py`](../phase1_detection/yolo11_search_space.py) | The policy validates protected fields, builds the model, hashes the built graph, rejects a no-op child, and keys cached fitness to the same protocol and seed. Wire it via `CANDIDATE_POLICY_MODULE`. |
| **Create an independent baseline.** | New family baseline runner, or a carefully generalized [`baseline_runner.py`](../phase1_detection/baseline_runner.py) | Seeds 0, 1, and 2 run through `candidate_runner.py --family <key> --model network`; results, recipe, and code identity are checked in a separate family output folder. The current runner hardcodes YOLO11. |
| **Record dependencies and test.** | [`protocol_fingerprint.py`](../phase1_detection/protocol_fingerprint.py), family tests, and the [Pareto plotter](../../../analysis/scripts/fred_pareto_graphs.py) | A changed external model-package version cannot silently resume; adapter/search-space/baseline tests pass; plot CSV paths identify real seed/candidate files. |

The policy functions the controller calls are `built_architecture`, `differs_from_parents`, `evaluation_cache_key`, and `write_cached_fitness`; the operator also needs response parsing/repair helpers suited to the new genome. Without a policy, the older generic operator path can run, but it lacks YOLO11's built-graph checks, duplicate reuse, and eight-distinct-graph generation-0 refill. Give the new family its **own** tests for one seed build/forward/backward pass, an invalid proposal, training results, and resume identity before a long run.

Normally **no changes are needed** to `export_split.py`, `map_reporter.py`, `run_improved.py`, `server.py`, `server.sh`, or `run.sh` when the family uses the same Event-frame split and existing interfaces. The current `rtdetr.py` and `faster_rcnn.py` adapter files are empty placeholders, so those families are **not ready to launch yet**.
