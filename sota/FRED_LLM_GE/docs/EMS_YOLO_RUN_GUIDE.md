# Preliminary EMS-YOLO FRED run guide

This family is an infrastructure pilot, not a Gen1 reproduction. It starts from
the public EMS-ResNet34-style graph and neuron equations, initializes every FRED
run from scratch, and uses the existing released FRED Event-frame search export.
Each frame is letterboxed to 320x320 and repeated across four timesteps. The
adapter returns detections in the original 1280x720 coordinates to the shared
FRED evaluator.

## What is and is not comparable

EMS-YOLO uses the same challenging-train search sequences, single `drone` class,
scratch policy, mAP evaluator, random-seed mechanism, and LLM-GE objectives as
the YOLO11 run. Its eight-epoch, 320x320, four-timestep recipe is family-specific
and therefore is not an equal-compute comparison with the 20-epoch, 1280-wide
YOLO11 protocol. Compare evolved EMS candidates with the EMS seed first.

No `.pt` input, checkpoint download, or pretrained initialization exists in the
EMS adapter. The pinned upstream commit and compatibility port are included in
the protocol fingerprint.

## 1. Select the family

Run from the repository root:

```bash
module load uv
uv sync

export FRED_SEED_FAMILY=ems_yolo
export FRED_NUM_GENERATIONS=4
export FRED_RUN_TAG=ems_yolo_$(date +%Y%m%d)_eot0_pilot1
export LLMGE_PORT=8476
```

`FRED_NUM_GENERATIONS=4` means generation 0 plus generations 1--3. Increase it
only after the seed baseline and a short pilot finish within the allocation.

Check the resolved setup:

```bash
uv run python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.RUN_TAG, c.num_generations, c.SEED_NETWORK, c.CANDIDATE_POLICY_MODULE)'
uv run python -m sota.FRED_LLM_GE.phase1_detection.candidate_runner --help | grep ems_yolo
```

## 2. Verify the existing released Event-frame export

EMS-YOLO reuses the same export as YOLO11; it does not use the BN-SNN raw cache.

```bash
grep -F 'search_train_step50: 152 sequences, 10053 frames' run_job_outputs/export/finalize-search.log
grep -F 'search_val_step10: 20 sequences, 6733 frames' run_job_outputs/export/finalize-search.log
```

If either line is absent, follow the export and finalize commands in
`PHASE1_RUN_CHECKLIST.md` before submitting a baseline.

## 3. Run local-only contracts

```bash
uv run pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_ems_yolo.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_result_contract.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_protocol_fingerprint.py
```

These tests build the seed, run a small forward/backward pass, verify box
conversion and loss, exercise a one-step tiny model, reject unsafe genomes, and
check registration/provenance. They do not submit SLURM jobs.

## 4. Run seed 0 before the three-seed array

Create the log directory and submit one H200 job:

```bash
mkdir -p run_job_outputs/baseline

EMS_BASELINE_0=$(sbatch --parsable --export=ALL -G 1 -C H200 -c 12 --mem=48G -t 4:00:00 \
  --output=run_job_outputs/baseline/ems-seed0-%j.out \
  --wrap 'module load uv; uv run python -m sota.FRED_LLM_GE.phase1_detection.ems_yolo_baseline run-seed --seed 0')

echo "$EMS_BASELINE_0"
tail -f "run_job_outputs/baseline/ems-seed0-$EMS_BASELINE_0.out"
```

A successful log ends with finite `mAP50`, `mAP50:95`, a positive parameter
count, a saved model path, and `job done`. The first job is also the runtime and
memory measurement. Eight epochs at 320x320 are expected to take roughly one to
three hours on one H200, but use the measured seed-0 wall time for later planning.

## 5. Run seeds 1 and 2

After seed 0 succeeds:

```bash
EMS_BASELINE_12=$(sbatch --parsable --export=ALL --array=1-2 -G 1 -C H200 -c 12 --mem=48G -t 4:00:00 \
  --output=run_job_outputs/baseline/ems-seed-%A_%a.out \
  --wrap 'module load uv; uv run python -m sota.FRED_LLM_GE.phase1_detection.ems_yolo_baseline run-seed --seed $SLURM_ARRAY_TASK_ID')

echo "$EMS_BASELINE_12"
squeue -j "$EMS_BASELINE_12"
```

Summarize only after both array tasks finish successfully:

```bash
uv run python -m sota.FRED_LLM_GE.phase1_detection.ems_yolo_baseline summarize
uv run python -m json.tool sota/FRED_LLM_GE/baseline_validation/phase1_detection/ems_yolo_stock/summary.json | head -80
```

Do not begin evolution if the seed produces nonfinite results, empty predictions
throughout validation, a flat/nondecreasing training loss, or cannot finish its
eight epochs within the candidate allocation.

## 6. Start the LLM server and evolution controller

Use a new tag for a new run. Keep the selected family, tag, generation count,
and code commit unchanged when resuming that run.

```bash
SERVER_JOB_ID=$(SUBMIT_ISLAND_CONTROLLER=0 sbatch --parsable --export=ALL server.sh)
export LLMGE_SERVER_JOB_ID="$SERVER_JOB_ID"
echo "$SERVER_JOB_ID"
tail -f "run_job_outputs/server/slurm-$SERVER_JOB_ID.out"
```

Wait for `Uvicorn running`, then submit exactly one controller:

```bash
RUN_JOB_ID=$(sbatch --parsable --export=ALL run.sh)
echo "$RUN_JOB_ID"
tail -f "run_job_outputs/islands/slurm-$RUN_JOB_ID.out"
```

Monitor all relevant jobs and failure reasons:

```bash
squeue -u "$USER" -o '%.18i %.24j %.9T %.12M %.30R'
grep -aE 'Finished Generation|Best Fitness|invalid_architecture|retry_exhausted|train_or_eval_failed|depleted' \
  "run_job_outputs/islands/slurm-$RUN_JOB_ID.out" | tail -40
```

Generation 0 should include the unmodified EMS seed and distinct successful
built graphs. Candidate evaluations use the same eight-epoch recipe as the
baseline. If one candidate approaches four hours, stop planning a longer search
until the evaluation allocation or proxy recipe is deliberately revised under a
new tag and protocol fingerprint.
