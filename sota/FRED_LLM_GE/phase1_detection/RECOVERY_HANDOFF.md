# YOLO26n-P2 and D-FINE recovery code handoff

This publishes the runtime corrections used for the October 19 five-hour
experiments on `fred-custom-seeds`. The eleven changed runtime files match the
verified runtime overlay in `fred-yolo26-p2-full-outputs-20261019T160847Z-3122367.tar.gz`.
The experiments ran base commit `3f72a5ae6415d5b3af5766932c949c79fd58ef3f`
plus that overlay, rather than the new commit publishing this handoff. Keep the
original reports and manifests intact when reproducing those results.

## Included corrections

- YOLO26n-P2 mutation/crossover derive channel choices from the seed, validate
  complete replacement sections, and retry up to three times with concrete
  rejection feedback. Adjacent `.proposal_attempts.json` files retain rejected
  proposals. No channel rounding or unchanged-parent fallback is used.
- Slurm templates preserve assigned CUDA devices for YOLO26n-P2 and D-FINE.
  LLM startup probes every allocated GPU and uses measured free memory. The
  singleton is published only after initialization succeeds.
- Local LLM HTTP requests have timeouts for those families. Managed controllers
  stop when their identified server job dies or readiness times out.
- Opt-in bounded sessions atomically write population/global-state pairs with
  SHA-256 commit markers. Resume ignores interrupted newer uncommitted pairs
  and refuses damaged committed pairs. Generation zero saves the pending
  initial population; it is setup, not a completed generation.
- Workers enforce an absolute deadline. Controller cleanup targets only the
  current user's jobs in that exact checkout.

The additional `analysis/scripts/fred_pareto_graphs_2d.py` leaves the existing
3D script intact. Follow the [analysis commands](../../../analysis/README.md)
and explicitly pass `--family YOLO26n-P2` or `--family D-FINE`: its CLI defaults
to a YOLO11 title. A title alone does not establish model identity.

## Integration and resume requirements

Baseline execution remains independent through `candidate_runner.py`. Select
`FRED_SEED_FAMILY=yolo26_p2` or `FRED_SEED_FAMILY=dfine` before generating Slurm
templates or starting processes. The shared configuration module is named
`constants_fred_yolo11.py` but selects all four families using that variable.

`src/fred_recovery_controller.py` requires a launcher to prepare frozen data
exports and a verified baseline, allocate the LLM server separately, and supply
`LLMGE_SERVER_JOB_ID` and a consistent `LLMGE_PORT`. Set
`LLM_SERVER_READY_TIMEOUT=1800` for cold LLM startup and
`LLMGE_EXPECTED_GPUS` to the server's actual allocation.

Bounded launchers must additionally:

1. Set `LLMGE_BOUNDED_SESSION=1` in the controller and arrange termination
   signals for cleanup.
2. Have the server publish an absolute Unix timestamp in
   `five-hour-deadline-${SLURM_JOB_ID}.txt` in the run directory. The original
   launcher used server start plus 17,760 seconds (4h56m).
3. Wrap both worker templates with `.venv/bin/python -m src.fred_session
   <worker-command>` and inherit `LLMGE_SESSION_DEADLINE` from the controller.
4. Wait for all owned jobs to drain before starting the next family; renewal
   is explicitly manual.

These modules do not replace data preparation or provide a complete new-job
launcher. Preserve the original launcher package for existing runs: resume pins
runtime, protocol and baseline hashes. Do not pull changed code into an existing
experiment and rewrite its hashes to force resume. Use separate experiment
directories for future changes. Training interrupted within an unfinished
generation may repeat; detector training does not resume midway through an epoch.

## Observed results and remaining work

YOLO26 completed its bounded session with checkpoint generations 0–21, but only
four candidate architectures obtained verified evaluations. All four were
larger and less accurate than the seed. Rejected proposals are not new evaluated
architectures, and evaluated-candidate failure counts exclude proposal failures.

D-FINE's controller failed during a Slurm RPC timeout after 1h31m. Scheduler
cleanup also failed, leaving its server allocated. This handoff still stops on
an unverified scheduler response; RPC retry and robust cleanup remain follow-up
work. Four candidate evaluations completed independently.

Two D-FINE candidates used exactly the baseline source, including the
highest-scoring record, with different trained weights and scores. Population
generation still needs unchanged/duplicate filtering; the smoke-generation AST
check alone does not provide that guarantee. Repeat-training variability needs
investigation. These results do not establish an architectural improvement.

Metrics use search validation, not held-out test data. One-epoch plumbing tests
are not interchangeable with the 20-epoch evaluations.

## Verification

Use pinned dependencies (`pyproject.toml`/`uv.lock`), including Ultralytics
8.4.175. From the repository root:

```bash
USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python -m pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_fred_recovery.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_fred_session.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_ultralytics_yolo26_p2.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_yolo26_controller_fitness.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_dfine_adapter.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_dfine_controller_fitness.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_family_isolation.py \
  analysis/tests/test_fred_pareto_graphs_2d.py
```

The handoff suite passed all 100 tests locally with Torch 2.6.0 and Ultralytics
8.4.175. Offline CUDA/scheduler fixtures use mocks; they do not establish a fresh
live allocation. Publishing this code does not launch new training jobs.
