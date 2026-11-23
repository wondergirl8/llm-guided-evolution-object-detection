# Faster R-CNN V2

Separate, frozen-source experiments with 20-epoch scratch training and input
width genes 640/960/1280. V1 source representations remain supported unchanged.
The shared SGD helper accepts optional batch-progress callbacks; old callers
retain their original epoch-level behavior.

## First Rollout

Search trains on 152 official challenging-training sequences, every fifth
frame; validation holds out the same 20 official-training sequences, every
tenth frame. Native-size verified exports are reused lazily. No test frames
are accessed during search.

After GPU smoke at all widths plus the largest legal configuration, run nine
stock baselines (three widths, seeds 0/1/2), then five LLM variants. The three
seed-zero stock runs plus five variants form eight unique initial candidates.
This is 14 twenty-epoch training evaluations, not merely eight GPU jobs.
The workflow stops at `awaiting_budget_approval` with initial Pareto, baseline,
throughput, memory, and compute estimates. It does not authorize itself to run
generations 1-20 or test.

```bash
# Print plan only; no files or jobs are created.
.venv/bin/python -m sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.cli launch \
  --tag faster_rcnn_scratch_20261108_eot0_v2_resolution20

# Create immutable source/config snapshot and start smoke-gated rollout.
.venv/bin/python -m sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.cli launch \
  --tag faster_rcnn_scratch_20261108_eot0_v2_resolution20 --submit

# Inspect the existing frozen workflow.
.venv/bin/python -m sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.cli status \
  --run fred_runs/faster_rcnn_scratch_20261108_eot0_v2_resolution20
```

Use `resume --run ...` for an interrupted supervisor. `stop --run ...` cancels
this run's jobs and its queued continuation. Explicitly stopped or blocked runs
are not blindly restarted. Use `approve-continuation --run ...` only after
reviewing `reports/initial_review.json`: this explicitly authorizes 20 more
generations, six globally novel successful candidates each, for 128 unique
seed-zero evaluations in total. Three additional seeds are not required for
every evolved architecture; baseline seed variation is reported separately.

## Eight-Hour Allocations

The CPU supervisor pre-arms an `afterany` successor before submitting workers.
It permits at most two concurrent one-H200 training/evaluation allocations.
All jobs are at most eight hours. A signal three minutes before timeout saves
training at the next completed optimizer update. Checkpoints also occur every
200 updates or five minutes and at epoch boundaries. They include the sample
permutation/cursor, optimizer momentum, and Python/NumPy/CPU/CUDA RNG state.

The supervisor adopts existing jobs by unique run/task/attempt identifiers,
resubmits timeouts/preemptions from saved progress, and stops after two
allocations without progress or more than three infrastructure retries.
Numerical/model failures do not receive valid fitness. LLM proposals are
recorded with raw responses/rejections; repeated or failed configurations cannot
consume new candidate slots. No random-search fallback is used.

Later working-tree changes do not change the executable snapshot. Recorded
dependency versions and external data-marker hashes must still match. JSON
records are compared structurally, not by whitespace. Environments are reused,
so do not upgrade their packages while the run is active.

## Approved Finalization

After generation 20, freeze every all-archive three-objective Pareto candidate,
including resolution, and add stock controls at the finalist resolutions.
Fresh final training uses all 172 official-training sequences at frame step 5
for 20 epochs, then official testing covers all 59 test sequences at every
frame. Test predictions resume per sequence and only held-out reports are
written. V2 performs no interim official-test evaluations. Historical interim
test exposure is recorded in the experiment protocol and cannot supply fitness.

Resolution changes compute without changing parameter count. Runtime and peak
memory are reported, but are not additional Pareto objectives. All three AP /
parameter objectives remain unchanged. Twenty epochs is the chosen budget,
not a guarantee of convergence or paper reproduction.
