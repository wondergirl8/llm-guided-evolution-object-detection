# Continue a checkpointed FRED run across server allocations

The optional [continuation supervisor](../scripts/continue_run.py) uses the existing
`server.sh`, `run.sh`, `server.py`, and `run_improved.py` without editing them.
It does not alter genomes, training budgets, fitness, or checkpoint contents.
Its default `start` command is a read-only plan. Only `--submit` launches jobs.

## Current SpikeYOLO continuation point

As checked on November 6, run `spikeyolo_20261101_eot0_trial1` has matching
generation-4 checkpoint files. The real checkpoint loader verifies their
fingerprint and resumes at generation 5 with five population entries.
The configured total is 21 generations, numbered 0 through 20. Failures remain
failures and are handled by the existing evolution loop. The checkpoint loader
always selects the latest matching pair, so these example numbers will advance.

The first automatic chain started server `6824766` and controller `6824810`.
Its CPU supervisor `6824767` suffered Slurm `NODE_FAIL` at 10:05 on November 5.
The server/controller continued until their time limits at 17:38, but no live
supervisor remained to renew them. Generations 3 and 4 were saved successfully.
There are five saved generations including generation 0. Start a new chain with
the command below to resume; the old chain and its failure evidence are retained.
Do not try to run the old supervisor with changed source: its source hash is frozen.

Controller `6818751` saved generation 2 and then failed on
`Unrecognized server time remaining from squeue: '40:57'`. Slurm uses `MM:SS`
below one hour, while the existing controller expects `HH:MM:SS`. The supervisor
recognizes this exact error as an allocation-refresh condition **only after a new
generation checkpoint exists**. It preserves the traceback and records the reason.
It does not patch or replace the shared time parser.

## Review, then submit

From the repository root, first inspect the plan:

```bash
.venv/bin/python sota/FRED_LLM_GE/scripts/continue_run.py start \
  --family spikeyolo --run-tag spikeyolo_20261101_eot0_trial1 \
  --generations 21 --cycles 3 --port 8479
```

Expect `next_generation: 5`, `population: 5`, and `submit: false` at the current
checkpoint. No jobs are submitted by this command.

To authorize a bounded automatic continuation, run the same command with `--submit`:

```bash
.venv/bin/python sota/FRED_LLM_GE/scripts/continue_run.py start \
  --family spikeyolo --run-tag spikeyolo_20261101_eot0_trial1 \
  --generations 21 --cycles 3 --port 8479 --submit
```

This permits **at most three new eight-hour server windows** (up to 24 server
allocation hours, each using the existing two-H200 request). It stops earlier
when generation 20 completes or an unexpected error occurs. Choose a different
cycle limit explicitly if a larger allocation budget is intended. Queue wait
times still apply; automatic submission does not guarantee immediate resources.

Keep the same tag, generation count, Git commit, dependencies, protocol, and
execution code throughout this experiment. Do not commit the new supervisor
mid-experiment: the original checkpoint fingerprint includes the Git commit.
The chain also records hashes of its own source and the shared execution files.

## What each window does

1. Submit the unchanged server script with `SUBMIT_ISLAND_CONTROLLER=0` and the
   frozen FRED environment. `UV_NO_SYNC=1` preserves installed dependencies.
2. Submit a one-CPU, 4-GB supervisor in `pace-cpu`, dependent on the server's
   start (`after:<server-job-id>`). Its 18-hour wall limit allows time for model
   loading, controller queueing/execution, and draining existing child jobs.
   The supervisor is eligible for Slurm requeue after a CPU node failure, with
   at most two restarts per supervisor. It resumes monitoring an already recorded
   controller and does not submit another one. Logs append across restarts.
3. Verify this server's log reports application startup, its HTTP endpoint is
   ready, and `hostname.log` identifies the same node. Then submit **one** `run.sh`
   controller with the new `LLMGE_SERVER_JOB_ID`.
4. Wait for the controller. Verify a new matching checkpoint pair before renewal.
   Accept a clean server-window pause, controller walltime expiration after
   checkpoint progress, or the specifically recognized server-window exceptions.
5. Release the server owned by this cycle, wait for its release, and wait for
   this run's existing child jobs to finish before queuing the next pair.

The supervisor may release an allocation early when the controller stops with
too little server time remaining. It never cancels model-evaluation jobs.
When first started, it refuses existing server/controller/operator/evaluation
jobs from this checkout to avoid conflicts with the shared hostname file. Do not start
another manual controller or server while a chain is active.

## Monitor and stop

```bash
.venv/bin/python sota/FRED_LLM_GE/scripts/continue_run.py status \
  --run-tag spikeyolo_20261101_eot0_trial1
squeue -u "$USER" -o '%i %j %T %R %L'
```

The status command prints the newest chain's state file, live Slurm states,
server/controller/supervisor IDs, cycle count, checkpoint progress, and any error. Its records and
supervisor logs live under `fred_runs/<tag>/continuation/<timestamp>/`.
Original server, controller, operator, and evaluation logs stay in their usual
locations. Check Slurm as well as the state file; a killed supervisor cannot
update its own status afterward. If Slurm reports a terminal supervisor while its
recorded status still says it is working, `status` reports `supervisor_stopped`
and preserves the old value as `recorded_status` without rewriting the state file.

To stop automatic renewal, cancel the **current supervisor ID** printed by
`status`. Cancel its listed controller/server too if those jobs should stop.
Use the specific IDs rather than cancelling unrelated jobs by name. Any model
evaluation jobs are separate and remain running unless deliberately cancelled.

The chain stops on startup/CUDA failures, unexpected controller failures, manual
cancellation, changed fingerprints/code, missing checkpoint pairs, or a window
that produces no new generation checkpoint. Child-job draining is bounded at
five hours; it stops for review if those jobs have not finished. The supervisor
does not extend allocation limits or automatically repair partial generations.

Requeue applies to the CPU supervisor only; it does not request extra GPU server
windows beyond `--cycles`, and scheduling after node failure is not guaranteed.
If a supervisor dies between submitting a job and durably recording its ID, it
stops for manual review rather than risk a duplicate submission. Repeated CPU
node failures, supervisor walltime expiration, and manual cancellation may still
require a new chain once existing jobs have finished. Unknown application or
scientific failures continue to stop the chain.

## Checkpoint boundary limitation

Resume uses the latest **completed generation**, not arbitrary in-memory state.
If a controller times out mid-generation, finished work after the latest saved
checkpoint remains on disk but is not automatically reconstructed into the
population or cache. Some offspring work may therefore be repeated. The chain
waits for existing jobs to finish to avoid overlapping old and new controllers.
Generation 0 without a checkpoint still requires explicit audited recovery;
this utility intentionally refuses to start a fresh experiment.

For manual continuation, use the same environment, start a new server, wait for
readiness, export its new `LLMGE_SERVER_JOB_ID`, and submit `run.sh` once. A separate
`--resume` flag is unnecessary.
