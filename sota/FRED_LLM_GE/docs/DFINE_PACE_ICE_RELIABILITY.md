# D-FINE PACE-ICE operator and evaluation timing

This note describes the D-FINE request and evaluation-clock behavior for a
new run. It applies to the legacy and `motifs_v1` D-FINE search paths.

## Local LLM requests

Both local generation clients use the bounded retry helper for D-FINE.
Connection and read timeouts are retried only after a transport timeout or
connection error. A completed HTTP response, including an error status, is
returned without retry. An exhausted request raises an error so the operator
slot is recorded as an infrastructure failure rather than silently producing
a candidate.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `LLMGE_LLM_REQUEST_ATTEMPTS` | `2` | Total request attempts for a transient transport failure. |
| `LLMGE_LLM_CONNECT_TIMEOUT_SECONDS` | `10` | Connection timeout for each attempt. |
| `LLMGE_LLM_HTTP_TIMEOUT_SECONDS` | `600` | D-FINE read timeout for each attempt. |

The retry delay is two seconds. This preserves the D-FINE per-attempt read
limit while giving one transient stalled connection a second chance.

## Evaluation queue and runtime

After submission, the controller reads `JobState` and `RunTime` from Slurm for
each D-FINE evaluation job. `PENDING` and `CONFIGURING` time is recorded as
queue time and does not consume the evaluation runtime allowance. The default
runtime allowance remains 3.5 hours and is checked against Slurm's `RunTime`
once the job is `RUNNING`. A controller restart recovers the job ID from the
durable evaluation-submission journal and re-reads its current Slurm runtime.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS` | `12600` | Runtime limit after the job starts. |
| `LLMGE_EVAL_QUEUE_TIMEOUT_SECONDS` | `0` | Optional independent queue limit; zero disables it. |
| `LLMGE_EVAL_TERMINAL_GRACE_SECONDS` | `300` | Wait for a completed or failed job's log/result before recording a missing result. |

The controller records the observed phase, submitted time, queue seconds,
running start estimate, runtime seconds, and scheduler errors in checkpoint
state. A temporary inability to verify Slurm state does not cancel or score
the candidate as a model failure. A verified runtime or configured queue
limit causes a visible failed outcome and attempts to cancel that job.

In bounded D-FINE sessions, confirmed evaluation jobs are preserved when the
controller allocation pauses; operator jobs and the local LLM server remain
subject to session cleanup. The next controller reuses the recorded job ID.
Tagged D-FINE sessions use their tagged checkpoint directory during recovery.

These changes have offline tests for transient and exhausted LLM requests,
Slurm state/runtime parsing, pending and running clocks, terminal grace,
checkpoint recovery, and session cleanup. They have not been exercised against
live PACE-ICE queue and GPU jobs in this workspace.
