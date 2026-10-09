# FRED YOLO11 candidate-generation check

The current next step is the combined seed-training/evaluation check in
[`INTEGRATION_RUNBOOK.md`](INTEGRATION_RUNBOOK.md). It includes a no-commit ICE
patch handoff and CPU/GPU job commands. The proposal and candidate checks below
already succeeded as ICE jobs 6009285 and 6009377; they need not be repeated.

The bounded seed job has already demonstrated one-epoch YOLO11m training on
ICE. `jobs/propose_yolo11_candidate.sbatch` is the next engineering check: it
uses the public `Qwen/Qwen2.5-Coder-7B-Instruct` model on one allocated H100 to
propose a single mutation of the three-gene YOLO11 `GENOME`. It does not use an
API key or an LLM server. The local model's resolved revision, exact prompt,
raw response, seed hash, and source commit are saved with the candidate.

The proposal is parsed as a literal assignment. Only one approved repeat may
change by one step. The complete generated source is checked against the
protected seed before the existing YOLO11m model-construction preflight runs.
If all responses are invalid, or if preflight fails, the Slurm job fails. No training
metric or fitness is assigned by this job.

From the checked-out `fred-yolo11-infrastructure` branch on ICE, after pulling
the commit that adds this job and confirming the earlier bounded data export
and `yolo11m.pt` still exist:

```bash
mkdir -p data/logs
sbatch sota/FRED_LLM_GE/phase1_detection/jobs/propose_yolo11_candidate.sbatch
```

The job writes `data/logs/fred-yolo11-propose-<jobid>.out` and, if the LLM
responds, `data/fred_generation_smoke/<jobid>/`. A valid source is named
`network_P<jobid>.py`. The directory contains `proposal.json` and
`prompt.txt`; `preflight_passed.txt` appears only after construction succeeds.
Outputs are ignored by Git and a repeated job gets a new candidate ID.

After a successful proposal job, train and evaluate that exact candidate for
one epoch on the existing bounded export to check the candidate execution
path. Pass the proposal job ID as the sole argument:

```bash
sbatch sota/FRED_LLM_GE/phase1_detection/jobs/train_proposed_yolo11_smoke.sbatch 6009285
```

The trainer writes `proposal_smoke_P<proposal-job-id>_results.csv` and a
matching JSON provenance record under the YOLO11 seed `results/` directory.
The output prefix prevents `run_improved.py` from interpreting this technical
score as an evolutionary candidate's fitness. This check does not select a
parent or advance a generation.

This is **candidate generation only**. The bring-up split and its 32/32 image
export cannot supply evolution fitness. Before a fitness-selected generation,
the project needs the Phase 0 leakage/grouping audit and frozen split, the
controlled Phase 1 training/fitness protocol, a documented non-evolutionary
baseline, and the LLM-GE execution bridge and pilot gate. The existing
`run_improved.py` entry point is not part of this check.
