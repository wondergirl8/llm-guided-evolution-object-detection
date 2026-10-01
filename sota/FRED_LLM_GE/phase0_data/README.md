# Phase 0 — FRED data infrastructure

This package implements the remote-first Phase 0 plan. It pins the official FRED
GitHub and Hugging Face revisions, inventories the remote release without downloading
it, verifies protected split files, materializes only explicitly requested sequence
archives, and exposes a model-neutral canonical sample contract.

## Safety properties

- no command downloads the complete 205 GB repository implicitly;
- every data read is pinned to the configured full dataset commit;
- cache entries are bounded, whole-entry evicted, and atomically promoted;
- the coordinator blocks preparation and eviction while an immutable prepared window is active;
- warm sequence reads use that active window without reopening the source ZIP or mutating cache state;
- ZIP traversal, duplicate members, and symlinks are rejected;
- runtime extraction allowlists RGB frames, released event frames, and `coordinates.txt`;
- raw `Event/events.hdf5` presence is validated from pinned archive metadata without routine extraction;
- malformed annotations and pairing/coordinate failures are blocking findings;
- the explicit `--allow-partial-out-of-bounds` exception is limited to named
  BRINGUP sequences; it warns on partially visible boxes without changing raw
  coordinates, while fully outside boxes and all other errors still block;
- released frame naming, sequence identity, event counters, duplicates, and interior gaps are validated before pairing;
- validation reports require complete expected sequence coverage and record execution failures separately from data findings;
- official challenging membership is checksum protected;
- project train/validation membership must be supplied with a `DG-P0-02` approval reference;
- candidate-facing evaluation mode never returns annotations;
- runtime loading reuses one read-only manifest connection per worker and decodes only the requested modality;
- held-out annotation overlays are prohibited.

## Setup

From the repository root, install the project dependencies. Configuration is in
`sota/FRED_LLM_GE/configs/phase0/default.yaml`. Runtime output defaults to
`data/fred_phase0/` and can be redirected with `FRED_PHASE0_ROOT`. Public access
needs no credential; if required by the provider, set the environment variable
named by `source.token_environment_variable` without committing its value.

## Canonical workflows

```bash
python -m sota.FRED_LLM_GE.phase0_data verify-source
python -m sota.FRED_LLM_GE.phase0_data inventory
python -m sota.FRED_LLM_GE.phase0_data build-official-splits \
  --inventory data/fred_phase0/metadata/source_inventory.json
```

Fetching is sequence-scoped and explicit:

```bash
python -m sota.FRED_LLM_GE.phase0_data fetch-sequence 0 \
  --inventory data/fred_phase0/metadata/source_inventory.json
```

## Prepared-sequence lifecycle

The source API separates mutable coordinator work from runtime reads:

- `HFFredSource.prepare_sequence(...)` downloads only when necessary, validates and
  selectively extracts the pinned sequence, then atomically publishes versioned
  metadata and a completion marker;
- `HFFredSource.activate_prepared_window(...)` validates and activates one immutable
  coordinator-owned sequence window;
- `HFFredSource.open_prepared_sequence(...)` reads an immutable handle from that
  active window without cache metadata access, lease files, downloads, extraction, or
  eviction;
- `HFFredSource.cleanup_cache(...)` performs coordinator-owned whole-entry capacity
  cleanup while preserving explicitly protected entries.

This is the sole sequence runtime lifecycle. The superseded extracted-sequence and
per-reader lease formats have distinct cache identities, are never accepted as
prepared entries, and are harmless if left in an existing disposable cache. Normal
bounded-cache pressure evicts them as inactive entries; deleting an inactive Phase 0
cache is also safe because the pinned remote release remains source truth.

`FREDDataset` consumes active prepared entries only. Sample access never downloads,
extracts, evicts, creates per-sample lease files, or updates cache metadata; an
inactive or missing prepared sequence raises an actionable error instead of falling
back to mutable cache work. Coordinators must prepare every sequence, enter one
`activate_prepared_window(...)` context, start and join all workers inside that
context, then close the window before cleanup or preparation resumes.

Inspection, manifest building, and smoke testing additionally require an approved
project-split JSON file with this shape:

```json
{
  "version": "project-split-v1",
  "grouping_unit": "sequence",
  "train": ["..."],
  "validation": ["..."],
  "held_out_test": ["..."],
  "approval_reference": "DG-P0-02 decision record"
}
```

The train and validation lists must exactly partition official challenging-train;
held-out test must exactly equal official challenging-test. Phase 0 does not create
this decision automatically.

Build only named sequences unless a deliberate full audit is intended:

```bash
python -m sota.FRED_LLM_GE.phase0_data build-manifest \
  --inventory data/fred_phase0/metadata/source_inventory.json \
  --official-split data/fred_phase0/metadata/official_challenging_split.json \
  --project-split /path/to/approved_project_split.json \
  --sequence 0
```

A complete scan requires both `--all-sequences` and `--confirm-full-scan`. It still
uses bounded sequence-level materialization rather than a permanent source mirror.

## Strict development-sequence audit on ICE

The bounded YOLO11 bring-up uses a named partial-box exception. It is only a
plumbing check; its manifest is not an approved research split. Before freezing a
development split, inspect the official challenging-train sequences without that
exception. The array job below fetches one pinned sequence per task through the
existing bounded cache and writes a separate report for each sequence. It never
inspects the held-out challenging-test sequences.

Start with eight tasks, one at a time, from the repository root
after the Phase 0 environment and source inventory are present:

```bash
mkdir -p data/logs
if test -x data/.venv-yolo11/bin/python && \
   test -s data/fred_phase0/metadata/source_inventory.json && \
   test -s data/fred_phase0/metadata/official_challenging_split.json && \
   test -s sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json; then
  sbatch --array=0-7%1 sota/FRED_LLM_GE/phase0_data/jobs/audit_development_sequences.sbatch
else
  echo "Phase 0 environment or pinned source metadata is missing"
fi
```

Check the submitted array ID and logs with `squeue -u "$USER"` and
`tail -n 80 data/logs/fred-dev-audit-<array-id>_<task-id>.out`. A task can
complete successfully with `data_status=failed`: that means its inspection
finished and recorded blocking source-data findings. An execution failure or
incomplete report still fails the Slurm task.

Summarize the pilot after its eight tasks finish:

```bash
REV=$(git rev-parse --short=12 HEAD)
REPORT_DIR="data/fred_phase0/validation/development_audit_${REV}"
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase0_data.development_audit \
  summarize --report-dir "$REPORT_DIR"
data/.venv-yolo11/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("inspected", d["inspected_sequence_count"], "/", d["expected_sequence_count"]); print("invalid data sequences", d["invalid_sequence_ids"]); print("finding counts", d["finding_counts"])' \
  "$REPORT_DIR/summary.json"
```

The pilot summary will say `complete=false` because it covers eight of 172
development sequences. After reviewing cost and findings, submit the remaining
indexes in batches that fit the ICE submission quota, then repeat the summary.
The `%1` array throttle limits concurrent execution but all array elements
still count toward Slurm's submitted-job limit. By default the report
directory names the checked-out code revision; the summary hashes each report
and the pinned inventory, official split, and project split. A complete scan is
evidence for annotation-policy and split review;
it does not approve a split. Class/condition balance, session or repeated-scene
grouping, visual review, and the Phase 0 decision gate still require review.

The first ICE pilot used code revision `6cadd2db302f` and found 846 partly
visible out-of-bounds boxes across eight sequences. The audit summary now
classifies such findings for review without changing their blocking severity.
To continue that same evidence set after updating the code, keep its report
directory explicitly. The 164-task array was rejected by ICE's per-user
submission limit; the first eight-task continuation is:

```bash
sbatch --export=ALL,FRED_AUDIT_REPORT_DIR=data/fred_phase0/validation/development_audit_6cadd2db302f \
  --array=8-15%1 sota/FRED_LLM_GE/phase0_data/jobs/audit_development_sequences.sbatch
data/.venv-yolo11/bin/python -m sota.FRED_LLM_GE.phase0_data.development_audit \
  summarize --report-dir data/fred_phase0/validation/development_audit_6cadd2db302f
```

Run `summarize` after the submitted tasks finish. Its `complete=true` means
all development sequences have reports, even if some have data-quality errors.
The `out_of_bounds_categories` field separates partly visible, fully outside,
and unclassified findings. See `docs/decisions/DG-P0-04-audit-findings.md` for
the provisional issue record; no formal bounds or export policy is approved.
After the `16-23%1` batch finishes, one packed eight-task array can cover the
remaining indexes without queueing 148 separate Slurm jobs:

```bash
sbatch --export=ALL,FRED_AUDIT_REPORT_DIR=data/fred_phase0/validation/development_audit_6cadd2db302f \
  --array=24-31%1 sota/FRED_LLM_GE/phase0_data/jobs/audit_development_packed.sbatch
```

Each array task scans every eighth sequence index starting at its task ID, up
to 171, within the same four-hour allocation. The shared cache therefore has
one active inspector at a time. If a task reaches its wall-time limit or fails
for an execution reason, resubmit the same `24-31%1` array after it has left
the queue: complete reports are verified and skipped, while missing reports
are retried. Inspect the new `fred-dev-pack-<array-id>_<task-id>.out` logs and
rerun the summary to see remaining coverage. A data-quality failure recorded
in a complete report does not fail the Slurm task.

## Current gate status

The implementation foundation is available, but Phase 0 is not frozen. Dataset-wide
content verification, the project-owned split decision, questionable-data policy,
visual review, held-out integrity run, and cold/warm performance acceptance remain
required by the completion gate.
