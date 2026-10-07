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

## Current gate status

The implementation foundation is available, but Phase 0 is not frozen. Dataset-wide
content verification, the project-owned split decision, questionable-data policy,
visual review, held-out integrity run, and cold/warm performance acceptance remain
required by the completion gate.
