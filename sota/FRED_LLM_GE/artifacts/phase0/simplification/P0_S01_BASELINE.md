# P0-S01 Known-Good Baseline Record

**Recorded at:** 2026-09-22T00:37:59Z  
**Task:** `P0-S01`  
**Git branch:** `MosesTheRedSea-main`  
**Git HEAD:** `3747efb5efcfbee5354de67d32f81035b8858e29`  
**Exit-gate status:** Passed — real-data fixture explicitly deferred to `P0-S08`

## Scope and change boundary

This record establishes the pre-simplification Phase 0 baseline. No runtime,
configuration, split, manifest, or validation implementation was changed by `P0-S01`.

The working tree was already dirty before `P0-S01`. All modified and untracked Phase 0
implementation, configuration, artifact, and test files other than the simplification
guide and this baseline record predate the simplification effort. In particular, the
existing edits to `docs/PHASE0_DATA_PLAN.md` and `docs/PHASE1_DETECTION_PLAN.md` are
user-owned planning changes and must remain untouched by simplification tasks.

No Git commit, stash, reset, checkout, or branch change was performed.

## Recovery snapshot

A non-destructive snapshot of `pyproject.toml`, `uv.lock`, and the complete
`sota/FRED_LLM_GE` tree was created outside the repository:

```text
/tmp/fred-phase0-p0-s01-baseline-20260922T003759Z.tar.gz
SHA-256: 39641aab45216e5267394bfc8fdb5c939dc1b3498cc8a943d2623a716de9c027
Size: 193 KiB
Archive entries: 170
```

The archive excludes generated `__pycache__` directories and `.DS_Store` files. It is
a local recovery snapshot, not a source-controlled research artifact. The canonical
per-file hashes for implementation, configuration, tests, and dependency files are in
`P0_S01_FILES.sha256`.

Before a future task changes a baseline file, its current hash must either match this
record or the difference must be identified and preserved as a newer user change.

## Runtime and interface identity

```text
Python: 3.12.4
uv: 0.6.17
Phase 0 extraction version: fred-runtime-selective-v2
```

Current public package exports:

```text
AccessMode
FREDDataset
FREDSample
HFFredSource
Modality
Phase0Config
ProjectSplit
load_config
```

## Verification results

| Check | Result | Evidence |
|---|---|---|
| Complete Phase 0 tests | Passed | 55 passed in 0.44 seconds |
| Flake8 | Passed | `phase0_data`, maximum line length 127 |
| Python compilation | Passed | complete `phase0_data` package |
| YAML parsing | Passed | 6 configuration YAML files parsed |
| Dependency lock | Passed | `uv lock --check`, 158 packages resolved |
| Simplification-scope diff check | Passed | implementation, configuration, lock, and guide are clean |
| Repository-wide diff check | Pre-existing failure | trailing whitespace in the status line of each user-modified Phase 0 and Phase 1 plan |

The two repository-wide whitespace findings were not introduced or corrected by this
task:

```text
sota/FRED_LLM_GE/docs/PHASE0_DATA_PLAN.md:4
sota/FRED_LLM_GE/docs/PHASE1_DETECTION_PLAN.md:4
```

## Existing local FRED evidence

The following previously produced local-only evidence exists under
`/tmp/fred-phase0-verification-20260921`:

| Artifact | SHA-256 |
|---|---|
| `metadata/source_verification.json` | `b8aa13cc9ceea50a820b7642afacbed4e0ff2317b31d9c195ed55617d07fe42a` |
| `metadata/source_inventory.json` | `2eecb09e078fb64babd108b7091b508a9c40eae7b6a54cbd2611ed2dbc36a0ff` |
| `metadata/official_challenging_split.json` | `2ea0070b5e343a9777ee9646c7ec4ebb0fa84f508b7dc651e658c93ba391a615` |
| sequence-21 archive cache metadata | `e5f2e2f1ade39a1b85e6c5ce4a378ee3bbd6728bcc907ef40fa635cf6ab18eee` |
| sequence-21 legacy extraction metadata | `a5f2d605f6be47996be8342732b5f5ca6c45e70af67c2f8edd14cb3cbdf251a0` |

Sequence 21 is a member of the official challenging-test split. Its locally cached
content is protected trusted-side evidence and must not be used as the ordinary
simplification development or human-visual comparison fixture. No existing canonical
manifest or complete validation report was found in the repository or the local
verification directory.

## Development fixture deferred to `P0-S08`

Sequence 67 is recorded as the initial low-transfer challenging-train candidate for a
future representative development fixture:

```text
sequence_id: 67
remote_path: train/67.zip
size_bytes: 334861415
sha256: 746065def15d92fa3b3205837d9c40d084dd30b79b2257e956806c7571611230
official_membership: challenging_train
local_materialization: absent
```

This task did not download sequence 67 because the simplification guide requires
explicit authorization for network-dependent materialization. On 2026-09-22 UTC, the
project directed work to continue to `P0-S02`, accepting the documented option to
defer real-data fixture materialization to `P0-S08`. Selection as a final
visual-validation fixture remains subject to the approved project split and
`DG-P0-05`; the sequence may be used earlier only for trusted automated development
equivalence checks consistent with project policy.

## Exit-gate assessment

The implementation baseline, verification results, hashes, dirty-tree boundary, and
recovery snapshot are complete. The project explicitly accepted deferring real-data
fixture materialization to `P0-S08` while retaining the current synthetic contract
tests as the immediate regression baseline. The `P0-S01` exit gate therefore passes.

The deferred fixture is still mandatory for `P0-S08`; this decision does not waive
real-data equivalence or authorize use of held-out sequence 21 as a development
fixture.
