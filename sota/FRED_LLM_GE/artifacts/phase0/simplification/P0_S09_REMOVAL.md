# P0-S09 Legacy-Path Removal Record

**Recorded at:** 2026-09-22T18:54:54Z  
**Task:** `P0-S09`  
**Removal status:** Implemented and regression-tested  
**Final simplification gate:** Pending the previously waived real-data `P0-S08` run

## Authorization and scope

The project explicitly directed `P0-S09` to begin and requested removal of redundant
code and files after being shown that the required real sequence-67 comparison was
still unavailable. This record treats that direction as authorization to remove the
recoverable legacy comparison path. It does not claim that the missing real-data
acceptance run occurred.

The change is limited to the Phase 0 cache/source lifecycle, its direct CLI and
validation consumers, focused tests, operational documentation, and verification
records. The existing user-owned edits to `PHASE0_DATA_PLAN.md` and
`PHASE1_DETECTION_PLAN.md` were not modified.

## Removed production code

The following superseded runtime implementation was deleted:

- `SequenceMaterialization` and its duplicate archive/extraction state;
- `RUNTIME_EXTRACTION_VERSION` and the `fred-runtime-selective-v2` identity path;
- `HFFredSource.materialize_sequence(...)`;
- `HFFredSource.open_sequence(...)`;
- `_extraction_identity(...)`;
- `_materialization_from_extracted_entry(...)`;
- `_remember_materialization(...)` and `_sequence_materializations`;
- `CacheEntryUnavailable` and the legacy retry/fallback branches;
- per-reader `BoundedCache.lease(...)` markers;
- the `leases/` cache directory, UUID marker creation/deletion, and stale-reader scan.

Archive preparation no longer creates a reader lease. Under the selected single-
coordinator model, `prepare_sequence(...)` keeps its source archive explicitly
protected during atomic prepared-entry publication. Writer/capacity locks remain;
they are required for atomic publication and bounded whole-entry eviction.

## Sole remaining lifecycle

All repository runtime consumers now use one sequence lifecycle:

1. `prepare_sequence(...)` performs mutable, coordinator-owned remote access,
   validation, selective extraction, metadata creation, and atomic publication.
2. `activate_prepared_window(...)` establishes the immutable active window.
3. `open_prepared_sequence(...)` supplies read-only handles to inspection, datasets,
   and spawned workers.
4. `cleanup_cache(...)` runs only outside an active window.

The CLI inspection, manifest-build, and smoke workflows were converted to this path.
Manifest construction remains sequence-windowed, so it does not require a complete
local FRED mirror.

`inspect_sequence(...)` now consumes a `PreparedSequence` directly. Raw-HDF5 findings
use the pinned inventory archive reference instead of compatibility fields from the
deleted materialization object.

## Tests removed or converted

Four legacy-implementation tests were removed:

- per-reader lease eviction protection;
- interrupted legacy extraction publication;
- obsolete legacy extraction-version rejection;
- spawned legacy readers sharing lease markers.

The prepared-sequence selective-extraction test was converted rather than removed.
Safety coverage remains for atomic publication, interrupted preparation, incompatible
prepared markers, active-window eviction prohibition, warm read-only access,
two-worker spawned reads, provider isolation, cache bounds, modality selection, and
held-out label isolation.

The complete suite changed from 78 to 74 tests solely because the four legacy-only
tests no longer describe a supported interface.

## Legacy cache compatibility

Old `phase0_safe_zip_selective` entries have a different provider and logical identity
from `phase0_prepared_sequence` entries. Current code cannot accept them as prepared
content. They are safely ignored and remain ordinary inactive, disposable entries,
which bounded-cache pressure can evict as whole directories. Old `leases/` directories
and marker files are no longer inspected and cannot pin content.

Users who want immediate reclamation may remove an inactive Phase 0 cache and rebuild
on demand; this does not delete source truth because the pinned remote release remains
authoritative. No existing local cache or source data was deleted by this task.

## Verification

### Code and tests

| Check | Result |
|---|---|
| Focused cache/source/dataset/validation/CLI tests | `45 passed in 1.01s` |
| Complete Phase 0 suite | `74 passed in 1.07s` |
| Flake8, maximum line length 127 | passed |
| Python compilation | passed |
| Default configuration load | passed |
| YAML parsing | passed |
| `uv lock --check` | passed; 158 packages resolved |
| Changed-scope whitespace check | passed |
| Removed-symbol repository search | no production imports or calls found |

The repository-wide whitespace check still has the two pre-existing findings in the
untouched Phase 0 and Phase 1 plan status lines.

### Post-removal synthetic smoke

The exact four-sample synthetic fixture definition used by `P0-S08` was replayed
through the sole prepared path after deletion.

| Metric | Result |
|---|---:|
| Cold preparation | 0.010089 seconds |
| Warm decoded reads | 240 |
| Warm elapsed time | 0.069515 seconds |
| Warm sample rate | 3,452.47 samples/second |
| Warm provider calls | 0 |
| Warm ZIP opens | 0 |
| Persistent cache changes during warm reads | 0 |
| Cache usage / harness limit | 8,837 / 20,000,000 bytes |

The post-removal manifest record hash was
`514b5d1bbd1634335d4bfd76691d1fb6042dd5dfa2ffb9f9967b3b3fa00c8868`,
exactly matching the old/new synthetic comparison recorded by `P0-S08`.

The complete suite also includes the selected operating model's two-spawned-worker
test: both workers open the same active prepared entry read-only, provider access is
forbidden, cleanup is rejected during the active window, and both workers exit cleanly.

## Exit assessment

The requested redundant implementation has been removed: one preparation path and one
read-only runtime path remain, and the available semantic/safety/regression evidence
passes. No unrelated files were changed by this removal task.

The overall simplification cannot be represented as having satisfied every original
exit condition because the real sequence-67 `P0-S08` run was explicitly bypassed when
removal was directed. That limitation affects final real-data acceptance, not the fact
that the dual implementation and per-reader lease machinery are now gone.
