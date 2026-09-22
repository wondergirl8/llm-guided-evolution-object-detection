# P0-S08 Semantic and Performance Equivalence Evidence

**Recorded at:** 2026-09-22T17:39:48Z  
**Task:** `P0-S08`  
**Exit-gate status:** **Incomplete — required real-data comparison blocked**  
**Legacy-removal authorization:** **Not granted**

## Outcome

The old lease-based lifecycle and the prepared-window lifecycle produced exactly the
same canonical output on a four-sample synthetic FRED-compatible archive. The prepared
warm path performed no provider access, ZIP open, lease-file cycle, or persistent cache
mutation. Two spawned workers also completed read-only access successfully.

This is supporting contract evidence only. It does not pass `P0-S08`: the mandatory
pinned real-data fixture was unavailable locally, and permission to download it was not
granted during this task. The legacy implementation must therefore remain available,
and `P0-S09` must not start.

## Required real fixture and blocker

`P0-S01` deferred the following permitted challenging-train fixture to this task:

| Field | Value |
|---|---|
| Sequence | `67` |
| Pinned object | `train/67.zip` |
| Size | 334,861,415 bytes |
| SHA-256 | `746065def15d92fa3b3205837d9c40d084dd30b79b2257e956806c7571611230` |
| Official membership | challenging-train |
| Local state checked during this task | absent |

The repository, `/tmp`, Desktop, and the local Hugging Face cache were checked for this
exact object; no copy was found. A request to download the pinned 319 MiB object was
declined. The cached sequence 21 was not substituted because it is protected
challenging-test data. No held-out labels or content were inspected by this comparison.

The pinned metadata evidence used to identify the fixture remains:

| Evidence | SHA-256 |
|---|---|
| `source_inventory.json` | `2eecb09e078fb64babd108b7091b508a9c40eae7b6a54cbd2611ed2dbc36a0ff` |
| `official_challenging_split.json` | `2ea0070b5e343a9777ee9646c7ec4ebb0fa84f508b7dc651e658c93ba391a615` |

## Synthetic semantic comparison

The comparison generated one local FRED-shaped sequence-67 archive containing four
48x32 RGB/event pairs, four annotations with track metadata, `coordinates.txt`, and a
non-empty `Event/events.hdf5` member. Both implementations consumed the same archive,
configuration, official/project split values, and manifest metadata. The project split
used here was explicitly synthetic and is not a `DG-P0-02` approval.

Every required comparison was exact:

| Compared output | Result |
|---|---|
| Sequence membership | exact |
| Sample count and order | exact (`4`) |
| Sample IDs | exact |
| Frame indexes and timestamps | exact |
| RGB logical/member references and dimensions | exact |
| Event logical/member references and dimensions | exact |
| Annotations, boxes, classes, and track IDs | exact |
| Official and project split values | exact |
| Provenance | exact |
| Validation findings | exact (both empty) |
| Manifest record hash | exact |
| Manifest content identity | exact |

Canonical manifest identities:

```text
records_sha256: 514b5d1bbd1634335d4bfd76691d1fb6042dd5dfa2ffb9f9967b3b3fa00c8868
content_identity: 95f1c3054299fee02dffce30e3811986cc6e89d6a0bf6a5658a9a9e6abdc44ff
```

## Synthetic lifecycle measurements

Environment: Darwin 24.6.0 arm64, Python 3.12.4, multiprocessing start method
`spawn`. The archive was seeded locally before each cold lifecycle measurement, so
network calls were intentionally zero. These tiny-fixture timings are directional and
must not be used as the `DG-P0-06` performance acceptance result.

### Cold preparation

| Metric | Legacy extraction | Prepared sequence |
|---|---:|---:|
| Elapsed seconds | 0.009022 | 0.009710 |
| Local materializer calls | 1 | 1 |
| Network calls | 0 | 0 |
| ZIP opens | 2 | 2 |
| Cache-metadata atomic writes | 4 | 4 |
| Prepared metadata writes | 0 | 1 |
| Prepared completion-marker writes | 0 | 1 |
| Persistent files created after archive seeding | 10 | 12 |
| Final cache usage | 8,325 bytes | 8,899 bytes |

The prepared format has two intentional cold-only files: versioned sequence metadata
and the atomic completion marker.

### Warm repeated access

Both paths decoded the same RGB and event pair 240 times.

| Metric | Legacy extraction | Prepared window |
|---|---:|---:|
| Startup seconds | 0.001012 | 0.000977 |
| Repeated-read seconds | 0.154022 | 0.069085 |
| Samples/second | 1,558.22 | 3,473.98 |
| Provider calls | 0 | 0 |
| ZIP opens | 0 | 0 |
| Lease create/delete cycles | 240 | 0 |
| Persistent cache state changed | no | no |

Each legacy lease cycle creates and then deletes a marker, so the 240 legacy reads
caused 480 transient filesystem mutation operations. The prepared active window caused
none. The rate difference is useful evidence that the removed hot-path lease behavior
is not hiding a regression, but the absolute rates are not representative of real FRED
archives or the intended Phase 1 loader.

### Spawned workers and cache bound

Two spawned workers each decoded 40 samples on each lifecycle. All four worker
processes exited with code 0. No lease file remained. The prepared workers inherited
the coordinator's active immutable entry and did not access the provider.

The harness cache limit was 20,000,000 bytes. Final usage was 8,325 bytes for the
legacy cache and 8,899 bytes for the prepared cache, both within the limit. No
full-dataset mirror was created. Separately, `P0-S02` already establishes that the
149.90 GiB challenging-train archive set cannot fit in the committed 50 GiB cache;
the selected coordinated-window design remains necessary.

## Regression verification

| Check | Result |
|---|---|
| Phase 0 tests | `78 passed in 1.47s` |
| Python bytecode compilation | passed |
| Default configuration load | passed |
| `uv lock --check` | passed; 158 packages resolved |
| Ruff | unavailable in the current environment; no lint result claimed |

The repository-wide `git diff --check` still reports two pre-existing trailing-space
findings in `PHASE0_DATA_PLAN.md` and `PHASE1_DETECTION_PLAN.md`. This evidence task did
not alter those files.

## Exit-gate decision

Synthetic semantics, warm-path behavior, spawned-worker access, and bounded-cache
behavior are consistent with the simplification contract. However, this task cannot
establish real sequence membership/counts, real validation findings, real manifest
hash equivalence, actual transfer/decompression costs, representative disk usage, or
`DG-P0-06` loader performance.

To complete `P0-S08`, provide or authorize retrieval of the exact pinned sequence-67
archive, rerun both paths against it, and record the real cold/warm evidence. Until
then, retain the legacy comparison path and do not execute `P0-S09` removal.

## Subsequent project direction

After this record was written, the project explicitly directed `P0-S09` to begin and
the redundant legacy code to be removed despite the unresolved real-data comparison.
That direction authorizes the removal work but does not retroactively make the
`P0-S08` real-data exit gate pass. The synthetic equivalence results above remain the
available pre-removal comparison evidence; the missing sequence-67 acceptance run
remains a recorded limitation.
