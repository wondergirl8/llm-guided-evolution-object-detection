# P0-S02 Cache and Runtime Operating-Model Decision

**Recorded at:** 2026-09-22T00:51:42Z  
**Task:** `P0-S02`  
**Decision:** Model B — coordinated active window  
**Exit-gate status:** Passed for operating-model selection; environment-specific
worker and prefetch tuning remains a required `DG-P0-06` benchmark

## Scope and change boundary

This record selects the permitted cache operating model from inventory-derived and
local runtime evidence. `P0-S02` makes no cache-policy, runtime, configuration, split,
manifest, dependency, or test change. It does not download FRED data.

The existing Phase 0 implementation remains the active baseline. Later tasks must not
treat this design decision as permission to weaken object verification, atomic
publication, path safety, bounded-capacity enforcement, or deterministic sample
semantics.

## Evidence

### Representative workload and bounded cache

The representative Phase 1 workload for this decision is the complete official
challenging-train set. It is the conservative workload required to decide whether a
complete run-scoped materialization can fit; it is not a new or approved project split.

The pinned source inventory and official split record under
`/tmp/fred-phase0-verification-20260921/metadata` provide the following archive
footprints:

| Inventory group | Archives | Archive bytes | GiB |
|---|---:|---:|---:|
| Official challenging-train | 172 | 160,955,735,126 | 149.9017 |
| Official challenging-test | 59 | 43,707,436,584 | 40.7057 |
| Complete pinned release | 231 | 204,663,171,710 | 190.6074 |

The committed default cache limit is 53,687,091,200 bytes (50 GiB). Challenging-train
archives alone require 2.998 times that limit, before selective extraction output,
entry metadata, or temporary atomic-publication space is counted. Consequently, the
complete official challenging-train workload cannot remain prepared and pinned in the
default bounded cache.

Local sequence 21 supplies an extraction sanity check, not a dataset-wide estimator:

| Sequence-21 quantity | Value |
|---|---:|
| Source archive | 89,823,266 bytes |
| Selective runtime members | 6,963 files |
| Selective runtime member bytes | 120,231,854 bytes |
| Runtime-member/archive ratio | 1.338538 |
| Optional raw HDF5 member | 26,436,347 bytes |

Sequence 21 is official challenging-test data, so it is protected evidence and is not
a simplification development fixture. Its ratio is not extrapolated across all 172
training archives. It does show that extraction is additional capacity pressure, not
evidence that the 149.90 GiB archive workload could fit into 50 GiB.

At measurement time the current volume had 668,671,796 KiB (approximately 637.70 GiB)
available. That host-level free space does not override the configured 50 GiB cache
contract or justify an unbounded local mirror.

Evidence identities inherited from `P0-S01`:

| Evidence | SHA-256 |
|---|---|
| `source_inventory.json` | `2eecb09e078fb64babd108b7091b508a9c40eae7b6a54cbd2611ed2dbc36a0ff` |
| `official_challenging_split.json` | `2ea0070b5e343a9777ee9646c7ec4ebb0fa84f508b7dc651e658c93ba391a615` |

### Execution environment and unresolved tuning inputs

The current environment is macOS 15.7.4 on arm64 with 12 logical CPUs. Python reports
the multiprocessing start method as `spawn`.

Phase 1 is still a zero-byte scaffold. It defines no DataLoader worker count, prefetch
factor, persistent-worker setting, batch size, approved project split, or sequence
ordering. The Phase 0 and Phase 1 plans explicitly require those environment-specific
settings to be measured and frozen through `DG-P0-06`; inventing them in this
pre-implementation decision would provide false assurance.

The operating model therefore establishes these assumptions now:

- one Phase 0 coordinator owns preparation, pinning, release, and eviction;
- spawned DataLoader workers are read-only cache consumers and never contact the
  provider, publish entries, or evict entries;
- the active-window unit is a complete FRED sequence because the current source and
  runtime materialization unit is a sequence archive;
- ordering is deterministic and supplied by the eventual approved project split;
- active entries remain pinned until all consumers for that window have finished;
- a next window is admitted only when its worst-case publication space fits without
  evicting an active entry;
- the exact window width, worker count, prefetch depth, and persistent-worker setting
  remain unset until the intended Phase 1 hardware and loader can be benchmarked.

This is a deliberately incomplete tuning profile, not an implicit choice of zero
workers or disabled prefetching.

## Model evaluation

### Model A — run-scoped read-only cache

Model A is rejected for the representative workload under the committed default
capacity. Its prerequisite—preparing and pinning the complete workload before the
run—fails on archive bytes alone by 107,268,643,926 bytes, before extracted entries
and atomic-publication headroom.

This does not prohibit Model A for a future explicitly approved workload whose measured
archive, extraction, metadata, and publication headroom all fit within its configured
capacity. Such a change would require a new recorded `DG-P0-06` decision.

### Model B — coordinated active window

Model B is selected. It preserves bounded storage for a workload larger than the
cache and gives cache mutation a single owner. The coordinator prepares and pins an
explicit sequence window; workers only read those pinned complete entries. After all
workers release the window, the coordinator may evict inactive entries and prepare the
next deterministic window.

Expected access pattern:

1. On a cold start, the coordinator downloads and safely publishes each required
   sequence in the current window before exposing it to workers.
2. During the active window, every sample read is local and read-only. Workers perform
   no provider request, source-ZIP reopening, extraction, publication, or eviction.
3. Repeated access within a retained warm window reuses the same verified complete
   entries.
4. At a window boundary, the coordinator waits for leases to drain, releases pins,
   evicts only inactive entries when required, and atomically prepares the next window.
5. A repeated run may reuse compatible warm entries, but correctness and sample
   identity cannot depend on cache warmth.

## Frozen safety boundary for later tasks

Later simplification work may reduce implementation complexity only if it preserves
the following observable operating contract:

- exactly one coordinator mutates the cache;
- worker processes receive read-only access to explicitly pinned complete entries;
- active entries cannot be evicted;
- incomplete or incompatible entries are never exposed;
- provider access and extraction occur outside per-sample worker reads;
- capacity includes archive, extraction, metadata, and atomic-publication headroom;
- cold and warm access produce identical canonical samples and provenance; and
- cache windowing cannot change membership, ordering, annotations, modalities, or
  scientific identity.

## Exit-gate assessment

Inventory-derived evidence supports Model B and disproves Model A for the complete
official challenging-train workload under the 50 GiB default. The `P0-S02` operating-
model selection gate passes, so `P0-S03` may freeze this safety contract in tests.

The tuning portion of `DG-P0-06` remains open. Tasks that implement or simplify cache
lifecycle behavior must not hard-code a window width, worker count, prefetch factor,
persistent-worker policy, or throughput claim until the approved split and intended
Phase 1 execution environment are available for cold/warm benchmarks.
