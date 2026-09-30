# DG-P0-02-BRINGUP-2026-09-29: bounded YOLO11 smoke split

**Scope:** A technical bring-up using at most 32 Event frames from one project-train
sequence and 32 Event frames from one project-validation sequence, followed by a
one-epoch YOLO11 seed run. The user delegated this split choice on 2026-09-29.
This is not the frozen research split and its metrics are not benchmark results.

## Membership decision

- Pin the FRED repository at `2bf89c5376eda528431b62d6c60f2c13d8f95ab4`.
- Use the official challenging-train set as the development pool (172 sequences).
- Assign `challenging-train ∩ canonical-train` to project train (139 sequences).
- Assign `challenging-train ∩ canonical-test` to project validation (33 sequences).
- Keep the complete official challenging-test set held out (59 sequences).
- Group by complete sequence; never split frames from a sequence across sets.
- Record the exact membership in
  `configs/phase0/project_split_bringup_v1.json`.

The four pinned source split files have SHA-256 values:

| FRED split file | SHA-256 |
| --- | --- |
| `challenging/challenging_train_split.txt` | `6d8f705c1b2ec5c3521b7d85697a0e5ae5d4f2b11772c99ddd0cce95e52e2ab7` |
| `challenging/challenging_test_split.txt` | `a1f408317eaabbf034ea4c53e40f1cc5fbbc507e58881131ada85768c6afcc56` |
| `canonical/train_split.txt` | `35f425b9ee2438df6f99ad9f9ada7533ea66cf158b6709a9ffd663769532a510` |
| `canonical/test_split.txt` | `3b0ebb512bcb6b6e5d934c42223695cf1f9e395fec45f46195597b75a011b1b1` |

The FRED authors describe their canonical 80/20 split as balanced across
scenarios. Intersecting it with challenging-train yields an approximately
81/19 development split; scenario balance after intersection has not been
measured. Source: Magrini et al., *FRED: The Florence RGB-Event Drone Dataset*,
Section 3, https://arxiv.org/pdf/2506.05163.

## Narrow exception and remaining gate

The Phase 0 plan requires a leakage/grouping audit before freezing DG-P0-02,
including sequence lengths, drone/condition distribution, recording sessions,
and repeated scenes. Those data are not established by the source inventory or
the published split lists. The user requested an immediate split decision for
the technical run; postponing the full audit is limited to this smoke run.

The risk is that two sequence IDs may share a recording session or near-identical
scene. Therefore the validation score must not drive architecture selection,
LLM-GE evolution, or formal baseline claims. A later research run must replace
this file with a fully audited and frozen DG-P0-02 split. Smoke outputs remain
separate and disposable.

Before the smoke run, validate source hashes, exact partition coverage,
non-overlap, pinned dataset identity, and distinct train/validation sequences.
If any check fails, stop rather than substituting another split.

## Sequence selection for the technical smoke

The first preparation attempt selected train sequence `0`. Its pinned
`coordinates.txt` produced 183 out-of-bounds boxes against 1280×720 frames, so
Phase 0 correctly refused to publish a manifest. A second job inspected the
first eight train sequences and all eight failed solely on out-of-bounds boxes.
Train sequences `3` and `5` each had only two partial overlaps with the image
edge; their reports are under `data/fred_phase0/validation/bringup_6007084/`.

For this technical smoke only, the job explicitly enables the
`--allow-partial-out-of-bounds` policy. A partially visible box becomes a
warning in the validation report, while a fully outside box and all other
integrity errors remain blocking. The canonical box coordinates remain
unchanged. The YOLO11 adapter excludes frames with any out-of-bounds box; it
does not clip or repair labels. The manifest records the bring-up-only policy
and cannot be treated as a formal research manifest. This policy requires a
BRINGUP approval reference and named sequences; a complete-inventory build
cannot use it.

The job tries train sequence `3` first, then other sequence IDs among the first
eight in the assigned train split. It tries at most the first eight sequence
IDs in the assigned validation split. It records every inspection under the
job-specific validation directory. If none passes, the job stops. Split
membership does not change.

The export takes up to 32 in-bounds annotated frames from each selected
sequence so the one-epoch seed exercises labels. Full-sequence out-of-bounds
findings remain a separate data-quality issue for the formal research split and
target policy.

## GPU seed smoke observation

ICE job `6008570` completed one epoch on 32 train and 32 validation images and
wrote a checkpoint and result. Ultralytics assumed `n` and built about 2.59
million parameters even though the input checkpoint was `yolo11m.pt`. This is
a plumbing success only, not a valid YOLO11m baseline. Job `6008761` then
failed the new scale preflight before training: Ultralytics 8.4.165 overwrites
the YAML `scale` field from the model filename, and `candidate.yaml` has no
size suffix. The trainer now writes `yolo11m.yaml`, checks the rebuilt model
against the checkpoint's parameter count, and uses `seed_mscale_v1` so the
earlier artifacts are preserved.
