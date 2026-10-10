# DG-P0-02-INFRASTRUCTURE-BRINGUP-2026-10-10: working scene-group split

## Approval and scope

Bill approved the reviewed 129/43 proposal in Codex on 2026-10-10:

> sure for now, i mean we're doing everything in the infrastructure (my branch) anyways

This approval adopts the explicit membership below for continued development on
`fred-yolo11-infrastructure`. It is a versioned working split, not a declaration
that Phase 0 is frozen or that the loader and baseline/evolution protocols are
approved. The prior sequence-only bring-up split and its completed run artifacts
remain available for reproduction. New development manifest builds should pass
`configs/phase0/project_split_scene_groups_infrastructure_v1.json` explicitly.

## Reviewed evidence

ICE scene-resume job **6140043** completed with exit `0:0` and a 172/172 summary.
The downloaded evidence package contained 172 records, 688 RGB thumbnails and
15 overview pages. All thumbnail hashes, snapshot-index coverage, summary/record
equality and partition coverage were checked locally. All four supplied samples
per sequence were visually reviewed on the overview pages; selected thumbnail
views were also inspected. All 33 prior validation sequences had a recognizable
scene represented in training. Similarity ranks were used only for triage.

| Evidence | SHA-256 |
|---|---|
| `fred-scene-review-6140043.tar.gz` | `ddb70b649e793e3a91e3e7f0a136f83a3fedebb86d2986eea891ced5a888a10c` |
| `summary.json` | `747f2addbbe3807b729ec2f953cdb85c96b35da2e907557091428604720784e3` |
| original bring-up split | `c9bbcb6ffd0c3e745a3fe6e361b80676e63e0f70e93865f3386ffe6413c64520` |

136 reused records preserve producing revision
`ac33341b2e72346f176bb3b730d0a691e7a4f93a`; 36 newly collected records use
`fc6b9854cf50946ac12e83debcd79e3ed0d2b265`. Reconstructing the original JSON
for all 136 reused records reproduced their recorded source hashes. This checks
lineage consistency; the original ICE directory and raw source images were not
independently examined locally. The archive contains no held-out test snapshots.

Examples supporting stronger grouping include 7/8 and 225/230 (courtyard well
and green shutters), 38/39 and 48/51 (trees and railings), 55/60 and 92/93
(skyline dome/cranes), 25/154 (yellow building, gravel lane and parked car),
124/127 (yellow facade and picnic tables), 155/162 (garden table and backboard),
181/187 and 180/196 (arches, notice boards and orange chairs). These identify
repeated scenes, not exact duplicate frames or certified recording sessions.

## Membership decision

Use five explicit groups and place each whole group on one side. The existing
manifest schema calls this `recording_group`; the identity basis here is reviewed
scene content, not verified session IDs. Membership is listed explicitly in the
new JSON. It is not assigned from sequence-number proximity, hash thresholds or
model fitness. No random seed is involved.

| Group | Sequences | Paired frames | Side |
|---|---:|---:|---|
| Courtyard facades, well, stairs and roof views | 58 | 191,221 | training |
| Outdoor gardens, railings, yellow building, tables, parking and trees | 63 | 207,728 | training |
| City skyline | 34 | 114,174 | validation |
| Utility corridor | 8 | 22,352 | training |
| Arched hall | 9 | 30,865 | validation |

Training totals **129 sequences / 421,301 paired frames**. Validation totals
**43 sequences / 145,039 paired frames**: 75/25 by sequence and about 74.4/25.6
by paired frame. All 172 challenging-train sequences remain included; the exact
59 challenging-test IDs remain unchanged. No frames within one sequence are
split between sets. Source annotation identity counts are retained in the
manifest's evidence; the target remains the single `drone` class.

Courtyard roof views 218/219 are conservatively merged with other courtyard
views. Uncertain connections among outdoor garden/parking/railing views are
also conservatively merged; one common physical site is not asserted.

## Limits and remaining decisions

- Validation covers skyline and hall backgrounds, not every environment. The
  corridor remains training data. Geographic/session relationships between the
  corridor and hall remain unknown; session independence is not established.
- Source identity DJI Mini 2 occurs only in training. Validation contains the
  raw strings Betafpv air75, DJI Mini 3, DJI Tello EDU, DarwinFPV cineape20 and
  DarwinFPV cineape20ger. This decision does not normalize identity strings.
- Four RGB samples per sequence can miss scene changes or frame duplicates.
  Zero sampled exact cross-split RGB hash matches does not prove absence of
  duplicates in full streams. Condition labels and session IDs are unverified.
- The official split and inventory hashes were recorded in the supplied summary;
  their source JSON files were not included in the download. The ICE check must
  compare the existing official manifest with that recorded input hash.
- Final research freeze, authoritative loader selection, and controlled
  baseline/evolution protocol decisions remain separate. Working-split approval
  does not turn previous bring-up scores into selection fitness.

## Implementation and verification contract

The split loader retains legacy sequence-only hashes and validates explicit
group membership: groups must be non-empty, disjoint, cover development IDs
exactly, exclude held-out IDs and remain wholly within training or validation.
The new group mapping participates in manifest identity.

`phase0_data/jobs/check_scene_split.sh` verifies existing ICE evidence and official
membership without collecting data or submitting training. Saved audit records
keep their original bring-up membership/context. Legacy audit defaults continue
to use the old split so completed receipts remain reproducible. Future manifest
builds select this working split with `--project-split` and use new output paths.
The `BRINGUP` approval tag preserves non-freeze status in approved-policy exports.
