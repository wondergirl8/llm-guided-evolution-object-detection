# DG-P0-04 evidence: extended FRED boxes (provisional)

**Status:** evidence collection; no formal annotation or inclusion policy approved.
**Source:** FRED dataset revision `980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd` and
FRED repository revision `2bf89c5376eda528431b62d6c60f2c13d8f95ab4`.

## Observed development-data issue

The strict ICE pilot at code revision `6cadd2db302f` inspected eight of the
172 official challenging-train sequences, IDs `0` through `7`. All eight
reports completed but failed strict validation. The user-provided summary
counted 846 `annotation.out_of_bounds` errors, and a second read-only
classification of those same reports found 846 boxes partly overlapping the
1280x720 image, zero fully outside, and zero unparsed. The reports and summary
are under `data/fred_phase0/validation/development_audit_6cadd2db302f/` on
PACE ICE. These counts describe only the pilot, not the full development set.

The pinned upstream [FRED annotation documentation](https://raw.githubusercontent.com/miccunifi/FRED/2bf89c5376eda528431b62d6c60f2c13d8f95ab4/README.md)
describes `coordinates.txt` as extended boxes that include padded-area cases,
and `coordinates_rgb.txt` as boxes excluding padding. This supports the
interpretation that at least some boundary extensions are intentional; it does
not establish that every one of the 846 boxes is correct or how a detector
should use them.

## Pending decision and evidence

`DG-P0-04` is a PROJECT/RESEARCH gate. The final blocking/warning categories
and any inclusion or exclusion rule require the full audit, issue frequencies,
affected sequences/samples, upstream evidence, and correctness assessment.
Until that decision, the strict reports remain failed. Canonical annotations
retain their released coordinates. The bounded YOLO11 smoke run's separate
bring-up exception and its in-bounds frame selection do not authorize a
formal data policy or a fitness result.

The audit summary classifies existing bounds findings as `partly_visible`,
`fully_outside`, or `unclassified` for evidence only. Classification does not
change report severity, publish a manifest, clip a box, or exclude a sample.

## Complete development scan and pairing diagnostic

The completed strict scan covered all 172 official challenging-train
sequences. User-provided summary output reported 9,440 partly visible
out-of-bounds boxes and no fully outside boxes. It also reported 100 sequences
with `pairing.event_filename_unrecognized` and 279,843
`annotation.unmatched_timestamp` findings. A per-sequence check showed that
every unmatched timestamp came from those same 100 sequences; there were no
unmatched timestamps in the other 72.

Representative released filenames are `Video_116_33333.png`,
`Video_116_66666.png`, and `Video_116_99999.png`. The original Phase 0
validator accepted only `Video_<sequence>_frame_<counter>.png`. A rejected
filename made pairing return no frame pairs, after which annotation
association emitted a secondary unmatched-timestamp finding for every source
annotation. The [pinned upstream loader](https://raw.githubusercontent.com/miccunifi/FRED/2bf89c5376eda528431b62d6c60f2c13d8f95ab4/src/data/data.py)
uses natural ordering of `.png` event frames and does not require the
`_frame_` token. This establishes a validator
format mismatch and a diagnostic cascade, not 279,843 independently verified
timestamp errors. The new parser accepts both released filename forms while
retaining sequence/counter checks, and pairing failures no longer generate
secondary unmatched-timestamp findings.

The original reports remain immutable evidence. A targeted ICE recheck of the
100 affected sequences is required before their new pairing and timestamp
outcomes can be stated. The other 72 reports can be reused with their paths
and hashes recorded in the combined audit summary. Box handling remains a
separate `DG-P0-04` research decision.
