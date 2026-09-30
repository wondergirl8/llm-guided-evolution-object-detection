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
