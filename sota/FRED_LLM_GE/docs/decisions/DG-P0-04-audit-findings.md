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

The original reports remain immutable evidence. The targeted ICE recheck at
code revision `fa6815b20050` inspected the 100 affected sequences. The
combined summary reused the other 72 original reports, recording each report
path and hash. It is complete for all 172 challenging-train sequences, with
no missing IDs. The filename finding cleared, and unmatched timestamps fell
from 279,843 to two. Rechecking formerly unpaired frames raised the observed
partly visible box count from 9,440 to 10,254. There were no fully outside
boxes. Of the 83 sequences with strict errors, 81 have partly visible box
findings; sequences `225` and `230` have one unmatched annotation each, both
at line 1 with timestamp `0.0`.

## Proposed research policy (not yet approved)

The [pinned upstream loader](https://raw.githubusercontent.com/miccunifi/FRED/2bf89c5376eda528431b62d6c60f2c13d8f95ab4/src/data/data.py)
assigns frame index 0 timestamp `0.033333`; it cannot associate a `0.0`
annotation with a released frame. It would not use either of these two
records as a training target. Their exact coordinate contents have not been
reviewed, so this establishes only that they precede the first paired frame.

Recommend a versioned `DG-P0-04` policy that keeps all 172 development
sequences, retains the released coordinates of every frame-paired annotation
verbatim in the canonical manifest, and treats a positive-area overlap with
the image as a recorded warning. For YOLO targets, clip the *corners* of such
a box to the verified image rectangle before normalization; record both boxes,
clipped-label count, and policy version. A zero-area result, fully outside
box, malformed record, or any other unmatched timestamp remains blocking.
The two `0.0` records would be recorded as source annotations without a
paired frame and omitted from frame labels; neither entire sequence needs
exclusion on that ground. Apply the same transform to train and validation
targets. Do not inspect held-out test annotations to make this choice.

This is a proposal for the `DG-P0-04` PROJECT/RESEARCH owner. Existing strict
reports remain failed and no canonical manifest or formal fitness result is
authorized by this note. Visual checks across both released filename forms
and representative clipped boxes remain necessary before a Phase 0 freeze.
