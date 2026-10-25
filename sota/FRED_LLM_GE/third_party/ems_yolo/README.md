# EMS-YOLO source snapshot

This directory contains the minimal model/loss dependency snapshot from
<https://github.com/BICLab/EMS-YOLO> at commit
`fbfc2caeac5ff4d60cbc2ad23864911df63d0f85` (GPL-3.0).

Only Python import statements were rewritten to make the snapshot an isolated
package; the original license is preserved. The FRED runtime does not import
these legacy PyTorch 1.10 modules directly. Its PyTorch 2.x compatibility port
is `phase1_detection/seeds/ems_yolo/`, where the origin and intentional changes
are documented. Both the snapshot and compatibility port are included in the
protocol fingerprint.

The preliminary FRED experiment is not a Gen1 reproduction. It uses the public
`resnet34-cat.yaml` topology and EMS neuron equations, random initialization,
the released FRED Event-frame search split, and the shared FRED evaluator.
