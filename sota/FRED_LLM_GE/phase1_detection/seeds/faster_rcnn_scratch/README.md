# Faster R-CNN from-scratch experiment

This isolated experiment uses the bounded MMDetection Faster R-CNN R50-FPN
genome without COCO or other pretrained weights. Every candidate is initialized
deterministically with seed 0, trained for three epochs on sequences
`5,18,30,42,57,66,96,104`, and ranked on sequences `1,50,72`. It uses every
tenth event frame, width 640, batch size 2, and the shared COCO evaluator.

`num_generations=21` yields generation checkpoints 1 through 20. Population is
4, with 2 elites and the existing YOLO-compatible mutation/crossover machinery.
All paths use the `faster_rcnn_scratch` identity and cannot overlap pretrained
Faster R-CNN or YOLO outputs.

After generation 20, `finalize.py` selects highest validation mAP50:95, then
mAP50, then fewer parameters. It freezes that choice, rebuilds the architecture
from scratch, trains 20 epochs on all eleven challenging-training sequences,
verifies the pinned official test split, and writes a held-out JSON report that
cannot feed evolutionary fitness.

Setup and tests:

```bash
bash scripts/faster_rcnn_scratch/setup.sh
.venv-faster-rcnn/bin/python -m pytest sota/FRED_LLM_GE/phase1_detection/tests -q
```

When an experiment is intentionally ready to run:

```bash
.venv/bin/python scripts/faster_rcnn_scratch/launch.py
```

If a scheduler window ends after a completed generation, use `--resume`.
