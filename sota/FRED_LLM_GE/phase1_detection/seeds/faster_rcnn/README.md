# Faster R-CNN Event development pilot

This path implements MMDetection 3.3.0 Faster R-CNN R50-FPN for one drone
class. It is an exploratory integration, not a validated formal baseline.

Setup builds MMCV 2.1.0 with CUDA 12.1.1 and `TORCH_CUDA_ARCH_LIST=9.0`
for H200 GPUs; the public wheel lacks compatible H200 kernels. The first build uses four compiler workers and can take several minutes;
subsequent setup runs reuse the checked wheel. Build tools
and dependencies are pinned in `requirements.lock`, and the compiled wheel
has a checksum under the scratch cache. PACE CUDA modules are required.

The protected builder loads the official COCO checkpoint, transferring only
shape-compatible weights. The single-class output layers and any incompatible
architecture layers retain explicit MMDetection initialization. Each run saves
all loaded/skipped keys and checkpoint SHA256 under `reports/<gene_id>/`.

LLM-GE changes literal settings in three gene classes, corresponding to FPN,
RPN/anchors, and RoI/box-head configuration. `genome_spec.yaml` lists conservative
prototype choices. Validation rejects executable mutations and protected-source
changes before candidate import. All candidates keep ResNet-50, native losses,
SGD (lr 0.0025, momentum 0.9, weight decay 0.0001), seed 0, and evaluation fixed.
Anchor alternatives are predefined development choices, not fitted to validation.

Setup from repository root:

```bash
bash scripts/faster_rcnn/setup.sh
.venv-faster-rcnn/bin/python -m pytest sota/FRED_LLM_GE/phase1_detection/tests -q
.venv-faster-rcnn/bin/python sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/train_eval.py --smoke --device cpu
```

Standalone baseline (GPU allocation recommended):

```bash
.venv-faster-rcnn/bin/python sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/train.py --device cuda
```

Bounded Slurm baseline and dependent pilot:

```bash
.venv/bin/python scripts/faster_rcnn/launch.py
```

The baseline job performs a GPU smoke test then real FRED seed training. Only
successful completion releases the pilot and optional dedicated LLM server.
Launch records prevent duplicate submission. After all previous jobs finish,
`launch.py --retry` archives their IDs and permits a repaired launch. No automatic restart chains run.
Use `squeue` and `run_job_outputs/` to inspect jobs; the launch JSON records IDs.

The pilot uses one island, population 4, two breeding generations, 3 epochs,
batch 2, width 640, frame step 10, and YOLO11's provisional sequence membership.
Evaluation jobs request one H200 GPU, 12 CPUs, 16 GB RAM, and 4 hours. Controller
and server limits are 8 hours; a new LLM server requests two H200 GPUs, 160 GB.
Partition/account are inherited from the existing Slurm defaults. Budget changes
must apply equally to seed and candidates; memory or environment failures stop
launches rather than silently changing training settings.

Family selection is `LLMGE_DETECTOR=faster_rcnn`, inherited by subprocesses;
unset selects YOLO11. Dedicated prompts, caches of evolutionary state, outputs,
server hostname/port, and Slurm configuration prevent mixing the two runs.
Data caches remain shared with YOLO11 through `event_data.py`.

Input images are RGB-order uint8 event frames; MMDetection normalizes and pads
them once. Targets use resized xyxy pixels, class 0. Inference rescales boxes to
original pixels using per-frame metadata, then the shared reporter evaluates
and writes `results/<gene_id>_results.csv`: mAP50, mAP50:95, parameter count.
Failures have a separate status/traceback and legacy worst-case CSV; failures
are never reported as successful runs. Checkpoints are saved atomically.
Held-out test data is not used. Formal study protocol/resource/genome gates
remain unresolved; three-epoch pilot scores are exploratory only.
