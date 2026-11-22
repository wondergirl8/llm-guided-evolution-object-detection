# Faster R-CNN on the Official Challenging Split

This experiment trains MMDetection Faster R-CNN R50-FPN from random weights.
The ResNet stem and all four stages are trainable. BatchNorm uses fixed running
statistics (`norm_eval=True`) with trainable affine parameters for batch size 2.
No COCO checkpoint or parent weights are loaded.

| Stage | Official membership | Frame step | Epochs |
|---|---|---:|---:|
| Search training | 152 challenging-training sequences | 50 | 3 |
| Search validation | Fixed 20 challenging-training sequences | 10 | - |
| Final training | All 172 challenging-training sequences | 5 | 20 |
| Held-out evaluation | All 59 challenging-test sequences | 1 | - |

Width is 640, batch size 2, and SGD uses learning rate 0.0025, momentum 0.9,
and weight decay 0.0001. Every search candidate uses seed 0. The independent
stock baseline uses seeds 0, 1, and 2; its verified seed-0 result supplies the
stock reference in generation 0. Evolution retains four distinct initial
architectures, two elites, and generations 0 through 20.

The shared official resolver verifies pinned split checksums and maps sequence
IDs to their physical Hugging Face folders. Those folders are not the official
challenging partitions. Data uses released event PNGs and the existing clipped
pixel-box annotation policy. This is a controlled scratch study, not a claim of
published-paper reproduction.

## Setup and Verification

Run from the repository root:

```bash
bash scripts/faster_rcnn_scratch/setup.sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LLMGE_AUTO_START_SERVER=0 \
  .venv-faster-rcnn/bin/python -m pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_faster_rcnn_scratch.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_faster_rcnn_official.py
```

The standalone integration check needs no dataset or LLM. CPU checks use tiny
images; CUDA checks also exercise the 640-wide batch-2 input:

```bash
.venv-faster-rcnn/bin/python scripts/faster_rcnn_scratch/smoke.py --device cpu
# Inside an H200 allocation:
.venv-faster-rcnn/bin/python scripts/faster_rcnn_scratch/smoke.py --device cuda
```

## Launch and Resume

Choose a fresh tag and unused server port. Previewing submits nothing:

```bash
export FRED_RUN_TAG=faster_rcnn_scratch_20261022_eot0_official1
export LLMGE_PORT=8295
unset FRED_EVENTS
.venv/bin/python scripts/faster_rcnn_scratch/launch.py
# Submit the reviewed workflow:
.venv/bin/python scripts/faster_rcnn_scratch/launch.py --submit
```

The dependency chain is GPU smoke plus eight CPU export shards, export
verification, three stock baseline jobs, server, evolution, Pareto finalization,
and plots. Exports are shared with the existing released-frame benchmark and
include final training data. Baseline failure prevents evolution from starting.
A baseline summary is verified again before the controller starts.

Source hashes are frozen before queued jobs start, including uncommitted edits.
Do not change source, dependency versions, protocol, sampling, prompts, or seed
under the same tag. Outputs live under `fred_runs/$FRED_RUN_TAG/`:

- `baseline/seed_{0,1,2}/` and `baseline/summary.json`
- `candidates/`, `checkpoints/`, and `operator_jobs/prompt_traces/`
- `final/finalists.json`, frozen sources, and one directory per architecture
- `final/held_out_reports/`, `final/comparison.csv`, and `final/status.json`

After the old jobs exit, retain the same tag, code, and port settings:

```bash
.venv/bin/python scripts/faster_rcnn_scratch/launch.py --resume --submit
```

Completed baseline seeds are validated and reused. Failed or partial baseline
directories are preserved for inspection and are not silently overwritten.
Evolution resumes only matching checkpoint pairs. Final training resumes
completed epochs with optimizer and random state; test inference resumes
completed sequence predictions.

The launcher's default is a dry run. A completed experiment requires generation
20, followed by a `complete` final status and reports for every frozen finalist.

## Evolution Checks and Prompts

The eight bounded gene fields in `genome_spec.yaml` are grouped into FPN,
proposal, and RoI classes after `# --OPTION--` markers. AST validation rejects
changes to the protected preamble, class interfaces, dictionary keys, or allowed
literal values. Backbone architecture, losses, training, data, and evaluator
cannot be evolved.

Mutation uses `templates/FRED_FasterRCNNScratch/Normal/1.txt` with the family
constant rules. Crossover uses the two templates in that family's `CrossOver/`
folder. Replies must contain a complete Python gene class. The family policy
builds the resolved MMDetection architecture, rejects parent-equivalent
children, and keys duplicate fitness reuse to the architecture and frozen run.
Rendered prompts, raw replies, sampling settings, and proposal rejection
records are saved. EoT and LLM quality-control rewriting are disabled.

Each candidate passes model construction and synthetic finite-loss,
backward, and prediction checks before real training. Preflight restores
parameters, buffers, gradients, mode, and random state. Numerical failures
receive failed fitness rather than a successful score.

## Pareto Finalization

All successful archived candidates are compared across validation mAP50,
mAP50:95 (both maximized), and parameter count (minimized). Every nondominated
architecture is frozen before accessing test labels. Identical architectures
share one final run and retain aliases; the stock model is included as a
labeled control even when dominated.

Each finalist starts freshly and trains 20 epochs on all official training
sequences. The finalizer never loads search-trained weights. Held-out reports
cannot produce evolutionary fitness CSVs.

The finalizer is independently runnable after evolution completes:

```bash
.venv-faster-rcnn/bin/python -m \
  sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.finalize \
  --run-root "fred_runs/$FRED_RUN_TAG" --freeze-only
# Omit --freeze-only inside a GPU allocation to train and test.
```

The old interim results, proxy trainer, and hardcoded Pareto comparison scripts
are historical. They are not inputs to this official-split run and their scores
must not be mixed with the new experiment.
