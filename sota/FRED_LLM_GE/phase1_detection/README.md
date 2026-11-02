# Phase 1 object detection

## Models

Model-specific sources live together under `seeds/`, following the simple
model-folder layout used by `sota/MujocoRL`. Shared data, scoring, and runners
stay in this directory.

| Model / experiment | Seed source | Execution path |
| --- | --- | --- |
| Official Ultralytics YOLO11n | [ultralytics_yolo11/network.yaml](seeds/ultralytics_yolo11/network.yaml) | Shared runner, `--family yolo11` |
| Tutorial-sized YOLO-inspired detector | [yolo11/network.py](seeds/yolo11/network.py) | Shared runner, `--family tiny_yolo11` |
| D-FINE-inspired Event detector | [dfine/network.py](seeds/dfine/network.py) | Shared runner, `--family dfine` |
| COCO-initialized Faster R-CNN pilot | [faster_rcnn/seed.py](seeds/faster_rcnn/seed.py) | [Model guide](seeds/faster_rcnn/README.md) |
| Random-initialization Faster R-CNN | [faster_rcnn_scratch/seed.py](seeds/faster_rcnn_scratch/seed.py) | [Experiment guide](seeds/faster_rcnn_scratch/README.md) |

The Faster R-CNN `network.py` files contain protected model builders; their
`seed.py` files contain the evolvable literal genomes. The `yolo11/` directory
keeps its historical name so existing candidate and cache locations stay valid.
RT-DETR remains a planned family, without runnable code in this checkout.

## Shared runner

Run these commands from the **repository root** using its Python environment.
Prepare and finalize the search export once:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize
```

Train and evaluate the unchanged official YOLO11n seed without starting LLM-GE:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family yolo11 --model network \
  --sota_root sota/FRED_LLM_GE/phase1_detection/seeds/ultralytics_yolo11
```

For the tutorial detector use `--family tiny_yolo11` and the `seeds/yolo11`
output root. For D-FINE use `--family dfine` and the `seeds/dfine` output root.
Always set `--sota_root` to the intended family or run directory; the runner's
historical default is `seeds/yolo11`. Generated candidates additionally use
`--model network_<gene_id> --variant_dir <run>/models`.

The shared [protocol](../configs/phase1/protocol.yaml) owns data splits and
training settings. Results go to `<sota_root>/results/<gene_id>_results.csv`,
reports to `<sota_root>/reports/`, and weights to `<sota_root>/trained_models/`.
The [baseline runner](baseline_runner.py) handles repeated stock-YOLO baselines;
[final_eval.py](final_eval.py) handles frozen Ultralytics selections on held-out
data. Model-specific Faster R-CNN evaluation follows its own linked guide.

## Seed path migration

Seed contents are unchanged. The former `FRED_LLM_GE/seeds/seed_*` sources now
live in the model folders listed above. Configuration, validators, reporting,
and baseline execution use these new paths. CLI family names and existing
candidate, cache, and output locations remain the same.

Seed hashes, protocol contents, and the code revision contribute to run fingerprints. Existing
research runs should use their recorded code revision; use a new run identity
for this layout rather than mixing fingerprints with an earlier checkpoint.

## D-FINE-inspired Event seed

The model is one file, [`seeds/dfine/network.py`](seeds/dfine/network.py). The
`dfine` family adapter converts the shared runner's `uint8` Event images and
pixel `xyxy` boxes to the seed's normalized training inputs. The runner uses
the same challenging-search export, sample IDs, evaluator, and result format as
the `yolo11` family. Its D-FINE model, matching, FGL, and GO-LSD losses live in
the seed file; generated candidates may replace only its three architecture
classes.

Run the unchanged seed without LLM-GE:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network \
  --sota_root sota/FRED_LLM_GE/phase1_detection/seeds/dfine \
  --variant_dir sota/FRED_LLM_GE/phase1_detection/seeds/dfine/models
```

The result and checkpoint are written under the selected `sota_root`. For a
local interface check without downloading FRED, run:

```bash
python -m pytest -q sota/FRED_LLM_GE/phase1_detection/tests/test_dfine_adapter.py
```

`FRED_SEED_FAMILY=dfine bash run.sh` selects the D-FINE LLM-GE configuration,
but formal evolution requires a validated non-evolutionary baseline and the
post-core `DG-P1-10` decision. The current shared data path clips out-of-bounds
annotations and is provisional relative to the unresolved Phase 0 policy; do
not treat proxy scores as an approved FRED benchmark result. The existing
`final_eval.py` currently handles Ultralytics YOLO11 only.

## Faster R-CNN integration

The Faster R-CNN Event pilot uses MMDetection R50-FPN, bounded literal genomes,
and the same detection result contract and mAP reporter as YOLO11. See
[its setup, baseline, and pilot guide](seeds/faster_rcnn/README.md).

Select the evolutionary family with `LLMGE_DETECTOR=faster_rcnn`; the default
remains YOLO11. Faster R-CNN dependencies and H200 kernels are isolated from the
main environment. Its initial three-epoch pilot is exploratory.

The separate [random-initialization 20-generation experiment](seeds/faster_rcnn_scratch/README.md)
uses the same bounded architecture genome without loading pretrained weights.
