# Plan for `seed_dfine.py`

**Status:** Prototype seed and `fred-run-setup` adapter implemented. The scientific baseline and post-core expansion gates remain open.

## 1. Deliverable and scope

Write one Python implementation file:

```text
sota/FRED_LLM_GE/seeds/seed_dfine.py
```

Use the attached `seed_yolo11.py` as a **file-layout and LLM-GE marker example**, not as a source of model operations or input assumptions. The chosen design is a compact, self-contained **D-FINE-inspired Event-frame detector**. It will not import the official D-FINE repository or claim to reproduce its published architecture or scores.

The model implementation remains one file. Model-specific matching and losses belong in its **protected portion**. The newer `fred-run-setup` branch uses a separate family adapter, shared protocol, and runner; those integration files are necessary to execute this seed without embedding data access or training orchestration in the model file.

The governing Phase 1 plan places D-FINE under post-core expansion. The user explicitly requested this prototype implementation and runner integration; formal experiments and reported comparisons remain subject to `DG-P1-10` after the core Phase 1 freeze. The mandatory YOLO11, RT-DETR, and Faster R-CNN baselines retain their separate status.

## 2. What the seed should consume and return

| Boundary | Planned contract |
| --- | --- |
| Input | `event: torch.Tensor` with shape `[B, C, H, W]`, finite values, and channels matching the released Event PNG encoding. A live `train/36` frame decoded as three-channel `RGB` mode; this is an Event image, not an RGB-camera frame. Validate other selected sequences before freezing `C=3`. No separate RGB input argument. |
| `forward(event)` | Final `pred_logits` `[B,Q,1]`, normalized `pred_boxes` `[B,Q,4]` in `cxcywh` order, plus training outputs for every decoder layer: reference boxes and four edge-distribution logits per query (`[B,Q,4,reg_max+1]`). Define one stable `aux_outputs` schema for the fixed criterion. |
| `predict(event)` | One detection record per input image: `{"boxes": [...], "scores": [...], "labels": [...]}`. Use label `0` for `drone`, the frozen confidence/postprocessing policy, and model-input pixel `xyxy` boxes. Return three empty lists when nothing is detected. The evaluation boundary uses trusted spatial metadata to map these boxes back to the original frame. |
| `DFINECriterion(outputs, targets)` | Fixed one-to-one matching, classification and box losses, fine-grained localization supervision, and GO-LSD self-distillation. Accept empty-target images; return named finite loss terms. The training harness calls it without allowing candidate code to change it. |
| Ground truth | The training boundary must pair each Event frame with its trusted Phase 0 boxes and label mapping, apply the same approved spatial transform as the image, then supply `targets` as one dictionary per image: normalized model-input `cxcywh` boxes `[N,4]` and single-class integer labels `[N]`. The seed does not read labels, choose splits, or fetch FRED data. |

### Current collector compatibility

`data/fred_stream.py` accepts `(split, sequence_id, bad_timestamps_path=None)` and yields `(PIL.Image.Image, timestamp_s)` in counter order for one sequence. It scans RGB filenames for diagnostics but never returns RGB pixels. It writes `bad_timestamps.csv` before yielding frames; the default path is overwritten on each call. A live check of its first `train/36` Event frame returned timestamp `0.033333`, size `1280×720`, mode `RGB`, and bands `R/G/B`. Its observed per-band extrema were `(30,255)`, `(37,255)`, and `(52,255)`; the sequence's timestamp report had only its header. These observations establish a three-channel `uint8` image for that sampled frame only. In particular, zero is absent from that frame's channel ranges, so zero pixels cannot be assumed to mean inactivity. The observation does not establish the semantic meaning of each channel, the full dataset range, or a normalization rule.

The seed's protected preamble should include a small `event_pil_to_tensor(image)` helper that preserves the released channel order, checks mode/shape against the frozen Phase 0 specification, converts `[H,W,C]` to `[C,H,W]`, and applies the **same documented numeric scaling** at training and inference. It must not silently coerce other modes or assume that the three Event-image channels are color-camera measurements. The caller adds a batch dimension or stacks several results before `forward` or `predict`; the collector itself remains responsible only for remote frame access and timestamps. The protected helper is inside the single seed file, so this inference bridge does not require another implementation file.

The collector has **no annotations, sample ID, split manifest, tensor conversion, or target-to-frame join**. Its relative `timestamp_s` is a `float`, and its caller supplies the `sequence_id`; neither alone is a trusted project sample identity. It cannot currently feed supervised `DFINECriterion` or a valid LLM-GE fitness run. `phase0_data/coco_adapter.py` is a standalone annotation-to-YOLO converter and is not connected to this stream or the planned D-FINE targets. It clamps coordinates and skips malformed rows, so it must not serve as the trusted target source while Phase 0's annotation policy is unresolved. A future Phase 0 model-neutral loader must provide stable identity, validated Event-frame/annotation association, trusted targets, and spatial transform metadata. Do not parse `coordinates.txt`, reconstruct a sample ID from float timestamps, choose splits, or call the remote API inside the seed. The CSV's RGB count/interval checks are diagnostics, not proof of exact Event/annotation synchronization. Retain the report per sequence and review its findings before building training samples. [Collector scope](../phase0_data/STREAMING_PLAN.md), [collector README](../phase0_data/README.md)

**Compatibility verdict:** the current collector can feed a **single-frame inference smoke check** once `event_pil_to_tensor` and the model exist in a PyTorch-enabled runtime: `PIL Event frame → [C,H,W] tensor → [1,C,H,W] batch → predict`. It cannot yet feed a **supervised training or LLM-GE run**: `Event frame + stable sample ID + approved box policy + matched target + shared image/box transform` must first come from the canonical Phase 0 interface. This is an integration dependency, not another model file.

**`fred-run-setup` integration update:** `phase1_detection/proxy_data.py` now combines the collector with `coordinates.txt`, creates sample IDs, resizes images and boxes, and caches `uint8` Event tensors with pixel `xyxy` targets. The D-FINE adapter converts those shared runner targets to normalized `cxcywh` and trains through `candidate_runner.py`. This makes provisional proxy runs technically executable. The shared loader currently clips out-of-bounds annotations, which conflicts with the unresolved Phase 0 annotation policy above, and it is not the frozen canonical Phase 0 interface. Such runs must not be presented as an approved FRED baseline or formal LLM-GE evidence until that policy and the Phase 1 gates are resolved. The seed itself still does no provider access or annotation repair.

Before coding, inspect representative released frames to fix pixel mode, numeric range, dimensions, inactive-pixel meaning, and the caller's tensor conversion. Confirm that the Phase 0 target convention, timestamp alignment, and choice of FRED annotation coordinates match the Event-frame coordinate space. The initial canonical source is `coordinates.txt`; `coordinates_rgb.txt` is not an interchangeable fallback. If the caller resizes or pads an image, the evaluation boundary must reverse that transform before comparing with original-image boxes. [FRED dataset and annotation description](https://github.com/miccunifi/FRED), [Phase 0 annotation policy](PHASE0_DATA_PLAN.md)

Phase 0 has recorded an out-of-bounds released annotation and has **not** approved clipping, repair, filtering, or exclusion. The seed criterion must receive targets only after a project-approved target policy resolves this case; neither the PIL conversion helper nor the seed may silently alter boxes. [Known data issue](../artifacts/phase0/known_data_issues.yaml), [Phase 0 box contract](PHASE0_DATA_PLAN.md)

The project output standard requires original-image pixel `xyxy` boxes, score in `[0,1]`, and label `0`. The seed's raw training output is distinct from that evaluation record. Preserve D-FINE's one-to-one, query-based detection design; settle the project's NMS/no-NMS and confidence policy at the evaluation gate, then keep it outside evolvable code. Never silently discard malformed or non-finite boxes.

## 3. Single-file layout

The top of the file is protected. It contains the module description, imports, the PIL-to-tensor conversion helper, shared tensor/box helpers, the fixed `DFINECriterion` and matcher, and the stable `DFINEEventDetector` wrapper. Python resolves the component class names when the wrapper is instantiated, after the entire module has loaded.

```text
seed_dfine.py
├── Protected preamble
│   ├── imports and shape/finite-value checks
│   ├── event_pil_to_tensor: released Event PNG → model input
│   ├── fixed matching, FGL/GO-LSD losses, and loss weights
│   ├── fixed box decoding and prediction formatting
│   └── DFINEEventDetector: builds components and defines forward/predict
├── marker 1 → EventBackbone
├── marker 2 → MultiScaleEncoder
└── marker 3 → DistributionRefinementDecoder
```

Each marker is one literal top-level `# --OPTION--` line. It **starts** a mutable section; it is not a paired opening/closing delimiter. The three sections should contain one complete, nonempty, top-level class each. There is no trailing protected code after marker 3. Local smoke checks can import the file from outside rather than adding a mutable `__main__` block.

| Section | Initial architecture | Fixed interface to the wrapper |
| --- | --- | --- |
| `EventBackbone` | Lightweight CNN stem and stages. Prototype a stride-4 feature level for small drones; profile its memory and latency against a stride-8 start before freezing the design. | Constructor `(in_channels, width)`; accept `[B,C,H,W]` and return three ordered feature maps with fixed, documented strides/channels after that choice. |
| `MultiScaleEncoder` | Project-owned feature projection and cross-scale fusion, inspired by D-FINE's hybrid encoder. | Constructor `(backbone_channels, d_model)`; accept the three feature maps and return three maps of the same spatial sizes, each with `d_model` channels. |
| `DistributionRefinementDecoder` | Select initial queries from top-ranked encoded features, predict reference boxes, attend across encoded scales, and refine four edge distributions across decoder layers. Official D-FINE defaults to encoder-selected queries; learned-only queries would be a separate architecture ablation. | Constructor `(d_model, num_queries, num_layers, reg_max)`; accept the three encoded maps and return exactly the raw prediction keys and shapes above. |

The protected `DFINEEventDetector` constructor supplies these parameters and checks that component outputs obey the fixed contract. The final code must document exact spatial behavior for odd input sizes and use the same return structure in every candidate.

The prototype width, query count, number of decoder layers, refinement-bin count, feature strides, and allowed architecture ranges must be chosen from FRED training/development evidence and build/resource measurements, then frozen in the experiment definition. The official model's size and COCO settings are references, not automatic FRED hyperparameters. The `fred-run-setup` candidate runner is launched from the repository root and uses the root `pyproject.toml`; that environment now declares PyTorch and SciPy for D-FINE. The separate `sota/FRED_LLM_GE/.venv` remains a collector environment without those training dependencies and should not be used to run the seed.

## 4. D-FINE identity and scientific limits

The official D-FINE model composes a backbone, hybrid encoder, and transformer decoder. Its two defining mechanisms are fine-grained distribution refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). For this compact seed, retain their **mechanisms**, then document every simplification from the official implementation. [D-FINE paper, method](https://arxiv.org/html/2410.13842v1#S4)

**FDR design:** predict an initial reference box for each query. At each decoder layer, predict logits for `reg_max+1` bins for each of the top, bottom, left, and right edge offsets. Add later-layer logits as residual refinements; softmax and a fixed, nonuniform bin projection convert the distributions into edge corrections relative to the reference box. Produce both refined boxes and the intermediate distributions. A plain four-number box head or one-shot distribution head is insufficient to represent FDR. [D-FINE paper, §4.1](https://arxiv.org/html/2410.13842v1#S4.SS1), [official decoder](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py)

**Fixed supervised criterion:** use one-to-one Hungarian assignments with frozen class/box costs; supervise classification, boxes, and per-edge distributions with the D-FINE fine-grained localization idea (soft targets on adjacent bins, with the approved weighting). GO-LSD then uses the final layer's detached distributions as a teacher for earlier layers, with the paper's per-layer match union and matched/unmatched weighting. Implement and verify these mechanics in the protected criterion before calling the model a D-FINE-inspired trained seed. Merely returning auxiliary boxes is not GO-LSD. Handle zero-object frames and zero matched predictions without division-by-zero or invented positives. [D-FINE paper, §§4.1–4.2](https://arxiv.org/html/2410.13842v1#S4), [official criterion](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_criterion.py)

This is still a **D-FINE-inspired** project implementation, not an official D-FINE reproduction: its Event stem, compact widths, possible stride-4 level, and any simplified attention differ from the published model. If FGL or GO-LSD proves infeasible within the one-file training contract, revise the design and label the resulting model according to the mechanisms actually implemented. Baseline and evolution runs must use the same frozen criterion, data, initialization policy, and evaluator.

## 5. Insights from `references.md` applied to this seed

| Source | Decision for the first Event-only seed | Validation or later study |
| --- | --- | --- |
| [FRED](https://arxiv.org/abs/2506.05163) | Use the released Event frames and trusted Phase 0 boxes. Validate the Event coordinate/annotation choice and challenging split before training. | Report localization and missed tiny drones on project validation; isolate the official challenging test from search feedback. |
| [D-FINE](https://arxiv.org/abs/2410.13842) | Keep FDR, FGL supervision, GO-LSD, and one-to-one set prediction as the model's distinctive core. | Confirm each loss term contributes finite gradients and compare with a no-FDR/no-GO-LSD ablation under the same budget. |
| [UAV-DETR](https://arxiv.org/abs/2603.22841) and [small-target P2 design](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml) | Test a high-resolution feature path and cross-scale fusion because tiny drones can vanish after early downsampling. This is a FRED adaptation hypothesis, not a published D-FINE component. | Profile attention cost and compare stride-4 versus stride-8 starts before fixing the seed; local-window attention or feature recalibration can be later architecture ablations. |
| [DEIM](https://arxiv.org/abs/2412.04234) | Record Dense One-to-One matching and Matchability-Aware Loss as a promising training-cost study. | Test only as a separately versioned, fixed training-protocol ablation after the initial D-FINE-inspired baseline; it changes matching and loss, so it cannot be silently mixed into architecture evolution. |
| [NWD](https://arxiv.org/abs/2110.13389) | Record small-box IoU sensitivity as a diagnostic for FRED localization. | Try NWD or UAV-DETR's mixed loss only in a separate fixed-criterion ablation; keep `mAP50:95` evaluation unchanged. |
| [SET](https://github.com/HuixinSun/SET) | Measure clutter/background failures on Event frames. | Consider its training-only background/saliency method after checking that the released Event encoding makes the operation meaningful; it is not a free inference module. |
| [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR), [CFPT](https://arxiv.org/abs/2407.19696), [SAHI](https://arxiv.org/abs/2202.06934) | Their query selection, feature-pyramid interaction, and tiled inference ideas suggest targeted error analyses. | Version each as a distinct query/encoder/inference ablation with measured recall, compute, and latency. Tiling belongs to the inference protocol, outside the seed architecture. |

The RGB/Event fusion works (ER-DETR, SPFD, PEPR) answer a different modality question. EV-UAV, SAST, and SparseVoxelDet need raw events or a new representation. They should inform future studies rather than being inserted into this released-frame, Event-only seed. [FRED repository](https://github.com/miccunifi/FRED), [EV-UAV](https://arxiv.org/abs/2506.23575), [SAST](https://github.com/Peterande/SAST), [SparseVoxelDet](https://arxiv.org/abs/2603.21638)

## 6. LLM-GE mutation contract

The existing `src/llm_utils.py` splits source on every literal option marker. `src/llm_mutation.py` chooses one section after a marker for replacement. `src/llm_crossover.py` aligns sections by position. Therefore:

1. The source has exactly three markers and four nonempty split parts: protected preamble plus the three components. Do not mention the literal marker string elsewhere in the Python file, including its docstring.
2. Each generated candidate retains the same marker count, order, class names, signatures, output keys, and tensor contracts. Crossover is allowed only for compatible genome versions.
3. Candidate edits are limited to the selected complete class body. The wrapper, matcher, loss definitions and weights, input validation, box conversion, class mapping, and postprocessing policy remain protected. A mutation may change architecture only while preserving every output needed by the fixed criterion.
4. Reject candidates that fail parsing, construction, shape checks, finite-value checks, or the frozen resource limits. Do not convert invalid candidates into apparently successful scores.

This marker layout matches the current text-splitting mechanism; it is not by itself a complete semantic genome validator. Before evolution, the integration must enforce the protected contracts and record candidate identity, code state, parentage, and invalid outcomes.

## 7. Implementation sequence and acceptance checks

1. Verify the released Event PNG mode, bands, range, and dimensions on the selected project-training sequences; freeze the PIL-to-tensor scaling and check the collector's per-sequence timestamp report. Verify Event annotation coordinates and the exact data/target handoff from Phase 0, including its out-of-bounds-box policy. Quantify training/development box sizes and inactive-frame frequency before choosing strides or query count; do not inspect held-out test labels for these choices.
2. Resolve the declared PyTorch and matching dependencies, then implement the protected wrapper, matcher/criterion, and three architecture components in `seed_dfine.py` only. Fix one-to-one assignment costs, loss weights, output schema, and initialization policy before candidate runs.
3. Check Python parsing/import, marker count, class construction, and `forward` on synthetic Event batches at representative sizes, including an odd image size if supported. Check all final and intermediate tensor shapes, finite values, valid boxes, and a bounded query count. Separately, pass a bounded live training-sequence `(PIL image, timestamp_s)` from `stream_event_frames` through `event_pil_to_tensor` and `predict` without requiring a local dataset mirror; this is an **inference interface check**, not a training validation.
4. On trusted small labeled batches, including images with zero drones, check finite classification, box, FGL, and GO-LSD losses; run backward and confirm gradients reach the Event stem and decoder distribution heads. Verify that the GO-LSD teacher is detached and that each intended matched/unmatched path is exercised.
5. Check that `predict` gives valid input-pixel boxes, scores in `[0,1]`, one label `0` per detection, and a valid empty-result record. Check that the evaluation boundary correctly reverses a nontrivial resize/pad transform before producing the original-image records required by `phase1_detection/MODEL_OUTPUT_STANDARD.md`; verify the frozen NMS/no-NMS and confidence policy there.
6. Run the unchanged seed as a non-evolutionary baseline under the approved training/evaluation protocol before allowing LLM-GE mutation. Track project-validation `mAP50`, `mAP50:95`, higher-IoU localization, size-stratified recall, false positives, parameter count, memory, detector-only latency, and training time. Only then validate a mutated candidate and same-version crossover through the actual runner.

**Prototype implementation status:** the one-file seed and provisional `fred-run-setup` family adapter exist, and synthetic training/evaluation integration checks can run. **Ready for formal LLM-GE runs when:** the approved Phase 0 loader supplies annotated samples and stable IDs under a resolved box policy, the non-evolutionary D-FINE baseline is trained and evaluated, generated sections pass the protected-contract checks, resource/postprocessing choices are frozen, and `DG-P1-10` authorizes the expansion. The collector alone still cannot supply supervised fitness inputs.
