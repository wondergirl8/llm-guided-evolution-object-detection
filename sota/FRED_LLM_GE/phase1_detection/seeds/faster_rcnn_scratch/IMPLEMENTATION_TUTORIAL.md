# Faster R-CNN Scratch: Implementation Tutorial

This guide explains the implementation in this checkout, not an idealized
implementation of Faster R-CNN. It covers the current official challenging-split
experiment and identifies the older proxy code that remains for historical use.
Source line numbers describe the code inspected when writing this guide; function
names are the more durable navigation aid if files change later.

## 1. What "From Scratch" Means

There are two different meanings of this phrase:

1. Implement every neural-network operation, loss, proposal generator, and
   detector component ourselves.
2. Start a detector with random weights instead of loading pretrained weights.

**Our experiment uses meaning 2.** We did not implement ResNet, FPN, RoIAlign,
anchor matching, or Faster R-CNN losses from first principles. We use the
MMDetection implementation and MMCV's compiled operations. Our contribution is
the controlled architecture representation and the surrounding reproducible
training/evolution/evaluation workflow.

This distinction matters when describing the work in a report: "randomly
initialized MMDetection Faster R-CNN R50-FPN with LLM-guided architecture search"
is more accurate than "a Faster R-CNN implementation written from scratch."

### What comes from the framework

| Component | Implementation owner |
|---|---|
| ResNet-50 feature extractor | MMDetection |
| Feature Pyramid Network | MMDetection |
| Region Proposal Network | MMDetection |
| Anchor generation, target assignment, proposal sampling | MMDetection |
| RoI feature extraction | MMDetection plus MMCV RoIAlign |
| Classification and bounding-box losses | MMDetection |
| NMS kernels | MMCV |
| Gradients, tensors, SGD, checkpoint serialization | PyTorch |
| COCO-style AP calculation | pycocotools |

### What the project implements or integrates

The gene schema, protected/mutable boundary, AST validation, configuration
translation, random-initialization policy, lazy FRED data loading, family
adapter, manual training orchestration, LLM prompts/operators, architecture
identity/cache, baseline gate, experiment fingerprints, Slurm workflow,
Pareto selection, fresh final training, held-out reports, and regression tests.
Several shared pieces existed already; Faster R-CNN reuses and extends them.

## 2. The Whole Experiment In One Picture

```text
Pinned FRED official membership + physical archive mapping
    |
    +-- Export search train: 152 sequences, frame step 50
    +-- Export search validation: 20 sequences, frame step 10
    +-- Export final train: all 172 training sequences, frame step 5

Source snapshot + successful GPU smoke
    |
Independent stock search baselines: seeds 0, 1, 2; three epochs each
    |
Verified seed-0 stock score becomes evolution's stock reference
    |
Generation 0: stock + three distinct successful architecture variants
    |
Generations 1 through 20
    select parents -> LLM crossover/mutation -> validate literals/source
    -> construct/hash architecture -> reuse matching cached fitness OR
       fresh random initialization -> three-epoch training -> validation AP
    -> selection/elites -> archive/checkpoint
    |
Freeze ALL three-objective nondominated archived architectures
    + add stock control if absent
    + merge identical architectures, retain aliases
    |
For EACH frozen architecture:
    fresh initialization -> 20 epochs on all 172 training sequences
    -> official test: all 59 sequences, every frame
    -> provenance-checked held-out report
    |
Final comparison table
```

Evolution inherits **architecture choices**, not trained parent weights.
An already evaluated identical architecture can reuse its score within the
matching protocol. That is computation reuse, not child weight inheritance.

## 3. The Detector Itself

Read [network.py](network.py) alongside this section.

### 3.1 Backbone: ResNet-50

An input is a rendered event frame represented as three image channels. It is
not a stream of raw event tuples entering a spiking network. The current study
uses the released event-frame representation.

The backbone extracts feature maps at multiple resolutions. Its four output
stages have 256, 512, 1024, and 2048 channels. Later stages have lower spatial
resolution and greater semantic context. We keep the ResNet-50 architecture
fixed so the search focuses on detection components and stays affordable.

At native 1280 x 720, resizing to width 640 produces 640 x 360. MMDetection's
preprocessor then pads as necessary to a divisor of 32. Resizing is not the same
as padding: padding does not change the coordinates of existing pixels.

### 3.2 Neck: FPN

The FPN combines backbone features into a multi-scale feature pyramid with five
outputs. The inherited anchor strides are 4, 8, 16, 32, and 64. Each pyramid
level uses the selected common channel width, either 128 or 256.

This common width is an interface contract: if FPN outputs 128 channels, both
the RPN and RoI head must consume 128 channels. Our builder propagates that
choice rather than asking the LLM to keep several dimensions consistent.

### 3.3 First detection stage: RPN

The RPN predicts objectness and box adjustments for anchors at pyramid locations.
Its proposals say "there may be an object here" rather than providing the final
drone classification. The selected anchor scale and ratios affect which initial
box shapes it considers.

Anchor scale is not a literal box width in input pixels. It interacts with the
anchor generator's strides and ratios. For example, at stride 4 a square anchor
with scale 4 has nominal side length about 16 input pixels; scale 8 gives about
32. Smaller anchors are a plausible option for tiny drones, not a guaranteed
improvement.

### 3.4 Second detection stage: RoI head

RoIAlign extracts a fixed-size feature grid for each proposal. The selected grid
is 7 x 7 or 14 x 14. A shared two-fully-connected-layer box head uses these
features to classify a proposal and refine its coordinates.

`num_classes=1` means one foreground category: drone. The framework handles
background internally. It does not mean that background examples disappear.

### 3.5 Protected loss and matching defaults

The builder loads MMDetection's installed
`.mim/configs/_base_/models/faster-rcnn_r50_fpn.py`, then applies our overrides.
The inspected installed configuration supplies:

| Component | Inherited behavior |
|---|---|
| RPN classification | Sigmoid cross-entropy, weight 1 |
| RPN box regression | L1 loss, weight 1 |
| RoI classification | Softmax cross-entropy, weight 1 |
| RoI box regression | L1 loss, weight 1 |
| RPN matching | Positive IoU 0.7, negative IoU 0.3; low-quality matches enabled |
| RPN sampler | 256 samples, positive fraction 0.5 |
| RoI matching | Positive/negative boundary IoU 0.5 |
| RoI sampler | 512 samples, positive fraction 0.25; GT proposals added |
| RPN proposal NMS | IoU 0.7 |

These are not evolvable in this experiment. The actual scalar training loss is
assembled by MMDetection's `parse_losses`, including the framework's handling
of multi-level loss lists. AP is not a differentiable training loss here.

## 4. Why These Parts Are Mutable

Read [genome_spec.yaml](genome_spec.yaml) and the canonical
[seed](../../../seeds/seed_faster_rcnn_scratch.py).

We use a bounded search space rather than allowing an LLM to rewrite arbitrary
detector code. The design criteria are:

1. A meaningful accuracy/capacity/parameter-count tradeoff.
2. A plausible relationship to tiny-object detection.
3. Compatibility with the existing MMDetection detector interfaces.
4. A small, inspectable search space with affordable candidate training.
5. Fixed data, optimization, losses, and evaluation so candidates remain
   comparable within the experiment.

This is an engineering choice, not evidence that these eight fields are the
globally optimal search space.

| Class | Field | Allowed values | Meaning |
|---|---|---|---|
| `FeatureGenes` | `fpn_channels` | `128`, `256` | Common pyramid output width |
| `ProposalGenes` | `rpn_channels` | `128`, `256` | Hidden RPN feature width |
| `ProposalGenes` | `rpn_convs` | `1`, `2` | RPN convolution depth |
| `ProposalGenes` | `anchor_scales` | `[4]`, `[8]` | Anchor size multiplier |
| `ProposalGenes` | `anchor_ratios` | `[0.5, 1.0, 2.0]`, `[1.0]` | Three shapes versus square-only anchors |
| `RoIGenes` | `roi_size` | `7`, `14` | Spatial side length of extracted RoI features |
| `RoIGenes` | `roi_sampling_ratio` | `0`, `2` | Adaptive versus fixed RoIAlign sampling |
| `RoIGenes` | `box_fc_channels` | `512`, `1024` | Width of shared fully connected layers |

There are eight binary decisions: 2^8 = 256 nominal literal combinations.
Valid literal combinations must still build and execute successfully. We hash
the resolved model rather than assuming different source text is a new model.

### Why capacity can change sharply

The first fully connected RoI layer consumes approximately:

```text
fpn_channels * roi_size * roi_size
```

features. Its weight matrix therefore has approximately:

```text
(fpn_channels * roi_size^2) * box_fc_channels
```

parameters, before biases and other layers. Doubling `roi_size` from 7 to 14
quadruples this layer's input dimension. This helps explain why older evolved
models can approach 80 million parameters without changing ResNet-50.

### Protected means two different things

**Architecturally protected:** ResNet depth/stages, FPN level count, losses,
matching/sampling rules, input representation, class count, optimizer, seeds,
data splits, training budget, and metric implementation cannot be changed by a
candidate's gene class.

**Not frozen against gradient training:** Backbone weights remain trainable.
"The LLM cannot change the backbone architecture" does not mean "SGD cannot
update the backbone weights."

### The canonical seed, statement by statement

```python
def build_model_config():
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_config
    return build_config({**FeatureGenes.values, **ProposalGenes.values, **RoIGenes.values})
```

`def` defines a callable; it does not build the model immediately. The import
loads our trusted translator when the function is called. `**` unpacks three
dictionaries into one combined dictionary. `build_config` validates and maps
those values into an MMDetection config. The return value is a configuration,
not trained weights and not yet a live detector.

Each `# --OPTION--` is a delimiter for the evolutionary operator. It is only a
comment to Python, but our splitting/validation code gives it special meaning.
There must be three delimiters, producing one protected preamble and three
mutable sections.

The three classes are namespaces for their `values` dictionaries; they are not
`nn.Module` layers. The seed uses FPN 256, RPN width 256/depth 1, scales `[8]`,
three ratios, RoI size 7, adaptive sampling, and FC width 1024.

In the current adapter path we extract validated dictionaries and call the
trusted builder directly. We do not need to execute a candidate's Python module
to obtain its genes. The callable interface also supports the older path.

## 5. The Source Validator, Statement By Statement

Read [validator.py](validator.py). This is the enforceable boundary; prompt
instructions alone are not sufficient.

| Lines | Explanation |
|---|---|
| 1 | Module docstring describes its role; no operational effect. |
| 2 | Import Python's abstract syntax tree parser. |
| 3 | Import the JSON parser. |
| 4 | Import filesystem path handling. |
| 6 | Resolve the canonical seed path relative to this file, not the shell directory. |
| 7 | Read the schema. Although named `.yaml`, this file currently contains JSON; JSON is valid YAML. |
| 9 | Define validation for an already extracted gene dictionary. |
| 10 | Flatten class-group dictionaries into `field -> allowed choices`. |
| 11-12 | Require a dictionary with exactly the approved eight keys. Missing and extra keys both fail. |
| 13 | Inspect each supplied field. |
| 14-16 | Require matching type, value, and representation against an approved choice. This rejects `True` as `1`, `1.0` as integer `1`, and altered literal representations where relevant. |
| 17 | Return the validated dictionary for convenient composition. |
| 19 | Define validation for a whole candidate source file. |
| 20-21 | Split candidate and canonical seed at the mutable-section marker. |
| 22-23 | Require four parts and an exactly unchanged protected preamble. |
| 24 | Start the merged gene dictionary. |
| 25 | Pair each candidate section with its expected schema class in order. |
| 26 | Parse that section into an AST without executing it. |
| 27-28 | Require exactly one top-level class in the section. |
| 29 | Name the class node for subsequent checks. |
| 30-31 | Reject wrong names, inheritance, class keywords, and decorators. |
| 32 | Copy the class body into a local list. |
| 33-34 | Permit and discard one leading class docstring. |
| 35-36 | Require exactly one remaining statement: an ordinary assignment. Methods and extra statements fail. |
| 37 | Name the assignment node. |
| 38-39 | Require one target named `values`. Chained/alternative assignments fail. |
| 40 | Extract literals with `ast.literal_eval`, not arbitrary `eval`. Calls and computed expressions fail. |
| 41-42 | Require exactly the keys belonging to this class. |
| 43 | Merge this class's dictionary into the full genome. |
| 44 | Apply the exact allowed-value/type checks to the result. |

Allowed replacement example:

```python
class FeatureGenes:
    values = {"fpn_channels": 128}
```

Rejected examples include `fpn_channels=192`, `rpn_convs=True`, a new method,
`values = make_genes()`, moving `roi_size` into `ProposalGenes`, or changing the
protected import. Comments/formatting are not automatically new architectures.

This restricts this representation. It is not an operating-system sandbox for
all project code or a general proof that arbitrary generated Python is safe.
The trusted evaluator and protected files remain part of the trust boundary.

## 6. Building A Randomly Initialized Detector

Read [network.py](network.py). This short file contains four functions.

### 6.1 Imports and initialization removal

| Lines | Explanation |
|---|---|
| 1 | Module description. |
| 2 | `hashlib` provides SHA-256 for state identity. |
| 3 | `Path` locates the installed MMDetection configuration. |
| 4 | `torch` provides tensors, RNG handling, and the model framework. |
| 5 | Reuse the trusted dictionary validator. |
| 7 | Define recursive cleanup of initialization instructions. |
| 8-11 | For dictionaries, replace every `init_cfg` value with `None`; recursively inspect other children. |
| 12-13 | Recurse into lists and tuples as well. Nested initialization directives must not survive. |

The installed stock config contains a pretrained ResNet initialization reference.
Loading that configuration as data does not itself download its checkpoint;
removing `init_cfg` before building/initializing prevents that loading path.

### 6.2 `build_config`: every assignment

| Line | Effect and reason |
|---|---|
| 15 | Entry point receives literal genes, not a checkpoint. |
| 16 | Reject illegal genes before constructing framework objects. |
| 17-18 | Import the installed detector package and its configuration parser locally. |
| 19 | Locate the installed R50-FPN model-only config and take its `model` section. This is not MMEngine's entire training schedule. |
| 20 | Remove inherited initialization instructions recursively. |
| 21 | `frozen_stages=-1`: do not freeze the stem or residual stages. |
| 22 | `norm_eval=True`: keep BatchNorm running-stat behavior in evaluation mode during training. |
| 23 | Keep BatchNorm affine scale/bias trainable. This does not unfreeze running statistics. |
| 24 | Disable BGR-to-RGB conversion because our image pipeline already supplies RGB tensors. |
| 25 | Give the chosen FPN channel count the local name `width`. |
| 26 | Set every FPN output's channel width. |
| 27 | Match the RPN input channels to FPN output. |
| 28 | Set the internal RPN feature width. |
| 29 | Set RPN convolution depth. |
| 30 | Set anchor scale choice. |
| 31 | Set anchor aspect-ratio choice. |
| 32 | Match the RoI extractor's output-channel declaration to FPN. |
| 33 | Set RoIAlign's spatial output side length. |
| 34 | Set RoIAlign's sampling ratio; zero means adaptive sampling in this implementation. |
| 35 | Match the box head input channels. |
| 36 | Set the shared fully connected layer width. |
| 37 | Tell the box head the same spatial RoI size used by the extractor. |
| 38 | Use one foreground class. |
| 39 | Return the resolved model config. |

The inherited preprocessor still applies its fixed mean/std normalization and
padding. Removing pretrained weights does not remove those preprocessing
constants.

### 6.3 BatchNorm: an important deliberate compromise

Batch size two gives unreliable batch-stat estimates, so we retain the
`norm_eval=True` recipe while allowing BN affine parameters to learn. For a
randomly initialized model this means the running statistics are not first
learned by a pretrained backbone. This is a protocol choice, not a claim that it
is the optimal normalization strategy for scratch training. Changing it would
be a new controlled experiment, not a harmless runtime repair.

### 6.4 `state_sha256`: lines 41-47

Create a SHA-256 accumulator. Traverse `state_dict()` in sorted-name order so
ordering is stable. `state_dict` includes parameters and persistent buffers,
including BN buffers. `detach()` removes autograd association; `.cpu()` makes
bytes accessible on the host; `.contiguous()` gives a consistent memory layout.
Add the tensor name, dtype, shape, and bytes to the digest. Return its hex string.

This hashes actual state values. It is different from the architecture hash,
which identifies configuration/module types/tensor shapes without trained bytes.

### 6.5 `build_detector`: lines 49-60

`seed` is keyword-only because of `*`. The function does not accept a pretrained
checkpoint argument. Register MMDetection's module classes so names such as
`FasterRCNN`, `ResNet`, and `FPN` resolve through `MODELS`.

Inside a PyTorch RNG context, set the seed, build the model from the config,
clear any module-level `init_cfg`, and call framework-standard initialization.
Building here happens on CPU; the adapter moves the result to the requested
device afterward. Initialization is repeatable for a given architecture, seed,
and compatible environment. Different shapes do not imply that all shared
parameters receive identical values across different architectures.

Return both the model and explicit metadata: initialization policy, seed,
`pretrained=False`, `checkpoint=None`, and the initial state hash. The RNG context
keeps construction from consuming the caller's CPU RNG sequence. This is not a
universal guarantee of bitwise-identical GPU training across hardware/software.

## 7. Environment Setup And Why There Are Two Environments

Read [setup.sh](../../../../../scripts/faster_rcnn_scratch/setup.sh),
[requirements.lock](../faster_rcnn/requirements.lock), and
[build_mmcv.sh](../../../../../scripts/faster_rcnn/build_mmcv.sh).

The detector environment `.venv-faster-rcnn` uses Python 3.10 and pinned packages:

| Package | Pinned detector version |
|---|---|
| PyTorch | 2.1.2, CUDA 12.1 wheel |
| torchvision | 0.16.2, CUDA 12.1 wheel |
| MMDetection | 3.3.0 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| NumPy | 1.26.4 |
| pycocotools | 2.0.10 |

The repository/controller environment `.venv` serves the shared LLM-GE stack.
Separating these avoids forcing an older detection/CUDA-extension stack onto
the controller and LLM dependencies.

The setup script:

1. Enables strict shell failure handling and changes to the repository root.
2. Chooses a local package cache.
3. Creates the Python 3.10 detector environment if missing.
4. Installs the locked detector dependencies.
5. Calls the MMCV CUDA-extension build helper.
6. Checks that `nms`, `roi_align`, and MMDetection import successfully.

The MMCV build helper selects CUDA 12.1.1 and compiles with operations enabled,
`TORCH_CUDA_ARCH_LIST=9.0` for H200, and four build jobs. It caches the built wheel,
records/verifies its SHA-256, and installs that wheel. A successful `import torch`
alone would not establish that the detector's compiled NMS/RoIAlign kernels work.

These are the inspected local pins, not recommendations to upgrade packages
during an experiment. Dependency changes intentionally invalidate run identity.

## 8. Data: Membership, Storage, Labels, And Sampling

The main implementation is shared:
[fred_splits.py](../../fred_splits.py),
[export_split.py](../../export_split.py), and
[proxy_data.py](../../proxy_data.py).

### 8.1 Official membership is not the Hugging Face folder name

The official challenging split is 172 training and 59 test sequences. The
physical Hugging Face folders contain a different partition. A sequence stored
under HF `train/` can belong to the official challenging test set.

`official_sequences` reads the pinned official list, caches it, and checks its
checksum. `hf_folders` resolves which physical archive holds each sequence at
the pinned dataset revision. `challenging_split` combines membership and storage
location and checks overlap/missing entries. `downloaded_sequence` and
`sequence_frames` handle archive access and cleanup/iteration.

Never substitute "everything in HF train" for official challenging training.

### 8.2 The four partitions of this experiment

| Stage | Sequences | Frame step | Epochs |
|---|---:|---:|---:|
| Search training | 152 official-training sequences | 50 | 3 |
| Search validation | Fixed 20 official-training sequences | 10 | None |
| Final training | All 172 official-training sequences | 5 | 20 |
| Official held-out test | All 59 official-test sequences | 1 | None |

`dataset_parts` builds search train as official training minus validation IDs.
For final training it uses all official training sequences, including the former
search-validation sequences. That is allowed after architecture selection is
frozen; the official test sequences remain held out.

`candidate_runner.challenging_search` explicitly checks the Faster R-CNN
152/20/50/10 configuration and the use of released event frames. Family-specific
budget resolution prevents shared YOLO/D-FINE budgets from silently overriding
Faster R-CNN's three-epoch search recipe.

### 8.3 Annotations and empty images

`load_annotations` reads a sequence's `coordinates.txt`, parses boxes, clips them
to image boundaries, and discards nonpositive-area boxes. Annotation timestamps
are mapped to frame indices using `STEP_US`. Images with no visible annotated
drone retain an empty `(0, 4)` box tensor; they are not removed.

Boxes use `xyxy`: left, top, right, bottom. All foreground labels are integer
zero in our adapter/evaluation interface. The COCO conversion later maps them
to category ID one internally.

### 8.4 Export artifacts

`export_sequence` writes native-size RGB PNGs, compatible label files, and a
`done/<sequence>.json` marker. The marker records sequence ID, physical HF folder,
sampling step, image size, frame count, frame indices, and exact pixel boxes.
Completion markers are atomically installed after sequence export.

The shared export also writes YOLO-format labels and `data.yaml`. Their presence
does not mean Faster R-CNN trains with YOLO losses or uses normalized YOLO boxes.
Faster R-CNN reads exact boxes from JSON markers. The shared `trainer_val` folder
exists for Ultralytics integration; it is not Faster R-CNN's search-validation
partition.

### 8.5 Lazy images

`LazyImages` stores paths and a tensor-like shape, not all image pixels.
`__getitem__` opens just requested PNG files, converts HWC RGB arrays to CHW
tensors, and stacks the requested batch. It accepts slices and one-dimensional
integer-tensor indices.

`load_exported_split` checks expected sequence/folder/step/frame counts and image
existence, constructs box tensors/sample IDs, and returns a dictionary containing
`LazyImages`. Boxes and metadata remain in memory, but the full image dataset
does not. This differs from the old proxy loader, which concatenates tensors.

## 9. Adapter: Statement-By-Statement Walkthrough

Read [adapters/faster_rcnn_scratch.py](../../adapters/faster_rcnn_scratch.py).

### 9.1 Imports

`SimpleNamespace` converts named settings into the attribute interface expected
by the shared trainer. `torch` supplies resizing/checkpoint operations.
`FamilyAdapter` is the shared interface. The remaining imports reuse our trusted
builder/validator and the existing MMDetection batch/train/preflight helpers.
Importing helpers from `faster_rcnn/train_eval.py` does **not** execute that
module's pretrained CLI workflow; the scratch adapter calls only those helpers.

### 9.2 `ResizedImages`, lines 13-31

| Statement | Explanation |
|---|---|
| `self.images = images` | Keep the underlying image collection, which can itself be lazy. |
| Height calculation | Preserve aspect ratio approximately, rounding height to a multiple of four. |
| `self.shape = ...` | Expose `[N, 3, H, W]` without loading N images. |
| `__len__` | Return N from that declared shape. |
| `scalar = isinstance(index, int)` | Detect when the shared trainer requests one image. |
| Wrap an integer in a tensor | Adapt to `LazyImages`' batched indexing interface. |
| `raw = self.images[indices]` | Actually load the requested images. |
| `.float()` | Bilinear interpolation operates on floating-point values. |
| `interpolate(..., bilinear, align_corners=False, antialias=True)` | Resize with a fixed interpolation recipe. |
| `.round().clamp_(0,255).to(uint8)` | Restore byte-image range/type before detector preprocessing. |
| `resized[0] if scalar else resized` | Restore the requested single-image versus batch shape. |

The shared training helper requests images individually inside each batch;
inference can resize a whole already-stacked batch. Both avoid materializing the
entire dataset as a resized tensor.

### 9.3 `resized_data`, lines 34-38

Create the resize wrapper. Read old height/width. Build the coordinate multiplier
`[new_w/old_w, new_h/old_h, new_w/old_w, new_h/old_h]`. Return a shallow dictionary
copy with resized `images` and scaled training `boxes`. Leave `orig_boxes` and
`orig_size` unchanged for evaluation in original coordinates.

### 9.4 `preserving_preflight`, lines 41-56

Clone every tensor in the model state, including buffers, and remember training
mode. Save Python and NumPy RNG state. Run the synthetic preflight inside a
PyTorch CPU/CUDA RNG-preserving context. The `finally` block executes even if
preflight fails: restore Python/NumPy RNG, reload model state strictly, clear
gradients, and restore model mode.

Why do this? A synthetic check should not secretly train the model, alter BN
buffers, leave gradients behind, or shift the training random sequence. The
wrapper makes preflight an interface check, not an unrecorded warm-up update.

### 9.5 `FasterRCNNScratchAdapter.build`, lines 59-71

The class-level `name` registers the family. Reject any foreground class count
other than one. Extract genes with the AST validator, build a config, build a
randomly initialized detector using the configured seed, move it to CPU/CUDA,
and run preserving preflight. Store genes/initialization metadata on the adapter
and return the live model.

### 9.6 `train`, lines 73-77

Resize lazily, pack epochs/batch/lr/momentum/weight decay into `SimpleNamespace`,
and delegate to the shared trainer. Pass optional `resume_state` and `on_epoch`
through unchanged. The device argument is part of the adapter interface; the
already-placed model and its data preprocessor handle device movement.

### 9.7 `predict`, lines 79-95

Construct an inference data dictionary with empty target tensors and temporary
sample IDs. Crucially, its `orig_size` is the size of images supplied to the
adapter. Resize to detector width, set score threshold/NMS/max detections, switch
to eval mode, and disable gradient recording. Prepare the batch and invoke
`mode="predict"`. Check output count and return detached CPU `(boxes, scores)`
pairs. These boxes are in the supplied images' coordinate system because
MMDetection uses the metainfo scale factor to rescale them.

The shared candidate runner may additionally convert supplied-image coordinates
to true original coordinates if its input images were already resized. With the
current native-size official exports that outer multiplier is one. The final
test helper also supplies native-size images. Do not blindly add another factor
of two: coordinate conversion must occur once for each actual resize.

### 9.8 `save` and `postprocessing`, lines 97-103

Save the state dictionary plus initialization, genes, and family settings.
Return a readable description of confidence/NMS/max-detection settings for
report metadata. A search checkpoint and a final resumable training checkpoint
have different schemas and purposes.

## 10. The Training Loop, Every Operation

Read the helper functions in
[faster_rcnn/train_eval.py](../faster_rcnn/train_eval.py). This file also has an
older standalone pretrained runner; the scratch path reuses only selected helpers.

### 10.1 `prepare_batch`

Import `DetDataSample` for one image's metadata/targets and `InstanceData` for
its bounding boxes/labels. Start image/sample lists. For each requested index:

1. Load its CHW image, lazily if applicable.
2. Read resized height/width and original width/height.
3. Construct metainfo: `img_shape`, `ori_shape`, `(scale_x, scale_y)`, and image ID.
4. Clone its training-coordinate boxes into `gt_instances`.
5. Build a long-integer zero label for each foreground box; an empty image gets
   an empty label tensor.
6. Append the image and its sample record.

Finally call `model.data_preprocessor` with `training=True` or `False`. The
preprocessor moves/casts/normalizes/pads inputs as implemented by MMDetection.
We pass the unpadded image shape in metadata; padding is a preprocessing detail.

### 10.2 `synthetic_data`

Create two small random byte images, one positive target and one empty target,
matching original boxes, original dimensions, and two sample IDs. This checks
mechanics without a network download and deliberately exercises empty targets.
It is not used as real training data in the official experiment.

### 10.3 `train`: setup and resume

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr,
    momentum=getattr(args, "momentum", 0.9),
    weight_decay=getattr(args, "weight_decay", 0.0001))
```

`model.parameters()` provides learnable tensor objects to SGD. `getattr` retains
defaults for older callers while the scratch adapter supplies explicit values.
We do not instantiate an MMEngine training Runner: this project owns the loop.

Initialize `first_epoch=0`. If resuming final training, read `completed_epochs`,
validate its range, restore optimizer state including momentum buffers, and
restore saved PyTorch CPU/CUDA RNG states. The caller restores the model state
before invoking this helper. This loop does not independently load a file.

Call `model.train()`. MMDetection's backbone implementation applies its
`norm_eval` behavior when training mode is set.

### 10.4 `train`: epoch and batch operations

```python
for epoch in range(first_epoch, args.epochs):
    order = torch.randperm(len(data["images"])).tolist()
    for start in range(0, len(order), args.batch_size):
        batch = prepare_batch(model, data, order[start:start + args.batch_size], True)
        optimizer.zero_grad()
        loss, _ = model.parse_losses(model(**batch, mode="loss"))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise ValueError("Non-finite gradients")
        optimizer.step()
```

| Operation | What it means |
|---|---|
| `range(first_epoch, args.epochs)` | Continue from completed epochs up to the fixed total; do not add another 20 epochs on resume. |
| `randperm(N)` | Shuffle all training-frame indices once per epoch. |
| `range(..., batch_size)` | Process two images at a time; a final partial batch is allowed. |
| `prepare_batch(..., True)` | Build framework-compatible images/targets. |
| `zero_grad()` | Remove previous batch gradients before accumulating this batch's derivatives. |
| `model(**batch, mode="loss")` | Expand the batch dictionary into detector keyword arguments and execute training loss computation. |
| `parse_losses` | Reduce framework loss tensors/lists to one scalar objective; `_` discards the accompanying log dictionary. |
| `isfinite(loss)` | Reject NaN and infinity instead of reporting a misleading score. |
| `backward()` | Differentiate the scalar loss through the detector's trainable graph. |
| Gradient check | Check every gradient that exists for NaN/infinity. This alone does not prove every stage receives gradients; smoke checks that separately. |
| `step()` | Update parameters using SGD, momentum, and weight decay. |

After each epoch print the **final batch's loss**, not an epoch-average loss.
If an epoch callback exists, call it with the completed-epoch count and live
optimizer so the finalizer can atomically save a resumable checkpoint.

The current scratch loop has constant learning rate 0.0025, momentum 0.9, weight
decay 0.0001, batch two, and width 640. It does not add a scheduler, warm-up, AMP,
gradient accumulation, random geometric augmentation, early stopping, or
best-validation-checkpoint selection. Do not infer such features from the fact
that MMDetection also supports them.

### 10.5 A training example versus an architecture candidate

An image supplies supervision to SGD. A candidate supplies an architecture to
the experiment. For a new uncached candidate, the sequence is:

```text
genes -> configuration -> fresh weights -> preflight -> three epochs
      -> final search checkpoint -> validation AP -> fitness tuple
```

There is no "train stock, then fine-tune children from that stock checkpoint"
step in this experiment. Training stock first establishes a reference/gate.

## 11. Shared Candidate Runner: Function-By-Function

Read [candidate_runner.py](../../candidate_runner.py).

| Function / block | Responsibility |
|---|---|
| Package/direct-execution imports | Support both `python -m ...` and Slurm calling the file directly. |
| `ADAPTERS` | Map family strings to adapter classes, including scratch Faster R-CNN. |
| `FAILED_RESULT` | Worst-direction infinite fitness, excluded from valid selection. |
| `load_protocol` | Parse YAML, normalize sequence ID strings, reject basic invalid budgets/overlap. |
| `sha256_file`, `code_revision` | File identity and producing Git revision. Dirty source is additionally captured by fingerprints. |
| `write_failure_results` | Write an explicit worst-case result with a reason and terminal log marker. |
| `predict_frames` | Iterate validation batches; convert/clip boxes; preserve frames with empty predictions/targets; build evaluator records. |
| `report_metadata` | Record data/split/annotation/model/checkpoint/code/postprocessing/training identities. |
| `proxy_data` | Historical small in-memory proxy data path; not the official scratch path. |
| `challenging_search` | Resolve disjoint official-training subsets and verified lazy exports; return deferred validation loader. |
| `family_protocol` | Deep-copy and apply Faster R-CNN's search budget without mutating other families' protocols. Preserve baseline seed overrides. |
| `run` setup | Select adapter, set RNG seeds, resolve data, validate/build/preflight, save args and initialization, count parameters. |
| `run` execution | Train, save checkpoint, predict all validation frames, invoke common reporter. |
| `run` error paths | Distinguish setup/build failure from train/eval failure; write invalid fitness instead of fabricated accuracy. |
| `main` | Parse CLI, derive gene ID, guard CUDA-unavailable allocations, apply optional seed/event settings, invoke `run`. |

The family override is essential: shared protocol files also contain other
families' epoch counts. The scratch search uses three epochs, not whichever
generic `challenging.search.epochs` happens to appear before resolution.

## 12. How We Know A Model Works

There are several separate questions, with different checks:

| Question | Check |
|---|---|
| Did the LLM stay inside the permitted representation? | AST/source and literal validation |
| Are dimensions/configuration constructible? | Resolve config and build detector |
| Is it a new architecture? | Resolved architecture hash versus parents |
| Does forward/backward/prediction execute? | Synthetic preflight |
| Does the actual GPU stack work? | Run-level CUDA smoke |
| Are all backbone stages trainable in practice? | Nonzero finite gradients checked per backbone stage in smoke |
| Do real training updates remain numerically valid? | Finite loss and gradient checks every batch |
| Does it detect drones usefully? | Validation AP, not mere successful execution |
| Did evaluation cover declared frames? | Expected/observed sample ID checks |

Read [smoke.py](../../../../../scripts/faster_rcnn_scratch/smoke.py). It builds
stock, checks initialization hash before/after preserving preflight, runs a
positive and empty target at the GPU recipe resolution, checks gradients in
`conv1` and `layer1` through `layer4`, performs an optimizer update, checks that
state changes, saves/reloads state, and validates prediction bounds. It records
source/dependency identities and rejects reuse of stale smoke results.

An absence of runtime errors does not prove an architecture is accurate, and
finite gradients do not establish convergence after three epochs.

## 13. Metrics And Coordinate Checks

Read [fitness.py](../../fitness.py), [map_reporter.py](../../map_reporter.py),
and [result_contract.py](../../result_contract.py).

For a predicted box and a true box:

```text
IoU = intersection area / union area
```

AP summarizes the precision-recall curve after confidence ranking, rather than
the proportion of images guessed correctly. In this one-foreground-class task:

- mAP50 is AP at IoU threshold 0.50.
- mAP50:95 averages AP over 0.50, 0.55, ..., 0.95.
- Parameters are counted with `sum(p.numel() for p in model.parameters())`.
  This is parameter count, not latency, FLOPs, checkpoint byte size, or peak RAM.

`fitness.evaluate_frames` rejects nonfinite/invalid coordinates, wrong labels,
invalid confidence scores, repeated IDs, missing IDs, unexpected IDs, and a
partition without any GT drones. It converts valid pixel `xyxy` boxes to COCO
`xywh`, constructs GT and detection datasets (including empty predictions),
then calls `COCOeval.evaluate`, `accumulate`, and `summarize`.

AP is computed globally over the declared partition, not by averaging per-batch
or per-sequence AP. Reports declare max detections `[1, 10, 100]` and 101 recall
thresholds. Values are stored as fractions: `0.348495` is `34.8495%`.

`map_reporter` validates provenance fields and mode/split declarations, writes
atomic JSON summaries, and emits the controller CSV only in validation mode.
Held-out mode writes to `held_out_reports`, not the evolutionary fitness feed.
These declarations enforce interface conventions, not label-access security.

## 14. Independent Baselines

Read [faster_rcnn_baseline.py](../../faster_rcnn_baseline.py).

| Function | Explanation |
|---|---|
| `digest` | SHA-256 identity of an artifact. |
| `identity` | Compute the stock-seed/run protocol fingerprint. |
| `validate_seed` | Check CSV fitness, metrics/report agreement, optimizer/data recipe, scratch seed, source/config identity, and checkpoint hash. Return compact validated facts. |
| `run_seed` | Accept only seed 0/1/2; validate existing record on resume or run a fresh stock candidate. Freeze record only after training and identity verification. |
| `verify_summary` | Require exact provenance and seed-record agreement; allow narrowly bounded roundoff only in derived aggregate statistics. |
| `summarize` | Validate all three independent seed runs, check same parameter count, compute mean/sample SD, and freeze/validate summary. |
| `main` | Expose `run-seed` and `summarize` commands. |

All three stock models use the identical search data/budget, but independent
initializations and training RNG seeds. Seed zero supplies evolution's matching
reference, not an average of seed-zero and other seeds. Seeds one/two quantify
variability; three seeds alone do not establish a confidence interval for every
evolved architecture. The baseline module requires no LLM server.

## 15. Architecture Policy And Duplicate Reuse

Read [faster_rcnn_search_space.py](../../faster_rcnn_search_space.py).

| Function | What each part does |
|---|---|
| `_inspect` | Validate genes, build config/model with seed zero, serialize config plus module names/types and state tensor names/shapes/dtypes, hash that representation, count parameters. |
| `_built` | Cache inspection by path/source hash. Inspect in-process only in detector env; otherwise launch detector Python and parse its final JSON line, avoiding dependency conflicts. |
| `built_architecture` | Resolve path, validate source, hash file bytes to invalidate memoization when source changes, return architecture hash and parameter count. |
| `differs_from_parents` | Require the child's architecture hash to differ from every parent; record accepted/no-op/invalid reasons. |
| `_record`, `record_rejection` | Append source/rejection details to proposal JSONL alongside candidate files. |
| `evaluation_cache_key` | Hash architecture identity together with the entire protocol fingerprint. |
| `write_cached_fitness` | Validate integral parameter count, convert CSV float counts to `int`, write the standard result contract. |
| `candidate_chunks` | Extract complete fenced Python proposal blocks from an LLM response. |
| `repair_channel_values` | Return the proposal unchanged: illegal values are rejected, not silently repaired. |
| `changed_layer_count` | Compare genes per mutable class and count changed classes. Despite the generic name, the unit here is a gene group, not one neural layer. |
| `retry_prompt` | Add the previous rejection reason and ask for a legal changed class. |
| `seed_baseline` | Verify the matching summary, copy verified seed-zero artifacts into candidate paths, and supply its fitness. |
| `validate_run` | Require matching successful CUDA smoke, all-stage gradient evidence, and valid independent baseline summary. |
| CLI `--inspect` | Subprocess entry point that prints architecture hash/parameter count as JSON. |

Whitespace changes or a renamed candidate ID do not justify another GPU
evaluation. Conversely, the same architecture under different data, seed,
dependencies, or recipe must not reuse old fitness. Pending architectures are
also tracked to avoid launching simultaneous duplicate training jobs.

## 16. Exact Prompts And How The LLM Is Used

Read [mutation template](../../../../../templates/FRED_FasterRCNNScratch/Normal/1.txt),
[constant rules](../../../../../templates/FRED_FasterRCNNScratch/ConstantRules.txt),
and [crossover templates](../../../../../templates/FRED_FasterRCNNScratch/CrossOver/crossover.txt).

Mutation starts with:

```text
Improve this bounded randomly initialized Faster R-CNN gene class for tiny drones in FRED event frames.
Favor validation mAP50:95 while considering parameter count. Change at least one legal value.
Current class:
```

The selected current class is inserted in a Python fence. The controller appends
constant rules listing the permitted fields/literals, forbidding extra code,
and protecting data/losses/initialization/training/evaluation.

### Mutation call path

1. `run_improved.generate_template` chooses the domain prompt and appends rules.
2. An operator job calls `src/llm_mutation.py:augment_network`.
3. Split parent source at markers and select one mutable class.
4. Render that class into the prompt; preserve the other sections.
5. Ask the LLM; extract complete candidate blocks.
6. Assemble a candidate file with one replacement section.
7. Require one changed gene group and a valid architecture different from parent.
8. Retry rejected proposals with feedback, or fail clearly after the bound.

The generic operator logs sometimes say "YAML block" or "layer" even for this
Python-gene family. Those are inherited message labels, not YOLO YAML parsing.
The selected policy uses Python literals and MMDetection construction.

### Crossover call path

`src/llm_crossover.py` pairs corresponding mutable sections from two parents,
prefers a differing section, randomly picks one of the Faster R-CNN templates,
renders both classes, appends constant rules, and requests one replacement
class. It retains parent X's other sections, validates the assembled detector,
and rejects a child architecturally identical to either parent.

This is LLM-mediated class replacement, not tensor averaging or conventional
one-point binary crossover. Binary choices can make some selected parent/group
combinations hard to change relative to both parents; retries are bounded.

### Generation settings and traces

The inspected configuration has proposal attempt limit five, starting
temperature 0.1 and top-p 0.15. Retry attempts increase each by 0.1, subject to
caps 0.8/0.9. Local response token budget defaults to 1648 for this family unless
overridden. Rendered prompts/replies and sampling settings are saved under
`operator_jobs/prompt_traces/`; proposal reasons are saved alongside candidates.

The legacy dispatch name can be `mixtral` even though the configured local model
path defaults to Llama-3.3-70B-Instruct. A dispatch alias is not reliable evidence
of actual model weights. Check the server's configured model path/log and the
saved traces for the particular run.

The current mutation prompt supplies a gene class and task/rules, **not** a
numerical table of previous fitness values. Performance affects which parents
are selected by the controller. The LLM is not being fine-tuned during evolution,
and it does not compute AP or perform SGD updates. EoT and LLM quality-control
passes are disabled for this experiment; structural checks remain enabled.

## 17. Evolution Controller And Selection

Read [run_improved.py](../../../../../run_improved.py),
[scratch constants](../../../../../src/cfg/constants_fred_faster_rcnn_scratch.py),
and [shared constants](../../../../../src/cfg/constants_fred_yolo11.py).

The shared constants filename is historical. Family registration explicitly
selects the scratch seed, Faster R-CNN policy, prompts, crossover templates,
candidate runner, run paths, and detector Python command. The scratch wrapper
uses `runpy` with a family override and sets population four, offspring two,
elites two, run-scoped server hostname, and scratch Slurm settings.

### 17.1 Fitness and generation zero

Fitness order is `(mAP50, mAP50_95, parameters)` with direction weights
`(+1, +1, -1)`. These indicate maximize/maximize/minimize; they are not a scalar
weighted sum of AP and parameter count.

Generation zero reserves a stock slot and generates three initial variants.
The verified stock seed-zero evaluation is reused. Distinct built-architecture
checks and a bounded initial refill try to obtain four distinct successful
models; depletion is recorded explicitly instead of silently accepting a bad
initial population. Refill attempt limit is 24 in the inspected configuration.

### 17.2 Each generation, in code order

1. Check the paired LLM-server allocation has enough remaining time.
2. Remove invalid fitness entries from the breeding population.
3. Select two unmodified elites using SPEA2 across all three objectives.
4. Select up to two offspring parents through the registered selection path.
5. Clone DEAP individuals so children do not overwrite parent records.
6. Archive current metadata.
7. Apply crossover probabilistically (configured 0.5) and wait for operator jobs.
8. Apply mutation probabilistically (configured 0.8) to eligible offspring.
9. Reattach elites and remove duplicate individual IDs.
10. Evaluate uncached new architectures, reuse matching cache entries, and collect
    explicit failures.
11. Update population, archive, Pareto hall of fame, checkpoints, and status.

The controller's parent-selection helper uses NSGA-II sorting and a
dominance/crowding tournament with count adjustment. Elite selection uses SPEA2.
`KEEP_BEST_ELITE=False`: there is no reserved slot for a single accuracy winner.

Four is the configured target/capacity, not a guarantee that every later
generation contains four distinct successful architectures: duplicates/failures
can reduce survivors. Initial graph deduplication is stronger than later
gene-ID duplicate removal. Parameter count is not normalized by the direction
weights; SPEA2 distance calculations can be influenced by objective scales.

`num_generations=21` means checkpoint generation zero plus evolutionary
generations one through twenty, not 21 rounds after generation zero.

### 17.3 Key controller responsibilities

| Function / state | Responsibility |
|---|---|
| `candidate_preflight` | Build/hash, reserve pending architecture, reuse completed fitness, seed baseline special case. |
| `check4results` | Poll result/log artifacts, parse CSV, classify failures, update cache and job state. |
| `GLOBAL_DATA` | Candidate execution/status/fitness records used by the controller. |
| `GLOBAL_DATA_HIST` | Archived candidate records, not just current population. |
| `GLOBAL_DATA_ANCESTRY` | Parent/operator/prompt lineage. |
| `EVAL_CACHE` | Protocol-keyed architecture evaluations. |
| `EVAL_PENDING` | Evaluations already in progress, preventing concurrent duplicates. |
| `select_elites` | Multiobjective elite choice. |
| `save_checkpoint` | Save population/hall-of-fame and global archive, with matching fingerprint. |
| `load_checkpoint` | Load latest matching pair, require same generation/identity, start at saved generation plus one. |
| `run_fingerprint` | Add evolutionary settings to the detector/data/source identity. |
| `write_run_status` | Record generation completion or explicit run state. |
| `check_server_window` | Parse Slurm remaining allocation time and pause before insufficient time remains. |

The printed "Best Individual" uses DEAP's single ordering for display. It is not
the full three-objective Pareto set and is not the finalization selection rule.

## 18. Fingerprints, Provenance, And Resume

Read [protocol_fingerprint.py](../../protocol_fingerprint.py).

`scratch_code_identity` hashes relevant source/config/prompt/script files while
excluding generated models, caches, results, reports, and similar directories.
`compute_fingerprint` combines protocol/seed/split files, Git revision, source
hashes, detector package versions, dataset revision, archive mapping, and export
marker identities. The controller adds population/evolution settings.

The frozen `code_snapshot.json` captures dirty working-tree source too: a Git
commit alone would not identify these local changes. Strict equality rejects
resumes after unapproved source/protocol/dependency changes.

### Two different kinds of resume

**Evolution resume:** load the latest population and matching archive checkpoint
and continue at the next generation. Do not rebuild generation zero or rerun
verified baseline evaluations. The controller checkpoint schema does not save
all Python/NumPy/controller RNG state, so this is a valid saved-population resume,
not a promise of the exact hypothetical uninterrupted stochastic trajectory.

**Final training resume:** reload model, optimizer/momentum, completed epochs,
and PyTorch CPU/CUDA RNG state. Resume at an epoch boundary. Python/NumPy RNG
states are not included in this final training checkpoint; the current batch
shuffle uses PyTorch. A mid-epoch interruption reruns the unfinished epoch from
the last completed-epoch checkpoint.

Test-prediction resume reuses completed per-sequence JSON only when checkpoint
and provenance identities match. These are separate mechanisms.

## 19. Freezing Finalists And Official Testing

Read [selection.py](selection.py) and [finalize.py](finalize.py).

### 19.1 Pareto dominance

Architecture A dominates B when A has at least as high mAP50 and mAP50:95,
at most as many parameters, and is strictly better in at least one objective.
The front is every successful archived candidate not dominated by another.
It is not a scalar winner, not only the final generation, and not a fixed count.

`valid_fitness` rejects missing/nonfinite metrics, invalid AP ordering/range,
nonintegral parameter counts, and the historical failure sentinel.
`pareto_front` filters eligible records then performs deterministic pairwise
dominance checks. Equal fitness tuples do not dominate one another.
`load_generation` combines history and current records from generation 20.
`select_winner` remains a legacy helper, not the current finalist policy.

`freeze_manifest` serializes deterministically, compares an existing file exactly,
or writes a temporary file and atomically installs it using a hard link without
overwriting an existing manifest. The finally block removes the temporary file.

### 19.2 `freeze_finalists`

Require successful generation-20 status, both checkpoint files, a matching
archive identity, and verified baseline summary. Find the all-archive Pareto
front and append stock if absent. Rebuild each selected architecture to verify
its parameter count. Deduplicate identical architectures while keeping alias
gene IDs. Copy selected source into an immutable final source directory.

Freeze validation fitness, source hash, architecture hash, lineage, archive and
population checkpoint hashes, training settings, final sampling, and protocol
identity in `final/finalists.json`. No official test metric participates here.

### 19.3 `evaluate_finalist`

Resolve frozen source and verify its hash. Merge the family final settings with
evaluation settings. Build a **new** model under the final seed and record its
fresh initial hash. Verify all 172 official-training exports and their markers.

If a finished `model.pt` exists, require matching provenance and load it. If only
`training.pt` exists, validate its recipe/initialization, restore weights, and
pass optimizer/RNG/epoch state to training. Otherwise start fresh. The per-epoch
callback atomically saves resumable state; after twenty epochs save `model.pt`.
Search-trained weights are not loaded into the fresh final model.

Resolve all 59 official test sequences and process one sequence at a time with
the shared `final_eval.predict_test_split` helper. That helper loads frames in
batches, pairs predictions with clipped GT boxes, retains empty outputs, and
constructs sample IDs. Save a provenance-keyed predictions JSON per sequence.
Do not run the YOLO-specific `final_eval.py` CLI for scratch final training; we
reuse its generic inference helper, not its YOLO training entry point.

Finally, aggregate the test records for global AP and produce a held-out report.
Already completed matching reports can be verified/reused on resume rather than
creating a second independent test trial.

**Memory caveat:** raw image pixels are processed incrementally, but prediction
records/expected IDs are accumulated for global COCO evaluation. "Streaming
images" does not mean constant memory for every metadata structure. Also,
expected test IDs are generated by the pinned frame-reading path; completeness
checks do not independently prove the underlying archive reader decoded every
intended source frame.

### 19.4 `run` and `cli`

Acquire a nonblocking filesystem lock to prevent concurrent finalizers for one
run. Freeze/validate selection, optionally stop at freeze-only, otherwise process
each finalist sequentially. Write an atomic `comparison.csv` and a final-stage
`status.json` only after all finalist reports complete. `cli` exposes run root,
device, and freeze-only options.

Search `status.json == complete` means evolution is complete. It does not imply
`final/status.json == complete`; these are different stage-completion markers.

## 20. Cluster Workflow: Every Script's Job

All paths below are under `scripts/faster_rcnn_scratch/` unless stated otherwise.

| File | Responsibility |
|---|---|
| `setup.sh` | Create/install the detector environment and verify compiled operations. |
| `launch.py` | Validate fresh tag/resume rules, freeze source snapshot, reject active duplicate launches, submit job dependencies, persist launch IDs. Default is a dry-run plan. |
| `smoke.sbatch` | Allocate a GPU for the recorded smoke gate. |
| `smoke.py` | Execute the independent synthetic integration checks described earlier. |
| `export.sbatch` | Eight CPU array shards export both search and final training datasets. |
| `export_finalize.sbatch` | Verify all sequences exported and finalize shared dataset manifests/cache. |
| `baseline.sbatch` | Array seeds 0, 1, 2; one stock search training per task with resume checks. |
| `server.sbatch` | Two H200 GPUs for the local LLM, run-scoped hostname/port, eight-hour allocation. |
| `wait_server.py` | Wait for the paired run's server endpoint to become ready. |
| `evolve.sbatch` | CPU controller, baseline gate, server readiness wait, `run_improved.py`; cleanup trap stops paired server. |
| `stop_server.py` | Clean up only the paired server allocation identified by environment/run metadata. |
| `finalize.sbatch` | One H200 GPU, eight-hour allocation, official frozen-front fresh training/test workflow. |
| `report.sbatch` | Generate search/Pareto plots from archived checkpoints. This is separate from finalizer's held-out comparison table. |
| `compare_pareto.sbatch`, `compare_pareto_table.sbatch` | Historical comparison launchers; not current automatic finalist selection. |
| `train_20epoch.sbatch` | Historical standalone training launcher, not the current official pipeline's finalizer. |

The dependency graph is smoke + exports -> baselines -> server/controller ->
finalizer -> plotting/report job. The controller depends on server start, then
waits for HTTP readiness; start does not mean the LLM is already loaded.

`set -euo pipefail` makes shell scripts fail on unsuccessful commands, undefined
variables, and failed pipeline components. `#SBATCH` directives request scheduler
resources; a `.sbatch` file does not train a model by itself. `SLURM_ARRAY_TASK_ID`
selects export shard or baseline seed. `--dependency=afterok` gates downstream
work on success, avoiding test reports from an incomplete search.

### Reading launcher behavior carefully

`launch.workflow` returns stage names/dependencies; its final-only path is used
when generation 20 is already complete. `main` validates tag/flags, computes
search-complete status, prints a plan when `--submit` is absent, then freezes
source and checks jobs before submission. Job IDs are written incrementally so
a partially submitted launch remains inspectable.

A generic `--resume` before search completion includes smoke/export/baseline
stages with their reuse checks; it is not automatically a downstream-only
submission. Our parser-repair resume used a separate audited downstream-only
script so completed baselines were not retrained.

An eight-hour finalization allocation may not finish every 20-epoch finalist.
Epoch/sequence resume makes continuation possible; the scripts do not guarantee
automatic self-resubmission through arbitrary wall-time limits.

## 21. Historical Code: What Not To Confuse With The Current Path

| File / directory | Role |
|---|---|
| `faster_rcnn_scratch/train.py` | Tiny entry point forwarding to the older standalone scratch trainer. |
| `faster_rcnn_scratch/train_eval.py` | Historical small-proxy candidate trainer; also supplies JSON/seed helpers imported by current finalizer. |
| `faster_rcnn_scratch/compare_pareto.py` | Historical hardcoded comparison of old candidate IDs. |
| `faster_rcnn_scratch/official_split.py` | Earlier pinned official-test resolver/static physical-folder mapping; current workflow uses shared `fred_splits.py`. |
| `results/fred_faster_rcnn_scratch_interim/` | Historical proxy scores/plots, not fitness input to the new run. |
| `faster_rcnn/train_eval.py` standalone CLI | Older pretrained-family workflow. Only selected train/batch/preflight helpers are reused by scratch. |

The older top proxy scores used eight training sequences, three validation
sequences, frame step ten, three epochs, 2,705 training frames, and 1,022
validation frames. Archived configs show `frozen_stages=1` despite random
weights, unlike the current `-1`. The newer search has a different validation
population and training protocol; comparing AP numbers directly does not prove
training on fewer sequences is better.

## 22. Bugs Encountered And What They Teach

### Summary roundoff across Python versions

Independent baselines finished, but one Python version serialized a derived
standard deviation as `0.000247989995314225`, another as
`0.00024798999531422503`. Exact summary comparison rejected otherwise matching
artifacts. `verify_summary` now keeps identities/seed records exact and uses
`rel_tol=1e-12`, `abs_tol=1e-15` only for finite derived aggregate numbers.

Lesson: byte-level/provenance equality and numerically equivalent derived
statistics need different comparison policies.

### Integral parameter count crossing a CSV boundary

The controller parsed all CSV columns as floats. A valid count like
`32368725.0` then violated the result contract's integral type requirement during
cache reuse. `write_cached_fitness` now rejects invalid/bool/fractional/nonfinite
counts and converts valid integral counts to `int` before writing.

Lesson: numeric value correctness is not sufficient when interfaces have type
contracts. Do not silently truncate fractional counts.

### Slurm's short remaining-time format

After generation 15, `squeue` returned `59:50`, not `00:59:50`. The parser only
accepted three fields and crashed before the graceful low-time pause. It now
accepts two fields as minutes/seconds by prepending zero hours, retaining
hours/day handling and malformed-value rejection. Regression tests cover the
actual failing value and threshold cases.

The generation-15 repair retained original source/metadata/checkpoint backups
and an audit, verified the parser was the sole frozen-source change, checked
protocol/dependencies/data unchanged, updated compatibility identities/cache
keys, and verified existing artifact/fitness/population integrity before resume.
That exception does not permit arbitrary fingerprint rewriting for a changed
training protocol. The new server/controller resumed at generation 16.

## 23. Tests: Which Claim Each One Supports

Read the actual tests rather than treating "tests passed" as a proof of every
scientific property.

| Test file | Main coverage |
|---|---|
| [test_faster_rcnn_scratch.py](../../tests/test_faster_rcnn_scratch.py) | Seed/schema, scratch initialization API/determinism, legacy selection helpers, immutable manifest, official parser, checkpoint provenance. |
| [test_faster_rcnn_official.py](../../tests/test_faster_rcnn_official.py) | Family budget isolation, Pareto/ties/failures, illegal/protected/no-op source, protocol-keyed cache, prompt parsing, lazy resizing, preflight state preservation, frozen finalist manifests, interrupted finalization, epoch optimizer/RNG resume, fingerprints, baseline summary gate/roundoff. |
| [test_faster_rcnn.py](../../tests/test_faster_rcnn.py) | Shared MMDetection-family validation/training/inference helper behavior. |
| `tests/operators/test_faster_rcnn_cached_fitness.py` | Integral float/NumPy counts, invalid counts, controller CSV/cache reuse integration. |
| `tests/operators/test_fred_population_integrity.py` | Parent preservation, duplicate/pending cache, failure/retry accounting, scheduler-time parser, population safeguards. |
| `scripts/faster_rcnn_scratch/smoke.py` | Real detector/CUDA operations, backward/update/reload, empty target, all-stage gradients, coordinate bounds. |

Some tests use fake/mocked models to isolate contracts. They do not substitute
for the run-level real GPU smoke. A historical test named for a single winner
does not mean current finalization selects only one model.

## 24. Practical Reading And Inspection Commands

These are reference commands, not instructions to relaunch the active run.
Do not change frozen code/dependencies while jobs are using this checkout.

Read core files with line numbers:

```bash
nl -ba sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py
nl -ba sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/network.py
nl -ba sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/validator.py
nl -ba sota/FRED_LLM_GE/phase1_detection/adapters/faster_rcnn_scratch.py
```

Inspect an architecture without training or touching the test set:

```bash
.venv-faster-rcnn/bin/python -m sota.FRED_LLM_GE.phase1_detection.faster_rcnn_search_space \
  --inspect sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py
```

Run isolated controller tests without accidentally starting an LLM server:

```bash
LLMGE_AUTO_START_SERVER=0 .venv/bin/python -m pytest \
  tests/operators/test_fred_population_integrity.py \
  tests/operators/test_faster_rcnn_cached_fitness.py -q
```

Plan a future fresh run without submitting jobs:

```bash
.venv/bin/python scripts/faster_rcnn_scratch/launch.py \
  --run-tag faster_rcnn_scratch_YYYYMMDD_eot0_experiment
```

Replace `YYYYMMDD` with eight digits. Submission requires explicit `--submit`;
the launcher refuses reusing an existing directory without resume. Do not use
the generic command to bypass a fingerprint mismatch.

## 25. Where To Find Evidence In A Run Directory

For the current official workflow, inspect `fred_runs/<run_tag>/`:

```text
code_snapshot.json                    Frozen source-file hashes
launch.json                           Scheduler job IDs
smoke.json                            GPU smoke evidence and identity
status.json                           Evolution stage status
baseline/seed_0|seed_1|seed_2/          Independent stock models/reports/args
baseline/summary.json                  Validated seed records and statistics
operator_jobs/                        Rendered operator jobs and prompt traces
candidates/models/network_<id>.py      Candidate source
candidates/models/*.proposals.jsonl    Proposal acceptance/rejection evidence
candidates/training_args/<id>/args.json Search settings and initialization
candidates/trained_models/<id>.pt      Search-trained checkpoint
candidates/results/<id>_results.csv    Controller fitness feed
candidates/results/*_evaluation.json   Search-validation report
checkpoints/checkpoint_gen_<n>.pkl     Population and hall of fame
checkpoints/global_gen_<n>.pkl         Archive, ancestry, cache, fingerprint
final/finalists.json                   Frozen all-archive Pareto selection
final/sources/                        Frozen selected source files
final/<id>/training.pt                 Epoch-boundary resumable final training
final/<id>/model.pt                    Completed final model
final/<id>/predictions/<sequence>.json Resumable official-test predictions
final/held_out_reports/                Official-test reports
final/comparison.csv                   Final cross-model test comparison
final/status.json                      Finalization completion status
repairs/                              Explicit audited repair records, if any
```

Scheduler logs are under `run_job_outputs/`, not necessarily inside the run.
`torch.load`/pickle should only be used with these trusted local artifacts, not
untrusted downloaded files: Python checkpoint formats can execute code.

## 26. Suggested Study Order And Self-Checks

1. Read the seed and schema. Write down all eight mutable choices without looking.
2. Read `network.py`. Trace how FPN width 128 changes all dependent interfaces.
3. Read `validator.py`. Explain why a `values` dictionary containing a function
   call cannot reach model construction through the scratch adapter.
4. Read adapter resizing and `prepare_batch`. Trace one box through native,
   resized, padded, and predicted-original coordinate systems.
5. Read `train`. Distinguish finite-loss execution from quality/convergence.
6. Read `fitness` and `map_reporter`. Explain why empty prediction frames remain.
7. Read baseline and architecture policy. Distinguish state/source/architecture/
   protocol hashes and why cache keys need more than a gene ID.
8. Read prompts/operators/controller. Identify where LLM suggestions end and
   validation-based decisions begin.
9. Read selection/finalization. Explain why stock can be tested even when it is
   dominated, and why fresh final weights do not load search checkpoints.
10. Read launcher/fingerprints/tests. Explain what can safely resume and what
    must create a new experiment identity.

The main conceptual separation is:

```text
LLM proposes legal architecture choices.
MMDetection defines detector mechanics and losses.
PyTorch SGD learns weights independently for each uncached candidate.
Validation metrics guide search and freeze finalists.
Official test metrics measure frozen finalists; they do not guide search.
```

This guide gives statement-level coverage of the core representation, validator,
builder, adapter, and training loop, plus function-level coverage of the larger
workflow files. It does not claim to annotate every line of the shared LLM-GE
repository or external framework internals; those are separate codebases with
their own implementations. The linked source remains authoritative.
