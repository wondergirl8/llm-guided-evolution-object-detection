# Plan for `seed_rtdetr.py`

**Status:** Seed, family adapter, runner/LLM-GE wiring and unit tests implemented.
A from-scratch smoke test of the unmutated seed is the next step.

## 1. Deliverable and scope

One Python implementation file, following the `seed_dfine.py` layout:

```text
sota/FRED_LLM_GE/seeds/seed_rtdetr.py
```

Unlike the D-FINE seed, this is a **port of the original RT-DETR (v1)**, not an
"inspired" design. Its architecture, matcher, criterion and contrastive denoising
are copied from the official PyTorch implementation the FRED paper cites
(reference [30], *DETRs Beat YOLOs on Real-time Object Detection*):

| Source | Pin |
| --- | --- |
| Repository | [`lyuwenyu/RT-DETR`](https://github.com/lyuwenyu/RT-DETR), folder `rtdetr_pytorch` (v1, not RT-DETRv2) |
| Commit | `df2a96c8c3a098567a1149bf0248da6efe6d495d` |
| License | Apache-2.0 |

The seed is self-contained: it does not import the official repository, and its
config/registry system is replaced by explicit constructor arguments.

## 2. Single-file layout

```text
seed_rtdetr.py
├── Protected preamble
│   ├── imports, box utilities, activation and init helpers
│   ├── official building blocks: ConvNormLayer, BasicBlock, BottleNeck, Blocks,
│   │   RepVggBlock, CSPRepLayer, TransformerEncoderLayer, TransformerEncoder,
│   │   MLP, MSDeformableAttention, TransformerDecoderLayer, TransformerDecoder
│   ├── contrastive denoising group (training-only query construction)
│   ├── HungarianMatcher and RTDETRCriterion (VFL + L1 + GIoU, official weights)
│   ├── depth presets taken from the official configs
│   └── RTDETREventDetector: builds the components, validates their contracts,
│       defines forward(event, targets) and predict(event)
├── marker 1 → PResNet          (backbone)
├── marker 2 → HybridEncoder    (encoder)
└── marker 3 → RTDETRTransformer (decoder)
```

As in the D-FINE seed, each marker starts one mutable section holding exactly
one top-level class, and nothing follows marker 3. The protected building blocks
play the role of D-FINE's `ConvNormAct`: candidates may use them or build new
layers from `torch.nn` inside their class.

| Section | Constructor | Contract checked by the wrapper |
| --- | --- | --- |
| `PResNet` | `(depth)` | `out_channels`, `out_strides == [8, 16, 32]`; returns three maps at those strides |
| `HybridEncoder` | `(in_channels, hidden_dim, expansion, dim_feedforward)` | `out_channels`; returns three maps with the input spatial sizes |
| `RTDETRTransformer` | `(num_classes, hidden_dim, num_queries, feat_channels, num_decoder_layers, num_denoising)` | `pred_logits [B,Q,1]`, normalized `pred_boxes [B,Q,4]` (cxcywh); in training, `num_decoder_layers` `aux_outputs` entries and, when targets exist, `dn_aux_outputs` + `dn_meta` |

## 3. Backbone depth setting

`RTDETREventDetector(depth=...)` selects the official preset for the seed's
starting backbone. Mutations may then change the backbone code freely.

| Depth | Backbone channels | Encoder width / FFN / expansion | Decoder width / layers | Params (1 class) |
| --- | --- | --- | --- | --- |
| 18 | 128, 256, 512 | 256 / 1024 / 0.5 | 256 / 3 | 20.1M |
| 34 | 128, 256, 512 | 256 / 1024 / 0.5 | 256 / 4 | 31.3M |
| 50 | 512, 1024, 2048 | 256 / 1024 / 1.0 | 256 / 6 | 42.8M |
| 101 | 512, 1024, 2048 | 384 / 2048 / 1.0 | 256 / 6 | 76.5M |

Parameter counts match the official model table (20M / 31M / 42M / 76M).

Order for the smoke test: R50 first; fall back to a smaller depth only if R50 is
impractical from scratch, and record the measurement either way.

## 4. Deliberate differences from the official training setup

These follow the team's current protocol (`configs/phase1/protocol.yaml`) and
the existing families, and are recorded so results are not mistaken for an
official reproduction.

| Official RT-DETR | This seed / adapter | Why |
| --- | --- | --- |
| ImageNet-pretrained backbone, frozen norm (R50) | **Random init, trainable BatchNorm, nothing frozen** | Team decision: every family trains from scratch. If RT-DETR is unusable from scratch, report that as a finding. |
| 72 epochs, EMA, multi-scale training, photometric/IoU-crop/flip augmentation | Protocol epochs (search: 20), no EMA, no augmentation, single scale | Same proxy budget and data handling as the D-FINE family. |
| Square 640×640 resize | Resize to the protocol `img_width` keeping aspect ratio, then zero-pad right/bottom to a multiple of 32 | The encoder upsamples by exactly 2×, so the input must be divisible by 32; padding keeps box geometry undistorted. |
| Lower backbone LR (pretrained) | One AdamW learning rate for all parameters | No pretrained weights to protect. |
| NMS-free top-k postprocessing | Top-k, then the shared proxy `eval_conf` + NMS | Same scoring path as the other families. |
| COCO (80 classes) | One class (drone, label 0) | FRED detection task. |

## 5. LLM-GE mutation contract

Same mechanism as D-FINE (`src/llm_utils.py`, `src/llm_mutation.py`,
`src/llm_crossover.py`): exactly three markers, four nonempty parts, the
preamble byte-identical to the seed's (checked by the adapter), and each section
one class with the fixed name and constructor. Invalid candidates fail with
worst-case fitness; they are never repaired into apparent successes.
