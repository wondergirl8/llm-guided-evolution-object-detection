# Phase 1 — Drone Detection Research References
## FRED + LLM-GE Research Project

**Research snapshot:** 2026-09-11  
**Purpose:** Curated references for planning Phase 1 (drone detection), emphasizing recent object-detection advances, tiny/small drone detection, RGB-event fusion, event-based detection, real-time efficiency, and methods that may improve or diversify the project's LLM-GE seed population.

> This file is a research reference index, not a governing planning document. Fixed project policy remains in `MASTER_PROJECT_PLAN.md`, `ENGINEERING_IMPLEMENTATION_STANDARD.md`, and the phase plans.

## Priority / phase-fit labels

- **HIGH** — investigate while planning Phase 1 because it may materially change the seed/search strategy.
- **MEDIUM** — useful technique or alternative seed, but not necessarily part of the first controlled LLM-GE study.
- **WATCH** — promising but immature, code-limited, recent/preprint-only, or outside the current Phase 0 representation.
- **Phase 1 v1** — potentially compatible with the frozen Phase 0 released-frame representation.
- **Custom seed** — especially useful for a later additional seed family.
- **Training/search technique** — may be reusable without defining a completely new seed architecture.
- **Future Phase 0 extension** — relies on raw-event or alternative event representations and must not be silently inserted into Phase 1 v1.

---

# 1. Current Project Repositories

## LLM-GE Framework
**Repository:** https://github.com/jasonzutty/llm-guided-evolution-fork/tree/MosesTheRedSea-main

- Current project evolutionary framework for mutation, crossover, EoT/feedback, population management, lineage and fitness handling.
- Phase 1 should connect detector families through controlled representations/adapters rather than allow unrestricted edits across entire external repositories.

## FRED
**Repository:** https://github.com/miccunifi/FRED  
**Paper:** https://arxiv.org/abs/2506.05163

- Current benchmark authority for RGB-event drone detection, tracking and forecasting.
- Its published YOLO11, RT-DETR, Faster R-CNN and ER-DETR results are the main reproduction anchors for Phase 1.

## Ultralytics — YOLO11 Seed Family
**Repository:** https://github.com/ultralytics/ultralytics

- Current YOLO11 seed ecosystem.
- The same repository now also contains YOLO26, allowing a controlled comparison between the FRED-comparable YOLO11 seed and a substantially newer Ultralytics detector.

## RT-DETR
**Repository:** https://github.com/lyuwenyu/RT-DETR

- Current Transformer/query-based seed ecosystem.
- Also contains newer RT-DETRv2 work, so the exact seed version should be an explicit Phase 1 decision.

## Faster R-CNN Candidate Frameworks
**Torchvision:** https://github.com/pytorch/vision  
**Detectron2:** https://github.com/facebookresearch/detectron2  
**MMDetection:** https://github.com/open-mmlab/mmdetection

- Faster R-CNN preserves two-stage/proposal-based architectural diversity.
- Framework choice matters because it changes how easily small-object FPN modules, custom heads and LLM-GE mutation boundaries can be implemented.

---

# 2. What the Existing Three Seeds Cover

| Seed family | Paradigm | Important 2024–2026 work not automatically represented |
|---|---|---|
| YOLO11 | Dense one-stage, multi-scale real-time detection | YOLO26 STAL/P2/training changes; event-native processing; modern RGB-event fusion |
| RT-DETR | Real-time end-to-end DETR/query detection | RT-DETRv4 distillation; D-FINE localization; DEIM fast convergence; drone-specific DETR variants |
| Faster R-CNN | Two-stage proposal + ROI detection | Tiny-object FPN denoising/transformer modules; modern spectral/background training techniques |

**Implication:** retain all three for controlled FRED reproduction, but do not treat them as complete coverage of 2026 object-detection research.

---

# 3. Highest-Priority Modern Detector Updates

## 3.1 YOLO26 + YOLO26-P2
**Priority:** HIGH  
**Phase fit:** Phase 1 v1 / modern comparison seed / LLM-GE mutation ideas

**Official documentation:** https://docs.ultralytics.com/models/yolo26/  
**Official repository:** https://github.com/ultralytics/ultralytics  
**P2 architecture:** https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml  
**Paper:** https://arxiv.org/abs/2606.03748

- Newer than YOLO11 and introduces Small-Target-Aware Label Assignment (STAL), Progressive Loss, MuSGD, DFL-free box regression, and optional NMS-free end-to-end inference.
- Ultralytics ships a P2/4 detection-head YAML specifically exposing a higher-resolution small-object detection scale, making it directly relevant to tiny/far-away drones.
- Strong candidate for a modern comparator or source of evolvable components, while YOLO11 remains the FRED-reproduction anchor.

## 3.2 RT-DETRv2
**Priority:** HIGH  
**Phase fit:** Phase 1 v1 / updated RT-DETR comparison

**Official repository:** https://github.com/lyuwenyu/RT-DETR  
**Paper:** https://arxiv.org/abs/2407.17140

- Direct follow-up to RT-DETR with improved training strategy and deformable-attention behavior while preserving the real-time DETR design.
- Because it lives in the same official ecosystem, it is a natural modern baseline to compare with the FRED-era RT-DETR seed.

## 3.3 RT-DETRv4
**Priority:** HIGH  
**Phase fit:** Modern comparison seed / custom seed / evolution reference

**Official repository:** https://github.com/RT-DETRs/RT-DETRv4  
**Paper:** https://arxiv.org/abs/2510.25257

- Accepted at ECCV 2026; uses Vision Foundation Model knowledge/distillation to strengthen lightweight real-time detectors.
- Particularly relevant because it aims to improve detector quality without adding inference-time cost.
- Useful either as a modern Transformer comparator or as a source of distillation/search ideas.

## 3.4 D-FINE
**Priority:** HIGH  
**Phase fit:** Phase 1 v1 / custom seed / RT-DETR-family mutation inspiration

**Official repository:** https://github.com/Peterande/D-FINE  
**Paper:** https://arxiv.org/abs/2410.13842

- Reframes DETR box regression as Fine-grained Distribution Refinement (FDR) and adds Global Optimal Localization Self-Distillation (GO-LSD).
- Strongly relevant to tiny drones because a few pixels of localization error can sharply reduce IoU and `mAP50:95`.
- Components may be valuable as evolvable localization modules even if D-FINE is not added as a full fourth seed.

## 3.5 DEIM / DEIMv2
**Priority:** HIGH  
**Phase fit:** Training/search technique / modern DETR custom seed

**Official repository:** https://github.com/Intellindust-AI-Lab/DEIM  
**Paper:** https://arxiv.org/abs/2412.04234

- Introduces Dense One-to-One matching and Matchability-Aware Loss to accelerate DETR convergence while improving accuracy.
- Reported to reduce training time by about 50% when used with RT-DETR/D-FINE variants—especially valuable because LLM-GE may train many candidates.
- DEIMv2 expands the family with more model sizes and DINOv3-based features/pretraining.

## 3.6 RF-DETR
**Priority:** HIGH  
**Phase fit:** Custom seed / modern comparison seed

**Official repository:** https://github.com/roboflow/rf-detr  
**Paper:** https://arxiv.org/abs/2511.09554

- ICLR 2026 real-time Transformer using a DINOv2 backbone, with strong accuracy/latency and domain-shift performance reported by the project.
- Provides substantially different search material from YOLO11, classic RT-DETR, and Faster R-CNN.
- Especially interesting because its development involved architecture search/NAS, creating a useful comparison point for LLM-guided evolution.

## 3.7 LW-DETR
**Priority:** MEDIUM-HIGH  
**Phase fit:** Custom seed / modern comparison seed

**Official repository:** https://github.com/Atten4Vis/LW-DETR  
**Paper:** https://arxiv.org/abs/2406.03459

- Lightweight detector built from a ViT encoder, projector and shallow DETR decoder with interleaved window/global attention.
- Offers a relatively clean Transformer architecture and is part of the architectural lineage behind RF-DETR.

---

# 4. Drone / Tiny-Object-Specific Research

## 4.1 UAV-DETR — DETR for Anti-Drone Target Detection
**Priority:** HIGH  
**Phase fit:** Custom seed / mutation-component inspiration

**Official repository:** https://github.com/wd-sir/UAVDETR  
**Paper:** https://arxiv.org/abs/2603.22841

- 2026 architecture explicitly targeted at anti-drone detection rather than generic COCO detection.
- Includes small-target-oriented ideas such as local/sliding-window attention, cross-scale feature fusion/recalibration and tiny-box localization changes.
- Very useful for a future drone-specific custom DETR seed, though newer/less mature than the established baseline families.

## 4.2 SET — Spectral Enhancement for Tiny Object Detection
**Priority:** HIGH  
**Phase fit:** Training/search technique / custom modules

**Official repository:** https://github.com/HuixinSun/SET  
**Venue:** CVPR 2025

- Suppresses high-frequency background noise and increases object-feature saliency, specifically targeting tiny-object detection.
- The official implementation applies its modules during training only, with no intended inference-time burden.
- Highly relevant to FRED's clutter, distractors and difficult lighting; worth a controlled FRED ablation.

## 4.3 DNTR — DeNoising FPN + Transformer R-CNN
**Priority:** HIGH  
**Phase fit:** Faster R-CNN modernization / custom seed

**Official repository:** https://github.com/hoiliu-0801/DNTR  
**Paper:** https://arxiv.org/abs/2406.05755

- Adds a DeNoising FPN and Transformer-style R-CNN head for tiny objects.
- The repository includes Faster R-CNN configurations using DN-FPN, making this a concrete modernization path for the project's two-stage seed.
- Strong source of FPN/ROI-head mutation ideas for LLM-GE.

## 4.4 DQ-DETR — Dynamic Query for Tiny Object Detection
**Priority:** MEDIUM-HIGH  
**Phase fit:** Custom DETR seed / query-mechanism inspiration

**Repository:** https://github.com/hoiliu-0801/DQ-DETR

- Targets tiny-object DETR weaknesses through dynamic query design.
- Query construction/selection is a natural controlled mutation target for an RT-DETR-family LLM-GE population.

## 4.5 CFPT — Cross-Layer Feature Pyramid Transformer
**Priority:** MEDIUM-HIGH  
**Phase fit:** FPN/neck module / two-stage or custom seed

**Official repository:** https://github.com/duzw9311/CFPT  
**Paper:** https://arxiv.org/abs/2407.19696

- Designed for small-object detection in aerial images and improves cross-level feature-pyramid interaction.
- Relevant to drones occupying very few pixels and could inform neck/FPN mutations in Faster R-CNN or YOLO-derived custom seeds.

## 4.6 Improving Small Drone Detection Through Multi-Scale Processing and Data Augmentation
**Priority:** HIGH  
**Phase fit:** Phase 1 v1 ablation / training and inference strategy

**Paper:** https://arxiv.org/abs/2504.19347  
**Project page:** https://raysonlaroca.github.io/supp/drone-vs-bird/

- Uses YOLO11m, making it unusually close to one of our existing seed families.
- Combines full-frame and segmented/zoomed processing, prediction aggregation, copy-paste augmentation and temporal consistency for small drone-vs-bird detection.
- Multi-scale/copy-paste ideas deserve direct FRED ablation; any temporal method must first be checked against FRED's no-future-information detection protocol.

## 4.7 SAHI — Slicing Aided Hyper Inference
**Priority:** MEDIUM  
**Phase fit:** Diagnostic / controlled ablation

**Official repository:** https://github.com/obss/sahi  
**Paper:** https://arxiv.org/abs/2202.06934

- Generic tiled inference keeps small targets larger relative to detector resolution and supports Ultralytics, Torchvision, Detectron2 and MMDetection ecosystems.
- Useful for diagnosing whether FRED misses arise mainly from spatial downscaling rather than architecture.
- Slicing increases inference cost, so it must be evaluated against FRED's runtime/protocol constraints rather than adopted automatically.

---



## 4.8 NWD — Normalized Gaussian Wasserstein Distance for Tiny Object Detection
**Priority:** HIGH  
**Phase fit:** Localization/loss technique / Faster R-CNN and custom-seed mutation idea

**Official repository:** https://github.com/jwwangchn/NWD  
**Paper/reference:** https://arxiv.org/abs/2110.13389

- NWD was designed specifically for tiny-object detection because ordinary IoU becomes extremely unstable when a bounding box contains only a few pixels; a very small displacement can cause a disproportionate IoU drop.
- It represents boxes as Gaussian distributions and compares them using normalized Wasserstein distance, providing a more stable similarity/localization signal for tiny objects.
- Particularly relevant to FRED because tiny drone localization quality strongly affects `mAP50:95`; it is also directly demonstrated with Faster R-CNN in the official implementation, making it a useful loss/assignment idea for that evolutionary branch.

# 5. RGB–Event / Multimodal Detection Research

## 5.1 Neuromorphic Drone Detection: an Event-RGB Multimodal Approach
**Priority:** HIGH  
**Phase fit:** Existing FRED multimodal comparator / custom-seed foundation

**Paper:** https://arxiv.org/abs/2409.16099

- Directly studies drone detection with synchronized RGB and event data and is closely related to the multimodal ER-DETR baseline used in FRED.
- Establishes why the modalities are complementary: event sensing is strong under high-speed/adverse illumination, while RGB supplies texture and appearance when event activity is weak.
- Any new RGB-event custom seed should be compared against this lineage rather than treating fusion as unexplored.

## 5.2 SPFD — Shared/Private Feature Decoupling for RGB-Event Detection
**Priority:** HIGH  
**Phase fit:** Custom multimodal seed

**Official repository:** https://github.com/git-KeYw/SPFD  
**CVPR 2026 paper:** https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html

- Separates shared, RGB-private and event-private information instead of simply concatenating the two modalities.
- Its TriAdapt Encoder and TriInject Decoder dynamically emphasize texture-rich RGB versus motion-sensitive event features.
- One of the strongest current references for a future FRED-specific fusion seed because it directly tackles the modality-fusion problem exposed by FRED.

## 5.3 PEPR — Privileged Event-Based Predictive Regularization
**Priority:** HIGH  
**Phase fit:** Custom training strategy / robustness study

**Official repository:** https://github.com/miccunifi/PEPR  
**Paper:** https://arxiv.org/abs/2602.04583

- From the same University of Florence research ecosystem as FRED; uses event information during training to improve a model that uses RGB only at inference.
- Directly targets domain generalization (for example day-to-night shifts), making it highly relevant to the challenging split.
- Offers a different research path from always fusing both sensors at inference: events can provide privileged robustness supervision.

**Maturity note:** at the 2026-09-11 research snapshot, the repository says detection code is forthcoming, so this is currently more useful as a research direction than drop-in code.

---

# 6. Event-Native / Raw-Event Research — Future Phase 0 Extension

> **Important:** Phase 0 v1 fixes the released FRED event-frame representation. The references below are highly relevant scientifically, but methods requiring raw events, voxelization, point clouds or new temporal aggregation must not be silently incorporated into Phase 1 v1.

## 6.1 EV-UAV / EV-SpSegNet
**Priority:** HIGH for future research  
**Phase fit:** Future Phase 0 extension / event-native custom seed

**Official repository:** https://github.com/ChenYichen9527/EV-UAV  
**Paper:** https://arxiv.org/abs/2506.23575

- ICCV 2025 work focused specifically on event-based tiny UAV detection, with reported targets averaging only about 6.8 × 5.4 pixels.
- EV-SpSegNet exploits sparse event point-cloud structure and uses a spatiotemporal-correlation loss to retain coherent moving-target events while rejecting isolated noise.
- Extremely relevant to the physics of tiny moving drones, but its input representation differs from the frozen Phase 0 v1 event-frame path.

## 6.2 SAST — Scene Adaptive Sparse Transformer
**Priority:** MEDIUM-HIGH for future research  
**Phase fit:** Future Phase 0 extension / event-native custom seed

**Official repository:** https://github.com/Peterande/SAST  
**Venue:** CVPR 2024

- Sparse Transformer designed for event-based object detection with window/token co-sparsification.
- Useful architecture reference for exploiting event sparsity rather than treating event data purely as dense image-like frames.

## 6.3 SparseVoxelDet — Fully Sparse Event-Camera Detection
**Priority:** HIGH / WATCH  
**Phase fit:** Future Phase 0 extension / efficiency research

**Paper:** https://arxiv.org/abs/2603.21638

- 2026 work evaluated directly on FRED and uses sparse 3D convolutions through backbone, feature pyramid and detection head.
- Reports competitive FRED detection with very sparse active data and large memory/storage reductions; its error analysis emphasizes localization near-misses.
- Particularly interesting for a future efficient event-native LLM-GE study, but it changes the representation and therefore belongs outside Phase 1 v1.

**Maturity caution:** recent preprint; verify code/reproducibility before implementation planning.

## 6.4 OpenEvDET
**Priority:** MEDIUM  
**Phase fit:** Literature/resource hub / future event-native research

**Official repository:** https://github.com/Event-AHU/OpenEvDET

- Maintains event-stream detection benchmarks, methods and a living paper list.
- Useful as a reference hub when selecting future event-native modules or benchmark comparators.

---

# 7. Domain Surveys / Landscape Resources

## 7.1 Drone Detection with Event Cameras
**Priority:** HIGH  
**Phase fit:** Phase 1 research framing / future seed design

**Paper:** https://arxiv.org/abs/2508.04564

- 2025 survey focused specifically on event-camera drone detection, including representations, detectors, tracking, forecasting and neuromorphic sensing.
- Useful for checking whether custom seed ideas are genuinely grounded in counter-UAV/event research rather than only generic object detection.

## 7.2 Small Object Detection and Tracking Resources
**Priority:** MEDIUM  
**Phase fit:** Literature discovery

**Repository:** https://github.com/ChenYichen9527/small_object_detection_resources

- Living collection of small/tiny object papers, code, datasets and workshops.
- Useful after this one-time research snapshot as new tiny-object work continues to appear.

## 7.3 OpenEvDET Paper List
**Priority:** MEDIUM  
**Phase fit:** Event-detection literature discovery

**Repository:** https://github.com/Event-AHU/OpenEvDET/blob/main/Paper_list.md

- Curated list of event-based object-detection algorithms, surveys and datasets.
- Useful for surveying future raw-event architectures before selecting additional event-native seeds.

---


## 7.4 Securing the Skies — Anti-UAV Survey, Benchmarking, and Future Directions
**Priority:** HIGH  
**Phase fit:** Domain research map / Phase 1 problem framing

**CVPR 2025 Workshop paper:** https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Dong_Securing_the_Skies_A_Comprehensive_Survey_on_Anti-UAV_Methods_Benchmarking_CVPRW_2025_paper.html  
**arXiv:** https://arxiv.org/abs/2504.11967

- A recent anti-UAV survey covering detection, classification, tracking, multimodal sensing, benchmarks, and emerging directions such as multimodal fusion, self-supervised learning, vision-language methods, and synthetic data.
- Useful as a high-level domain check when deciding whether a Phase 1 custom seed or experiment addresses a real anti-drone research gap rather than only improving generic COCO-style detection.
- Highlights continuing challenges in real-time performance, stealth/small-target detection, robustness, and multi-UAV settings, which are directly relevant to how Phase 1 should evaluate detector quality beyond a single AP number.

# 8. Main Gaps in the Current Three-Seed Setup

## YOLO11 gap

YOLO11 predates important YOLO26 ideas:

- STAL small-target label assignment;
- a supplied P2 small-object detection-head architecture;
- Progressive Loss;
- MuSGD;
- DFL-free regression;
- optional native NMS-free inference.

**Phase 1 implication:** preserve YOLO11 for FRED reproduction, then seriously evaluate YOLO26-P2 or selected YOLO26 mechanisms as a modern comparison/custom seed.

## RT-DETR gap

Original RT-DETR predates or does not inherently contain:

- RT-DETRv4 Vision Foundation Model distillation;
- D-FINE distribution-based localization refinement;
- DEIM Dense O2O / Matchability-Aware training;
- UAV-DETR's drone-specific tiny-target changes.

**Phase 1 implication:** the Transformer branch has a particularly large set of 2024–2026 improvements worth evaluating.

## Faster R-CNN gap

Classic Faster R-CNN does not inherently include recent tiny-object-specific:

- denoised FPNs;
- Transformer R-CNN heads;
- spectral background suppression;
- cross-layer feature-pyramid Transformers.

**Phase 1 implication:** do not discard the two-stage seed. DNTR, SET and CFPT provide useful modernization/evolution material while retaining architectural diversity.

---

# 9. Recommended Shortlist for Phase 1 Planning

To avoid exploding the project into too many seed families, investigate the literature in layers.

## Tier A — Mandatory controlled anchors

1. **YOLO11**
2. **RT-DETR matching the FRED/reference setup**
3. **Faster R-CNN matching the FRED/reference setup**

These preserve direct comparison with the published FRED benchmark.

## Tier B — Highest-value modern additions / techniques

1. **YOLO26 / YOLO26-P2** — direct modern upgrade to the YOLO branch with explicit small-target changes.
2. **D-FINE + DEIM** — stronger localization plus substantially faster DETR convergence.
3. **RT-DETRv4** — latest major RT-DETR direction with VFM knowledge distillation.
4. **DNTR components** — direct tiny-object modernization path for the Faster R-CNN branch.
5. **SET** — tiny-object background/saliency training improvements with no intended inference overhead.

These do not all need to become independent seed families; some may instead become allowed search-space components or ablations.

## Tier C — Candidate custom / additional seed families

1. **RF-DETR** — modern VFM-backed real-time Transformer.
2. **UAV-DETR** — domain-specific anti-drone Transformer.
3. **SPFD** — modern RGB-event shared/private fusion.
4. **PEPR-style model** — event-privileged training with RGB-only inference.
5. **LW-DETR** — lightweight Transformer family.

## Tier D — Separate future raw-event branch

1. **EV-UAV / EV-SpSegNet**
2. **SAST**
3. **SparseVoxelDet**
4. **OpenEvDET methods**

These require a deliberate Phase 0 research-extension decision before formal use.

---

# 10. Suggested Phase 1 Research Strategy After This Review

```text
Layer 1 — Reproduce FRED
YOLO11 / RT-DETR / Faster R-CNN
RGB + Event baselines

             ↓

Layer 2 — Controlled LLM-GE study
family-specific evolution with fixed data/modality/protocol

             ↓

Layer 3 — Modern-detector comparison
YOLO26-P2
RT-DETRv4 / D-FINE+DEIM
DNTR-enhanced two-stage detector
(possibly RF-DETR)

             ↓

Layer 4 — Custom multimodal/drone seeds
SPFD-inspired RGB/Event fusion
PEPR-inspired privileged-event training
UAV-DETR-inspired tiny-drone architecture

             ↓

Layer 5 — Separate event-native research
EV-UAV / SAST / SparseVoxelDet-style raw-event processing
```

This preserves interpretability: improvements from LLM-GE can be separated from gains caused simply by replacing an older seed or changing the input representation.

---

# 11. Questions to Resolve While Drafting Phase 1

1. Should YOLO11 remain the mandatory FRED anchor while YOLO26-P2 is added as a modern comparator?
2. Should the original RT-DETR be the only evolutionary RT-DETR seed, or should RT-DETRv2/v4 receive separate modern-baseline status?
3. Can DEIM reduce candidate-training cost enough to increase LLM-GE population/generation size without changing architecture rankings?
4. Should D-FINE-style localization refinement enter the allowed mutation space because tiny drone boxes are highly IoU-sensitive?
5. Should Faster R-CNN evolution explicitly expose DN-FPN / cross-layer FPN mechanisms?
6. Does SET materially help FRED's difficult backgrounds/lighting?
7. Does tiled/multi-scale processing help enough to justify its inference cost?
8. Should the first custom multimodal seed be ER-DETR-like, SPFD-like, or another deliberately simpler fusion baseline?
9. Can PEPR improve robust RGB detection while allowing RGB-only inference?
10. Which methods improve localization versus recall? FRED reports `mAP50:95`, so this distinction matters.
11. Which innovations fit cleanly into the LLM-GE structured candidate representation?
12. Which techniques reduce training cost enough to improve the evolutionary search itself?

---

# 12. Research Guardrail

A detector should not be adopted merely because it reports higher COCO AP.

Before becoming a Phase 1 seed, mutation component or training rule, a method should be evaluated for:

- relevance to tiny / fast drones;
- compatibility with the FRED challenging protocol;
- compatibility with frozen Phase 0;
- code/reproduction maturity;
- training and inference cost;
- ability to fit a controlled LLM-GE representation;
- licensing;
- whether it changes available information or preprocessing in a way that makes comparisons unfair.

The three FRED-comparable seeds therefore remain valuable anchors even when newer methods are added.
