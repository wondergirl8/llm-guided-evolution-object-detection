# Detector integration

The Faster R-CNN Event pilot uses MMDetection R50-FPN, bounded literal genomes,
and the same detection result contract and mAP reporter as YOLO11. See
[its setup, baseline, and pilot guide](seeds/faster_rcnn/README.md).

Select the evolutionary family with `LLMGE_DETECTOR=faster_rcnn`; the default
remains YOLO11. Faster R-CNN dependencies and H200 kernels are isolated from the
main environment. Its initial three-epoch pilot is exploratory.

The separate [random-initialization 20-generation experiment](seeds/faster_rcnn_scratch/README.md)
uses the same bounded architecture genome without loading pretrained weights.
