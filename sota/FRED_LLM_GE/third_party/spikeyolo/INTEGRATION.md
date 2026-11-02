# Official SpikeYOLO Gen1 snapshot

Source: https://github.com/BICLab/SpikeYOLO/tree/dc51f35d0b6e970cd157080c98157a41af4392f5/SpikeYOLO_for_Gen1

The complete upstream ultralytics package and bundled SpikingJelly Python package
are preserved, with upstream train.py, README.md, requirements.txt and licenses.
UPSTREAM.json records the source commit and SHA-256 of every included source file.
Do not install a generic Ultralytics/SpikingJelly release as a substitute.

Three upstream files differ. compat.patch records all four changes:
remove the unconditional model.cuda(), probe strides on CPU, and synchronize CUDA
only for CUDA training. The LetterBox override forcing 320x320 is removed so
configured FRED resolutions are honored in training and inference. No layer, neuron, loss, parser, or model YAML is rewritten.

Project-owned changes live in phase1_detection/adapters/spikeyolo.py:
isolate label-cache writes and runtime settings; disable external tracking;
reset state before every independent training batch and evaluation frame; use
the shared FRED evaluator and final-epoch FP32 checkpoint instead of the upstream
Gen1 validator/best-checkpoint selection. The official training loop, optimizer,
scheduler, gradient clipping, augmentation, target assignment, and loss are used.

Upstream train.py selects snn_yolov8s.yaml, resolved by the official parser to
cfg/models/v8/snn_yolov8.yaml at scale s. Its MS_GetT arguments [3,1] mean T=1.
The README's separately downloadable Gen1 checkpoint says T=2; it is not loaded.
FRED changes nc from 80 to 1. Input is released RGB Event PNG, not Gen1 NumPy
event volumes. The shared FRED budget replaces Gen1's 100 training epochs.

The source is AGPL-3.0; bundled SpikingJelly retains its own LICENSE separately
as SPIKINGJELLY_LICENSE. Keep all notices with redistributed source.

The project-owned spikeyolo-graph-v2 genome exposes the stock backbone and neck
YAML through validated constructor choices. Evolution changes the config passed
to the official parser, never this source snapshot. The unmodified seed still
resolves to the exact same official single-class s graph. See
../../docs/SPIKEYOLO_RUN_GUIDE.md for the v2 search boundaries and checks.
