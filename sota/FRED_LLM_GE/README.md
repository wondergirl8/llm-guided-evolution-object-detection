# FRED + LLM-GE

Object detection on FRED Event frames, with standalone seed training and an
optional LLM-Guided Evolution layer.

## Start here

- [Detection models and commands](phase1_detection/README.md)
- [FRED data streaming and export](phase0_data/README.md)
- [Cluster run checklist](docs/PHASE1_RUN_CHECKLIST.md)
- [Research plans and engineering rules](docs/AGENTS.md)

## Layout

```text
FRED_LLM_GE/
├── data/                 Shared FRED streaming and event rendering
├── phase0_data/          Data export adapter and streaming checks
├── phase1_detection/
│   ├── seeds/            One folder per implemented model or experiment
│   │   ├── ultralytics_yolo11/  Official YOLO11n YAML
│   │   ├── yolo11/       Tutorial-sized detector (tiny_yolo11)
│   │   ├── dfine/        D-FINE-inspired detector
│   │   ├── faster_rcnn/  COCO-initialized MMDetection pilot
│   │   └── faster_rcnn_scratch/  Random-initialization experiment
│   ├── adapters/         Model interfaces for the shared candidate runner
│   └── tests/            Detection regression and interface checks
├── configs/              Shared protocol and repository configuration
├── baseline_validation/  Baseline evidence, separate from evolution
├── experiments/          Experiment manifests and summaries
├── artifacts/            Verification records and durable result references
├── docs/                 Research plans, contracts, and run instructions
└── archive/              Historical Phase 0 implementation and tests
```

Each model folder owns its seed source and any model-specific training code.
Data loading, export, metrics, and candidate orchestration remain shared.
RT-DETR is planned in the research documents; it has no implemented seed here.
Active evolution prompts live in the repository's `templates/FRED*/` folders.
The `prompts/` and `logs/` directories reserve locations from the research plan.

Raw FRED data stays remote and immutable. Runtime caches, generated candidates,
weights, and run outputs are not source files and are not committed to Git.
Historical code and verification evidence under `archive/` and `artifacts/`
remain available for reproducibility.

## Python environment

Detection commands below use the repository-root environment declared in the
root `pyproject.toml`. From the repository root, run `uv sync`, then use `uv run`.
The MMDetection experiments have their own setup and environment; follow the
linked model guides before running them.

For the standalone Phase 0 streaming tools, this directory has a smaller
`pyproject.toml` and `uv.lock`. From **this directory**, run:

```sh
uv sync
uv run pytest phase0_data/test_fred_stream.py -q
uv run python data/fred_stream.py train 36 --limit 2
```

`.vscode/settings.json` selects that smaller environment when this directory is
opened in VS Code. If needed, use **Python: Select Interpreter** and choose this
directory's `.venv/bin/python`.
