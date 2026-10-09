# FRED + LLM-GE

## Python environment

From this directory, run `uv sync` to create `.venv` from `pyproject.toml` and
`uv.lock`. Use `uv run` for FRED commands. If this directory is open in VS Code,
`.vscode/settings.json` selects the environment. The `data` package lives in this
project directory, so Pylance can resolve it directly.
If VS Code has already selected another interpreter for this workspace, run
**Python: Select Interpreter** and choose this directory's `.venv/bin/python`.

```sh
uv run pytest phase0_data/test_fred_stream.py -q
uv run python data/fred_stream.py train 36 --limit 2
```

This directory contains the staged implementation of the FRED data, detection,
tracking, forecasting, and LLM-Guided Evolution research project.

The team streaming loader is described in [`phase0_data/README.md`](phase0_data/README.md).
The Event-only YOLO11 integration check is described in
[`phase1_detection/INTEGRATION_RUNBOOK.md`](phase1_detection/INTEGRATION_RUNBOOK.md).
It uses the prepared-data compatibility backend in `archive/phase0_data_old`
through `data.yolo_export`; this does not decide the team's authoritative loader.
The team's fusion seed and rules remain available separately. Planning and
governance live under [`docs/`](docs/).

The official FRED source remains remote and immutable. Runtime data and caches are
not committed to Git.
