# FRED event frames

Open `sota/FRED_LLM_GE` as the editor workspace and run `uv sync` there to create
the project `.venv`. The workspace settings select that interpreter and add the
repository root to editor import paths. The project pytest configuration adds
the same path when running tests.

The top-level `data/fred_stream.py` reads released event PNG frames from one FRED
sequence through the Hugging Face `datasets` streaming API. It needs no local FRED dataset copy.
The ZIP stays remote; the script first scans member paths, then decodes only the
event frames requested by the caller.

From a Python process started at the repository root with
`uv run --project sota/FRED_LLM_GE python`, inspect the first frame with:

```python
from data.fred_stream import stream_event_frames
image, timestamp_s = next(stream_event_frames("train", 36))
print(timestamp_s, image.size)
```

Or run a bounded smoke check from the FRED project directory:

```sh
uv run python ../../data/fred_stream.py train 36 --limit 2
```

Each result is a Pillow image and its relative timestamp in seconds. The script
creates `data/bad_timestamps.csv` when run; pass a third argument to
`stream_event_frames` to choose another location. Missing counter ranges and
skipped malformed, unexpected, duplicate, or wrong-sequence frame names are
reported there. The report is replaced on each run.
RGB filenames are scanned only as metadata: the loader reports malformed RGB
timestamps, RGB/event frame-count differences, and implausible adjacent RGB
timestamp gaps. It never decodes RGB images. The event counter is the output
timestamp; the RGB clock is not used to rewrite it. The count and interval checks
flag possible alignment problems, but they are not a dataset-wide proof of exact
RGB/event synchronization.

The dataset revision is pinned in `fred_stream.py` to the project Phase 0 config.
Access requires a network connection. Reading frames necessarily transfers those
frame bytes, but a full sequence ZIP or dataset mirror is not saved locally.
