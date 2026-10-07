# Simple FRED event-frame streaming plan

## Goal

Implement one `data/fred_stream.py` script with an event-frame iterator and a small command-line entry point. It should read the released FRED event-frame PNGs from the official Hugging Face dataset through its remote API and yield each usable frame with its relative timestamp. It must work without a local FRED dataset copy. The archived Phase 0 implementation is outside this task.

This is an event-frame access step, not a claim that the broader Phase 0 data and evaluation gates are complete.

## Scope and proposed interface

- Input: an explicit FRED split and sequence ID, such as `train/36.zip`, from `GabrieleMagrini/FRED`. Use the revision already pinned in `configs/phase0/default.yaml` so the data source is reproducible. Start with one sequence; do not scan the entire dataset to get a sample.
- Output: a Python iterator of decoded event PNG images, each accompanied by its timestamp in seconds. The sequence ID is an input and the timestamp is derived from the frame counter; no new framework or manifest is needed.
- Read only released `Event/Frames/Video_<sequence>_frame_<counter>.png` members for the output. Inspect RGB filenames as metadata to flag missing/misaligned timestamps, but do not decode RGB frames, generate event frames from `Event/events.hdf5`, parse annotations, or implement model adapters in this script.
- Do not require or create a full local dataset mirror or extract sequence archives into a dataset directory. Requested frame bytes must still travel over the network; library-managed temporary caching or byte-range reads may occur.
- Create `data/bad_timestamps.csv` as the one local report artifact. Recreate it for each run, including its header when no problems are found, so reruns do not duplicate findings. Do not commit the generated report.

## First verification: remote access

1. Install the `datasets` dependency in the intended environment if absent; it is not currently declared in the root `pyproject.toml` or installed in the repository virtual environment.
2. Try `datasets.load_dataset(..., streaming=True)` against **one explicitly selected, pinned FRED sequence ZIP**. Check the resulting row schema, whether an event member's path and bytes are available, and whether the stream can be restricted to or filtered for `Event/Frames/*.png` without decoding RGB images. Do not infer this from the dataset card alone.
3. Observe whether fetching the first few event frames causes a full archive download or substantial unwanted RGB transfer. If the `datasets` path cannot meet the event-only, remote-access requirement, record the observed limitation before choosing another official Hugging Face API access method. Do not hide a full archive download behind the word "streaming."
4. Check the order of event members. If the API does not yield them in counter order, choose a way to order frame references without holding all image payloads in memory before implementing gap detection.

The source publishes one ZIP per sequence, containing RGB, released event frames, and raw HDF5 events. Hugging Face documents remote ZIP streaming, but that does not establish the exact behavior for this repository's mixed ZIP contents. This check determines the simplest workable access path.

## Timestamp and reporting behavior

- Parse the integer counter from each event PNG name. For the verified FRED naming convention, the first expected counter is `33333` microseconds and the step is `33333` microseconds; `timestamp_s = counter / 1_000_000`. Verify this against the selected remote sequence before relying on it generally.
- A malformed name, wrong sequence ID, duplicate counter, or counter off the expected grid is an unexpected-counter finding. Skip that frame and add one row to `bad_timestamps.csv`.
- If valid counters have a gap, log the missing expected counter(s) and continue with the next valid frame. There is no frame to yield or skip for a missing counter. Do not manufacture images or timestamps for gaps.
- The report should identify the split, sequence, member name when present, issue, expected counter, and observed counter when present. A missing frame after the last available member cannot be inferred without an independent expected frame count; do not report one speculatively.
- A remote-access failure is distinct from a bad timestamp: surface it as an error rather than silently treating the whole sequence as skipped.

## Implementation sequence

1. Run the one-sequence remote-access check above and record what the API actually returns.
2. Implement a single event-frame iterator and the CSV reporter in `fred_stream.py`; keep the public interface small and avoid carrying over the archived cache, manifest, split, or lifecycle system.
3. Add only focused checks for filename/counter parsing, gap and duplicate reporting, and iterator output. Use a tiny test fixture plus a bounded live smoke check (a few frames from one sequence) when network access is available.
4. Confirm the smoke check yields event images with timestamps, produces a readable `bad_timestamps.csv`, does not need local FRED files, and does not trigger a dataset-wide download.

## References and limits

- [Official FRED dataset card](https://huggingface.co/datasets/GabrieleMagrini/FRED): sequence ZIP structure and released event-frame data.
- [Hugging Face Datasets loading reference](https://huggingface.co/docs/datasets/package_reference/loading_methods): `streaming=True` and remote ZIP support.
- Existing `artifacts/phase0/verification_register.yaml` records that sequence 21 used `Video_21_frame_33333.png` and that the upstream 30 Hz timestamp formula matched its sampled annotations. This is evidence for a starting convention, not a completed dataset-wide audit.

## Implementation check (2026-10-05)

`fred_stream.py` now implements this scoped plan. With `datasets==4.8.5`, a live
`streaming=True` read of pinned `train/36.zip` exposed member paths without image
bytes. The archive contained 3,408 event-frame paths; their raw iteration order
was not chronological, so the script sorts those paths by counter before decoding.
All 3,408 counters in that sequence were on the `33333`-microsecond grid with no
gaps. The script decoded the first two event PNGs at 1280×720 with timestamps
`0.033333` and `0.066666` seconds. A separate one-frame measurement returned
8,217,998 bytes across 13 remote range requests from a 314,402,579-byte archive;
the temporary Hugging Face cache contained no dataset archive. This verifies one
sequence, not every FRED sequence.

The later task check also scanned the 3,408 RGB filenames in `train/36.zip`.
They have six-digit microsecond timestamps, one per event frame. Adjacent RGB
intervals ranged from 33,288 to 40,104 microseconds; the RGB clock accumulated
36,447 microseconds of relative offset across the sequence. Therefore the script
checks RGB count and local intervals, but does not require the two timestamp
clocks to be exactly equal or silently shift event timestamps. It reports possible
alignment defects in `bad_timestamps.csv` while returning only valid event frames.

The codebase example in `sota/Titanic/` was checked for its parser and entry-point
structure. Its local CSV input assumption does not apply to the remote FRED loader.
