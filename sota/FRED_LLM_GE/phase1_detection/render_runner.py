"""Score one evolved event-frame renderer (FRED_SEED_FAMILY=renderer).

    uv run python sota/FRED_LLM_GE/phase1_detection/render_runner.py \\
        --model network_<gene_id> --variant_dir <models dir> --sota_root <sota root>

"--model network" scores the seed renderer (seeds/seed_renderer.py). The
renderer draws every frame of the challenging search split from the raw events
cached and sliced by cache_events.py; the frames go to node-local scratch ($TMPDIR) in the
layout export_split.py writes, reusing the released export's frame list and
labels, so only the images differ. The fixed YOLO11n seed is then trained and
scored on them by candidate_runner.run, and the frames are deleted.

Results go to <sota_root>/results/<gene_id>_results.csv as
(mAP50, mAP50:95, WINDOW_US): FITNESS_WEIGHTS (1, 1, -1) then favour accurate
frames from short windows. The renderer only ever receives the events inside
its own window, so WINDOW_US is the true amount of time each frame waits for.
"""

import argparse
import importlib.util
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    from .candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, load_protocol, run,
                                   write_failure_results)
    from .cache_events import MAX_WINDOW_US, slices_dir
    from .export_split import STEP_US, dataset_parts, finalize
    from .result_contract import DetectionResult, write_results
except ImportError:
    from candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, load_protocol, run,
                                  write_failure_results)
    from cache_events import MAX_WINDOW_US, slices_dir
    from export_split import STEP_US, dataset_parts, finalize
    from result_contract import DetectionResult, write_results
from fred_events import HEIGHT, WIDTH  # on sys.path via proxy_data

SEED_RENDERER = FRED_DIR / "seeds" / "seed_renderer.py"
MIN_WINDOW_US = 1000
RELEASED = {"source": "released", "window_us": STEP_US, "encoding": "palette"}


def load_renderer(path):
    """The renderer module at path, with its WINDOW_US and render checked."""
    spec = importlib.util.spec_from_file_location(f"renderer_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = getattr(module, "WINDOW_US", None)
    if isinstance(window, bool) or not isinstance(window, int) or not MIN_WINDOW_US <= window <= MAX_WINDOW_US:
        raise ValueError(f"WINDOW_US must be an int from {MIN_WINDOW_US} to {MAX_WINDOW_US}, got {window!r}")
    if not callable(getattr(module, "render", None)):
        raise ValueError("renderer must define render(events, window_us)")
    return module


def window_events(events, end_us, window_us):
    """A copy of the events with end_us - window_us < t < end_us, t made relative to end_us."""
    t = events["t"]
    lo = np.searchsorted(t, end_us - window_us, side="right")
    hi = np.searchsorted(t, end_us, side="left")
    window = events[lo:hi].copy()
    window["t"] -= end_us
    return window


def render_frame(renderer, events, end_us):
    """(image, number of events in its window)."""
    window = window_events(events, end_us, renderer.WINDOW_US)
    image = renderer.render(window, renderer.WINDOW_US)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (HEIGHT, WIDTH, 3):
        shape = getattr(image, "shape", None)
        raise ValueError(f"render must return a uint8 array of shape ({HEIGHT}, {WIDTH}, 3), "
                         f"got {type(image).__name__} {getattr(image, 'dtype', '')} {shape}")
    return image, len(window)


_renderer = None


def _init_worker(path):
    global _renderer
    _renderer = load_renderer(path)


def _render_sequence(job):
    """Render one sequence's frames; returns (frames, events used over all frames)."""
    events_path, records, image_dir = job
    events = np.load(events_path)
    used = 0
    for record in records:
        image, count = render_frame(_renderer, events, record["frame_index"] * STEP_US)
        used += count
        Image.fromarray(image).save(Path(image_dir) / f"{record['name']}.png", compress_level=1)
    return len(records), used


def render_split(renderer_path, protocol, workdir, workers, timeout_s):
    """Render the search split into workdir; returns ((train_root, val_root), stats)."""
    challenging = dict(protocol["challenging"], events=RELEASED)
    data_root = FRED_DIR / challenging["data_root"]
    cache = slices_dir(challenging)
    roots, jobs, parts = [], [], []
    for released, sequences, _ in dataset_parts(challenging, data_root, "search"):
        out = Path(workdir) / released.name
        (out / "images").mkdir(parents=True)
        (out / "done").mkdir()
        # Same frames and boxes as the released export: share its labels, copy its markers.
        (out / "labels").symlink_to(released / "labels", target_is_directory=True)
        for sequence, _ in sequences:
            done = released / "done" / f"{sequence}.json"
            events = cache / f"{sequence}.npy"
            for path, hint in ((done, "export_split.py --set search"), (events, "cache_events.py --slice")):
                if not path.is_file():
                    raise FileNotFoundError(f"{path} missing: run {hint}")
            shutil.copy2(done, out / "done")
            jobs.append((str(events), json.loads(done.read_text())["records"], str(out / "images")))
        roots.append(out)
        parts.append((sequences, out))
    start = time.time()
    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (str(renderer_path),)) as pool:
        # A renderer that is too slow (e.g. a Python loop over events) fails here.
        results = pool.map_async(_render_sequence, jobs, chunksize=1).get(timeout=timeout_s)
    frames = sum(n for n, _ in results)
    stats = {"frames": frames, "render_s": round(time.time() - start, 1),
             "events_per_frame": round(sum(u for _, u in results) / frames)}
    for sequences, out in parts:
        finalize(sequences, out, challenging["trainer_val_frames"])
    return tuple(roots), stats


def main():
    parser = argparse.ArgumentParser(description="Score one evolved event-frame renderer")
    parser.add_argument("--model", default="network",
                        help='Renderer module name like "network_XXXX"; "network" scores the seed')
    parser.add_argument("--variant_dir", default=None, help="Directory where LLM-GE writes renderers")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--sota_root", required=True, help="Where results/ and renders/ are written")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gene_id = args.model.split("network_")[1] if "network_" in args.model else "seed"
    start = time.time()
    if device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        write_failure_results(args.sota_root, gene_id, start, "GPU allocated but CUDA is unavailable")
        raise SystemExit(1)
    protocol = load_protocol(args.protocol)
    settings = protocol["families"]["renderer"]
    renderer_path = (SEED_RENDERER if args.model == "network"
                     else Path(args.variant_dir) / f"{args.model}.py")
    workdir = Path(tempfile.mkdtemp(prefix=f"render_{gene_id}_", dir=os.environ.get("TMPDIR")))
    try:
        # This is LLM Guided Code: a broken renderer gets worst-case fitness.
        try:
            renderer = load_renderer(renderer_path)
            roots, stats = render_split(renderer_path, protocol, workdir,
                                        settings["render_workers"], settings["render_timeout_s"])
        except Exception as e:
            write_failure_results(args.sota_root, gene_id, start, f"renderer failed: {e!r}")
            return
        stats["window_us"] = renderer.WINDOW_US
        print(f"rendered {stats['frames']} frames in {stats['render_s']} s, window {renderer.WINDOW_US} us, "
              f"{stats['events_per_frame']} events per frame", flush=True)

        # Train and score the fixed seed model on the rendered frames, in the workdir.
        eval_root = workdir / "eval"
        try:
            run(gene_id, "network", None, settings["model_family"], protocol, eval_root, device, roots)
            map50, map50_95, params = (float(v) for v in
                                       (eval_root / "results" / f"{gene_id}_results.csv").read_text().split(","))
        except Exception as e:
            params, error = None, repr(e)
        else:
            error = "training or scoring failed"
        if params in (None, 999999999):  # 999999999: candidate_runner.FAILED_RESULT
            write_failure_results(args.sota_root, gene_id, start, error)
            return
        renders = Path(args.sota_root) / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        (renders / f"{gene_id}.json").write_text(json.dumps(dict(stats, mAP50=map50, mAP50_95=map50_95)))
        for sub in ("reports", "trained_models"):
            if (eval_root / sub).is_dir():
                shutil.copytree(eval_root / sub, Path(args.sota_root) / sub, dirs_exist_ok=True)
        write_results(args.sota_root, gene_id, DetectionResult(map50, map50_95, renderer.WINDOW_US))
        print(f"fitness: mAP50 {map50:.4f}, mAP50:95 {map50_95:.4f}, window {renderer.WINDOW_US} us "
              f"({time.time() - start:.0f} s)", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
