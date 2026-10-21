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
import hashlib
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


def code_key(path):
    """The renderer's editable code without comments or blank lines: equal keys draw equal frames."""
    code = Path(path).read_text().split("# --OPTION--", 1)[-1]
    lines = (line.rstrip() for line in code.splitlines())
    return "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#"))


def earlier_score(renderer_path, sota_root, variant_dir):
    """(gene_id, render stats) of an already scored renderer with the same code, or None.

    Rendering and training are deterministic (fixed seed), so LLM-GE's frequent
    exact copies, e.g. crossovers of two identical parents, need not be retrained.
    A twin that failed gives stats {"failed": True}.
    """
    key = code_key(renderer_path)
    for done in sorted((Path(sota_root) / "results").glob("*_results.csv")):
        gene = done.name[:-len("_results.csv")]
        path = Path(variant_dir or "") / f"network_{gene}.py"
        if done.read_text().strip().endswith(",999999999") and path.is_file() \
                and Path(path) != Path(renderer_path) and code_key(path) == key:
            return gene, {"failed": True}
    for done in sorted((Path(sota_root) / "renders").glob("*.json")):
        gene = done.stem
        path = SEED_RENDERER if gene == "seed" else Path(variant_dir or "") / f"network_{gene}.py"
        stats = json.loads(done.read_text())
        if path.is_file() and Path(path) != Path(renderer_path) and "duplicate_of" not in stats \
                and code_key(path) == key:
            return gene, stats
    return None


def wait_for_twin(renderer_path, sota_root, gene_id, poll_s=30, max_wait_s=3 * 3600):
    """Register this renderer as running; if an identical one is already being
    scored, wait for it to finish. Returns the claim file to delete when done.

    Twins submitted together both see each other's claim, so the one with the
    smaller gene id runs and the other waits.
    """
    running = Path(sota_root) / "renders" / "running"
    running.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(code_key(renderer_path).encode()).hexdigest()
    claim = running / f"{gene_id}.{digest}"
    claim.touch()
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        twins = [c for c in running.glob(f"*.{digest}") if c.name.split(".")[0] < gene_id
                 and time.time() - c.stat().st_mtime < max_wait_s]
        if not twins:
            break
        print(f"waiting for identical renderer {twins[0].name.split('.')[0]}", flush=True)
        time.sleep(poll_s)
    return claim


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
    try:
        claim = wait_for_twin(renderer_path, args.sota_root, gene_id)
    except Exception as e:
        print(f"twin check skipped: {e!r}", flush=True)
        claim = None
    try:
        earlier = earlier_score(renderer_path, args.sota_root, args.variant_dir)
    except Exception as e:
        print(f"duplicate check skipped: {e!r}", flush=True)
        earlier = None
    if earlier and earlier[1].get("failed"):
        write_failure_results(args.sota_root, gene_id, start, f"same code as {earlier[0]}, which failed")
        claim and claim.unlink(missing_ok=True)
        return
    if earlier:
        twin, stats = earlier
        claim and claim.unlink(missing_ok=True)
        renders = Path(args.sota_root) / "renders"
        (renders / f"{gene_id}.json").write_text(json.dumps(dict(stats, duplicate_of=twin)))
        write_results(args.sota_root, gene_id,
                      DetectionResult(stats["mAP50"], stats["mAP50_95"], stats["window_us"]))
        print(f"same code as {twin}: reusing its fitness mAP50 {stats['mAP50']:.4f}, "
              f"mAP50:95 {stats['mAP50_95']:.4f}, window {stats['window_us']} us", flush=True)
        return
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
        if claim:
            claim.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
