"""Score one yolo11_window candidate (FRED_SEED_FAMILY=yolo11_window).

    uv run python sota/FRED_LLM_GE/phase1_detection/window_runner.py \\
        --model network_<gene_id> --variant_dir <models dir> --sota_root <sota root>

The gene (window_search_space.py) is a YOLO11 architecture plus window_us. The
search split is drawn as plain palette frames (seeds/seed_renderer.py) from
window_us of events per frame, into node-local scratch ($TMPDIR) as
render_runner.py does; the architecture is then trained and scored on them by
candidate_runner.run under the yolo11 protocol entry, and the frames are
deleted.

Results go to <sota_root>/results/<gene_id>_results.csv as
(mAP50, mAP50:95, params, window_us), matching FITNESS_WEIGHTS (1, 1, -1, -1):
accurate, small models on short windows are preferred.
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

try:
    from .candidate_runner import DEFAULT_PROTOCOL, load_protocol, run
    from .event_sweep import read_result
    from .render_runner import SEED_RENDERER, render_split
    from .result_contract import write_fitness_row
    from .window_search_space import write_architecture
except ImportError:
    from candidate_runner import DEFAULT_PROTOCOL, load_protocol, run
    from event_sweep import read_result
    from render_runner import SEED_RENDERER, render_split
    from result_contract import write_fitness_row
    from window_search_space import write_architecture

FAILED = (float("-inf"), float("-inf"), float("inf"), float("inf"))


def fail(sota_root, gene_id, start, reason):
    """Record worst-case fitness for a candidate that cannot be scored."""
    write_fitness_row(sota_root, gene_id, FAILED, reason=reason)
    print(f"ERROR evaluating generated model: {reason}")
    print(f"Wrote worst-case fitness after {time.time() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Score one architecture + window candidate")
    parser.add_argument("--model", required=True, help='Gene module name like "network_XXXX"')
    parser.add_argument("--variant_dir", required=True, help="Directory where LLM-GE writes genes")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--sota_root", required=True, help="Where results/ and reports/ are written")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gene_id = args.model.split("network_", 1)[1]
    start = time.time()
    if device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        fail(args.sota_root, gene_id, start,
             "gpu_unavailable: GPU allocated but CUDA is unavailable on this node")
        raise SystemExit(1)
    protocol = load_protocol(args.protocol)
    settings = protocol["families"]["renderer"]
    workdir = Path(tempfile.mkdtemp(prefix=f"window_{gene_id}_", dir=os.environ.get("TMPDIR")))
    try:
        # This is LLM Guided Code: a broken gene gets worst-case fitness.
        models = workdir / "models"
        models.mkdir()
        try:
            window = write_architecture(Path(args.variant_dir) / f"{args.model}.py",
                                        models / f"{args.model}.py")
            roots, stats = render_split(SEED_RENDERER, protocol, workdir / "frames",
                                        settings["render_workers"], settings["render_timeout_s"],
                                        window)
        except Exception as e:
            fail(args.sota_root, gene_id, start, f"setup_or_render_failed: {e!r}")
            return
        print(f"rendered {stats['frames']} frames in {stats['render_s']} s, window {window} us, "
              f"{stats['events_per_frame']} events per frame", flush=True)

        eval_root = workdir / "eval"
        try:
            run(gene_id, args.model, models, "yolo11", protocol, eval_root, device, roots)
            score, error = (read_result(eval_root / "results" / f"{gene_id}_results.csv"),
                            "training or scoring failed")
        except Exception as e:
            score, error = None, repr(e)
        if score is None:
            # candidate_runner records why (e.g. the YOLO11 size bounds); keep its reason.
            failed = eval_root / "results" / f"{gene_id}_results.csv"
            if failed.is_file() and len(failed.read_text().split(",")) > 3:
                error = failed.read_text().split(",", 3)[3].strip() or error
            fail(args.sota_root, gene_id, start, error)
            return
        map50, map50_95, params = score
        for sub in ("reports", "trained_models", "training_args"):
            if (eval_root / sub).is_dir():
                shutil.copytree(eval_root / sub, Path(args.sota_root) / sub, dirs_exist_ok=True)
        write_fitness_row(args.sota_root, gene_id, (map50, map50_95, params, window))
        print(f"fitness: mAP50 {map50:.4f}, mAP50:95 {map50_95:.4f}, params {params}, "
              f"window {window} us ({time.time() - start:.0f} s)", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
