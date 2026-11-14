"""Render the challenging_test frames for an event setting once, for final_eval.py --test_cache.

    uv run python sota/FRED_LLM_GE/phase1_detection/cache_test_frames.py \\
        --events raw_w10000_palette --workers 12

Frames drawn from raw events (e.g. 10 ms) take ~7.5 h to render per test run;
many seeds of a final eval (Experiment_Procedure: 30 per model) would redo
that every time. This CPU-only job renders them once with the same code
(final_eval.cached_sequence) into <data_root>/test_frames/<setting>_step<k>/,
one zip per test sequence, and skips sequences already there, so it can be
rerun to finish an interrupted cache.
"""

import argparse
import multiprocessing
import time

try:
    from .export_split import events_from_tag
    from .final_eval import (DEFAULT_PROTOCOL, FRED_DIR, cached_sequence, challenging_split,
                             load_protocol, test_cache_dir, test_jobs)
except ImportError:
    from export_split import events_from_tag
    from final_eval import (DEFAULT_PROTOCOL, FRED_DIR, cached_sequence, challenging_split,
                            load_protocol, test_cache_dir, test_jobs)


def main():
    parser = argparse.ArgumentParser(description="Render challenging_test frames once for --test_cache")
    parser.add_argument("--events", required=True, help='Event frames, e.g. "raw_w10000_palette"')
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--workers", type=int, default=4,
                        help="Processes rendering sequences (each holds one sequence's events)")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    challenging = protocol["challenging"]
    data_root = FRED_DIR / challenging["data_root"]
    events = events_from_tag(args.events)
    step = challenging["final"]["test_frame_step"]
    cache = test_cache_dir(data_root, events, step)
    cache.mkdir(parents=True, exist_ok=True)
    _, test = challenging_split(data_root)
    jobs = [job for job in test_jobs(test, {"test_frame_step": step, "events": events}, cache, cache)
            if not job[0].is_file()]
    print(f"{cache}: {len(test) - len(jobs)} of {len(test)} sequences cached; rendering {len(jobs)}",
          flush=True)
    start, frames = time.time(), 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for number, rendered in enumerate(pool.imap_unordered(cached_sequence, jobs), 1):
            frames += len(rendered)
            print(f"{number}/{len(jobs)} sequences, {frames} frames, {time.time() - start:.0f} s",
                  flush=True)
    missing = [sequence for sequence, _ in test if not (cache / f"{sequence}.zip").is_file()]
    if missing:
        raise SystemExit(f"sequences still missing from {cache}: {missing}")
    print(f"done: all {len(test)} test sequences in {cache}")


if __name__ == "__main__":
    main()
