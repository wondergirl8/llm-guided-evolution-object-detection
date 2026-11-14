"""The yolo11_window search space: a YOLO11n architecture plus its event-frame window.

A gene is the YOLO11 genome (seeds/seed_yolo11n.yaml) with a third mutable
section holding one key, window_us: how many microseconds of events each
palette frame is drawn from. A mutation changes one layer or the window; the
window comes from WINDOWS_US, the range event_sweep.py and the renderer runs
measured. The architecture keeps every yolo11_search_space rule, and a
candidate's identity (duplicate checks, cached fitness) is its built graph
together with its window. window_runner.py draws the frames and scores it.
"""

import hashlib
import json
import re
import tempfile
from pathlib import Path

import yaml

try:
    from . import yolo11_search_space as base
except ImportError:
    import yolo11_search_space as base

WINDOW_KEY = "window_us"
# 2, 3.3, 5, 10, 16.7 and 33.3 ms (FRED's released frames).
WINDOWS_US = (2000, 3333, 5000, 10000, 16667, 33333)
SEED_WINDOW_US = 33333
_WINDOW_LINE = re.compile(rf"^\s*{WINDOW_KEY}\s*:\s*(\S+)", re.MULTILINE)


def seed_gene_text(seed_text, window_us=SEED_WINDOW_US):
    """The yolo11_window seed: the YOLO11 seed file with a window section appended."""
    return (f"{seed_text.rstrip()}\n\n# --OPTION--\n"
            f"# Microseconds of events drawn into each frame; one of {list(WINDOWS_US)}.\n"
            f"{WINDOW_KEY}: {window_us}\n")


def split_gene(path):
    """(YOLO11 config without the window, window_us) of a gene file."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or WINDOW_KEY not in config:
        raise ValueError(f"yolo11_window gene needs a {WINDOW_KEY} section")
    window = config.pop(WINDOW_KEY)
    if window not in WINDOWS_US:
        raise ValueError(f"{WINDOW_KEY} must be one of {list(WINDOWS_US)}, got {window!r}")
    return config, window


def write_architecture(path, out_path):
    """Write the gene's YOLO11 config alone to out_path (for yolo11_search_space and the
    adapter); return its window_us."""
    config, window = split_gene(path)
    Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False))
    return window


def built_architecture(path):
    """Validate and build the architecture on CPU; return the hash of (graph, window)
    and the parameter count."""
    with tempfile.TemporaryDirectory(prefix="yolo11_window_") as directory:
        architecture = Path(directory) / "architecture.yaml"
        window = write_architecture(path, architecture)
        graph_hash, params = base.built_architecture(architecture)
    digest = hashlib.sha256(json.dumps([graph_hash, window]).encode()).hexdigest()
    return digest, params


def differs_from_parents(child_path, *parent_paths):
    """Reject a candidate whose graph and window both equal an input parent's."""
    child_hash, _ = built_architecture(child_path)
    return all(built_architecture(parent)[0] != child_hash for parent in parent_paths)


def _window_request(prompt):
    """The operator request for a window section, or None for a layer section."""
    match = _WINDOW_LINE.search(prompt)
    if not match:
        return None
    others = ", ".join(str(w) for w in WINDOWS_US if str(w) != match.group(1))
    return (f"An event camera's events are drawn into image frames for a YOLO11n drone "
            f"detector. Each frame currently uses this many microseconds of events:\n\n"
            f"```yaml\n{WINDOW_KEY}: {match.group(1)}\n```\n\n"
            f"Choose a different window from: {others}. A shorter window lets the detector "
            f"react sooner but gives it fewer events per frame; a longer one shows more "
            f"motion but waits longer. Return exactly one fenced yaml block containing only "
            f"`{WINDOW_KEY}: <value>` and no explanation.")


def retry_prompt(prompt, error, attempt):
    """Feedback on a rejected proposal; a window section gets the window request again."""
    window = _window_request(prompt)
    if window is None:
        return base.retry_prompt(prompt, error, attempt)
    return window if attempt == 0 else f"{window}\n\nPrevious proposal was rejected: {error}."


def mutation_prompt(prompt, error, attempt):
    """A layer section keeps the YOLO11 contract; a window section asks for a new window."""
    if _window_request(prompt) is None:
        return base.mutation_prompt(prompt, error, attempt)
    return retry_prompt(prompt, error, attempt)


candidate_chunks = base.candidate_chunks
evaluation_cache_key = base.evaluation_cache_key


def repair_channel_values(chunk):
    """Snap a window to the nearest allowed value; repair layer widths as YOLO11 does."""
    try:
        section = yaml.safe_load(chunk)
    except yaml.YAMLError:
        return chunk
    if isinstance(section, dict) and set(section) == {WINDOW_KEY}:
        window = section[WINDOW_KEY]
        if isinstance(window, (int, float)) and not isinstance(window, bool):
            nearest = min(WINDOWS_US, key=lambda allowed: abs(allowed - window))
            if nearest != window:
                print(f"Snapped {WINDOW_KEY} {window} to the allowed {nearest}", flush=True)
                return f"{WINDOW_KEY}: {nearest}\n"
        return chunk
    return base.repair_channel_values(chunk)


def changed_layer_count(child_path, parent_path):
    """Edited layers plus one if the window changed: a mutation must change exactly one."""
    with tempfile.TemporaryDirectory(prefix="yolo11_window_") as directory:
        child, parent = Path(directory) / "child.yaml", Path(directory) / "parent.yaml"
        child_window = write_architecture(child_path, child)
        parent_window = write_architecture(parent_path, parent)
        return base.changed_layer_count(child, parent) + (child_window != parent_window)


def write_cached_fitness(sota_root, gene_id, fitness):
    """Write a reused (mAP50, mAP50:95, params, window_us) as an ordinary result."""
    try:
        from .result_contract import write_fitness_row
    except ImportError:
        from result_contract import write_fitness_row
    write_fitness_row(sota_root, gene_id,
                      (fitness[0], fitness[1], int(fitness[2]), int(fitness[3])))
