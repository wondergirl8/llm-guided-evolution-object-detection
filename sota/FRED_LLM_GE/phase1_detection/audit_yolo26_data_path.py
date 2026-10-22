"""Read-only audit of the challenging-search YOLO export and image access.

The audit uses the export's ``done/*.json`` markers as its pixel-box record.
It cannot establish whether a duplicate box was already present in the original
FRED annotation ZIP unless that source is separately inspected.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from . import candidate_runner
    from .export_split import yolo_label
except ImportError:
    import candidate_runner
    from export_split import yolo_label


def _sample(paths, limit):
    """Take deterministic, evenly spaced paths across the complete split."""
    count = min(limit, len(paths))
    if count == 0:
        return []
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[i * (len(paths) - 1) // (count - 1)] for i in range(count)]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def probe_images(root, paths, limit):
    """Time stat and full-file read on a bounded sample, as Ultralytics does."""
    rows = []
    for path in _sample(paths, limit):
        start = time.perf_counter()
        size = os.stat(path).st_size
        stat_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        with path.open("rb") as stream:
            content = stream.read()
        read_ms = (time.perf_counter() - start) * 1000
        if len(content) != size:
            raise OSError(f"image changed while reading: {path}")
        rows.append({"image": str(path.relative_to(root)), "bytes": size,
                     "stat_ms": round(stat_ms, 3), "read_ms": round(read_ms, 3),
                     "read_mb_per_s": round(size / (1 << 20) / max(read_ms / 1000, 1e-9), 3)})
    return {
        "sample_count": len(rows),
        "stat_ms_mean": round(statistics.mean(row["stat_ms"] for row in rows), 3) if rows else None,
        "stat_ms_p95": _percentile([row["stat_ms"] for row in rows], .95) if rows else None,
        "read_ms_mean": round(statistics.mean(row["read_ms"] for row in rows), 3) if rows else None,
        "read_ms_p95": _percentile([row["read_ms"] for row in rows], .95) if rows else None,
        "samples": rows,
    }


def _add_issue(report, message):
    report["issue_count"] += 1
    if len(report["issues"]) < 100:
        report["issues"].append(message)


def _duplicate_groups(lines):
    positions = defaultdict(list)
    for index, line in enumerate(lines):
        parts = line.split()
        try:
            if len(parts) != 5 or not all(math.isfinite(float(x)) for x in parts):
                return None
            positions[tuple(map(float, parts))].append(index)
        except ValueError:
            return None
    return [indices for indices in positions.values() if len(indices) > 1]


def _check_trainer_val(root, report):
    copied = root / "trainer_val"
    if not copied.is_dir():
        _add_issue(report, "trainer_val directory missing")
        return
    paths = sorted((copied / "images").glob("*.png"))
    report["trainer_val_frames"] = len(paths)
    if not paths:
        _add_issue(report, "trainer_val has no images")
    for image in paths:
        for subdir, suffix in (("images", ".png"), ("labels", ".txt")):
            replica = copied / subdir / f"{image.stem}{suffix}"
            original = root / subdir / f"{image.stem}{suffix}"
            if not original.is_file() or not replica.is_file():
                _add_issue(report, f"trainer_val missing {subdir}/{image.stem}{suffix}")
            elif hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(
                    replica.read_bytes()).digest():
                _add_issue(report, f"trainer_val differs from train: {subdir}/{image.stem}{suffix}")
    extra_labels = {path.stem for path in (copied / "labels").glob("*.txt")} - {
        path.stem for path in paths}
    for stem in sorted(extra_labels):
        _add_issue(report, f"trainer_val label has no image: {stem}")


def _check_dataset_listing(root, expected_names, report):
    train_list = root / "train.txt"
    if not train_list.is_file():
        _add_issue(report, "train.txt missing")
    else:
        listed = train_list.read_text().splitlines()
        expected = [str(root / "images" / f"{name}.png") for name in sorted(expected_names)]
        if [str(Path(path).resolve()) for path in listed] != expected:
            _add_issue(report, "train.txt differs from marker image list or uses stale paths")
        report["train_list_frames"] = len(listed)
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        _add_issue(report, "data.yaml missing")
    else:
        try:
            data = yaml.safe_load(data_yaml.read_text())
        except yaml.YAMLError as exc:
            _add_issue(report, f"data.yaml invalid: {exc}")
        else:
            if (not isinstance(data, dict) or
                    Path(str(data.get("path", ""))).resolve() != root or
                    data.get("train") != "train.txt" or
                    data.get("val") != "trainer_val/images" or
                    data.get("names") != {0: "drone"}):
                _add_issue(report, "data.yaml differs from finalized export or uses a stale root")


def audit_split(root, sample_images=16, trainer_val=False, expected_step=None):
    """Check marker/label/image consistency without modifying the export."""
    root = Path(root).resolve()
    markers = sorted((root / "done").glob("*.json"))
    if not markers:
        raise FileNotFoundError(f"no export markers under {root / 'done'}")
    if sample_images < 0:
        raise ValueError("sample_images must be nonnegative")
    report = {"root": str(root), "markers": len(markers), "frames": 0,
              "background_frames": 0, "label_rows": 0, "duplicate_label_rows": 0,
              "duplicate_label_frames": 0, "marker_duplicate_rows": 0,
              "rounding_collision_rows": 0, "unclassified_duplicate_rows": 0,
              "duplicate_examples": [], "issue_count": 0, "issues": []}
    label_hash = hashlib.sha256()
    marker_hash = hashlib.sha256()
    expected_names = set()
    images = []
    for marker_path in markers:
        raw_marker = marker_path.read_bytes()
        marker_hash.update(marker_path.name.encode() + b"\0" + raw_marker + b"\0")
        marker = json.loads(raw_marker)
        sequence = str(marker["sequence"])
        width, height = marker["size"]
        records = marker["records"]
        if sequence != marker_path.stem or marker["frames"] != len(records):
            _add_issue(report, f"marker identity/count mismatch: {marker_path.name}")
        if expected_step is not None and marker["frame_step"] != expected_step:
            _add_issue(report, f"marker frame step differs from protocol: {marker_path.name}")
        if not all(type(value) is int and value > 0 for value in (width, height)):
            raise ValueError(f"invalid image size in {marker_path}")
        for record in records:
            name = record["name"]
            frame_index = record["frame_index"]
            if name != f"{sequence}_{frame_index:06d}" or name in expected_names:
                _add_issue(report, f"duplicate or mismatched frame identity: {name}")
            expected_names.add(name)
            report["frames"] += 1
            boxes = record["boxes"]
            report["background_frames"] += not bool(boxes)
            report["label_rows"] += len(boxes)
            for box in boxes:
                if (len(box) != 4 or not all(math.isfinite(x) for x in box)
                        or not (0 <= box[0] < box[2] <= width and
                                0 <= box[1] < box[3] <= height)):
                    _add_issue(report, f"invalid marker box: {name}")
            image = root / "images" / f"{name}.png"
            label = root / "labels" / f"{name}.txt"
            if not image.is_file() or image.stat().st_size == 0:
                _add_issue(report, f"missing or empty image: {name}.png")
            else:
                images.append(image)
            if not label.is_file():
                _add_issue(report, f"missing label: {name}.txt")
                continue
            label_text = label.read_text()
            label_hash.update(label.name.encode() + b"\0" + label_text.encode() + b"\0")
            expected = yolo_label(boxes, width, height)
            if label_text != expected:
                _add_issue(report, f"label differs from marker: {name}.txt")
            groups = _duplicate_groups(label_text.splitlines())
            if groups is None:
                _add_issue(report, f"invalid YOLO label row: {name}.txt")
                continue
            if groups:
                report["duplicate_label_frames"] += 1
            for indices in groups:
                extras = len(indices) - 1
                report["duplicate_label_rows"] += extras
                if label_text != expected or max(indices) >= len(boxes):
                    report["unclassified_duplicate_rows"] += extras
                    kind = "label_marker_mismatch"
                else:
                    marker_count = sum(tuple(boxes[index]) == tuple(boxes[indices[0]])
                                       for index in indices[1:])
                    report["marker_duplicate_rows"] += marker_count
                    report["rounding_collision_rows"] += extras - marker_count
                    kind = "marker_duplicate" if marker_count == extras else (
                        "rounding_collision" if marker_count == 0 else "mixed")
                if len(report["duplicate_examples"]) < 100:
                    report["duplicate_examples"].append(
                        {"frame": name, "rows_1_based": [index + 1 for index in indices],
                         "extra_rows": extras, "kind": kind})
    for subdir, suffix in (("images", ".png"), ("labels", ".txt")):
        present = {path.stem for path in (root / subdir).glob(f"*{suffix}")}
        for name in sorted(present - expected_names):
            _add_issue(report, f"untracked {subdir} file: {name}{suffix}")
    report["marker_sha256"] = marker_hash.hexdigest()
    report["label_sha256"] = label_hash.hexdigest()
    _check_dataset_listing(root, expected_names, report)
    if trainer_val:
        _check_trainer_val(root, report)
    try:
        report["image_probe"] = probe_images(root, sorted(images), sample_images)
    except OSError as exc:
        _add_issue(report, f"image read probe failed: {exc}")
        report["image_probe"] = None
    return report


def audit_search(data_root, protocol, sample_images=16):
    search = protocol["challenging"]["search"]
    root = Path(data_root).resolve()
    train = audit_split(root / f"search_train_step{search['train_frame_step']}",
                        sample_images, trainer_val=True,
                        expected_step=search["train_frame_step"])
    val = audit_split(root / f"search_val_step{search['val_frame_step']}",
                      sample_images, expected_step=search["val_frame_step"])
    return {"audit": "yolo26_search_data_path_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(), "data_root": str(root),
            "timing_note": "One bounded stat/read pass; affected by host load and OS cache. "
                           "Do not compare across hosts as detector latency.",
            "train": train, "search_val": val}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=candidate_runner.DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, help="challenging export parent")
    parser.add_argument("--sample-images", type=int, default=16)
    parser.add_argument("--output", type=Path, help="new JSON report path; existing files are refused")
    args = parser.parse_args()
    protocol = candidate_runner.load_protocol(args.protocol)
    data_root = args.data_root or candidate_runner.FRED_DIR / protocol["challenging"]["data_root"]
    if args.output and args.output.resolve().is_relative_to(Path(data_root).resolve()):
        parser.error("--output must be outside the audited export")
    try:
        result = audit_search(data_root, protocol, args.sample_images)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    if result["train"]["issue_count"] or result["search_val"]["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
