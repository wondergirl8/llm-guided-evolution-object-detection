"""Development-only background snapshots and repeated-scene review aids.

Similarity ranks are triage evidence, never automatic scene/session identities,
split assignments, exclusions, or evidence that leakage is absent.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from .annotation_policy import POLICY_VERSION, validate_policy_source
from .annotations import parse_coordinates_file
from .cli import DEFAULT_CONFIG
from .config import load_config
from .development_audit import DEFAULT_INVENTORY, DEFAULT_OFFICIAL, DEFAULT_PROJECT, load_inputs
from .fred_api import HFFredSource
from .provenance import atomic_write_bytes, atomic_write_json, sha256_file
from .synchronization import pair_sequence_frames


SCHEMA = "fred-development-scene-snapshots-v1"


def snapshot_indexes(frame_count: int) -> tuple[int, ...]:
    if frame_count < 1:
        raise ValueError("scene audit requires at least one paired frame")
    return tuple(dict.fromkeys(((frame_count-1)*numerator//denominator
                               for numerator, denominator in ((1, 4), (1, 2), (3, 4), (19, 20)))))


def difference_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8))
    values = [small.getpixel((col, row)) for row in range(8) for col in range(9)]
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (values[row*9+col] > values[row*9+col+1])
    return f"{bits:016x}"


def cross_split_candidates(records: list[dict]) -> list[dict]:
    """Two closest training backgrounds per validation sequence, for human review."""
    train = [r for r in records if r["project_split"] == "train"]
    validation = [r for r in records if r["project_split"] == "validation"]
    matches = []
    for val in validation:
        ranked = []
        for candidate in train:
            comparisons = [(int(a["difference_hash"], 16) ^ int(b["difference_hash"], 16)).bit_count()
                           for a in val["snapshots"] for b in candidate["snapshots"]
                           if a["grayscale_stddev"] >= 5 and b["grayscale_stddev"] >= 5]
            if comparisons:
                ranked.append((min(comparisons), candidate["sequence_id"]))
        for distance, sequence in sorted(ranked, key=lambda x: (x[0], int(x[1])))[:2]:
            matches.append({"train_sequence_id": sequence, "validation_sequence_id": val["sequence_id"],
                            "minimum_dhash_distance": distance, "decision": "manual_review_required"})
    return sorted(matches, key=lambda r: (r["minimum_dhash_distance"], int(r["validation_sequence_id"])))


def audit_inputs(config_path=DEFAULT_CONFIG, inventory_path=DEFAULT_INVENTORY,
                 official_path=DEFAULT_OFFICIAL, project_path=DEFAULT_PROJECT):
    config = load_config(config_path, workspace_root=Path.cwd())
    validate_policy_source(config, POLICY_VERSION)
    inventory, official, project = load_inputs(inventory_path, official_path, project_path)
    if (inventory.dataset_id != config.source.dataset_id
            or inventory.dataset_revision != config.source.revision
            or official.repository_revision != config.fred_repository.revision):
        raise ValueError("scene-audit source revisions differ from the approved pinned inputs")
    if (official.train_source_sha256 != config.fred_repository.challenging_train_sha256
            or official.test_source_sha256 != config.fred_repository.challenging_test_sha256):
        raise ValueError("scene-audit split source hashes differ from the pinned configuration")
    context = {"dataset_revision": inventory.dataset_revision,
               "repository_revision": official.repository_revision,
               "config_sha256": sha256_file(config_path),
               "inventory_sha256": sha256_file(inventory_path),
               "official_sha256": sha256_file(official_path), "project_sha256": sha256_file(project_path),
               "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "scene_audit_source_sha256": sha256_file(__file__),
               "sampling_policy": "paired_frames_quartiles_and_95_percent_v1"}
    return config, inventory, official, project, context


def verify_record(path: Path, sequence: str, context: dict, split: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("schema_version") != SCHEMA or record.get("context") != context
            or record.get("sequence_id") != sequence or record.get("project_split") != split
            or record.get("status") != "complete"):
        raise ValueError(f"stale or incomplete scene record: {path}")
    if not record.get("snapshots") or len(record["snapshots"]) > 4:
        raise ValueError(f"invalid snapshot coverage: {path}")
    if tuple(s["frame_index"] for s in record["snapshots"]) != snapshot_indexes(record["paired_frame_count"]):
        raise ValueError(f"scene record used a different sampling policy: {path}")
    for snapshot in record["snapshots"]:
        filename = snapshot["thumbnail"]
        if Path(filename).name != filename or sha256_file(path.parent / filename) != snapshot["thumbnail_sha256"]:
            raise ValueError(f"missing or modified scene thumbnail: {path}")
    return record


def collect(output: Path, shard: int, shards: int, summarize_when_complete: bool = False) -> None:
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid scene-audit shard")
    config, inventory, official, project, context = audit_inputs()
    source = HFFredSource(config)
    output.mkdir(parents=True, exist_ok=True)
    for sequence in official.challenging_train[shard::shards]:
        if sequence in official.challenging_test:
            raise ValueError("scene audit must not open a held-out test sequence")
        split = "train" if sequence in project.train else "validation"
        record_path = output / f"sequence_{sequence}.json"
        if record_path.exists():
            verify_record(record_path, sequence, context, split)
            print(f"Reused verified scene record {sequence}", flush=True)
            continue
        inventory_record = inventory.by_sequence()[sequence]
        source.prepare_sequence(sequence, inventory_record)
        with source.activate_prepared_window(((sequence, inventory_record),)):
            with source.open_prepared_sequence(sequence, inventory_record) as prepared:
                pairing = pair_sequence_frames(prepared.sequence_root, sequence_id=sequence,
                                               timestamp_policy=config.timestamp)
                if not pairing.pairs or any(f.severity == "error" for f in pairing.findings):
                    raise ValueError(f"sequence {sequence}: blocking pairing findings")
                # Class counts describe source annotations, including the two
                # recorded unpaired records. They do not define recording groups.
                coordinates = prepared.sequence_root / "coordinates.txt"
                parsed = parse_coordinates_file(coordinates)
                snapshots = []
                for index in snapshot_indexes(len(pairing.pairs)):
                    pair = pairing.pairs[index]
                    image_path = prepared.sequence_root / pair.rgb_relative_path
                    filename = f"{sequence}_{pair.frame_index:08d}.jpg"
                    with Image.open(image_path) as original:
                        image = original.convert("RGB")
                        gray = image.convert("L")
                        thumbnail = image.copy()
                        thumbnail.thumbnail((480, 270))
                        buffer = io.BytesIO()
                        thumbnail.save(buffer, format="JPEG", quality=85)
                        atomic_write_bytes(output / filename, buffer.getvalue())
                        snapshots.append({"frame_index": pair.frame_index, "timestamp": pair.timestamp,
                                          "rgb_archive_member": pair.rgb_relative_path, "image_size": image.size,
                                          "source_rgb_sha256": sha256_file(image_path),
                                          "thumbnail": filename, "thumbnail_sha256": sha256_file(output / filename),
                                          "difference_hash": difference_hash(image),
                                          "grayscale_stddev": ImageStat.Stat(gray).stddev[0]})
                record = {"schema_version": SCHEMA, "status": "complete", "context": context,
                          "sequence_id": sequence, "project_split": split,
                          "paired_frame_count": len(pairing.pairs),
                          "nominal_duration_seconds": len(pairing.pairs)*float(config.timestamp.frame_period_seconds),
                          "source_annotation_count": len(parsed.annotations),
                          "source_drone_class_counts": dict(sorted(Counter(a.original_class for a in parsed.annotations).items())),
                          "coordinates_sha256": sha256_file(coordinates), "snapshots": snapshots,
                          "recording_session_id": None, "condition_metadata": "not_verified"}
                atomic_write_json(record_path, record)
                print(f"Scene snapshots complete: {sequence} ({split}); frames={len(pairing.pairs)}", flush=True)
    if summarize_when_complete:
        if all((output / f"sequence_{s}.json").is_file() for s in official.challenging_train):
            summarize(output)
        else:
            print("Shard complete; the final successful shard will publish the scene summary.", flush=True)


def summarize(output: Path) -> dict:
    _, _, official, project, context = audit_inputs()
    records, missing = [], []
    for sequence in official.challenging_train:
        path = output / f"sequence_{sequence}.json"
        if not path.exists():
            missing.append(sequence)
            continue
        split = "train" if sequence in project.train else "validation"
        records.append(verify_record(path, sequence, context, split))
    pages = []
    if not missing:
        for start in range(0, len(records), 12):
            batch = records[start:start+12]
            canvas = Image.new("RGB", (1280, len(batch)*200), "#202020")
            draw = ImageDraw.Draw(canvas)
            for row, record in enumerate(batch):
                draw.text((6, row*200+4), f"Sequence {record['sequence_id']} | {record['project_split']}", fill="white")
                for col, snapshot in enumerate(record["snapshots"]):
                    with Image.open(output / snapshot["thumbnail"]) as image:
                        canvas.paste(image.resize((320, 180)), (col*320, row*200+20))
            filename = f"overview_{start//12+1:02d}.jpg"
            buffer = io.BytesIO(); canvas.save(buffer, format="JPEG", quality=85)
            atomic_write_bytes(output / filename, buffer.getvalue())
            pages.append(filename)
    result = {"schema_version": "fred-development-scene-summary-v1", "context": context,
              "complete": not missing, "expected_sequences": len(official.challenging_train),
              "inspected_sequences": len(records), "missing_sequence_ids": missing,
              "overview_pages": pages, "cross_split_candidates": cross_split_candidates(records),
              "records": records, "phase0_frozen": False, "split_changed": False,
              "known_cross_split_scene_observation": {"train": "225", "validation": "230",
                                                       "evidence": "DG-P0-05-review-6088625.md"},
              "limitations": ["Four frames per sequence may miss a moving camera or scene change.",
                              "Similarity ranks can miss repeated scenes and generate false matches.",
                              "Recording-session IDs and condition metadata remain unverified.",
                              "Human scene grouping and DG-P0-02 decision are required before freezing."]}
    atomic_write_json(output / "summary.json", result)
    print(f"Scene audit: {len(records)}/{len(official.challenging_train)}; complete={not missing}")
    print(f"Summary: {output / 'summary.json'}; overview pages: {len(pages)}; no split change or fitness")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--shard", type=int, required=True)
    collector.add_argument("--shards", type=int, default=8)
    collector.add_argument("--summarize-when-complete", action="store_true")
    commands.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.output, args.shard, args.shards, args.summarize_when_complete)
    elif not summarize(args.output)["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
