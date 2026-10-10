"""Resume a scene audit across commits without rewriting its source artifacts.

Only the recorded Git revision may differ. All input hashes and the unchanged
collector's source hash must match. Reused records keep their producing context
and source record hash alongside their current verification context.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import scene_audit
from .provenance import atomic_write_bytes, atomic_write_json, sha256_file


def stage_records(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"missing source scene audit: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate scene audit directories")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scene audit: {output}")
    _, _, official, project, current = scene_audit.audit_inputs()
    paths = sorted(source.glob("sequence_*.json"))
    if not paths:
        raise ValueError("source scene audit contains no records")
    expected = set(official.challenging_train)
    verified = []
    for path in paths:
        record = json.loads(path.read_text())
        sequence = record.get("sequence_id")
        if (sequence not in expected or sequence in official.challenging_test
                or path.name != f"sequence_{sequence}.json"):
            raise ValueError(f"unexpected scene record: {path}")
        saved = record.get("context", {})
        if not re.fullmatch(r"[0-9a-f]{40}", saved.get("code_revision", "")):
            raise ValueError(f"missing producing Git revision: {path}")
        differences = {key for key in set(saved) | set(current)
                       if saved.get(key) != current.get(key)}
        if differences - {"code_revision"}:
            raise ValueError(f"incompatible scene inputs/collector: {path}: {sorted(differences)}")
        split = "train" if sequence in project.train else "validation"
        scene_audit.verify_record(path, sequence, saved, split)
        verified.append((path, record, sha256_file(path)))

    # Validate the entire source before publishing any resumed record.
    output.mkdir(parents=True)
    lineage = []
    for path, record, record_hash in verified:
        for snapshot in record["snapshots"]:
            name = snapshot["thumbnail"]
            destination = output / name
            if destination.exists():
                raise ValueError(f"duplicate scene thumbnail: {name}")
            atomic_write_bytes(destination, (source / name).read_bytes())
            if sha256_file(destination) != snapshot["thumbnail_sha256"]:
                raise ValueError(f"scene thumbnail changed during resume: {name}")
        if sha256_file(path) != record_hash:
            raise ValueError(f"source scene record changed during resume: {path}")
        origin = {"record": str(path), "sha256": record_hash,
                  "context": record["context"]}
        resumed = {**record, "context": current,
                   "producing_context": record.get("producing_context", record["context"]),
                   "reused_from": origin}
        atomic_write_json(output / path.name, resumed)
        scene_audit.verify_record(output / path.name, record["sequence_id"], current,
                                 record["project_split"])
        lineage.append({"sequence_id": record["sequence_id"], **origin})
    reused = {record["sequence_id"] for _, record, _ in verified}
    receipt = {"schema_version": "fred-scene-audit-resume-v1", "source": str(source),
               "context": current, "resume_source_sha256": sha256_file(__file__),
               "reused_sequences": len(reused),
               "missing_sequence_ids": [s for s in official.challenging_train if s not in reused],
               "lineage": lineage, "source_modified": False, "split_changed": False}
    atomic_write_json(output / "resume_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = stage_records(args.source, args.output)
    print(f"Verified/reused {receipt['reused_sequences']} scene records; "
          f"collecting {len(receipt['missing_sequence_ids'])} missing sequences", flush=True)
    scene_audit.collect(args.output, 0, 1, summarize_when_complete=True)
    summary = json.loads((args.output / "summary.json").read_text())
    if not summary["complete"] or summary["missing_sequence_ids"]:
        raise ValueError("resumed scene audit is incomplete")
    print(f"SCENE AUDIT RESUME COMPLETE: {summary['inspected_sequences']}/"
          f"{summary['expected_sequences']}; source preserved; split unchanged", flush=True)


if __name__ == "__main__":
    main()
