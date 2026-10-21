"""Re-score a saved D-FINE checkpoint twice; optionally record error diagnostics."""

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

try:
    from .candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, load_protocol,
                                   predict_frames, runtime_environment, sha256_file)
    from .adapters.dfine import DFINEAdapter
    from .fitness import evaluate_frames
    from .proxy_data import load_exported_split
except ImportError:
    from candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, load_protocol,
                                  predict_frames, runtime_environment, sha256_file)
    from adapters.dfine import DFINEAdapter
    from fitness import evaluate_frames
    from proxy_data import load_exported_split


def validation_export(protocol, metadata):
    """Load the original validation sequence order from its local export."""
    val_ids = metadata["split_identity"].partition(";val=")[2].split(",")
    expected = {str(sequence) for sequence in protocol["challenging"]["search"]["val_sequences"]}
    if not val_ids or len(val_ids) != len(expected) or set(val_ids) != expected:
        raise ValueError("reference validation sequences differ from the protocol")
    challenging = protocol["challenging"]
    val_root = (FRED_DIR / challenging["data_root"] /
                f"search_val_step{challenging['search']['val_frame_step']}")
    sequences = []
    for sequence in val_ids:
        done = val_root / "done" / f"{sequence}.json"
        if not done.is_file():
            raise FileNotFoundError(f"{done} missing: copy the original search-validation export")
        sequences.append((sequence, json.loads(done.read_text())["hf_folder"]))
    return load_exported_split(val_root, sequences)


def render_examples(frames, val_data, diagnostics, output_dir, max_examples=3):
    """Draw recorded miss/localization cases on the original Event PNGs."""
    paths = getattr(val_data["images"], "paths", None)
    if paths is None or len(paths) != len(frames):
        raise ValueError("example overlays require the exported Event PNGs")
    by_sample = {frame["sample_id"]: (frame, path)
                 for frame, path in zip(frames, paths)}
    selected = []
    for category in ("false_negatives", "poorly_localized"):
        cases = diagnostics["examples"][category]
        chosen, seen_frames, seen_sequences = [], set(), set()
        for example in cases:
            sample_id = example["sample_id"]
            sequence = sample_id.rsplit(":", 1)[0]
            if sequence not in seen_sequences:
                chosen.append(example)
                seen_frames.add(sample_id)
                seen_sequences.add(sequence)
            if len(chosen) == max_examples:
                break
        for example in cases:
            if len(chosen) == max_examples:
                break
            if example["sample_id"] not in seen_frames:
                chosen.append(example)
                seen_frames.add(example["sample_id"])
        selected.extend((category, example) for example in chosen)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for category, example in selected:
        frame, path = by_sample[example["sample_id"]]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (frame["width"], frame["height"]):
            raise ValueError(f"Event frame size differs from boxes: {path}")
        draw = ImageDraw.Draw(image)
        for box, score in zip(frame["prediction"]["boxes"], frame["prediction"]["scores"]):
            draw.rectangle(box, outline="#ff9900", width=2)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="#ff9900")
        for box in frame["target"]["boxes"]:
            draw.rectangle(box, outline="#00ff00", width=3)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", example["sample_id"])
        name = f"{category}_{safe_id}.png"
        image.save(output_dir / name)
        manifest.append({"category": category, "sample_id": example["sample_id"],
                         "file": name, "target_boxes": len(frame["target"]["boxes"]),
                         "prediction_boxes": len(frame["prediction"]["boxes"])})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_checkpoint(checkpoint, reference_report, model_name, variant_dir, protocol_path):
    """Verify saved weights, source, and validation export against a run report."""
    checkpoint = Path(checkpoint)
    reference = json.loads(Path(reference_report).read_text())
    protocol = load_protocol(protocol_path)
    settings = protocol["families"]["dfine"]
    adapter = DFINEAdapter(settings, FRED_DIR)
    source = adapter.variant_path(model_name, variant_dir)
    metadata = reference["metadata"]
    if (reference["mode"] != "validation" or metadata.get("project_split") != "validation"
            or metadata.get("official_split") != "challenging_train"):
        raise ValueError("D-FINE diagnostics require a challenging_train validation report")
    if (sha256_file(checkpoint) != metadata["checkpoint_identity"] or
            sha256_file(source) != metadata["model_identity"] or
            adapter.postprocessing(protocol["proxy"]) != metadata["postprocessing"]):
        raise ValueError("checkpoint, source, or postprocessing differs from the reference report")
    proxy = protocol["proxy"]
    val_data = validation_export(protocol, metadata)
    sample_hash = hashlib.sha256(json.dumps(val_data["sample_ids"]).encode()).hexdigest()
    if sample_hash != metadata["validation_sample_ids_sha256"]:
        raise ValueError("validation frames differ from the reference report")
    return checkpoint, reference, adapter, source, proxy, val_data, sample_hash


def rescore(checkpoint, reference_report, model_name="network", variant_dir=None,
            protocol_path=DEFAULT_PROTOCOL, device="cpu", diagnostics=False,
            examples_dir=None, max_examples=3):
    if examples_dir is not None and not diagnostics:
        raise ValueError("example overlays require diagnostics")
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    checkpoint, reference, adapter, source, proxy, val_data, sample_hash = prepare_checkpoint(
        checkpoint, reference_report, model_name, variant_dir, protocol_path)
    metadata = reference["metadata"]

    passes = []
    details = None
    first_frames = None
    param_count = None
    for index in range(2):
        model = adapter.build(source, num_classes=1, device=device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True),
                              strict=True)
        frames = predict_frames(adapter, model, val_data, proxy, device)
        if index == 0:
            first_frames = frames
        param_count = adapter.param_count(model)
        evaluated = evaluate_frames(frames, params=param_count,
                                    expected_sample_ids=val_data["sample_ids"],
                                    diagnostics=diagnostics and index == 0)
        if diagnostics and index == 0:
            result, details = evaluated
        else:
            result = evaluated
        passes.append({
            "mAP50": result.map50,
            "mAP50_95": result.map50_95,
            "predictions_sha256": hashlib.sha256(
                json.dumps([(frame["sample_id"], frame["prediction"]) for frame in frames],
                           sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
    output = {
        "schema_version": "fred-dfine-rescore-v1",
        "source_report_schema_version": reference["schema_version"],
        "gene_id": reference["gene_id"],
        "mode": reference["mode"],
        "metadata": metadata,
        "metrics": {"mAP50": passes[0]["mAP50"], "mAP50_95": passes[0]["mAP50_95"],
                    "params": param_count},
        "metric_units": reference["metric_units"],
        "num_frames": len(val_data["sample_ids"]),
        "frames_sha256": hashlib.sha256(json.dumps(
            first_frames, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "evaluator": dict(reference["evaluator"],
                          version=importlib.metadata.version("pycocotools")),
        "checkpoint_identity": metadata["checkpoint_identity"],
        "model_identity": metadata["model_identity"],
        "validation_sample_ids_sha256": sample_hash,
        "runtime_environment": runtime_environment(device),
        "reference_metrics": {key: reference["metrics"][key]
                              for key in ("mAP50", "mAP50_95")},
        "passes": passes,
        "repeated_scoring_matches": passes[0] == passes[1],
        "reference_scoring_matches": all(
            passes[0][key] == reference["metrics"][key] for key in ("mAP50", "mAP50_95")),
    }
    if diagnostics:
        output["diagnostics"] = details
        if examples_dir is not None:
            output["example_overlays"] = render_examples(
                first_frames, val_data, details, examples_dir, max_examples)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics", action="store_true",
                        help="record AP75, small-object, miss, localization, and sequence results")
    parser.add_argument("--examples-dir", type=Path,
                        help="draw representative misses and localization cases on Event PNGs")
    parser.add_argument("--max-examples", type=int, default=3,
                        help="maximum frames per error category (default: 3)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.examples_dir is not None and args.examples_dir.exists():
        parser.error(f"refusing to overwrite {args.examples_dir}")
    result = rescore(args.checkpoint, args.reference_report, args.model,
                     args.variant_dir, args.protocol, args.device,
                     diagnostics=args.diagnostics, examples_dir=args.examples_dir,
                     max_examples=args.max_examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["repeated_scoring_matches"]:
        raise SystemExit("Repeated scoring of one checkpoint disagreed")


if __name__ == "__main__":
    main()
