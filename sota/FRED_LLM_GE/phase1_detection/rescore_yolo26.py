"""Re-score saved YOLO26-P2 checkpoints on the unchanged search-validation export.

This only runs inference. It writes versioned comparison reports to a separate
output root and never replaces the historical candidate fitness/results files.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from ultralytics import YOLO

try:
    from . import candidate_runner, map_reporter
    from .adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
    from .export_split import dataset_parts
    from .proxy_data import load_exported_split
    from .result_contract import validate_gene_id
except ImportError:
    import candidate_runner
    import map_reporter
    from adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
    from export_split import dataset_parts
    from proxy_data import load_exported_split
    from result_contract import validate_gene_id


def validation_export(protocol, data_root):
    """Use the candidate runner's search split and original frame loader."""
    search = protocol["challenging"]["search"]
    val_root = data_root / f"search_val_step{search['val_frame_step']}"
    if not val_root.is_dir():
        raise FileNotFoundError(
            f"{val_root} missing; restore the original search-validation export")
    (_, train, train_step), (val_root, val, val_step) = dataset_parts(
        protocol["challenging"], data_root, "search")
    data = load_exported_split(val_root, val)
    proxy = dict(protocol["proxy"], train_sequences=[s for s, _ in train],
                 val_sequences=[s for s, _ in val], frame_step=train_step,
                 val_frame_step=val_step, epochs=search["epochs"])
    return proxy, data


def reference_metadata(gene_id, checkpoint, reference_dir, adapter, proxy, frame_count):
    """Refuse comparisons when the checkpoint, architecture or split differs."""
    report_path = reference_dir / f"{gene_id}_evaluation.json"
    report = json.loads(report_path.read_text())
    if report.get("mode") != "validation" or report.get("num_frames") != frame_count:
        raise ValueError(f"{report_path}: not the matching search-validation report")
    variant = (adapter.variant_path("network", None) if gene_id == "seed" else
               checkpoint.parent.parent / "models" / f"network_{gene_id}.py")
    actual = candidate_runner.report_metadata(
        adapter, proxy, {"family": "yolo26_p2", "proxy": proxy}, variant, checkpoint)
    original = report["metadata"]
    for key in ("dataset_revision", "manifest_identity", "split_identity",
                "annotation_policy", "model_identity", "checkpoint_identity",
                "modality", "benchmark_split", "project_split", "official_split"):
        if original.get(key) != actual[key]:
            raise ValueError(f"{report_path}: {key} differs from the current export/checkpoint")
    return original, candidate_runner.sha256_file(report_path), report["metrics"]


def score_name(gene_id, head, conf, nms_iou):
    suffix = f"{head}__conf{conf:g}".replace(".", "p")
    if head == "one_to_many":
        suffix += f"__iou{nms_iou:g}".replace(".", "p")
    return f"{gene_id}__{suffix}"


def scoring_settings(args, proxy):
    return [(head, conf, iou)
            for head in args.heads
            for conf in args.confs
            for iou in (args.nms_ious if head == "one_to_many" else (proxy["nms_iou"],))]


def write_summary(args, records, sweep):
    """Gather every completed setting into one reviewable comparison table."""
    rows = []
    for gene_id, _, _, _, historical_metrics in records:
        for head, conf, iou in sweep:
            rescore_id = score_name(gene_id, head, conf, iou)
            report = json.loads((args.output_dir / "results" /
                                 f"{rescore_id}_evaluation.json").read_text())
            metrics = report["metrics"]
            rows.append({
                "source_gene_id": gene_id, "head": head, "conf": conf,
                "nms_iou": iou if head == "one_to_many" else "",
                "mAP50": metrics["mAP50"], "mAP50_95": metrics["mAP50_95"],
                "params": metrics["params"],
                "historical_mAP50": historical_metrics["mAP50"],
                "historical_mAP50_95": historical_metrics["mAP50_95"],
                "delta_mAP50_95": metrics["mAP50_95"] - historical_metrics["mAP50_95"],
            })
    output = args.output_dir / "head_rescore_summary.csv"
    temp = None
    try:
        with NamedTemporaryFile(mode="w", newline="", dir=args.output_dir,
                                prefix=".head_rescore_summary.", delete=False) as handle:
            temp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f"summary: {output}", flush=True)


def run(args):
    protocol = candidate_runner.load_protocol(args.protocol)
    proxy, data = validation_export(protocol, args.data_root)
    sweep = scoring_settings(args, proxy)
    sample_ids_hash = hashlib.sha256(json.dumps(data["sample_ids"]).encode()).hexdigest()
    source_dir = Path(candidate_runner.__file__).parent
    source_hash = hashlib.sha256(b"".join(
        (source_dir / name).read_bytes() for name in (
            "rescore_yolo26.py", "candidate_runner.py", "fitness.py", "map_reporter.py",
            "adapters/ultralytics_yolo26_p2.py", "adapters/ultralytics_yolo11.py",
            "adapters/yolo11.py"))).hexdigest()
    protocol_hash = candidate_runner.sha256_file(args.protocol)
    reference_dir = args.reference_results_dir or args.checkpoint_dir.parent / "results"
    if args.output_dir.resolve() == reference_dir.parent.resolve():
        raise ValueError("output_dir must be separate from the historical family results root")
    genes = args.gene_id or [path.stem for path in sorted(args.checkpoint_dir.glob("*.pt"))]
    if not genes:
        raise FileNotFoundError(f"no checkpoints found in {args.checkpoint_dir}")
    settings = protocol["families"]["yolo26_p2"]
    adapter = UltralyticsYolo26P2Adapter(settings, candidate_runner.FRED_DIR)
    records = []
    for gene_id in genes:
        validate_gene_id(gene_id)
        checkpoint = args.checkpoint_dir / f"{gene_id}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        original, report_hash, historical_metrics = reference_metadata(
            gene_id, checkpoint, reference_dir, adapter, proxy, len(data["sample_ids"]))
        records.append((gene_id, checkpoint, original, report_hash, historical_metrics))

    for gene_id, checkpoint, original, report_hash, historical_metrics in records:
        model = YOLO(str(checkpoint), task="detect")
        params = adapter.param_count(model)
        if params != historical_metrics["params"]:
            raise ValueError(f"{gene_id}: checkpoint parameter count differs from historical report")
        for head, conf, iou in sweep:
            adapter.inference_head = head
            eval_proxy = dict(proxy, eval_conf=conf, nms_iou=iou)
            rescore_id = score_name(gene_id, head, conf, iou)
            output = args.output_dir / "results" / f"{rescore_id}_evaluation.json"
            csv = args.output_dir / "results" / f"{rescore_id}_results.csv"
            postprocessing = adapter.postprocessing(eval_proxy)
            if output.is_file() and csv.is_file():
                previous_report = json.loads(output.read_text())
                previous = previous_report["metadata"]
                if (previous.get("checkpoint_identity") != original["checkpoint_identity"]
                        or previous.get("source_report_identity") != report_hash
                        or previous.get("postprocessing") != postprocessing
                        or previous.get("validation_sample_ids_sha256") != sample_ids_hash
                        or previous.get("rescore_source_identity") != source_hash
                        or previous.get("rescore_protocol_identity") != protocol_hash):
                    raise ValueError(f"existing rescore conflicts with this run: {output}")
                if not getattr(args, "diagnostics", False) or "diagnostics" in previous_report:
                    print(f"already scored: {rescore_id}", flush=True)
                    continue
            frames = candidate_runner.predict_frames(adapter, model, data, eval_proxy,
                                                     args.device)
            metadata = dict(
                original, code_revision=candidate_runner.code_revision(),
                postprocessing=postprocessing,
                source_gene_id=gene_id, source_report_identity=report_hash,
                historical_postprocessing_claim=original["postprocessing"],
                historical_metrics=historical_metrics,
                validation_sample_ids_sha256=sample_ids_hash,
                rescore_source_identity=source_hash,
                rescore_protocol_identity=protocol_hash,
                ultralytics_version=importlib.metadata.version("ultralytics"),
                rescore_version="yolo26-head-sweep-v1")
            result, _ = map_reporter.report(
                {"metadata": metadata, "frames": frames, "params": params,
                 "expected_sample_ids": data["sample_ids"]},
                sota_root=args.output_dir, gene_id=rescore_id, mode="validation",
                diagnostics=getattr(args, "diagnostics", False))
            print(f"{rescore_id}: mAP50={result.map50:.6f} "
                  f"mAP50:95={result.map50_95:.6f}", flush=True)
            if (head == "one_to_many" and conf == proxy["eval_conf"]
                    and iou == proxy["nms_iou"]):
                delta = result.map50_95 - historical_metrics["mAP50_95"]
                print(f"  historical NMS setting: mAP50:95 delta={delta:+.6f}",
                      flush=True)
                if abs(delta) > 0.005:
                    print("  WARNING: historical score differs by >0.005; "
                          "check the restored validation export and runtime", flush=True)
    write_summary(args, records, sweep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=candidate_runner.DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="challenging export root; defaults to protocol challenging.data_root")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference-results-dir", type=Path,
                        help="historical evaluation JSON directory; defaults to checkpoint sibling results/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gene-id", action="append", help="checkpoint stem; omit to score all checkpoints")
    parser.add_argument("--heads", nargs="+", choices=("one_to_many", "one_to_one"),
                        default=["one_to_many", "one_to_one"])
    parser.add_argument("--confs", nargs="+", type=float, default=[0.001, 0.05])
    parser.add_argument("--nms-ious", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostics", action="store_true",
                        help="include AP75, small-object, sequence and missed-box diagnostics")
    args = parser.parse_args()
    for conf in args.confs:
        if not math.isfinite(conf) or not 0 <= conf < 1:
            parser.error("each confidence threshold must be finite and in [0, 1)")
    for iou in args.nms_ious:
        if not math.isfinite(iou) or not 0 < iou < 1:
            parser.error("each NMS IoU must be finite and in (0, 1)")
    if args.data_root is None:
        args.data_root = candidate_runner.FRED_DIR / candidate_runner.load_protocol(
            args.protocol)["challenging"]["data_root"]
    try:
        run(args)
    except FileNotFoundError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
