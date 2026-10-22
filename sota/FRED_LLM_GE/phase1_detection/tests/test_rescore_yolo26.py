import csv
import json
from argparse import Namespace

import pytest
import torch

pytest.importorskip("ultralytics")

from sota.FRED_LLM_GE.phase1_detection import candidate_runner, rescore_yolo26
from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import (
    UltralyticsYolo26P2Adapter,
)


def test_rescore_keeps_historical_report_and_scores_both_heads(tmp_path, monkeypatch):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    adapter = UltralyticsYolo26P2Adapter(protocol["families"]["yolo26_p2"],
                                        candidate_runner.FRED_DIR)
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    checkpoint_dir = tmp_path / "trained_models"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "seed.pt"
    model.save(checkpoint)

    proxy = dict(protocol["proxy"], img_width=96, batch_size=1)
    data = {"images": torch.zeros(1, 3, 64, 96, dtype=torch.uint8),
            "orig_boxes": [torch.tensor([[10.0, 10.0, 20.0, 20.0]])],
            "sample_ids": ["train/0:0"], "orig_size": (96, 64)}
    monkeypatch.setattr(rescore_yolo26, "validation_export", lambda *_: (proxy, data))

    reference_dir = tmp_path / "results"
    reference_dir.mkdir()
    reference_path = reference_dir / "seed_evaluation.json"
    historical = {"mode": "validation", "num_frames": 1,
                  "metadata": candidate_runner.report_metadata(
                      adapter, proxy, {"family": "yolo26_p2"},
                      adapter.variant_path("network", None), checkpoint),
                  "metrics": {"mAP50": 0.0, "mAP50_95": 0.0,
                              "params": adapter.param_count(model)}}
    historical["metadata"]["postprocessing"] = (
        "ultralytics YOLO26 end2end predict imgsz=96 conf=0.05; no NMS")
    reference_path.write_text(json.dumps(historical))
    historical_bytes = reference_path.read_bytes()
    args = Namespace(protocol=candidate_runner.DEFAULT_PROTOCOL, data_root=tmp_path,
                     checkpoint_dir=checkpoint_dir, reference_results_dir=reference_dir,
                     output_dir=tmp_path / "rescore", gene_id=["seed"],
                     heads=["one_to_many", "one_to_one"], confs=[0.05, 0.1],
                     nms_ious=[0.5, 0.7], device="cpu")

    rescore_yolo26.run(args)
    reports = sorted((args.output_dir / "results").glob("*_evaluation.json"))
    assert len(reports) == 6
    assert reference_path.read_bytes() == historical_bytes
    assert {json.loads(path.read_text())["metadata"]["postprocessing"].split("head=")[1]
            .split()[0] for path in reports} == {"one_to_many", "one_to_one"}
    with (args.output_dir / "head_rescore_summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert {row["conf"] for row in rows} == {"0.05", "0.1"}
    assert {row["nms_iou"] for row in rows if row["head"] == "one_to_many"} == {"0.5", "0.7"}
    assert sum(row["head"] == "one_to_one" and row["nms_iou"] == "" for row in rows) == 2
    assert all(row["mAP75"] == "" for row in rows)
    rescore_yolo26.run(args)  # matching reports are safely reused
    assert len(list((args.output_dir / "results").glob("*_evaluation.json"))) == 6

    args.diagnostics = True
    rescore_yolo26.run(args)  # opt in to diagnostics on existing comparison reports
    for path in reports:
        details = json.loads(path.read_text())["diagnostics"]
        assert details["overall"]["false_negatives_iou50"] == 1
        assert details["by_sequence"]["train/0"]["ground_truth"] == 1
    with (args.output_dir / "head_rescore_summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert all(row["mAP75"] != "" and row["mAP_small"] != "" for row in rows)
    assert all(row["false_negatives_iou50"] == "1" for row in rows)
    baseline = next(row for row in rows if row["head"] == "one_to_many"
                    and row["conf"] == "0.05" and row["nms_iou"] == "0.5")
    assert float(baseline["delta_vs_baseline_mAP50_95"]) == 0.0
    assert float(baseline["delta_vs_baseline_mAP75"]) == 0.0
    with (args.output_dir / "head_rescore_sequences.csv").open(newline="") as file:
        sequences = list(csv.DictReader(file))
    assert len(sequences) == 6
    assert all(row["sequence"] == "train/0" and row["ground_truth"] == "1"
               for row in sequences)
    assert reference_path.read_bytes() == historical_bytes
    rescore_yolo26.run(args)  # completed diagnostic reports are reused
    assert len(list((args.output_dir / "results").glob("*_evaluation.json"))) == 6


def test_summary_shows_policy_deltas_and_undefined_small_object_ap(tmp_path):
    args = Namespace(output_dir=tmp_path)
    (tmp_path / "results").mkdir()
    proxy = {"eval_conf": 0.05, "nms_iou": 0.5}
    sweep = [("one_to_many", 0.05, 0.5), ("one_to_one", 0.05, 0.5)]
    records = [("seed", None, None, None,
                {"mAP50": 0.8, "mAP50_95": 0.4})]
    for head, ap50, ap5095, ap75, small in (
            ("one_to_many", 0.8, 0.4, 0.5, None),
            ("one_to_one", 0.82, 0.43, 0.54, None)):
        report = {
            "metrics": {"mAP50": ap50, "mAP50_95": ap5095, "params": 100},
            "diagnostics": {
                "overall": {"mAP75": ap75, "mAP_small": small,
                            "recall_small": None, "false_negatives_iou50": 3,
                            "poorly_localized_iou50_to75": 2},
                "by_sequence": {"train/4": {"frames": 10, "ground_truth": 5,
                                             "false_negatives_iou50": 3,
                                             "poorly_localized_iou50_to75": 2,
                                             "mAP50": ap50, "mAP75": ap75,
                                             "mAP50_95": ap5095}},
            },
        }
        name = rescore_yolo26.score_name("seed", head, 0.05, 0.5)
        (tmp_path / "results" / f"{name}_evaluation.json").write_text(json.dumps(report))

    rescore_yolo26.write_summary(args, records, sweep, proxy)
    with (tmp_path / "head_rescore_summary.csv").open(newline="") as file:
        rows = {row["head"]: row for row in csv.DictReader(file)}
    assert float(rows["one_to_many"]["mAP50_minus_mAP75"]) == pytest.approx(0.3)
    assert float(rows["one_to_one"]["delta_vs_baseline_mAP50_95"]) == pytest.approx(0.03)
    assert float(rows["one_to_one"]["delta_vs_baseline_mAP75"]) == pytest.approx(0.04)
    assert rows["one_to_one"]["mAP_small"] == ""
    assert rows["one_to_one"]["delta_vs_baseline_mAP_small"] == ""
