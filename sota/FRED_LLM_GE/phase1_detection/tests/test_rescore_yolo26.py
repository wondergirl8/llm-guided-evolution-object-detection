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
    rescore_yolo26.run(args)  # matching reports are safely reused
    assert len(list((args.output_dir / "results").glob("*_evaluation.json"))) == 6

    args.diagnostics = True
    rescore_yolo26.run(args)  # opt in to diagnostics on existing comparison reports
    for path in reports:
        details = json.loads(path.read_text())["diagnostics"]
        assert details["overall"]["false_negatives_iou50"] == 1
        assert details["by_sequence"]["train/0"]["ground_truth"] == 1
    assert reference_path.read_bytes() == historical_bytes
    rescore_yolo26.run(args)  # completed diagnostic reports are reused
    assert len(list((args.output_dir / "results").glob("*_evaluation.json"))) == 6
