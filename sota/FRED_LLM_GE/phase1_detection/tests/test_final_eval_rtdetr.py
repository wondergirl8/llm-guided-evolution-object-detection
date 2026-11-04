import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection import final_eval


_SMALL_MODEL = {
    "depth": 18, "num_queries": 16, "num_denoising": 8, "hidden_dim": 32,
    "dim_feedforward": 64, "decoder_dim": 32, "num_decoder_layers": 2,
}


def test_rtdetr_final_eval_trains_and_reports_test_split(tmp_path, monkeypatch):
    protocol = final_eval.load_protocol(final_eval.DEFAULT_PROTOCOL)
    protocol["families"]["rtdetr"].update(model=dict(_SMALL_MODEL), workers=0, amp=False)
    protocol["challenging"]["data_root"] = str(tmp_path)
    protocol["challenging"]["final"].update(epochs=1, batch_size=2, img_width=96)
    monkeypatch.setattr(final_eval, "load_protocol", lambda path: protocol)

    # A two-frame "final export" in the export_split.py layout.
    train_root = tmp_path / "final_train_step5"
    (train_root / "images").mkdir(parents=True)
    (train_root / "done").mkdir()
    (train_root / "data.yaml").write_text("path: unused\n")
    records = []
    for index in range(2):
        name = f"7_{index}"
        Image.fromarray(np.full((64, 96, 3), 60 + 40 * index, dtype=np.uint8)).save(
            train_root / "images" / f"{name}.png")
        records.append({"name": name, "frame_index": index, "boxes": [[10, 12, 22, 20]]})
    (train_root / "done" / "7.json").write_text(json.dumps({"size": [96, 64], "records": records}))
    (tmp_path / "splits").mkdir()
    for split in ("challenging_train", "challenging_test"):
        (tmp_path / "splits" / f"{split}_split.txt").write_text("7\n")
    monkeypatch.setattr(final_eval, "challenging_split",
                        lambda root: ([("7", "train")], [("9", "test")]))
    monkeypatch.setattr(final_eval, "dataset_parts",
                        lambda challenging, root, name: [(train_root, [("7", "train")], 5)])

    def frames(folder, sequence, report, step, downloads, events=None):
        for index in range(3):
            yield Image.fromarray(np.full((64, 96, 3), 90, dtype=np.uint8)), index / 30
    monkeypatch.setattr(final_eval, "sequence_frames", frames)
    monkeypatch.setattr(final_eval, "load_annotations",
                        lambda folder, sequence, width, height: {0: [[10.0, 12.0, 22.0, 20.0]]})
    reported = {}

    def report(payload, sota_root, gene_id, mode):
        reported.update(payload=payload, gene_id=gene_id, mode=mode)
        return SimpleNamespace(map50=0.1, map50_95=0.05, params=payload["params"]), tmp_path / "r.json"
    monkeypatch.setattr(final_eval.map_reporter, "report", report)
    monkeypatch.setattr(sys, "argv", ["final_eval.py", "--family", "rtdetr",
                                      "--name", "rtdetr_unit", "--device", "cpu"])
    final_eval.main()

    assert (tmp_path / "final_runs" / "rtdetr_unit" / "model.pt").is_file()
    payload = reported["payload"]
    assert reported["mode"] == "held-out" and reported["gene_id"] == "rtdetr_unit"
    assert len(payload["frames"]) == 3 == len(payload["expected_sample_ids"])
    assert all(0 <= box[0] < box[2] <= 96 for frame in payload["frames"]
               for box in frame["prediction"]["boxes"])
    assert payload["metadata"]["official_split"] == "challenging_test"
    assert isinstance(torch.load(tmp_path / "final_runs" / "rtdetr_unit" / "model.pt"), dict)
