from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection import candidate_runner
from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import (
    DFINEAdapter, _nms, _targets_from_pixels,
)
from sota.FRED_LLM_GE.phase1_detection.proxy_data import LazyImages


def _adapter():
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    settings = dict(protocol["families"]["dfine"])
    settings["model"] = {
        "width": 16, "d_model": 32, "num_queries": 32,
        "num_layers": 2, "reg_max": 8,
    }
    return DFINEAdapter(settings, candidate_runner.FRED_DIR)


def _data():
    images = torch.randint(0, 256, (3, 3, 64, 96), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(1))
    boxes = [torch.tensor([[10.0, 12.0, 22.0, 20.0]]),
             torch.empty(0, 4), torch.tensor([[30.0, 20.0, 36.0, 26.0]])]
    return {
        "images": images, "boxes": boxes,
        "orig_boxes": [box * 2 for box in boxes],
        "sample_ids": [f"train/0:{i}" for i in range(len(images))],
        "orig_size": (192, 128),
    }


def test_dfine_target_conversion_and_rejection():
    target = _targets_from_pixels(torch.tensor([[10.0, 12.0, 22.0, 20.0]]), 96, 64, "cpu")
    assert torch.allclose(target["boxes"], torch.tensor([[1 / 6, 1 / 4, 1 / 8, 1 / 8]]))
    assert target["labels"].tolist() == [0]
    assert _targets_from_pixels(torch.empty(0, 4), 96, 64, "cpu")["boxes"].shape == (0, 4)
    with pytest.raises(ValueError, match="out-of-bounds"):
        _targets_from_pixels(torch.tensor([[-1.0, 0.0, 20.0, 10.0]]), 96, 64, "cpu")


def test_dfine_candidate_protects_preamble(tmp_path):
    adapter = _adapter()
    seed = adapter.variant_path("network", None)
    source = Path(seed).read_text()
    assert source.count("# --OPTION--") == 3
    mutable = tmp_path / "network_mutable.py"
    mutable.write_text(source.replace("Three-scale CNN Event backbone",
                                      "Three-scale CNN Event backbone variant", 1))
    adapter.build(mutable, num_classes=1, device="cpu")

    changed = tmp_path / "network_changed.py"
    changed.write_text(source.replace("Compact D-FINE-inspired detector",
                                      "Changed D-FINE-inspired detector", 1))
    with pytest.raises(ValueError, match="protected preamble"):
        adapter.build(changed, num_classes=1, device="cpu")


def test_dfine_adapter_trains_and_reports_through_runner():
    assert candidate_runner.ADAPTERS["dfine"] is DFINEAdapter
    adapter = _adapter()
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    data = _data()
    proxy = {"epochs": 1, "batch_size": 2, "seed": 0,
             "eval_conf": 0.05, "nms_iou": 0.5}
    before = model.backbone.stem[0][0].weight.detach().clone()
    adapter.train(model, data, proxy, "cpu")
    assert not torch.equal(before, model.backbone.stem[0][0].weight)

    frames = candidate_runner.predict_frames(adapter, model, data, proxy, "cpu")
    assert [frame["sample_id"] for frame in frames] == data["sample_ids"]
    for frame in frames:
        prediction = frame["prediction"]
        assert len(prediction["boxes"]) == len(prediction["scores"]) == len(prediction["labels"])
        assert all(label == 0 for label in prediction["labels"])
        assert all(0 <= score <= 1 for score in prediction["scores"])
        assert all(0 <= x1 < x2 <= 192 and 0 <= y1 < y2 <= 128
                   for x1, y1, x2, y2 in prediction["boxes"])


def test_dfine_nms_uses_protocol_iou():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                          [1.0, 1.0, 11.0, 11.0],
                          [20.0, 20.0, 30.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert _nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_dfine_trains_from_exported_lazy_images(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((64, 96, 3), 40 + index * 20, dtype=np.uint8)).save(path)
        paths.append(path)
    images = LazyImages(paths, (96, 64))
    assert images[torch.tensor([1, 0])][:, 0, 0, 0].tolist() == [60, 40]
    data = {"images": images, "boxes": [torch.empty(0, 4),
                                        torch.tensor([[10.0, 12.0, 20.0, 22.0]])],
            "orig_boxes": [torch.empty(0, 4), torch.tensor([[10.0, 12.0, 20.0, 22.0]])],
            "sample_ids": ["train/0:0", "train/0:1"], "orig_size": (96, 64)}
    adapter = _adapter()
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    proxy = {"epochs": 1, "batch_size": 2, "seed": 0,
             "eval_conf": 0.05, "nms_iou": 0.5}
    adapter.train(model, data, proxy, "cpu")
    assert len(candidate_runner.predict_frames(adapter, model, data, proxy, "cpu")) == 2


def test_dfine_challenging_route_loads_export(monkeypatch, tmp_path):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["data_root"] = str(tmp_path)
    train_root, val_root = tmp_path / "search_train", tmp_path / "search_val"
    train_root.mkdir()
    (train_root / "data.yaml").write_text("path: ignored\n")
    monkeypatch.setattr(candidate_runner, "dataset_parts", lambda config, root, name: [
        (train_root, [("5", "train")], 50), (val_root, [("4", "train")], 10)])
    calls = []

    def load(root, sequences):
        calls.append((root, sequences))
        return {"images": "loaded export"}

    monkeypatch.setattr(candidate_runner, "load_exported_split", load)
    proxy, training, load_val = candidate_runner.challenging_search(protocol, "dfine")
    assert training == {"images": "loaded export"}
    assert calls == [(train_root, [("5", "train")])]
    assert proxy["train_sequences"] == ["5"]
    load_val()
    assert calls[-1] == (val_root, [("4", "train")])


def test_dfine_seed_runs_through_candidate_runner(monkeypatch, tmp_path):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["families"]["dfine"]["model"] = _adapter().settings["model"]
    protocol["families"]["dfine"].pop("data")
    protocol["proxy"].update({"epochs": 1, "batch_size": 2, "img_width": 96})
    monkeypatch.setattr(candidate_runner, "proxy_data", lambda proxy: (_data(), _data()))
    reported = []

    def report(payload, sota_root, gene_id, mode):
        reported.append((payload, gene_id, mode))
        return candidate_runner.DetectionResult(0.1, 0.05, payload["params"]), tmp_path / "report.json"

    monkeypatch.setattr(candidate_runner.map_reporter, "report", report)
    candidate_runner.run("seed", "network", tmp_path / "models", "dfine",
                         protocol, tmp_path, "cpu")
    assert (tmp_path / "trained_models" / "seed.pt").is_file()
    assert len(reported) == 1
    payload, gene_id, mode = reported[0]
    assert gene_id == "seed" and mode == "validation"
    assert len(payload["frames"]) == 3
    assert payload["metadata"]["model_identity"]
