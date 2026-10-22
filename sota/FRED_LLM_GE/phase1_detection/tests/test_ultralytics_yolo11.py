import pytest
import torch
import yaml

pytest.importorskip("ultralytics")

from sota.FRED_LLM_GE.phase1_detection import candidate_runner
from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo11 import (
    UltralyticsYolo11Adapter,
    export_dataset,
)


def _adapter():
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    return UltralyticsYolo11Adapter(protocol["families"]["yolo11"], candidate_runner.FRED_DIR)


def test_seed_builds_yolo11n_with_one_class():
    adapter = _adapter()
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    assert model.model.yaml["nc"] == 1
    assert model.model.stride.tolist() == [8.0, 16.0, 32.0]
    assert adapter.param_count(model) == 2_590_035


def test_seed_has_protected_preamble_and_two_mutable_sections():
    adapter = _adapter()
    parts = adapter.variant_path("network", None).read_text().split("# --OPTION--")
    assert len(parts) == 3
    assert "backbone:" in parts[1] and "head:" in parts[2]


def test_invalid_candidate_fails_to_build(tmp_path):
    adapter = _adapter()
    seed = yaml.safe_load(adapter.variant_path("network", None).read_text())
    seed["head"][-1][0] = [16, 19, 99]  # Detect reads a layer that does not exist
    variant = tmp_path / "network_bad.py"
    variant.write_text(yaml.safe_dump(seed))
    with pytest.raises(Exception):
        adapter.build(variant, num_classes=1, device="cpu")


def test_export_writes_yolo_labels_once(tmp_path):
    images = torch.zeros(3, 3, 40, 80, dtype=torch.uint8)
    boxes = [torch.tensor([[8.0, 4.0, 24.0, 20.0]]), torch.zeros(0, 4), torch.zeros(0, 4)]
    data = {"images": images, "boxes": boxes}
    proxy = {"hf_split": "train", "train_sequences": ["0"], "frame_step": 10, "img_width": 80}
    data_yaml = export_dataset(data, proxy, tmp_path)
    root = data_yaml.parent
    assert len(list((root / "train" / "images").glob("*.png"))) == 3
    assert (root / "train" / "labels" / "000000.txt").read_text() == \
        "0 0.200000 0.300000 0.200000 0.400000"
    assert (root / "train" / "labels" / "000001.txt").read_text() == ""
    # A second call reuses the finished export instead of rewriting it.
    assert export_dataset(data, proxy, tmp_path) == data_yaml
    assert [p.name for p in (tmp_path / "ultralytics").iterdir()] == [root.name]
