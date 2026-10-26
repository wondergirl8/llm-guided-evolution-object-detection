import sys
from unittest.mock import patch

import pytest
import torch
import yaml

pytest.importorskip("ultralytics")

from sota.FRED_LLM_GE.phase1_detection import candidate_runner


def _adapter():
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    return candidate_runner.ADAPTERS["yolo26_p2"](
        protocol["families"]["yolo26_p2"], candidate_runner.FRED_DIR)


def test_seed_builds_one_class_with_p2_detection():
    adapter = _adapter()
    assert adapter.variant_path("network", None).read_text().count("# --OPTION--") == 2
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    assert model.model.yaml["nc"] == 1
    assert model.model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model.model[-1].reg_max == 1
    assert hasattr(model.model.model[-1], "one2one_cv2")


def test_candidate_cannot_change_fixed_detection_mode(tmp_path):
    adapter = _adapter()
    config = yaml.safe_load(adapter.variant_path("network", None).read_text())
    config["end2end"] = False
    variant = tmp_path / "network_changed.py"
    variant.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="fixed"):
        adapter.build(variant, num_classes=1, device="cpu")


def test_bounded_mutation_builds_and_routing_change_is_rejected(tmp_path):
    adapter = _adapter()
    config = yaml.safe_load(adapter.variant_path("network", None).read_text())
    config["head"][8][1] = 3  # P2 C3k2 repeats at absolute layer 19.
    variant = tmp_path / "network_mutated.py"
    variant.write_text(yaml.safe_dump(config))
    model = adapter.build(variant, num_classes=1, device="cpu")
    assert model.model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    config["head"][8][0] = 16
    variant.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="routing"):
        adapter.build(variant, num_classes=1, device="cpu")


def test_preflight_requires_one_valid_layer_edit_and_normalizes_hash(tmp_path):
    adapter = _adapter()
    seed = adapter.variant_path("network", None)
    assert adapter.preflight(seed, seed, stock_seed=True) == adapter.architecture_hash(seed)
    with pytest.raises(ValueError, match="exactly one layer"):
        adapter.preflight(seed, seed)
    config = yaml.safe_load(seed.read_text())
    config["head"][8][1] = 3
    variant = tmp_path / "network_candidate.py"
    variant.write_text(yaml.safe_dump(config, sort_keys=False))

    digest = adapter.preflight(variant, seed)
    variant.write_text("# A formatting-only change\n" + yaml.safe_dump(config))
    assert adapter.architecture_hash(variant) == digest

    config["head"][9][3][1] = 5
    variant.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="exactly one layer"):
        adapter.preflight(variant, seed)

    config["head"][9][3][1] = 3
    config["head"][8][3][0] = 9999
    variant.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="out of bounds"):
        adapter.preflight(variant, seed)


def test_final_evaluation_selects_yolo26_adapter():
    from sota.FRED_LLM_GE.phase1_detection import final_eval

    assert final_eval.FINAL_ADAPTERS["yolo26_p2"] is candidate_runner.ADAPTERS["yolo26_p2"]
    assert _adapter().postprocessing({"img_width": 1280, "eval_conf": 0.05,
                                      "nms_iou": 0.5}) == (
        "ultralytics YOLO26 head=one_to_many nms=True imgsz=1280 conf=0.05 nms_iou=0.5")


def test_final_evaluation_uses_yolo26_training_overrides(monkeypatch, tmp_path):
    from sota.FRED_LLM_GE.phase1_detection import final_eval
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import (
        UltralyticsYolo26P2Adapter,
    )

    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["data_root"] = str(tmp_path)
    final = protocol["challenging"]["final"]
    data_dir = tmp_path / f"final_train_step{final['train_frame_step']}"
    data_dir.mkdir()
    (data_dir / "data.yaml").write_text("train: synthetic\n")
    variant = tmp_path / "network_selected.py"
    variant.write_text("nc: 1\n")
    options = {}

    class TrainingReached(Exception):
        pass

    class Model:
        def train(self, **kwargs):
            options.update(kwargs)
            raise TrainingReached

    monkeypatch.setattr(final_eval, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(final_eval, "challenging_split", lambda root: ([], []))
    monkeypatch.setattr(UltralyticsYolo26P2Adapter, "variant_path",
                        lambda self, model, directory: variant)
    monkeypatch.setattr(UltralyticsYolo26P2Adapter, "build",
                        lambda self, path, num_classes, device: Model())
    monkeypatch.setattr(UltralyticsYolo26P2Adapter, "param_count",
                        lambda self, model: 1)

    arguments = ["final_eval.py", "--family", "yolo26_p2", "--name", "selected_final",
                 "--model", "network_selected", "--epochs", "1", "--device", "cpu"]
    with patch.object(sys, "argv", arguments), pytest.raises(TrainingReached):
        final_eval.main()

    assert options["data"] == str(data_dir / "data.yaml")
    assert options["epochs"] == 1
    assert options["pretrained"] is False


def test_prediction_selects_each_real_yolo26_head():
    adapter = _adapter()
    model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    images = torch.zeros(1, 3, 64, 96, dtype=torch.uint8)
    proxy = {"img_width": 96, "eval_conf": 0.05, "nms_iou": 0.5}

    for head, nms in (("one_to_many", True), ("one_to_one", False)):
        adapter.inference_head = head
        predictions = adapter.predict(model, images, proxy, "cpu")
        assert len(predictions) == 1
        assert predictions[0][0].shape[1] == 4
        assert model.predictor.args.nms is nms
        assert model.predictor.model.end2end is not nms
        assert model.predictor.model.model.end2end is not nms

    assert "no NMS" in adapter.postprocessing(proxy)


def test_invalid_inference_head_is_rejected():
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    settings = dict(protocol["families"]["yolo26_p2"], inference_head="auto")
    with pytest.raises(ValueError, match="inference_head"):
        candidate_runner.ADAPTERS["yolo26_p2"](settings, candidate_runner.FRED_DIR)


def test_runner_defaults_to_separate_yolo26_output():
    arguments = ["candidate_runner.py", "--family", "yolo26_p2"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    assert run.call_args.args[2] == str(candidate_runner.FAMILY_SOTA_ROOTS["yolo26_p2"] / "models")
    assert run.call_args.args[5] == str(candidate_runner.FAMILY_SOTA_ROOTS["yolo26_p2"])


def test_runner_diagnostics_flag_is_opt_in():
    arguments = ["candidate_runner.py", "--family", "yolo26_p2", "--diagnostics"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    assert run.call_args.kwargs["diagnostics"] is True
