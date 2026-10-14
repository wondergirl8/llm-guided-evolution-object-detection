import sys
from unittest.mock import patch

import pytest
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


def test_final_evaluation_selects_yolo26_adapter():
    from sota.FRED_LLM_GE.phase1_detection import final_eval

    assert final_eval.FINAL_ADAPTERS["yolo26_p2"] is candidate_runner.ADAPTERS["yolo26_p2"]
    assert "no NMS" in _adapter().postprocessing({"img_width": 1280, "eval_conf": 0.001})


def test_runner_defaults_to_separate_yolo26_output():
    arguments = ["candidate_runner.py", "--family", "yolo26_p2"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    assert run.call_args.args[2] == str(candidate_runner.FAMILY_SOTA_ROOTS["yolo26_p2"] / "models")
    assert run.call_args.args[5] == str(candidate_runner.FAMILY_SOTA_ROOTS["yolo26_p2"])
