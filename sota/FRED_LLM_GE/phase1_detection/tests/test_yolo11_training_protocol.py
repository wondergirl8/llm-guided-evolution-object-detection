from pathlib import Path

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter


FRED_DIR = Path(__file__).resolve().parents[2]
PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.options = None

    def train(self, **options):
        self.options = options
        args = Path(options["project"]) / options["name"] / "args.yaml"
        args.parent.mkdir(parents=True)
        args.write_text(yaml.safe_dump(options))
        if self.fail:
            raise RuntimeError("training failed")


@pytest.mark.parametrize("fail", [False, True])
def test_yolo11_training_uses_protocol_and_preserves_trainer_args(tmp_path, fail):
    protocol = yaml.safe_load(PROTOCOL.read_text())
    adapter = UltralyticsYolo11Adapter(protocol["families"]["yolo11"], FRED_DIR)
    adapter.workdir = tmp_path / "work"
    model = FakeModel(fail=fail)
    args_path = tmp_path / "candidate" / "args.yaml"
    proxy = dict(protocol["proxy"], epochs=2)
    data = {"data_yaml": tmp_path / "data.yaml"}

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            adapter.train(model, data, proxy, "cpu", args_path=args_path)
    else:
        adapter.train(model, data, proxy, "cpu", args_path=args_path)

    assert model.options["optimizer"] == "SGD"
    assert model.options["lr0"] == 0.01
    assert model.options["momentum"] == 0.937
    assert model.options["mosaic"] == 1.0
    assert model.options["close_mosaic"] == 10
    assert model.options["pretrained"] is False
    assert yaml.safe_load(args_path.read_text()) == model.options
