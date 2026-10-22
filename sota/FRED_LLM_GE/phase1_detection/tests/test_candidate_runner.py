import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sota.FRED_LLM_GE.phase1_detection import candidate_runner
from sota.FRED_LLM_GE.phase1_detection.adapters.tiny_yolo11 import TinyYolo11Adapter

FRED_DIR = Path(candidate_runner.__file__).resolve().parents[1]
TRAIN_EVAL = FRED_DIR / "phase1_detection" / "seeds" / "yolo11" / "train_eval.py"


def test_direct_script_imports_adapters_without_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(candidate_runner.__file__)), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--family" in completed.stdout


@pytest.mark.parametrize("events_tag", [None, "raw_w10000_polarity"])
def test_main_evaluates_once_with_requested_events(monkeypatch, tmp_path, events_tag):
    protocol = {"proxy": {"seed": 0}, "challenging": {"events": {"source": "released"}}}
    monkeypatch.setattr(candidate_runner, "load_protocol", lambda _: protocol)
    calls = []
    monkeypatch.setattr(candidate_runner, "run", lambda *args: calls.append(args[4]["challenging"]["events"].copy()))
    argv = ["candidate_runner.py", "--family", "yolo11", "--model", "network",
            "--device", "cuda", "--sota_root", str(tmp_path)]
    if events_tag:
        argv.extend(["--events", events_tag])
    monkeypatch.setattr(sys, "argv", argv)

    candidate_runner.main()

    assert calls == ([candidate_runner.events_from_tag(events_tag)] if events_tag
                     else [{"source": "released"}])


def _load_train_eval():
    spec = importlib.util.spec_from_file_location("legacy_train_eval", TRAIN_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_data(n=5, height=64, width=96):
    generator = torch.Generator().manual_seed(1)
    images = torch.randint(0, 256, (n, 3, height, width), dtype=torch.uint8, generator=generator)
    boxes = [torch.tensor([[10.0, 12.0, 22.0, 20.0]]), torch.zeros(0, 4)] * n
    boxes = boxes[:n]
    return {
        "images": images,
        "boxes": boxes,
        "orig_boxes": [b * 2 for b in boxes],
        "sample_ids": [f"train/0:{i}" for i in range(n)],
        "orig_size": (width * 2, height * 2),
    }


def test_protocol_matches_legacy_train_eval_settings():
    legacy = _load_train_eval()
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    proxy = protocol["proxy"]
    assert proxy["train_sequences"] == legacy.TRAIN_SEQUENCES.split(",")
    assert proxy["val_sequences"] == legacy.VAL_SEQUENCES.split(",")
    assert proxy["epochs"] == legacy.PROXY_EPOCHS
    assert proxy["frame_step"] == legacy.FRAME_STEP
    assert proxy["eval_conf"] == legacy.EVAL_CONF
    assert proxy["benchmark_split"] == legacy.BENCHMARK_SPLIT
    assert proxy["official_split"] == legacy.OFFICIAL_SPLIT
    assert (FRED_DIR / proxy["cache_dir"]).resolve() == (TRAIN_EVAL.parent / "data_cache").resolve()
    assert (FRED_DIR / protocol["families"]["tiny_yolo11"]["seed_file"]) == legacy.SEED_FILE


def test_tiny_yolo11_adapter_reproduces_legacy_training_and_predictions():
    legacy = _load_train_eval()
    data = _synthetic_data()
    proxy = {"epochs": 2, "batch_size": 2, "seed": 0, "eval_conf": legacy.EVAL_CONF,
             "nms_iou": 0.5}

    torch.manual_seed(0)
    module, _ = legacy.load_variant("network", None)
    old_model = module.DroneDetector(num_classes=1)
    stride = legacy.model_stride(old_model, data["images"][:1].float() / 255)
    old_model.output_stride = stride
    legacy.train(old_model, data, proxy["epochs"], proxy["batch_size"], 1e-3, "cpu", stride)
    old_frames = legacy.predict_frames(old_model, data, proxy["batch_size"], "cpu")

    adapter = TinyYolo11Adapter({"seed_file": "seeds/seed_yolo11.py", "lr": 1e-3}, FRED_DIR)
    torch.manual_seed(0)
    new_model = adapter.build(adapter.variant_path("network", None), num_classes=1, device="cpu")
    adapter.train(new_model, data, proxy, "cpu")
    new_frames = candidate_runner.predict_frames(adapter, new_model, data, proxy, "cpu")

    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    assert old_state.keys() == new_state.keys()
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key
    assert new_frames == old_frames
