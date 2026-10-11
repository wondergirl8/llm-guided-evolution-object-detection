"""Option-B split protection, cache reuse and candidate reporting."""

import json
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11"))
import option_b
import train_ultralytics as runner


def test_frozen_partition_and_canonical_archive_mapping():
    manifest = option_b.load_manifest()
    train, val = set(manifest["train_sequences"]), set(manifest["val_sequences"])
    test = {s.rstrip("/") for s in manifest["official_test_text"].splitlines()}
    assert len(train) == 152 and len(val) == 20 and len(test) == 59
    assert not train & val and not (train | val) & test
    # Canonical HF test-folder archives can belong to official challenging-train.
    assert "8" in train and manifest["archive_sources"]["8"] == "test"
    assert "36" not in train | val


@pytest.mark.parametrize("change", ["test-leak", "official-source", "overlap", "sampling"])
def test_invalid_manifest_is_rejected(tmp_path, monkeypatch, change):
    manifest = json.loads(option_b.MANIFEST_PATH.read_text())
    if change == "test-leak":
        manifest["train_sequences"][0] = "36"
    elif change == "official-source":
        manifest["official_train_text"] += "36/\n"
    elif change == "overlap":
        manifest["val_sequences"][0] = manifest["train_sequences"][0]
    else:
        manifest["frame_step"] = 10
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(option_b, "MANIFEST_PATH", path)
    with pytest.raises(ValueError):
        option_b.load_manifest()


def test_option_b_cli_defaults_and_stale_a_settings(monkeypatch):
    for name in ("FRED_TRAIN_SEQUENCES", "FRED_VAL_SEQUENCES", "FRED_FRAME_STEP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["trainer", "--data_profile", "b"])
    args = runner.parse_args()
    assert len(args.train_sequences) == 152 and len(args.val_sequences) == 20 and args.frame_step == 50
    assert args.val_frame_step == 10
    assert (args.scale, args.epochs, args.img_width, args.batch_size) == ("n", 20, 1280, 16)
    monkeypatch.setenv("FRED_TRAIN_SEQUENCES", "5,18,30,42,57,66,96,104")
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_option_b_reuses_sparse_cache_and_original_annotation_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shared, "SCRIPT_DIR", tmp_path)
    calls = []

    def stream(split, sequence, report, frame_step, **options):
        calls.append((split, sequence, frame_step, options))
        yield Image.new("RGB", (128, 72)), 33333 / 1_000_000

    monkeypatch.setattr(runner.shared, "stream_event_frames", stream)
    monkeypatch.setattr(runner.shared, "load_annotations", lambda *a: {1: [[10, 10, 30, 20]]})
    kwargs = {"sequence_sources": {"8": "test"}, "use_archive_index": True}
    first = runner.shared.load_split("train", ["8"], 50, 64, **kwargs)
    second = runner.shared.load_split("train", ["8"], 50, 64, **kwargs)
    assert calls == [("test", "8", 50, {"use_archive_index": True})]
    assert first["sample_ids"] == second["sample_ids"] == ["test/8:1"]
    torch.testing.assert_close(first["orig_boxes"][0], torch.tensor([[10., 10., 30., 20.]]))
    torch.testing.assert_close(first["boxes"][0], torch.tensor([[5., 5., 15., 10.]]))
    assert (tmp_path / "data_cache/test_8_step50_w64_indexed_v1.pt").is_file()


def test_option_b_training_records_frozen_manifest_and_source_mapping(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import synthetic_data

    for name in ("FRED_TRAIN_SEQUENCES", "FRED_VAL_SEQUENCES", "FRED_FRAME_STEP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["trainer", "--data_profile", "b", "--scale", "n",
                                      "--device", "cpu", "--epochs", "1", "--img_width", "96", "--batch_size", "2"])
    args = runner.parse_args()
    monkeypatch.setattr(runner, "SCRIPT_DIR", tmp_path)
    calls = []

    def fake_load(manifest, width):
        calls.append((manifest["train_sequences"], manifest["frame_step"], manifest["val_frame_step"]))
        return synthetic_data(manifest["train_sequences"]), synthetic_data(manifest["val_sequences"])

    monkeypatch.setattr(runner.option_b, "load_exported_data", fake_load)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        runner.main(args)
    finally:
        torch.set_num_threads(previous)
    assert len(calls) == 1 and len(calls[0][0]) == 152
    assert calls[0][1:] == (50, 10)
    metadata = json.loads((tmp_path / "results/seed_evaluation.json").read_text())["metadata"]
    config = metadata["training_config"]
    assert config["data_profile"] == "b"
    assert config["val_frame_step"] == 10
    assert config["loader"] == "team-exported-native-png-lazy-batches"
    assert config["option_b_manifest_sha256"] == option_b.load_manifest()["sha256"]
    assert metadata["manifest_identity"] == config["option_b_manifest_sha256"]
    assert config["archive_sources"]["8"] == "test"
