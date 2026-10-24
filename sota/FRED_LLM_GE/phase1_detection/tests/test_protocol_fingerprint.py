from types import SimpleNamespace

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint


def test_fingerprint_changes_with_protocol_and_search_manifest(tmp_path, monkeypatch):
    fred = tmp_path / "sota" / "FRED_LLM_GE"
    protocol_path = fred / "configs" / "phase1" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol = {"families": {"yolo11": {"data": "challenging_search"}},
                "challenging": {"data_root": "cache",
                                "search": {"train_frame_step": 50, "val_frame_step": 10}}}
    protocol_path.write_text(yaml.safe_dump(protocol))
    seed = fred / "seeds" / "seed.yaml"
    seed.parent.mkdir()
    seed.write_text("backbone: []\nhead: []\n")
    cache = fred / "cache"
    for relative in ("splits/challenging_train_split.txt",
                     "search_train_step50/train.txt", "search_val_step10/train.txt"):
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("one\n")
    monkeypatch.setattr(protocol_fingerprint.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="commit-1\n"))
    monkeypatch.setattr(protocol_fingerprint.importlib.metadata, "version", lambda _: "8.4.175")

    first = protocol_fingerprint.compute_fingerprint(tmp_path, "yolo11", "run_1", seed)
    assert first["ultralytics_version"] == "8.4.175"
    assert "challenging_test" not in first["files"]
    assert first == protocol_fingerprint.compute_fingerprint(tmp_path, "yolo11", "run_1", seed)

    (cache / "search_val_step10" / "train.txt").write_text("two\n")
    assert first["sha256"] != protocol_fingerprint.compute_fingerprint(
        tmp_path, "yolo11", "run_1", seed)["sha256"]
    assert first["sha256"] != protocol_fingerprint.compute_fingerprint(
        tmp_path, "yolo11", "run_2", seed)["sha256"]


def test_resume_rejects_missing_or_changed_fingerprint():
    expected = {"sha256": "first"}
    protocol_fingerprint.verify_fingerprint({"protocol_fingerprint": expected}, "checkpoint", expected)
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        protocol_fingerprint.verify_fingerprint({}, "checkpoint", expected)
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        protocol_fingerprint.verify_fingerprint(
            {"protocol_fingerprint": {"sha256": "different"}}, "checkpoint", expected)
