import hashlib
import json

import pytest
import torch
import yaml
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection import export_split, fred_splits


def test_parse_split_file_reads_fred_format():
    assert fred_splits.parse_split_file("0/\n1/\n\n120/\n") == ["0", "1", "120"]


def test_official_sequences_rejects_changed_split_file(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "challenging_test_split.txt").write_text("9/\n10/\n")
    with pytest.raises(ValueError, match="checksum"):
        fred_splits.official_sequences("challenging_test", tmp_path)


def test_official_sequences_accepts_pinned_file(tmp_path, monkeypatch):
    content = b"9/\n10/\n"
    config = {"fred_repository": {"challenging_test_url": "unused",
                                  "challenging_test_sha256": hashlib.sha256(content).hexdigest()}}
    phase0 = tmp_path / "phase0.yaml"
    phase0.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(fred_splits, "PHASE0_CONFIG", phase0)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "challenging_test_split.txt").write_bytes(content)
    assert fred_splits.official_sequences("challenging_test", tmp_path) == ["9", "10"]


def test_challenging_split_maps_folders_and_rejects_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(fred_splits, "hf_folders", lambda _: {"1": "train", "2": "test", "3": "train"})
    lists = {"challenging_train": ["1", "2"], "challenging_test": ["3"]}
    monkeypatch.setattr(fred_splits, "official_sequences", lambda name, _: lists[name])
    assert fred_splits.challenging_split(tmp_path) == ([("1", "train"), ("2", "test")], [("3", "train")])
    lists["challenging_test"] = ["2"]
    with pytest.raises(ValueError, match="overlap"):
        fred_splits.challenging_split(tmp_path)


def test_yolo_label_normalizes_boxes():
    assert export_split.yolo_label([[8.0, 4.0, 24.0, 20.0]], 80, 40) == \
        "0 0.200000 0.300000 0.200000 0.400000"
    assert export_split.yolo_label([], 80, 40) == ""


def test_export_sequence_writes_frames_then_marks_done(tmp_path, monkeypatch):
    step_us = export_split.STEP_US
    frames = [(Image.new("RGB", (80, 40)), i * step_us / 1_000_000) for i in (1, 2, 3)]
    calls = []

    def fake_stream(folder, sequence, report, frame_step, download_dir, events=None):
        calls.append((folder, sequence, frame_step))
        yield from frames

    monkeypatch.setattr(export_split, "sequence_frames", fake_stream)
    monkeypatch.setattr(export_split, "load_annotations",
                        lambda folder, seq, w, h: {2: [[8.0, 4.0, 24.0, 20.0]]})
    root = tmp_path / "train_step5"
    assert export_split.export_sequence("120", "test", 5, root) == 3
    assert calls == [("test", "120", 5)]
    assert sorted(p.name for p in (root / "images").iterdir()) == \
        ["120_000001.png", "120_000002.png", "120_000003.png"]
    assert (root / "labels" / "120_000002.txt").read_text().startswith("0 0.200000")
    assert (root / "labels" / "120_000001.txt").read_text() == ""
    assert json.loads((root / "done" / "120.json").read_text())["frames"] == 3
    # Done sequences are not streamed again.
    assert export_split.export_sequence("120", "test", 5, root) == 3
    assert len(calls) == 1

    export_split.finalize([("120", "test")], root, val_frames=2)
    data = yaml.safe_load((root / "data.yaml").read_text())
    assert data["train"] == "train.txt" and data["names"] == {0: "drone"}
    assert len((root / "train.txt").read_text().split()) == 3
    assert data["val"] == "trainer_val/images"
    assert len(list((root / "trainer_val" / "images").glob("*.png"))) == 2
    assert len(list((root / "trainer_val" / "labels").glob("*.txt"))) == 2
    # Label caches are built once here, separately for train and the check frames.
    assert (root / "labels.cache").is_file() and (root / "trainer_val" / "labels.cache").is_file()
    with pytest.raises(SystemExit, match="not exported"):
        export_split.finalize([("120", "test"), ("121", "train")], root, val_frames=2)


def test_predict_test_split_scores_every_streamed_frame(monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection import final_eval

    step_us = final_eval.STEP_US
    frames = [(Image.new("RGB", (80, 40)), i * step_us / 1_000_000) for i in (1, 2, 3)]
    monkeypatch.setattr(final_eval, "sequence_frames", lambda *a, **k: iter(frames))
    monkeypatch.setattr(final_eval, "load_annotations",
                        lambda folder, seq, w, h: {3: [[1.0, 2.0, 11.0, 12.0]]})

    class Adapter:
        def predict(self, model, images, settings, device):
            assert images.shape[1:] == (3, 40, 80)
            return [(torch.tensor([[-5.0, 1.0, 90.0, 9.0]]), torch.tensor([0.9]))
                    for _ in images]

    settings = {"batch_size": 2, "test_frame_step": 1}
    out, expected = final_eval.predict_test_split(Adapter(), None, [("9", "test")], settings,
                                                  "cpu", __import__("pathlib").Path("/tmp"))
    assert expected == ["test/9:1", "test/9:2", "test/9:3"]
    assert [f["sample_id"] for f in out] == expected
    assert out[0]["prediction"]["boxes"] == [[0.0, 1.0, 80, 9.0]]  # clipped to the frame
    assert out[2]["target"]["boxes"] == [[1.0, 2.0, 11.0, 12.0]]


def test_search_set_holds_out_val_sequences(monkeypatch, tmp_path):
    train = [("4", "train"), ("5", "test"), ("14", "train"), ("20", "train")]
    monkeypatch.setattr(export_split, "challenging_split", lambda _: (train, []))
    challenging = {"search": {"val_sequences": [4, 14], "train_frame_step": 50,
                              "val_frame_step": 10},
                   "final": {"train_frame_step": 5}}
    (t_root, t_seqs, t_step), (v_root, v_seqs, v_step) = export_split.dataset_parts(
        challenging, tmp_path, "search")
    assert (t_root.name, t_seqs, t_step) == ("search_train_step50", [("5", "test"), ("20", "train")], 50)
    assert (v_root.name, v_seqs, v_step) == ("search_val_step10", [("4", "train"), ("14", "train")], 10)
    [(f_root, f_seqs, f_step)] = export_split.dataset_parts(challenging, tmp_path, "final")
    assert (f_root.name, f_seqs, f_step) == ("final_train_step5", train, 5)
    challenging["search"]["val_sequences"] = [999]
    with pytest.raises(ValueError, match="not in challenging_train"):
        export_split.dataset_parts(challenging, tmp_path, "search")


def test_load_exported_split_reads_frames_lazily_with_exact_boxes(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import load_exported_split

    step_us = export_split.STEP_US
    image = Image.new("RGB", (80, 40), (10, 20, 30))
    monkeypatch.setattr(export_split, "sequence_frames",
                        lambda *a, **k: iter([(image, i * step_us / 1_000_000) for i in (1, 2)]))
    monkeypatch.setattr(export_split, "load_annotations",
                        lambda folder, seq, w, h: {2: [[8.5, 4.25, 24.0, 20.0]]})
    export_split.export_sequence("7", "test", 10, tmp_path)
    data = load_exported_split(tmp_path, [("7", "test")])
    assert data["sample_ids"] == ["test/7:1", "test/7:2"]
    assert data["orig_size"] == (80, 40)
    assert data["images"].shape == (2, 3, 40, 80)
    assert data["boxes"][1].tolist() == [[8.5, 4.25, 24.0, 20.0]]
    batch = data["images"][0:2]
    assert batch.dtype == torch.uint8 and batch[0, :, 0, 0].tolist() == [10, 20, 30]
    with pytest.raises(FileNotFoundError):
        load_exported_split(tmp_path, [("8", "test")])


def test_runner_requires_search_export(monkeypatch, tmp_path):
    from sota.FRED_LLM_GE.phase1_detection import candidate_runner

    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["data_root"] = str(tmp_path)
    monkeypatch.setattr(candidate_runner, "dataset_parts", lambda c, root, name: [
        (tmp_path / "search_train_step50", [("5", "train")], 50),
        (tmp_path / "search_val_step10", [("4", "train")], 10)])
    with pytest.raises(FileNotFoundError, match="export_split.py --set search"):
        candidate_runner.challenging_search(protocol)
    root = tmp_path / "search_train_step50"
    root.mkdir()
    (root / "data.yaml").write_text("path: x\n")
    (root / "train.txt").write_text("a.png\nb.png\n")
    proxy, train_data, load_val = candidate_runner.challenging_search(protocol)
    assert proxy["train_sequences"] == ["5"] and proxy["val_sequences"] == ["4"]
    assert proxy["frame_step"] == 50 and proxy["epochs"] == protocol["challenging"]["search"]["epochs"]
    assert train_data == {"data_yaml": root / "data.yaml", "frames": 2}
