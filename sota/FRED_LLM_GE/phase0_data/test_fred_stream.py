import csv

from data import fred_stream


class FakeRows(list):
    def cast_column(self, *args):
        return self


class FakeImage:
    def __init__(self, decode=True):
        pass

    def decode_example(self, value):
        return value["path"]


def test_event_frames_are_ordered_and_bad_counters_reported(
    monkeypatch, tmp_path
):
    names = [
        "RGB/Video_36_20_04_25.124287.jpg",
        "RGB/Video_36_20_04_25.157621.jpg",
        "RGB/Video_36_20_04_25.190955.jpg",
        "Event/Frames/Video_36_frame_99999.png",
        "Event/Frames/Video_36_frame_33333.png",
        "Event/Frames/Video_36_frame_99999.png",
        "Event/Frames/Video_36_frame_50000.png",
        "Event/Frames/Video_36_frame_166665.png",
    ]
    rows = FakeRows(
        {"image": {"path": f"zip://36/{name}::hf://example/36.zip"}}
        for name in names
    )
    def fake_load_dataset(name, **kwargs):
        assert name == "imagefolder"
        assert kwargs["streaming"] is True
        assert kwargs["data_files"]["train"].startswith(
            "hf://datasets/GabrieleMagrini/FRED@"
        )
        return rows

    monkeypatch.setattr(fred_stream, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(fred_stream, "Image", FakeImage)
    report = tmp_path / "bad_timestamps.csv"

    frames = list(fred_stream.stream_event_frames("train", 36, report))

    assert [timestamp for _, timestamp in frames] == [
        0.033333, 0.099999, 0.166665
    ]
    with report.open(newline="") as file:
        findings = list(csv.DictReader(file))
    assert [row["issue"] for row in findings].count("missing_counter") == 2
    assert {row["issue"] for row in findings} == {
        "unexpected_counter", "missing_counter", "duplicate_counter"
    }


def test_rgb_timestamp_gap_is_reported_without_decoding_rgb(monkeypatch, tmp_path):
    names = [
        "RGB/Video_36_20_04_25.124287.jpg",
        "RGB/Video_36_20_04_25.157621.jpg",
        "RGB/Video_36_20_04_25.224289.jpg",
        "Event/Frames/Video_36_frame_33333.png",
        "Event/Frames/Video_36_frame_66666.png",
        "Event/Frames/Video_36_frame_99999.png",
    ]
    rows = FakeRows(
        {"image": {"path": f"zip://36/{name}::hf://example/36.zip"}}
        for name in names
    )
    monkeypatch.setattr(fred_stream, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(fred_stream, "Image", FakeImage)
    report = tmp_path / "bad_timestamps.csv"

    frames = list(fred_stream.stream_event_frames("train", 36, report))

    assert len(frames) == 3
    assert all("/Event/Frames/" in image for image, _ in frames)
    with report.open(newline="") as file:
        findings = list(csv.DictReader(file))
    assert [row["issue"] for row in findings] == ["rgb_interval_mismatch"]


def test_missing_rgb_frame_is_reported(monkeypatch, tmp_path):
    names = [
        "RGB/Video_36_20_04_25.124287.jpg",
        "Event/Frames/Video_36_frame_33333.png",
        "Event/Frames/Video_36_frame_66666.png",
    ]
    rows = FakeRows(
        {"image": {"path": f"zip://36/{name}::hf://example/36.zip"}}
        for name in names
    )
    monkeypatch.setattr(fred_stream, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(fred_stream, "Image", FakeImage)
    report = tmp_path / "bad_timestamps.csv"

    frames = list(fred_stream.stream_event_frames("train", 36, report))

    assert [timestamp for _, timestamp in frames] == [0.033333, 0.066666]
    with report.open(newline="") as file:
        findings = list(csv.DictReader(file))
    assert [row["issue"] for row in findings] == ["rgb_event_count_mismatch"]
