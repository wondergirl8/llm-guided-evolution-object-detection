import csv
import pytest

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


def test_option_b_indexes_both_frame_names_and_decodes_only_sampled_events(monkeypatch, tmp_path):
    import io
    import zipfile
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    names = []
    for index in range(1, 102):
        prefix = "frame_" if index % 2 else ""
        names.append(f"116/Event/Frames/Video_116_{prefix}{index * fred_stream.STEP_US}.png")
        names.append(f"116/RGB/Video_116_00_00_00.{index:06d}.jpg")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as source:
        for name in reversed(names):
            source.writestr(name, image.getvalue())
    content = payload.getvalue()
    connections = []

    class FakeFS:
        def open(self, *args, **kwargs):
            connections.append(args)
            return io.BytesIO(content)

    monkeypatch.setattr(fred_stream, "HfFileSystem", FakeFS)
    monkeypatch.setattr(fred_stream, "load_dataset", lambda *a, **k: pytest.fail("Option B read all dataset rows"))
    decoded = []

    original_open = zipfile.ZipFile.open
    def recording_open(source, name, *args, **kwargs):
        decoded.append(name)
        return original_open(source, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", recording_open)
    frames = list(fred_stream.stream_event_frames("train", 116, tmp_path / "report.csv", 50,
                                                 use_archive_index=True))
    assert [round(t * 1_000_000) for _, t in frames] == [n * fred_stream.STEP_US for n in (1, 51, 101)]
    assert len(decoded) == 3
    assert all("/Event/Frames/" in path for path in decoded)
    assert len(connections) == 1


def test_archive_index_reads_directory_without_opening_image_members(monkeypatch):
    import io
    import zipfile

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("116/Event/Frames/Video_116_33333.png", b"not decoded")
        archive.writestr("116/RGB/Video_116_00_00_00.000001.jpg", b"not decoded")
        archive.writestr("116/coordinates.txt", b"not an image")
    content = payload.getvalue()

    class FakeFS:
        def open(self, path, mode, block_size):
            assert path.startswith("datasets/GabrieleMagrini/FRED@")
            assert mode == "rb" and block_size == 65536
            return io.BytesIO(content)

    monkeypatch.setattr(fred_stream, "HfFileSystem", FakeFS)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("Image member was opened"))
    paths = list(fred_stream.indexed_archive_paths(
        f"hf://datasets/GabrieleMagrini/FRED@{fred_stream.REVISION}/train/116.zip"))
    assert len(paths) == 2
    assert all(path.startswith("zip://116/") for path in paths)
