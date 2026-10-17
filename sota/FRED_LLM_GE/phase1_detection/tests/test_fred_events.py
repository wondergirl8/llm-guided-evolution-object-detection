import numpy as np
import pytest

from sota.FRED_LLM_GE.phase1_detection import export_split  # puts data/ on sys.path
import fred_events

DTYPE = np.dtype({"names": ["x", "y", "p", "t"], "formats": ["<u2", "<u2", "<i2", "<i8"],
                  "offsets": [0, 2, 4, 8], "itemsize": 16})


def events(*rows):
    return np.array(list(rows), DTYPE)


def test_palette_window_is_open_at_both_ends_and_latest_event_wins():
    ev = events((1, 1, 1, 100),   # at window start: excluded
                (2, 1, 1, 150),
                (2, 1, 0, 160),   # same pixel, later negative event wins
                (3, 1, 1, 170),
                (4, 1, 1, 200))   # at window end: excluded
    image = np.asarray(fred_events.render(ev, end_us=200, window_us=100))
    assert image.shape == (720, 1280, 3)
    assert tuple(image[1, 1]) == fred_events.BACKGROUND
    assert tuple(image[1, 2]) == fred_events.NEGATIVE
    assert tuple(image[1, 3]) == fred_events.POSITIVE
    assert tuple(image[1, 4]) == fred_events.BACKGROUND
    assert (image != fred_events.BACKGROUND).any(-1).sum() == 2


def test_polarity_encoding_keeps_both_polarities_of_a_pixel():
    ev = events((5, 6, 1, 10), (5, 6, 0, 20), (7, 6, 0, 30))
    image = np.asarray(fred_events.render(ev, end_us=100, window_us=100, encoding="polarity"))
    assert tuple(image[6, 5]) == (255, 255, 0)
    assert tuple(image[6, 7]) == (0, 255, 0)
    assert image.sum() == 3 * 255


def test_render_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="encoding"):
        fred_events.render(events(), end_us=100, encoding="voxels")


def test_raw_events_get_their_own_dataset_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(export_split, "challenging_split", lambda _: ([("5", "train")], []))
    challenging = {"search": {"val_sequences": [], "train_frame_step": 50, "val_frame_step": 10},
                   "final": {"train_frame_step": 5},
                   "events": {"source": "released", "window_us": 33333, "encoding": "palette"}}
    [(root, _, _)] = export_split.dataset_parts(challenging, tmp_path, "final")
    assert root.name == "final_train_step5"
    challenging["events"] = {"source": "raw", "window_us": 10000, "encoding": "polarity"}
    train, val = export_split.dataset_parts(challenging, tmp_path, "search")
    assert (train[0].name, val[0].name) == ("search_train_step50_raw_w10000_polarity",
                                            "search_val_step10_raw_w10000_polarity")


@pytest.mark.parametrize("events_setting, message", [
    ({"source": "hdf5", "window_us": 33333, "encoding": "palette"}, "source"),
    ({"source": "raw", "window_us": 0, "encoding": "palette"}, "window_us"),
    ({"source": "raw", "window_us": 33333, "encoding": "rgb"}, "encoding"),
])
def test_events_suffix_rejects_bad_settings(events_setting, message):
    with pytest.raises(ValueError, match=message):
        export_split.events_suffix(events_setting)


@pytest.mark.parametrize("events_setting", [
    {"source": "released", "window_us": 33333, "encoding": "palette"},
    {"source": "raw", "window_us": 10000, "encoding": "polarity"},
])
def test_event_tag_round_trips(events_setting):
    from sota.FRED_LLM_GE.phase1_detection.event_sweep import tag
    assert export_split.events_from_tag(tag(events_setting)) == events_setting


def test_events_from_tag_rejects_unknown_tags():
    for bad in ("raw_w10000", "raw_w10000_voxels", "w10000_palette"):
        with pytest.raises(ValueError):
            export_split.events_from_tag(bad)


def test_sweep_export_downloads_each_sequence_once(tmp_path, monkeypatch):
    import contextlib
    import sys
    import yaml
    from PIL import Image

    sequences = [("4", "train"), ("5", "train")]
    monkeypatch.setattr(export_split, "challenging_split", lambda _: (sequences, []))
    downloads, rendered = [], []

    @contextlib.contextmanager
    def fake_download(folder, sequence, download_dir):
        downloads.append(sequence)
        yield f"{sequence}.zip"

    def fake_frames(archive, folder, sequence, report, frame_step, events):
        rendered.append((archive, events["window_us"]))
        return iter([(Image.new("RGB", (80, 40)), 1 * export_split.STEP_US / 1_000_000)])

    monkeypatch.setattr(export_split, "downloaded_sequence", fake_download)
    monkeypatch.setattr(export_split, "archive_frames", fake_frames)
    monkeypatch.setattr(export_split, "load_annotations", lambda *a: {})
    monkeypatch.setattr(export_split, "FRED_DIR", tmp_path)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"challenging": {
        "data_root": "data", "trainer_val_frames": 4,
        "search": {"val_sequences": [4], "train_frame_step": 50, "val_frame_step": 10},
        "event_sweep": [{"source": "raw", "window_us": w, "encoding": "palette"}
                        for w in (10000, 20000)]}}))
    argv = ["export_split.py", "--set", "search", "--sweep", "--protocol", str(protocol)]
    monkeypatch.setattr(sys, "argv", argv)
    export_split.main()
    assert sorted(downloads) == ["4", "5"]
    assert sorted(rendered) == [("4.zip", 10000), ("4.zip", 20000),
                                ("5.zip", 10000), ("5.zip", 20000)]
    assert (tmp_path / "data/search_val_step10_raw_w20000_palette/done/4.json").is_file()
    assert (tmp_path / "data/search_train_step50_raw_w10000_palette/done/5.json").is_file()
    # A rerun skips sequences already exported for every setting.
    export_split.main()
    assert sorted(downloads) == ["4", "5"]


def test_event_sweep_setup_and_summary(tmp_path, capsys):
    from sota.FRED_LLM_GE.phase1_detection import event_sweep

    protocol = {"proxy": {"seed": 0}, "challenging": {
        "events": event_sweep.CONTROL,
        "event_sweep": [{"source": "raw", "window_us": 10000, "encoding": "polarity"}]}}
    event_sweep.setup(protocol, tmp_path, [0, 1], "yolo11")
    jobs = (tmp_path / "jobs.txt").read_text().splitlines()
    assert len(jobs) == 4 and all("--model network" in job for job in jobs)
    import yaml
    run = yaml.safe_load((tmp_path / "raw_w10000_polarity/seed1/protocol.yaml").read_text())
    assert run["proxy"]["seed"] == 1 and run["challenging"]["events"]["window_us"] == 10000

    for name, seed, row in [("released", 0, "0.8,0.4,100"), ("released", 1, "0.9,0.5,100"),
                            ("raw_w10000_polarity", 0, "0.0,0.0,999999999")]:
        (tmp_path / name / f"seed{seed}" / "results").mkdir(parents=True)
        (tmp_path / name / f"seed{seed}" / "results" / "seed_results.csv").write_text(row)
    event_sweep.summary(protocol, tmp_path)
    rows = (tmp_path / "summary.csv").read_text().splitlines()
    assert rows[1].startswith("released,33.333,palette,2,0.850000,0.070711,0.450000,")
    assert rows[2] == "raw_w10000_polarity,10.0,polarity,0,,,,"  # failed run left out
