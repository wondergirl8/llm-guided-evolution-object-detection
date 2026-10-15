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
