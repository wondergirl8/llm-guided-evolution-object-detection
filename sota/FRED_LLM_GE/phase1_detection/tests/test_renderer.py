import numpy as np
import pytest

from sota.FRED_LLM_GE.phase1_detection import render_runner  # puts data/ on sys.path
import fred_events

DTYPE = np.dtype({"names": ["x", "y", "p", "t"], "formats": ["<u2", "<u2", "<i2", "<i8"],
                  "offsets": [0, 2, 4, 8], "itemsize": 16})


def events(*rows):
    return np.array(list(rows), DTYPE)


def renderer_file(tmp_path, body):
    path = tmp_path / "network_test.py"
    path.write_text("import numpy as np\nWIDTH, HEIGHT = 1280, 720\n" + body)
    return path


def test_seed_renderer_reproduces_the_released_palette_frames():
    seed = render_runner.load_renderer(render_runner.SEED_RENDERER)
    rng = np.random.default_rng(0)
    t = np.sort(rng.integers(0, 100_000, 5000))
    ev = np.zeros(len(t), DTYPE)
    ev["x"], ev["y"] = rng.integers(0, 1280, len(t)), rng.integers(0, 720, len(t))
    ev["p"], ev["t"] = rng.integers(0, 2, len(t)), t
    for end in (33_333, 66_666, 99_999):
        image, _ = render_runner.render_frame(seed, ev, end)
        assert np.array_equal(image, np.asarray(fred_events.render(ev, end, 33_333)))


def test_seed_has_one_editable_block_holding_window_and_render():
    parts = render_runner.SEED_RENDERER.read_text().split("# --OPTION--")
    assert len(parts) == 2
    assert "WINDOW_US = 33333" in parts[1] and "def render(" in parts[1]


def test_renderer_only_sees_its_own_window_with_relative_times():
    ev = events((1, 1, 1, 100), (2, 1, 1, 150), (3, 1, 0, 199), (4, 1, 1, 200))
    window = render_runner.window_events(ev, end_us=200, window_us=100)
    assert window["x"].tolist() == [2, 3]
    assert window["t"].tolist() == [-50, -1]
    assert ev["t"].tolist() == [100, 150, 199, 200]  # the cached events are untouched


@pytest.mark.parametrize("body, message", [
    ("WINDOW_US = 500\ndef render(events, window_us):\n    pass\n", "WINDOW_US"),
    ("WINDOW_US = 10000.0\ndef render(events, window_us):\n    pass\n", "WINDOW_US"),
    ("WINDOW_US = 10000\n", "render"),
])
def test_invalid_renderers_are_rejected(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        render_runner.load_renderer(renderer_file(tmp_path, body))


@pytest.mark.parametrize("returned", [
    "np.zeros((720, 1280, 3), np.float32)",
    "np.zeros((360, 640, 3), np.uint8)",
    "None",
])
def test_frames_of_the_wrong_type_or_size_are_rejected(tmp_path, returned):
    body = f"WINDOW_US = 10000\ndef render(events, window_us):\n    return {returned}\n"
    renderer = render_runner.load_renderer(renderer_file(tmp_path, body))
    with pytest.raises(ValueError, match="uint8"):
        render_runner.render_frame(renderer, events((1, 1, 1, 5)), end_us=10)


def test_render_frame_counts_the_events_it_was_given(tmp_path):
    body = "WINDOW_US = 1000\ndef render(events, window_us):\n    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)\n"
    renderer = render_runner.load_renderer(renderer_file(tmp_path, body))
    _, count = render_runner.render_frame(renderer, events((1, 1, 1, 500), (2, 2, 0, 1500), (3, 3, 1, 1999)),
                                          end_us=2000)
    assert count == 2


def test_slicing_keeps_exactly_the_events_any_frame_window_can_use():
    from sota.FRED_LLM_GE.phase1_detection import cache_events

    ends = np.array([1000, 5000], np.int64)
    t = np.array([0, 1, 999, 1000, 1001, 4000, 4001, 4999, 5000, 9000], np.int64)
    keep = cache_events.used_by_frames(t, ends, max_window_us=1000)
    assert t[keep].tolist() == [1, 999, 4001, 4999]


def test_frames_drawn_from_sliced_events_match_frames_from_all_events():
    from sota.FRED_LLM_GE.phase1_detection import cache_events

    seed = render_runner.load_renderer(render_runner.SEED_RENDERER)
    rng = np.random.default_rng(1)
    ev = np.zeros(20_000, DTYPE)
    ev["x"], ev["y"] = rng.integers(0, 1280, len(ev)), rng.integers(0, 720, len(ev))
    ev["p"], ev["t"] = rng.integers(0, 2, len(ev)), np.sort(rng.integers(0, 2_000_000, len(ev)))
    ends = np.array([400_000, 1_200_000, 1_999_999], np.int64)
    sliced = ev[cache_events.used_by_frames(ev["t"], ends)]
    assert len(sliced) < len(ev) / 2
    for end in ends:
        assert np.array_equal(render_runner.render_frame(seed, sliced, end)[0],
                              render_runner.render_frame(seed, ev, end)[0])


def test_identical_renderers_reuse_the_earlier_score(tmp_path):
    import json

    models, renders = tmp_path / "models", tmp_path / "renders"
    models.mkdir()
    renders.mkdir()
    body = "import numpy as np\n# --OPTION--\nWINDOW_US = 10000\ndef render(events, window_us):\n    return 1\n"
    (models / "network_a.py").write_text(body)
    (renders / "a.json").write_text(json.dumps({"mAP50": 0.8, "mAP50_95": 0.4, "window_us": 10000}))
    # Same code with other comments and blank lines: a duplicate.
    (models / "network_b.py").write_text(body.replace("WINDOW_US", "# a comment\n\nWINDOW_US", 1))
    gene, stats = render_runner.earlier_score(models / "network_b.py", tmp_path, models)
    assert gene == "a" and stats["mAP50"] == 0.8
    # A real change: no reuse.
    (models / "network_c.py").write_text(body.replace("10000", "8000"))
    assert render_runner.earlier_score(models / "network_c.py", tmp_path, models) is None


def test_a_twin_waits_for_the_identical_renderer_with_the_smaller_id(tmp_path):
    import threading

    path = renderer_file(tmp_path, "# --OPTION--\nWINDOW_US = 10000\n")
    first = render_runner.wait_for_twin(path, tmp_path, "a", poll_s=0.05)
    waited = []
    second = threading.Thread(target=lambda: waited.append(
        render_runner.wait_for_twin(path, tmp_path, "b", poll_s=0.05)))
    second.start()
    second.join(0.3)
    assert second.is_alive()  # "b" waits while "a" runs
    first.unlink()
    second.join(2)
    assert waited and not second.is_alive()


def test_a_twin_of_a_failed_renderer_fails_without_rerunning(tmp_path):
    models, results = tmp_path / "models", tmp_path / "results"
    models.mkdir()
    results.mkdir()
    body = "import numpy as np\n# --OPTION--\nWINDOW_US = 10000\ndef render(events, window_us):\n    return 1\n"
    (models / "network_a.py").write_text(body)
    (results / "a_results.csv").write_text("0.000000,0.000000,999999999")
    (models / "network_b.py").write_text(body)
    assert render_runner.earlier_score(models / "network_b.py", tmp_path, models) == ("a", {"failed": True})


def test_a_renderer_that_changes_the_fixed_window_fails_unscored(tmp_path, monkeypatch):
    import sys

    models = tmp_path / "models"
    models.mkdir()
    body = "import numpy as np\n# --OPTION--\nWINDOW_US = 10000\ndef render(events, window_us):\n    return 1\n"
    (models / "network_a.py").write_text(body)
    monkeypatch.setattr(render_runner, "render_split", lambda *a: pytest.fail("must not render"))
    monkeypatch.setattr(sys, "argv", ["render_runner.py", "--model", "network_a", "--variant_dir",
                                      str(models), "--sota_root", str(tmp_path), "--window_us", "2000",
                                      "--device", "cpu"])
    render_runner.main()
    assert (tmp_path / "results" / "a_results.csv").read_text().strip().endswith(",999999999")
