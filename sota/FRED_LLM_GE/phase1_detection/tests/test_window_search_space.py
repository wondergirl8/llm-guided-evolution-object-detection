import math
import sys

import pytest

from sota.FRED_LLM_GE.phase1_detection import window_runner
from sota.FRED_LLM_GE.phase1_detection import window_search_space as space
from sota.FRED_LLM_GE.phase1_detection.event_sweep import read_result
from sota.FRED_LLM_GE.phase1_detection.result_contract import (
    DetectionResult, write_fitness_row, write_results)


def gene(tmp_path, name="seed", window=None, old=None, new=None):
    text = space.seed_gene_text(space.base.SEED_YAML.read_text())
    if window is not None:
        text = text.replace(f"window_us: {space.SEED_WINDOW_US}", f"window_us: {window}")
    if old is not None:
        text = text.replace(old, new, 1)
    path = tmp_path / f"network_{name}.py"
    path.write_text(text)
    return path


WIDER = ("[-1, 1, Conv, [128, 3, 2]] # 1-P2/4", "[-1, 1, Conv, [192, 3, 2]] # 1-P2/4")


def test_the_seed_gene_is_the_yolo11_seed_plus_one_window_section(tmp_path):
    seed = gene(tmp_path)
    assert len(seed.read_text().split("# --OPTION--")) == 4
    config, window = space.split_gene(seed)
    assert window == space.SEED_WINDOW_US
    assert config == space.base.load_and_validate(space.base.SEED_YAML)


def test_the_window_is_part_of_a_candidates_identity(tmp_path):
    seed, other = gene(tmp_path), gene(tmp_path, "other", window=10000)
    (seed_hash, seed_params), (other_hash, other_params) = (
        space.built_architecture(seed), space.built_architecture(other))
    assert seed_params == other_params == space.base.SEED_PARAMS
    assert seed_hash != other_hash
    assert space.differs_from_parents(other, seed)
    assert not space.differs_from_parents(seed, gene(tmp_path, "copy"))


def test_a_mutation_changes_exactly_one_layer_or_the_window(tmp_path):
    seed = gene(tmp_path)
    assert space.changed_layer_count(gene(tmp_path, "w", window=5000), seed) == 1
    assert space.changed_layer_count(gene(tmp_path, "l", old=WIDER[0], new=WIDER[1]), seed) == 1
    both = gene(tmp_path, "b", window=5000, old=WIDER[0], new=WIDER[1])
    assert space.changed_layer_count(both, seed) == 2


def test_windows_outside_the_menu_are_rejected_and_near_misses_snapped(tmp_path):
    with pytest.raises(ValueError, match="window_us must be one of"):
        space.built_architecture(gene(tmp_path, window=7000))
    assert space.repair_channel_values("window_us: 7000") == "window_us: 5000\n"
    assert space.repair_channel_values("window_us: 5000") == "window_us: 5000"


def test_a_window_section_gets_a_window_request_and_layers_keep_the_yolo11_contract():
    request = space.mutation_prompt("Edit this section:\nwindow_us: 33333", None, 0)
    assert "Choose a different window from: 2000, 3333, 5000, 10000, 16667." in request
    layers = space.mutation_prompt("Edit this section:\nbackbone: []", None, 0)
    assert layers == space.base.mutation_prompt("Edit this section:\nbackbone: []", None, 0)
    assert "rejected: too long" in space.retry_prompt("window_us: 2000", "too long", 1)


def test_fitness_rows_of_any_length_match_write_results(tmp_path):
    write_results(tmp_path / "a", "x", DetectionResult(0.5, 0.25, 7))
    write_fitness_row(tmp_path / "b", "x", (0.5, 0.25, 7))
    assert (tmp_path / "a/results/x_results.csv").read_text() == \
           (tmp_path / "b/results/x_results.csv").read_text()
    write_results(tmp_path / "a", "f", DetectionResult(-math.inf, -math.inf, math.inf), reason="r")
    write_fitness_row(tmp_path / "b", "f", (-math.inf, -math.inf, math.inf), reason="r")
    assert (tmp_path / "a/results/f_results.csv").read_text() == \
           (tmp_path / "b/results/f_results.csv").read_text()
    write_fitness_row(tmp_path, "w", (0.5, 0.25, 7, 10000))
    assert (tmp_path / "results/w_results.csv").read_text() == "0.500000,0.250000,7,10000,\n"
    with pytest.raises(ValueError):
        write_fitness_row(tmp_path, "w", window_runner.FAILED)  # a failure needs a reason
    space.write_cached_fitness(tmp_path, "c", (0.5, 0.25, 7.0, 3333.0))
    assert (tmp_path / "results/c_results.csv").read_text() == "0.500000,0.250000,7,3333,\n"


def run_window_runner(tmp_path, monkeypatch, model, score=(0.8, 0.4, 2590035)):
    calls = {}

    def render(renderer, protocol, workdir, workers, timeout, window):
        calls["window"] = window
        return ("train", "val"), {"frames": 1, "render_s": 0.0, "events_per_frame": 1}

    def train(gene_id, model_name, variant_dir, family, protocol, sota_root, device, roots):
        calls.update(family=family, roots=roots,
                     architecture=(variant_dir / f"{model_name}.py").read_text())
        if score:
            write_results(sota_root, gene_id, DetectionResult(*score))

    monkeypatch.setattr(window_runner, "render_split", render)
    monkeypatch.setattr(window_runner, "run", train)
    monkeypatch.setattr(sys, "argv", ["window_runner.py", "--model", model, "--variant_dir",
                                      str(tmp_path), "--sota_root", str(tmp_path / "out"),
                                      "--device", "cpu"])
    window_runner.main()
    return calls


def test_the_runner_draws_frames_at_the_genes_window_and_trains_its_architecture(tmp_path, monkeypatch):
    gene(tmp_path, "g", window=5000)
    calls = run_window_runner(tmp_path, monkeypatch, "network_g")
    assert calls["window"] == 5000 and calls["family"] == "yolo11" and calls["roots"] == ("train", "val")
    assert "window_us" not in calls["architecture"]
    assert (tmp_path / "out/results/g_results.csv").read_text() == "0.800000,0.400000,2590035,5000,\n"


def test_the_runner_records_four_value_failures(tmp_path, monkeypatch):
    gene(tmp_path, "bad", window=7000)
    run_window_runner(tmp_path, monkeypatch, "network_bad")
    row = (tmp_path / "out/results/bad_results.csv").read_text()
    assert row.startswith("-inf,-inf,inf,inf,") and "window_us must be one of" in row
    gene(tmp_path, "untrained", window=5000)
    run_window_runner(tmp_path, monkeypatch, "network_untrained", score=None)
    assert read_result(tmp_path / "out/results/untrained_results.csv") is None
