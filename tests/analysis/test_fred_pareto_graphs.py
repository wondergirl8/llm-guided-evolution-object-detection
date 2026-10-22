"""Generation-zero seed and candidate coverage in the two FRED accuracy views."""

import pickle
import sys

import pandas as pd

from analysis.scripts import fred_pareto_graphs as plots


def sample_candidates():
    return pd.DataFrame([
        {"gene_id": "seed", "mAP50": .40, "mAP50_95": .20, "params": 2_000_000},
        {"gene_id": "same_score", "mAP50": .40, "mAP50_95": .20, "params": 2_000_000},
        {"gene_id": "best_50", "mAP50": .60, "mAP50_95": .19, "params": 3_000_000},
        {"gene_id": "best_95", "mAP50": .30, "mAP50_95": .30, "params": 2_500_000},
        {"gene_id": "dominated", "mAP50": .20, "mAP50_95": .10, "params": 4_000_000},
    ])


def test_checkpoint_keeps_seed_and_all_valid_descendants(tmp_path):
    checkpoint = tmp_path / "global_gen_1.pkl"
    state = {
        "GLOBAL_DATA_HIST": {
            "seed": {"fitness": (.4, .2, 2_000_000)},
            "earlier": {"fitness": (.5, .25, 2_800_000)},
            "failed": {"fitness": (-float("inf"), -float("inf"), float("inf"))},
        },
        "GLOBAL_DATA": {"current": {"fitness": (.6, .3, 3_000_000)}},
    }
    checkpoint.write_bytes(pickle.dumps(state))
    rows = plots.load_candidates(checkpoint)
    assert set(rows["gene_id"]) == {"seed", "earlier", "current"}


def test_each_panel_has_its_own_correct_front_and_retains_ties(tmp_path, monkeypatch):
    candidates = sample_candidates()
    assert set(plots.front_2d(candidates, "mAP50")["gene_id"]) == {
        "seed", "same_score", "best_50"}
    assert set(plots.front_2d(candidates, "mAP50_95")["gene_id"]) == {
        "seed", "same_score", "best_95"}
    assert plots.pareto_mask(candidates).tolist() == [True, True, True, True, False]
    limits = {"params": (1_000_000, 5_000_000), "mAP50": (0, 100),
              "mAP50_95": (0, 100)}
    target = tmp_path / "pareto2d.png"
    figures = []
    monkeypatch.setattr(plots.plt, "close", lambda figure: figures.append(figure))
    plots.plot_fronts_2d(candidates, "test", limits, target)
    assert target.is_file()
    assert plots.n_candidates(candidates) == 5
    for axis in figures[0].axes:
        assert sum(len(collection.get_offsets()) for collection in axis.collections) == 5
        assert any(text.get_text() == "×2" for text in axis.texts)


def test_plot_outputs_include_generation_zero_and_historical_candidates(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "global_gen_0.pkl").write_bytes(pickle.dumps({
        "GLOBAL_DATA": {"seed": {"fitness": (.4, .2, 2_000_000)},
                        "first": {"fitness": (.5, .25, 2_800_000)}}}))
    (checkpoints / "global_gen_1.pkl").write_bytes(pickle.dumps({
        "GLOBAL_DATA_HIST": {"seed": {"fitness": (.4, .2, 2_000_000)},
                             "first": {"fitness": (.5, .25, 2_800_000)}},
        "GLOBAL_DATA": {"later": {"fitness": (.6, .3, 3_000_000)},
                        "failed": {"fitness": (-float("inf"), -float("inf"), float("inf"))}}}))
    seed_file = tmp_path / "seed.yaml"
    seed_file.write_text("nc: 1\n")
    output = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", ["fred_pareto_graphs.py", "--checkpoints", str(checkpoints),
                                  "--models_dir", str(tmp_path / "models"),
                                  "--seed_file", str(seed_file), "--output", str(output), "--no_gif"])
    plots.main()
    assert (output / "pareto2d_gen_0.png").is_file()
    assert (output / "pareto2d_final.png").is_file()
    candidates = pd.read_csv(output / "candidates.csv")
    assert set(candidates["gene_id"]) == {"seed", "first", "later"}
    models = pd.read_csv(output / "pareto_models.csv")
    assert str(seed_file) in set(models["model_file"])
