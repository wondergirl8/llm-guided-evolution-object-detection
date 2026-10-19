"""Frontier semantics and the standalone post-run plotting interface."""

import hashlib
import importlib.util
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fred_pareto_graphs.py"
spec = importlib.util.spec_from_file_location("fred_pareto_graphs", SCRIPT)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def models():
    return pd.DataFrame([
        ("a", .80, .60, 1_000_000),
        ("b", .90, .50, 2_000_000),
        # c is on the joint front but dominated in BOTH 2D projections.
        ("c", .85, .55, 3_000_000),
        ("dominated", .70, .40, 4_000_000),
        ("a_tie", .80, .60, 1_000_000),
    ], columns=["gene_id", "mAP50", "mAP50_95", "params"])


def test_joint_front_and_projected_front_are_different():
    data = models()
    assert set(data[plots.pareto_mask(data)].gene_id) == {"a", "a_tie", "b", "c"}
    assert list(plots.view_front(data, "mAP50").gene_id) == ["a", "b"]
    assert list(plots.view_front(data, "mAP50_95").gene_id) == ["a"]
    assert set(data[plots.view_pareto_mask(data, "mAP50_95")].gene_id) == {"a", "a_tie"}


def test_tied_cost_keeps_best_score_and_rejects_equal_score_at_higher_cost():
    data = pd.DataFrame([
        ("a", .8, .4, 1_000), ("b", .9, .3, 1_000), ("c", .9, .6, 2_000),
    ], columns=["gene_id", "mAP50", "mAP50_95", "params"])
    assert list(plots.view_front(data, "mAP50").gene_id) == ["b"]
    assert list(plots.view_front(data, "mAP50_95").gene_id) == ["a", "c"]


@pytest.mark.parametrize("fitness", [None, (), (.8, .4), (.8, .4, 0),
    (float("nan"), .4, 100), (.8, float("inf"), 100), (1.1, .4, 100),
    (.8, -.1, 100), (0, 0, 999_999_999), ("invalid", .4, 100)])
def test_invalid_fitness_is_excluded(fitness):
    assert not plots.is_valid_fitness(fitness)


def test_loader_excludes_failed_and_unfinished_candidates(tmp_path):
    path = tmp_path / "global_gen_1.pkl"
    data = {"GLOBAL_DATA_HIST": {"previous": {"fitness": (.8, .4, 100)}},
            "GLOBAL_DATA": {
                "complete": {"fitness": (.9, .5, 200), "status": "completed"},
                "failed": {"fitness": (.9, .5, 200), "status": "FAILED: TIMEOUT"},
                "pending": {"fitness": (.9, .5, 200), "status": "running eval"},
                "sentinel": {"fitness": (0, 0, 999_999_999)},
                "malformed": {"fitness": ("bad", .5, 200)}}}
    path.write_bytes(pickle.dumps(data))
    assert set(plots.load_candidates(path).gene_id) == {"previous", "complete"}


def test_seed_participates_in_both_fronts(tmp_path):
    baseline = tmp_path / "seed_results.csv"
    baseline.write_text("0.95,0.65,900000\n")
    seed = plots.load_seed(baseline)
    data = pd.concat([models(), pd.DataFrame([{"gene_id": "seed", **seed}])], ignore_index=True)
    assert list(data[plots.pareto_mask(data)].gene_id) == ["seed"]
    assert list(plots.view_front(data, "mAP50").gene_id) == ["seed"]
    baseline.write_text("0,0,999999999\n")
    with pytest.raises(ValueError, match="Invalid baseline"):
        plots.load_seed(baseline)


@pytest.mark.parametrize("family", ["YOLO26n-P2", "D-FINE"])
def test_cli_creates_two_views_and_animations_without_changing_inputs(tmp_path, monkeypatch, family):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    hashes = {}
    for generation in (1, 2):
        data = {"GLOBAL_DATA": {
            row.gene_id: {"status": "completed", "fitness": (row.mAP50, row.mAP50_95, row.params)}
            for row in models().iloc[:generation + 1].itertuples()}}
        # Some controllers also retain the baseline in generation zero.
        data["GLOBAL_DATA"]["seed"] = {"status": "completed", "fitness": (.82, .58, 2_200_000)}
        path = checkpoints / f"global_gen_{generation}.pkl"
        path.write_bytes(pickle.dumps(data))
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    seed = tmp_path / "seed.csv"
    seed.write_text(".82,.58,2200000\n")
    output = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoints", str(checkpoints),
        "--seed_results", str(seed), "--output", str(output), "--family", family])
    plots.main()
    for metric, _ in plots.METRICS:
        with Image.open(output / f"pareto2d_{metric}_final.png") as image:
            assert image.width > 1000 and image.height > 1000
        with Image.open(output / f"pareto2d_{metric}_fronts.gif") as image:
            assert image.n_frames == 2
    assert not list(output.glob("pareto3d*"))
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    csv = pd.read_csv(output / "candidates.csv")
    c = csv.set_index("gene_id").loc["c"]
    assert c.pareto_front and not c.pareto_mAP50_params and not c.pareto_mAP50_95_params
    assert list(csv.gene_id).count("seed") == 1


def test_empty_search_has_clear_error_instead_of_fabricated_front(tmp_path, monkeypatch):
    checkpoint = tmp_path / "global_gen_1.pkl"
    checkpoint.write_bytes(pickle.dumps({"GLOBAL_DATA": {"failed": {"fitness": (0, 0, 999_999_999)}}}))
    output = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoints", str(tmp_path),
        "--seed_results", str(tmp_path / "missing.csv"), "--output", str(output)])
    with pytest.raises(SystemExit, match="No valid completed"):
        plots.main()
    assert not list(output.glob("*.png"))


def test_baseline_only_is_labeled_and_has_no_frontier_line(tmp_path, monkeypatch):
    checkpoint = tmp_path / "global_gen_1.pkl"
    checkpoint.write_bytes(pickle.dumps({"GLOBAL_DATA": {}}))
    seed = tmp_path / "seed.csv"
    seed.write_text(".82,.58,2200000\n")
    output = tmp_path / "plots"
    rendered = []
    savefig = plots.plt.Figure.savefig

    def capture(fig, *args, **kwargs):
        if "pareto2d" in str(args[0]):
            ax = fig.axes[0]
            rendered.append(ax.get_title().lower())
            assert not ax.lines
            assert not any("Pareto members" in label for label in ax.get_legend_handles_labels()[1])
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(plots.plt.Figure, "savefig", capture)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoints", str(tmp_path),
        "--seed_results", str(seed), "--output", str(output), "--no_gif"])
    plots.main()
    assert len(rendered) == 4
    assert all("baseline only" in title for title in rendered)
    assert list(pd.read_csv(output / "candidates.csv").gene_id) == ["seed"]


def test_missing_saved_generations_does_not_fabricate_a_front(tmp_path, monkeypatch):
    output = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoints", str(tmp_path),
        "--output", str(output)])
    with pytest.raises(SystemExit, match="No global_gen_"):
        plots.main()
    assert not output.exists()
