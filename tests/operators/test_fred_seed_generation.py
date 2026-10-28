"""The FRED stock seed enters GE as a real, unmodified candidate."""

import importlib.util
from pathlib import Path

import run_improved as evolution


def test_stock_seed_is_a_candidate_without_changing_source(tmp_path, monkeypatch):
    source = tmp_path / "seed.yaml"
    source.write_text("nc: 1\n")
    monkeypatch.setattr(evolution, "SEED_FAMILY", "yolo11", raising=False)
    monkeypatch.setattr(evolution, "SEED_NETWORK", str(source))
    monkeypatch.setattr(evolution, "VARIANT_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(evolution, "GLOBAL_DATA", {})
    monkeypatch.setattr(evolution, "GLOBAL_DATA_ANCESTRY", {})
    population = evolution.add_stock_seed_to_initial_population([])
    assert [individual[0] for individual in population] == ["seed"]
    assert (tmp_path / "models/network_seed.py").read_bytes() == source.read_bytes()
    assert evolution.GLOBAL_DATA["seed"]["status"] == "subbed file"
    assert evolution.GLOBAL_DATA_ANCESTRY["seed"]["MUTATE_TYPE"] == ["STOCK_SEED"]


def test_non_fred_population_is_unchanged(monkeypatch):
    monkeypatch.delattr(evolution, "SEED_FAMILY", raising=False)
    existing = [evolution.creator.Individual(["original"])]
    assert evolution.add_stock_seed_to_initial_population(existing) is existing


def test_tagged_fred_candidates_stay_in_the_tagged_run(monkeypatch):
    monkeypatch.setenv("FRED_SEED_FAMILY", "yolo11")
    monkeypatch.setenv("FRED_RUN_TAG", "yolo11_20261019_eot0_seedtest")
    monkeypatch.setenv("FRED_EVENTS", "raw_w10000_polarity")
    path = Path(evolution.__file__).parent / "src/cfg/constants_fred_yolo11.py"
    spec = importlib.util.spec_from_file_location("fred_test_constants", path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    assert constants.SOTA_ROOT == str(Path(constants.RUN_ROOT) / "candidates")
    assert constants.VARIANT_DIR == str(Path(constants.SOTA_ROOT) / "models")


def test_generation_zero_checkpoint_resumes_at_generation_one(tmp_path, monkeypatch):
    seed = evolution.creator.Individual(["seed"])
    seed.fitness.values = (.4, .2, 2_000_000)
    monkeypatch.setattr(evolution, "run_fingerprint", lambda: None)
    monkeypatch.setattr(evolution, "population", [seed], raising=False)
    monkeypatch.setattr(evolution, "hof", evolution.tools.ParetoFront(), raising=False)
    monkeypatch.setattr(evolution, "GLOBAL_DATA", {
        "seed": {"status": "completed", "fitness": seed.fitness.values}})
    monkeypatch.setattr(evolution, "GLOBAL_DATA_HIST", {})
    monkeypatch.setattr(evolution, "GLOBAL_DATA_ANCESTRY", {
        "seed": {"GENES": ["seed"], "MUTATE_TYPE": ["STOCK_SEED"]}})
    checkpoint_dir = tmp_path / "checkpoints"
    evolution.save_checkpoint(0, folder_name=checkpoint_dir, global_path=checkpoint_dir)
    saved, next_generation, global_data = evolution.load_checkpoint(
        folder_name=checkpoint_dir, global_path=checkpoint_dir)
    assert next_generation == 1
    assert saved["population"][0][0] == "seed"
    assert global_data["GLOBAL_DATA"]["seed"]["fitness"] == (.4, .2, 2_000_000)
