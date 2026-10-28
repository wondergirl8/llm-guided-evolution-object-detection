"""FRED leaves both elite slots to SPEA2 without changing other problems."""

import run_improved as evolution
from src.cfg import constants_fred_yolo11 as fred_config


def _individual(name, map50, map95, parameters):
    individual = evolution.creator.Individual([name])
    individual.fitness.values = (map50, map95, parameters)
    return individual


def test_fred_uses_spea2_for_both_elites(monkeypatch):
    assert fred_config.KEEP_BEST_ELITE is False
    assert not hasattr(fred_config, "ELITE_OBJECTIVE_INDEX")
    monkeypatch.setattr(evolution, "KEEP_BEST_ELITE", False, raising=False)
    population = [
        _individual("map50", 0.70, 0.30, 2_500_000),
        _individual("map95", 0.60, 0.40, 2_600_000),
        _individual("small", 0.50, 0.20, 1_500_000),
    ]

    expected = [individual[0] for individual in evolution.tools.selSPEA2(population, 2)]
    assert [individual[0] for individual in evolution.select_elites(population, 2)] == expected


def test_other_problems_keep_original_elite_selection(monkeypatch):
    monkeypatch.setattr(evolution, "KEEP_BEST_ELITE", True, raising=False)
    population = [
        _individual("best", 0.70, 0.30, 2_500_000),
        _individual("other", 0.60, 0.40, 1_500_000),
    ]

    assert evolution.select_elites(population, 1)[0][0] == "best"
