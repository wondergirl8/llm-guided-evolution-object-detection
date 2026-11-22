"""Faster R-CNN cache reuse must accept the controller's float CSV values."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import run_improved as evolution
from sota.FRED_LLM_GE.phase1_detection import faster_rcnn_search_space as policy


@pytest.mark.parametrize("params", [32368725, 32368725.0, np.float64(32368725), np.int64(32368725)])
def test_cached_parameter_count_is_written_as_an_integer(tmp_path, params):
    policy.write_cached_fitness(tmp_path, "duplicate", (.743669, .310304, params))
    with (tmp_path / "results/duplicate_results.csv").open() as handle:
        row = next(csv.reader(handle))
    assert row == ["0.743669", "0.310304", "32368725", ""]


@pytest.mark.parametrize("params", [32368725.5, -1, True, np.bool_(True),
                                   float("nan"), float("inf"), -float("inf"), "32368725"])
def test_invalid_cached_parameter_counts_are_not_coerced(tmp_path, params):
    with pytest.raises(ValueError):
        policy.write_cached_fitness(tmp_path, "invalid", (.7, .3, params))
    assert not (tmp_path / "results/invalid_results.csv").exists()


def test_csv_result_can_be_reused_by_duplicate_candidate(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "trained_results.csv").write_text("0.743669,0.310304,32368725,\n")
    logs = tmp_path / "evaluation"
    logs.mkdir()
    (logs / "slurm-123.out").write_text("job done\n")
    monkeypatch.setattr(evolution, "LOCAL", False)
    monkeypatch.setattr(evolution, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(evolution, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(evolution, "SLURM_OUTPUT_PATH", f"{tmp_path}/")
    monkeypatch.setattr(evolution, "GLOBAL_DATA", {
        "trained": {"results_job": "123", "local_output": None,
                    "status": "running eval", "fitness": None, "evaluation_cache_key": "same-key"},
        "duplicate": {"status": "subbed file", "fitness": None}})
    monkeypatch.setattr(evolution, "EVAL_CACHE", {})
    monkeypatch.setattr(evolution, "EVAL_PENDING", {"same-key": "trained"})
    monkeypatch.setattr(evolution, "run_fingerprint", lambda: {"sha256": "frozen"})
    evolution.check4results("trained")
    assert type(evolution.EVAL_CACHE["same-key"]["fitness"][2]) is float
    adapter = SimpleNamespace(
        built_architecture=lambda _: ("same-graph", 32368725),
        evaluation_cache_key=lambda *_: "same-key",
        write_cached_fitness=policy.write_cached_fitness)
    assert evolution.candidate_preflight("duplicate", adapter) is False
    assert evolution.GLOBAL_DATA["duplicate"]["status"] == "completed"
    assert evolution.GLOBAL_DATA["duplicate"]["cache_source"] == "trained"
    assert evolution.EVAL_PENDING == {}
    with (results / "duplicate_results.csv").open() as handle:
        assert next(csv.reader(handle))[2] == "32368725"
