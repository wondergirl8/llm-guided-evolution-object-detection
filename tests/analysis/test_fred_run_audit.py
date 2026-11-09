"""Archive analysis must separate failures, cache aliases, and model introduction."""
import pickle
from types import SimpleNamespace

import pytest

from analysis.scripts import fred_run_audit as audit


class Individual(list):
    def __init__(self, gene, fitness):
        super().__init__([gene])
        self.fitness = SimpleNamespace(values=fitness)


def save_pair(root, generation, genes, population, fingerprint=None):
    directory = root / "checkpoints"
    directory.mkdir(exist_ok=True)
    fingerprint = fingerprint or {"sha256": "fixed"}
    (directory / f"global_gen_{generation}.pkl").write_bytes(pickle.dumps(dict(
        GLOBAL_DATA=genes, GLOBAL_DATA_HIST={}, GLOBAL_DATA_ANCESTRY={},
        protocol_fingerprint=fingerprint)))
    (directory / f"checkpoint_gen_{generation}.pkl").write_bytes(pickle.dumps(dict(
        population=[Individual(g, genes[g]["fitness"]) for g in population], hof=[],
        protocol_fingerprint=fingerprint)))


def test_first_archive_and_population_appearances_are_distinct(tmp_path):
    first = dict(fitness=(.8, .4, 100), architecture_hash="A")
    later = dict(fitness=(.81, .41, 90), architecture_hash="B")
    save_pair(tmp_path, 0, {"seed": first, "intermediate": dict(fitness=None)}, ["seed"])
    save_pair(tmp_path, 1, {"seed": first, "intermediate": dict(fitness=None), "later": later}, ["later"])
    result = audit.collect_checkpoints(tmp_path)
    assert result["first"] == {"seed": 0, "intermediate": 0, "later": 1}
    assert result["population_first"] == {"seed": 0, "later": 1}
    assert result["history"][1]["unscored_ids"] == 1
    assert result["history"][1]["unique_valid_graphs"] == 2
    assert result["history"][1]["best_mAP50_95"] == .41


def test_failed_numeric_fitness_is_not_counted_as_success():
    assert audit.state_of(dict(fitness=(.8, .4, 100), failure_reason="invalid_architecture")) == "failed"
    assert audit.state_of(dict(fitness=(-float("inf"), -float("inf"), float("inf")), failure_reason="timeout")) == "failed"
    assert audit.state_of(dict(fitness=None, status="subbed file")) == "unscored"


@pytest.mark.parametrize("change", ["fitness", "fingerprint"])
def test_inconsistent_snapshots_are_rejected(tmp_path, change):
    first = dict(fitness=(.8, .4, 100), architecture_hash="A")
    save_pair(tmp_path, 0, {"seed": first}, ["seed"])
    next_attrs = dict(first, fitness=(.81, .4, 100)) if change == "fitness" else first
    save_pair(tmp_path, 1, {"seed": next_attrs}, ["seed"], {"sha256": "changed"} if change == "fingerprint" else None)
    with pytest.raises(ValueError, match="Fitness changed|fingerprint mismatch"):
        audit.collect_checkpoints(tmp_path)


def test_missing_population_pair_is_not_silently_skipped(tmp_path):
    save_pair(tmp_path, 0, {"seed": dict(fitness=(.8, .4, 100), architecture_hash="A")}, ["seed"])
    (tmp_path / "checkpoints/checkpoint_gen_0.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        audit.collect_checkpoints(tmp_path)


def test_cache_aliases_resolve_to_one_training_run():
    source = dict(fitness=(.8, .4, 100), architecture_hash="A")
    genes = {"source": source, "alias": dict(source, cache_source="source"),
             "second": dict(source, cache_source="alias")}
    assert {audit.resolve_source(gene, genes) for gene in genes} == {"source"}


@pytest.mark.parametrize("change", ["hash", "fitness", "cycle"])
def test_invalid_cache_provenance_is_rejected(change):
    source = dict(fitness=(.8, .4, 100), architecture_hash="A")
    genes = {"source": source, "alias": dict(source, cache_source="source")}
    if change == "hash":
        genes["alias"]["architecture_hash"] = "B"
    elif change == "fitness":
        genes["alias"]["fitness"] = (.9, .4, 100)
    else:
        genes["source"]["cache_source"] = "alias"
    with pytest.raises(ValueError, match="Cache graph|Cache fitness|Cyclic"):
        audit.resolve_source("alias", genes)
