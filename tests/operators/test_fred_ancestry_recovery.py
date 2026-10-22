"""Generation-zero refill lineage survives crossover and checkpoint resume."""

import pytest

import run_improved as evolution
from sota.FRED_LLM_GE.phase1_detection.ancestry_recovery import recover_population_ancestry


def test_refill_records_mutation_ancestry_before_first_crossover(tmp_path, monkeypatch):
    seed = evolution.creator.Individual(["seed"])
    seed.fitness.values = (0.8, 0.4, 2_000_000)
    records = {"seed": {"fitness": seed.fitness.values, "failure_reason": None,
                        "architecture_hash": "seed-graph"}}
    monkeypatch.setattr(evolution, "GLOBAL_DATA", records)
    monkeypatch.setattr(evolution, "GLOBAL_DATA_ANCESTRY", {
        "seed": {"GENES": ["seed"], "MUTATE_TYPE": ["STOCK_SEED"]}})
    monkeypatch.setattr(evolution, "GEN_COUNT", -1)
    monkeypatch.setattr(evolution, "OUTPUT_DIR", str(tmp_path / "operators"))
    monkeypatch.setattr(evolution, "VARIANT_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(evolution, "RUN_ROOT", None)
    monkeypatch.setattr(evolution, "FRED_MAX_REFILL_ATTEMPTS", 1)
    monkeypatch.setattr(evolution, "load_candidate_policy", lambda _: object())
    monkeypatch.setattr(evolution, "load_yaml", lambda: {
        "llm_bash_script": "{}\n", "gpu_selection": ""})
    monkeypatch.setattr(evolution, "generate_template", lambda *args: ("prompt", "1"))
    monkeypatch.setattr(evolution, "delayed_mutate_check", lambda children: children)
    monkeypatch.setattr(evolution, "log_failure_counts", lambda *_: None)

    def mutate(individual, llm_model):
        assert evolution.GEN_COUNT == 0
        evolution.write_bash_script(
            llm_model=llm_model, job_name="mutation_operation",
            input_filename_x=str(tmp_path / "models" / "network_seed.py"),
            output_filename=str(tmp_path / "models" / "network_child.py"),
            gpu="", python_file="src/llm_mutation.py", top_p=0.1, temperature=0.1)
        individual[0] = "child"
        records["child"] = {"fitness": None, "failure_reason": None,
                            "architecture_hash": "child-graph"}
        return individual

    def evaluate(children):
        for child in children:
            records[child[0]]["fitness"] = seed.fitness.values
            child.fitness.values = seed.fitness.values

    monkeypatch.setattr(evolution.toolbox, "mutate", mutate)
    monkeypatch.setattr(evolution, "check_and_update_fitness", evaluate)
    population = evolution.refill_initial_population([seed], 2, "mixtral")
    assert [individual[0] for individual in population] == ["seed", "child"]
    assert evolution.GLOBAL_DATA_ANCESTRY["child"] == {
        "GENES": ["seed", "child"], "MUTATE_TYPE": ["STOCK_SEED", "1"]}
    evolution.update_ancestry("grandchild", "child", evolution.GLOBAL_DATA_ANCESTRY,
                              gene_id_parent2="seed")
    assert evolution.GLOBAL_DATA_ANCESTRY["grandchild"]["MUTATE_TYPE"][-1] == "CrossOver"


def _saved_mutation(root, parent, child, template_name):
    operators = root / "operators" / "0"
    models = root / "models"
    operators.mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    command = (f"uv run python src/llm_mutation.py {models / f'network_{parent}.py'} "
               f"{models / f'network_{child}.py'} {operators / f'{child}_model.txt'} "
               "--top_p 0.1 --temperature 0.2")
    (operators / f"{child}.sh").write_text(f"#!/bin/bash\n{command}\n")
    template = root / "templates" / "Normal" / f"{template_name}.txt"
    rules = root / "templates" / "rules.txt"
    (operators / f"{child}_model.txt").write_text(template.read_text() + "\n" + rules.read_text())


def test_resume_recovers_exact_lineage_for_nested_refill_mutations(tmp_path, monkeypatch):
    normal = tmp_path / "templates" / "Normal"
    normal.mkdir(parents=True)
    (normal / "1.txt").write_text("change channels")
    (normal / "2.txt").write_text("change repeats")
    (tmp_path / "templates" / "rules.txt").write_text("keep Detect last")
    _saved_mutation(tmp_path, "seed", "refill_a", "1")
    _saved_mutation(tmp_path, "refill_a", "refill_b", "2")
    original = {"seed": {"GENES": ["seed"], "MUTATE_TYPE": ["STOCK_SEED"]}}

    recovered, repaired = recover_population_ancestry(
        [["seed"], ["refill_b"]], original, tmp_path, tmp_path / "operators",
        tmp_path / "models", "network", "templates/Normal/*.txt", "templates/rules.txt")

    assert repaired == ["refill_a", "refill_b"]
    assert recovered["refill_b"] == {
        "GENES": ["seed", "refill_a", "refill_b"],
        "MUTATE_TYPE": ["STOCK_SEED", "1", "2"]}
    assert original == {"seed": {"GENES": ["seed"], "MUTATE_TYPE": ["STOCK_SEED"]}}
    monkeypatch.setattr(evolution, "GLOBAL_DATA_ANCESTRY", recovered)
    monkeypatch.setattr(evolution, "load_yaml", lambda: {
        "llm_bash_script": "{}\n", "gpu_selection": ""})
    evolution.write_bash_script(
        llm_model="mixtral", job_name="crossover_operation",
        input_filename_x=str(tmp_path / "models" / "network_refill_b.py"),
        input_filename_y=str(tmp_path / "models" / "network_seed.py"),
        output_filename=str(tmp_path / "models" / "network_crossover.py"),
        gpu="", python_file="src/llm_crossover.py", top_p=0.1, temperature=0.1)
    assert evolution.GLOBAL_DATA_ANCESTRY["crossover"]["MUTATE_TYPE"] == [
        "STOCK_SEED", "1", "2", "CrossOver"]
    assert recover_population_ancestry(
        [["refill_b"]], recovered, tmp_path, tmp_path / "operators",
        tmp_path / "models", "network", "templates/Normal/*.txt",
        "templates/rules.txt") == (recovered, [])


def test_resume_rejects_unverifiable_refill_lineage(tmp_path):
    normal = tmp_path / "templates" / "Normal"
    normal.mkdir(parents=True)
    (normal / "1.txt").write_text("change channels")
    (tmp_path / "templates" / "rules.txt").write_text("keep Detect last")
    _saved_mutation(tmp_path, "seed", "refill", "1")
    (tmp_path / "operators" / "0" / "refill_model.txt").write_text("wrong prompt")
    original = {"seed": {"GENES": ["seed"], "MUTATE_TYPE": ["STOCK_SEED"]}}

    with pytest.raises(RuntimeError, match="Cannot identify one saved mutation template"):
        recover_population_ancestry(
            [["refill"]], original, tmp_path, tmp_path / "operators",
            tmp_path / "models", "network", "templates/Normal/*.txt", "templates/rules.txt")
    assert list(original) == ["seed"]
