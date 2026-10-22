"""Focused checks for YOLO26 fitness ingestion and parent eligibility."""

import pickle
import time

import pytest
import yaml

import run_improved as controller
from sota.FRED_LLM_GE.phase1_detection import candidate_runner


def candidate(gene_id, fitness):
    ind = controller.creator.Individual([gene_id])
    ind.fitness.values = fitness
    return ind


@pytest.fixture
def yolo26_state(monkeypatch):
    monkeypatch.setattr(controller, "SEED_FAMILY", "yolo26_p2")
    monkeypatch.setattr(controller, "GLOBAL_DATA", {})
    monkeypatch.setattr(controller, "GLOBAL_DATA_HIST", {})
    monkeypatch.setattr(controller, "GLOBAL_DATA_ANCESTRY", {})


@pytest.mark.parametrize(
    "row, expected_status",
    [
        ("0.000000,0.000000,999999999\n", "FAILED: INVALID FITNESS"),
        ("nan,0.5,1400000\n", "FAILED: INVALID FITNESS"),
        ("0.5,0.4\n", "FAILED: RESULT FILE"),
    ],
)
def test_invalid_csv_fitness_is_recorded_as_failure(
    tmp_path, monkeypatch, yolo26_state, row, expected_status,
):
    results = tmp_path / "results"
    results.mkdir()
    (results / "bad_results.csv").write_text(row)
    monkeypatch.setattr(controller, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(controller, "check_contents_for_error", lambda output: True)
    controller.GLOBAL_DATA["bad"] = {
        "sub_flag": True, "status": "running eval", "fitness": None,
        "local_output": "job done", "results_job": None, "start_time": time.time(),
    }

    controller.check4results("bad")

    record = controller.GLOBAL_DATA["bad"]
    assert record["status"] == expected_status
    assert record["fitness"] == controller.INVALID_FITNESS_MAX
    assert record["failure_reason"]
    assert (results / "bad_results.csv").read_text() == row
    ind = candidate("bad", controller.PLACEHOLDER_FITNESS)
    controller.check_and_update_fitness([ind], loop_delay=0)
    assert controller.valid_detection_population([ind]) == []


def test_valid_csv_fitness_remains_selectable(tmp_path, monkeypatch, yolo26_state):
    results = tmp_path / "results"
    results.mkdir()
    (results / "good_results.csv").write_text("0.800000,0.400000,1400000\n")
    monkeypatch.setattr(controller, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(controller, "check_contents_for_error", lambda output: True)
    controller.GLOBAL_DATA["good"] = {
        "sub_flag": True, "status": "running eval", "fitness": None,
        "local_output": "job done", "results_job": None, "start_time": time.time(),
    }

    controller.check4results("good")

    assert controller.GLOBAL_DATA["good"]["status"] == "completed"
    assert controller.GLOBAL_DATA["good"]["fitness"] == (0.8, 0.4, 1400000.0)


def test_failed_evaluation_job_is_not_selectable(monkeypatch, yolo26_state):
    monkeypatch.setattr(controller, "check_contents_for_error", lambda output: False)
    controller.GLOBAL_DATA["failed_job"] = {
        "sub_flag": True, "status": "running eval", "fitness": None,
        "local_output": "job failed", "results_job": None, "start_time": time.time(),
    }
    ind = candidate("failed_job", controller.PLACEHOLDER_FITNESS)

    controller.check_and_update_fitness([ind], loop_delay=0)

    assert ind.fitness.values == controller.INVALID_FITNESS_MAX
    assert controller.GLOBAL_DATA["failed_job"]["status"] == "FAILED: EVALUATION JOB"
    assert controller.valid_detection_population([ind]) == []


def test_resume_rejects_cached_failure_before_parent_selection(tmp_path, monkeypatch, yolo26_state):
    controller.GLOBAL_DATA.update({
        "good": {"sub_flag": True, "status": "completed",
                 "fitness": (0.8, 0.4, 1400000), "start_time": time.time()},
        "other": {"sub_flag": True, "status": "completed",
                  "fitness": (0.7, 0.3, 1500000), "start_time": time.time()},
        "bad": {"sub_flag": True, "status": "completed",
                "fitness": (0.0, 0.0, 999999999), "start_time": time.time()},
    })
    population = [candidate("good", controller.PLACEHOLDER_FITNESS),
                  candidate("other", controller.PLACEHOLDER_FITNESS),
                  candidate("bad", controller.PLACEHOLDER_FITNESS)]
    checkpoint_dir = tmp_path / "checkpoints"
    global_dir = tmp_path / "global"
    controller.save_checkpoint(
        46, folder_name=checkpoint_dir, global_path=global_dir,
        checkpoint_data={"population": population, "hof": controller.tools.ParetoFront()},
    )
    saved, start_gen, global_data = controller.load_checkpoint(
        folder_name=checkpoint_dir, global_path=global_dir,
    )
    assert start_gen == 47
    assert saved["selection_version"] == controller.YOLO26_SELECTION_VERSION
    assert global_data["selection_version"] == controller.YOLO26_SELECTION_VERSION
    assert saved["coverage_version"] == controller.YOLO26_COVERAGE_VERSION
    assert global_data["coverage_version"] == controller.YOLO26_COVERAGE_VERSION
    monkeypatch.setattr(controller, "GLOBAL_DATA", global_data["GLOBAL_DATA"])
    population = saved["population"]

    controller.check_and_update_fitness(population, loop_delay=0)
    eligible = controller.valid_detection_population(population)

    assert [ind[0] for ind in eligible] == ["good", "other"]
    assert {ind[0] for ind in controller.select_elites(eligible, 2)} == {"good", "other"}
    assert controller.GLOBAL_DATA["bad"]["raw_fitness"] == (0.0, 0.0, 999999999)
    assert controller.GLOBAL_DATA["bad"]["status"] == "FAILED: INVALID FITNESS"


def test_yolo26_selection_uses_map50_95_then_parameters(yolo26_state):
    high_map50 = candidate("high_map50", (0.95, 0.35, 900000))
    high_map5095 = candidate("high_map5095", (0.75, 0.45, 1800000))
    compact_tie = candidate("compact_tie", (0.70, 0.45, 1400000))
    population = [high_map50, high_map5095, compact_tie]

    assert [ind[0] for ind in controller.select_elites(population, 2)] == [
        "compact_tie", "high_map5095",
    ]
    assert [ind[0] for ind in controller.toolbox.select(population, 2)] == [
        "compact_tie", "high_map5095",
    ]


def test_yolo26_checkpoint_rejects_older_selection_versions(tmp_path, yolo26_state):
    checkpoint_dir = tmp_path / "checkpoints"
    global_dir = tmp_path / "global"
    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.save_checkpoint(
            2, folder_name=checkpoint_dir,
            checkpoint_data={"population": [], "hof": controller.tools.ParetoFront(),
                             "selection_version": "older_policy"},
        )
    controller.save_checkpoint(
        1, folder_name=checkpoint_dir, global_path=global_dir,
        checkpoint_data={"population": [], "hof": controller.tools.ParetoFront()},
    )
    checkpoint = checkpoint_dir / "checkpoint_gen_1.pkl"
    with checkpoint.open("rb") as file:
        saved = pickle.load(file)
    saved.pop("selection_version")
    with checkpoint.open("wb") as file:
        pickle.dump(saved, file)

    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.load_checkpoint(folder_name=checkpoint_dir, global_path=global_dir)
    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.save_checkpoint(
            1, folder_name=checkpoint_dir, global_path=global_dir,
            checkpoint_data={"population": [], "hof": controller.tools.ParetoFront()},
        )

    saved["selection_version"] = controller.YOLO26_SELECTION_VERSION
    with checkpoint.open("wb") as file:
        pickle.dump(saved, file)
    global_checkpoint = global_dir / "global_gen_1.pkl"
    with global_checkpoint.open("rb") as file:
        global_state = pickle.load(file)
    global_state.pop("selection_version")
    with global_checkpoint.open("wb") as file:
        pickle.dump(global_state, file)
    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.load_checkpoint(folder_name=checkpoint_dir, global_path=global_dir)
    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.save_checkpoint(
            1, folder_name=checkpoint_dir, global_path=global_dir,
            checkpoint_data={"population": [], "hof": controller.tools.ParetoFront()},
        )


def test_yolo26_checkpoint_rejects_older_coverage_policy(tmp_path, yolo26_state):
    checkpoint_dir = tmp_path / "checkpoints"
    global_dir = tmp_path / "global"
    controller.save_checkpoint(
        1, folder_name=checkpoint_dir, global_path=global_dir,
        checkpoint_data={"population": [], "hof": controller.tools.ParetoFront()},
    )
    checkpoint = checkpoint_dir / "checkpoint_gen_1.pkl"
    with checkpoint.open("rb") as stream:
        saved = pickle.load(stream)
    saved["coverage_version"] = "older_coverage"
    with checkpoint.open("wb") as stream:
        pickle.dump(saved, stream)
    with pytest.raises(ValueError, match="Start a fresh YOLO26 search"):
        controller.load_checkpoint(folder_name=checkpoint_dir, global_path=global_dir)


def test_other_family_checkpoint_keeps_previous_format(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "SEED_FAMILY", "yolo11")
    checkpoint_dir = tmp_path / "checkpoints"
    global_dir = tmp_path / "global"
    controller.save_checkpoint(
        1, folder_name=checkpoint_dir, global_path=global_dir,
        checkpoint_data={"population": [], "hof": controller.tools.ParetoFront()},
    )
    saved, _, global_state = controller.load_checkpoint(
        folder_name=checkpoint_dir, global_path=global_dir,
    )
    assert "selection_version" not in saved
    assert "selection_version" not in global_state


def test_refill_keeps_survivor_and_history(monkeypatch, yolo26_state):
    survivor = candidate("survivor", (0.7, 0.3, 1500000))
    controller.GLOBAL_DATA["survivor"] = {
        "sub_flag": True, "status": "completed", "fitness": survivor.fitness.values,
    }
    controller.GLOBAL_DATA_HIST["older"] = {"fitness": (0.6, 0.2, 1600000)}
    controller.GLOBAL_DATA_ANCESTRY["survivor"] = {"GENES": ["survivor"]}
    monkeypatch.setattr(controller, "GEN_COUNT", 46)
    monkeypatch.setattr(controller, "max_gen_attempts", 2)
    monkeypatch.setattr(controller, "start_population_size", 2)
    monkeypatch.setattr(controller.toolbox, "population",
                        lambda n, llm_model: [candidate("failed", (0, 0, 1)),
                                              candidate("replacement", (0, 0, 1))])
    monkeypatch.setattr(controller, "delayed_creation_check", lambda population: population)

    def evaluate(replacements):
        assert controller.GEN_COUNT == 46
        for ind in replacements:
            fitness = ((0.0, 0.0, 999999999) if ind[0] == "failed"
                       else (0.65, 0.25, 1450000))
            ind.fitness.values = controller.normalize_detection_fitness(fitness)
            controller.GLOBAL_DATA[ind[0]] = {
                "sub_flag": True,
                "status": "FAILED: INVALID FITNESS" if ind[0] == "failed" else "completed",
                "fitness": ind.fitness.values,
            }

    monkeypatch.setattr(controller, "check_and_update_fitness", evaluate)

    eligible = controller.refill_detection_population([survivor], "test-model")

    assert [ind[0] for ind in eligible] == ["survivor", "replacement"]
    assert controller.GEN_COUNT == 46
    assert "survivor" in controller.GLOBAL_DATA
    assert "older" in controller.GLOBAL_DATA_HIST
    assert controller.GLOBAL_DATA_ANCESTRY["survivor"] == {"GENES": ["survivor"]}


def test_refill_stops_after_configured_failed_batches(monkeypatch, yolo26_state):
    survivor = candidate("survivor", (0.7, 0.3, 1500000))
    controller.GLOBAL_DATA["survivor"] = {"status": "completed"}
    controller.GLOBAL_DATA_HIST["older"] = {"fitness": (0.6, 0.2, 1600000)}
    monkeypatch.setattr(controller, "GEN_COUNT", 46)
    monkeypatch.setattr(controller, "max_gen_attempts", 2)
    counter = iter(range(2))
    monkeypatch.setattr(
        controller.toolbox, "population",
        lambda n, llm_model: [candidate(f"failed_{next(counter)}", (0, 0, 1))],
    )
    monkeypatch.setattr(controller, "delayed_creation_check", lambda population: population)

    def evaluate(replacements):
        for ind in replacements:
            ind.fitness.values = controller.INVALID_FITNESS_MAX
            controller.GLOBAL_DATA[ind[0]] = {"status": "FAILED: INVALID FITNESS"}

    monkeypatch.setattr(controller, "check_and_update_fitness", evaluate)

    with pytest.raises(RuntimeError, match="after 2 refill batches"):
        controller.refill_detection_population([survivor], "test-model")

    assert controller.GEN_COUNT == 46
    assert "survivor" in controller.GLOBAL_DATA
    assert "older" in controller.GLOBAL_DATA_HIST


def test_preflight_blocks_invalid_and_duplicate_yolo26_before_submission(
    tmp_path, monkeypatch, yolo26_state,
):
    seed = candidate_runner.FRED_DIR / "seeds" / "seed_yolo26n_p2.yaml"
    monkeypatch.setattr(controller, "SEED_NETWORK", str(seed))
    monkeypatch.setattr(controller, "VARIANT_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "MODEL", "network")
    submitted = []

    def submit(gene_id):
        submitted.append(gene_id)
        controller.GLOBAL_DATA[gene_id]["status"] = "running eval"

    monkeypatch.setattr(controller, "submit_run", submit)
    config = yaml.safe_load(seed.read_text())
    config["head"][8][1] = 3
    for gene_id in ("first", "same", "invalid"):
        controller.GLOBAL_DATA[gene_id] = {
            "sub_flag": True, "status": "subbed file", "fitness": None,
            "start_time": time.time(),
        }
        (tmp_path / f"network_{gene_id}.py").write_text(yaml.safe_dump(config))

    controller.check4model2run("first")
    assert submitted == ["first"]
    assert len(controller.GLOBAL_DATA["first"]["architecture_sha256"]) == 64

    controller.check4model2run("same")
    assert submitted == ["first"]
    assert controller.GLOBAL_DATA["same"]["status"] == "FAILED: DUPLICATE ARCHITECTURE"
    assert "first" in controller.GLOBAL_DATA["same"]["failure_reason"]

    config["head"][8][3][0] = 9999
    (tmp_path / "network_invalid.py").write_text(yaml.safe_dump(config))
    controller.check4model2run("invalid")
    assert submitted == ["first"]
    assert controller.GLOBAL_DATA["invalid"]["status"] == "FAILED: PREFLIGHT"
    assert "out of bounds" in controller.GLOBAL_DATA["invalid"]["failure_reason"]

    controller.reject_detection_fitness(
        "first", None, "Evaluation job failed", status="FAILED: EVALUATION JOB")
    controller.GLOBAL_DATA["retry"] = {
        "sub_flag": True, "status": "subbed file", "fitness": None,
        "start_time": time.time(),
    }
    (tmp_path / "network_retry.py").write_text((tmp_path / "network_first.py").read_text())
    controller.check4model2run("retry")
    assert submitted == ["first", "retry"]
    assert controller.GLOBAL_DATA["retry"]["status"] == "running eval"


def test_preflight_uses_parent_and_hashed_checkpoint_history(
    tmp_path, monkeypatch, yolo26_state,
):
    seed = candidate_runner.FRED_DIR / "seeds" / "seed_yolo26n_p2.yaml"
    monkeypatch.setattr(controller, "SEED_NETWORK", str(seed))
    monkeypatch.setattr(controller, "VARIANT_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "MODEL", "network")
    monkeypatch.setattr(controller, "submit_run", lambda gene_id: pytest.fail("GPU submitted"))
    parent = yaml.safe_load(seed.read_text())
    parent["head"][8][1] = 3
    (tmp_path / "network_parent.py").write_text(yaml.safe_dump(parent))
    (tmp_path / "network_child.py").write_text(yaml.safe_dump(parent))
    controller.GLOBAL_DATA["child"] = {
        "sub_flag": True, "status": "subbed file", "fitness": None,
        "parent_gene_id": "parent", "start_time": time.time(),
    }

    controller.check4model2run("child")
    assert controller.GLOBAL_DATA["child"]["status"] == "FAILED: PREFLIGHT"
    assert "found 0" in controller.GLOBAL_DATA["child"]["failure_reason"]

    parent["head"][9][3][1] = 5
    (tmp_path / "network_child.py").write_text(yaml.safe_dump(parent))
    controller.GLOBAL_DATA["child"]["status"] = "subbed file"
    old_path = tmp_path / "network_old.py"
    old_path.write_text(yaml.safe_dump(parent))
    settings = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)["families"]["yolo26_p2"]
    adapter = candidate_runner.ADAPTERS["yolo26_p2"](settings, candidate_runner.FRED_DIR)
    controller.GLOBAL_DATA_HIST["old"] = {
        "status": "completed", "fitness": (0.8, 0.4, 1400000),
        "architecture_sha256": adapter.architecture_hash(old_path),
    }

    controller.check4model2run("child")
    assert controller.GLOBAL_DATA["child"]["status"] == "FAILED: DUPLICATE ARCHITECTURE"
    assert controller.GLOBAL_DATA_HIST["old"]["architecture_sha256"] == (
        controller.GLOBAL_DATA["child"]["architecture_sha256"]
    )

    (tmp_path / "network_child.py").write_text(seed.read_text())
    controller.GLOBAL_DATA["child"]["status"] = "subbed file"
    controller.check4model2run("child")
    assert controller.GLOBAL_DATA["child"]["status"] == "FAILED: DUPLICATE ARCHITECTURE"
    assert "duplicates seed" in controller.GLOBAL_DATA["child"]["failure_reason"]


def test_yolo26_operators_record_the_source_parent(monkeypatch, yolo26_state):
    generated_ids = iter(("mutated", "left_child", "right_child"))
    monkeypatch.setattr(controller, "DELAYED_CHECK", True)
    monkeypatch.setattr(controller, "LINKED_GENES", {})
    monkeypatch.setattr(controller, "load_yaml", lambda: {"gpu_selection": None, "LLM_GPU": None})
    monkeypatch.setattr(controller, "generate_random_string", lambda length: next(generated_ids))
    monkeypatch.setattr(controller, "submit_bash", lambda *args, **kwargs: (True, "job", None))

    controller.customMutation(candidate("mutation_parent", (0.8, 0.4, 1400000)), "test", 0.2)
    controller.customCrossover(
        candidate("left_parent", (0.8, 0.4, 1400000)),
        candidate("right_parent", (0.7, 0.3, 1500000)), "test",
    )

    assert controller.GLOBAL_DATA["mutated"]["parent_gene_id"] == "mutation_parent"
    assert controller.GLOBAL_DATA["left_child"]["parent_gene_id"] == "left_parent"
    assert controller.GLOBAL_DATA["right_child"]["parent_gene_id"] == "right_parent"
