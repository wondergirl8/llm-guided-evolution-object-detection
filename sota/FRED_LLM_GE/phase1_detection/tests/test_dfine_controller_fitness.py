"""D-FINE failure results must not enter evolution selection."""

import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import run_improved as controller
from sota.FRED_LLM_GE.phase1_detection import candidate_runner


@pytest.fixture
def dfine_state(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "SEED_FAMILY", "dfine")
    monkeypatch.setattr(controller, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(controller, "GLOBAL_DATA", {})
    monkeypatch.setattr(controller, "GLOBAL_DATA_HIST", {})
    (tmp_path / "results").mkdir()
    return tmp_path / "results"


def _candidate(gene_id):
    candidate = controller.creator.Individual([gene_id])
    candidate.fitness.values = controller.PLACEHOLDER_FITNESS
    return candidate


def test_runner_failure_is_recorded_and_excluded(dfine_state):
    (dfine_state / "bad_results.csv").write_text("0.000000,0.000000,999999999\n")
    (dfine_state / "good_results.csv").write_text("0.544000,0.167000,850040\n")
    for gene_id, output in (
        ("bad", "ERROR evaluating generated model: TypeError('bad ConvNormAct argument')\njob done"),
        ("good", "job done"),
    ):
        controller.GLOBAL_DATA[gene_id] = {
            "sub_flag": True, "status": "running eval", "fitness": None,
            "local_output": output, "results_job": None, "start_time": time.time(),
        }
    population = [_candidate("bad"), _candidate("good")]

    controller.check_and_update_fitness(population, loop_delay=0)
    eligible = controller.valid_detection_population(population)
    archive = controller.tools.ParetoFront()
    archive.update(eligible)

    assert [candidate[0] for candidate in eligible] == ["good"]
    assert [candidate[0] for candidate in archive] == ["good"]
    assert controller.GLOBAL_DATA["bad"]["status"] == "FAILED: INVALID FITNESS"
    assert controller.GLOBAL_DATA["bad"]["raw_fitness"] == (0.0, 0.0, 999999999.0)
    assert "bad ConvNormAct argument" in controller.GLOBAL_DATA["bad"]["failure_reason"]
    assert "infrastructure_retries" not in controller.GLOBAL_DATA["bad"]
    assert controller.GLOBAL_DATA["good"]["fitness"] == (0.544, 0.167, 850040.0)


def test_cached_failure_is_rejected_before_selection(dfine_state):
    controller.GLOBAL_DATA["old"] = {
        "sub_flag": True, "status": "completed", "fitness": (0.0, 0.0, 999999999),
        "start_time": time.time(),
    }
    candidate = _candidate("old")

    controller.check_and_update_fitness([candidate], loop_delay=0)

    assert candidate.fitness.values == controller.INVALID_FITNESS_MAX
    assert controller.valid_detection_population([candidate]) == []
    assert controller.GLOBAL_DATA["old"]["status"] == "FAILED: INVALID FITNESS"
    assert controller.GLOBAL_DATA["old"]["failure_reason"] == "Invalid cached fitness"


def test_dfine_preflight_rejects_invalid_candidates_before_submission(
    dfine_state, monkeypatch, tmp_path,
):
    monkeypatch.setattr(controller, "VARIANT_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "MODEL", "network")
    submitted = []
    monkeypatch.setattr(controller, "submit_run", submitted.append)
    seed = (candidate_runner.FRED_DIR / "seeds" / "seed_dfine.py").read_text()
    candidates = {
        "bad_constructor": seed.replace(
            "ConvNormAct(in_channels, width, stride=2)",
            "ConvNormAct(in_channels, width, kernel_size=3)", 1,
        ),
        "bad_training": seed.replace(
            "        tokens, coordinates = self._tokens(features)",
            '        if self.training: raise RuntimeError("training-only failure")\n'
            "        tokens, coordinates = self._tokens(features)", 1,
        ),
        "good": seed,
    }
    for gene_id, source in candidates.items():
        controller.GLOBAL_DATA[gene_id] = {"status": "subbed file", "fitness": None}
        (tmp_path / f"network_{gene_id}.py").write_text(source)
        controller.check4model2run(gene_id)

    assert submitted == ["good"]
    for gene_id, cause in (("bad_constructor", "kernel_size"),
                           ("bad_training", "training-only failure")):
        assert controller.GLOBAL_DATA[gene_id]["status"] == "FAILED: PREFLIGHT"
        assert cause in controller.GLOBAL_DATA[gene_id]["failure_reason"]
    assert controller.GLOBAL_DATA["good"]["status"] == "subbed file"


def test_dfine_gpu_job_keeps_slurm_assignment(monkeypatch):
    root = Path(controller.__file__).resolve().parent
    slurm = yaml.safe_load((root / "slurm-config/slurm_config.yaml").read_text())
    assert "export CUDA_VISIBLE_DEVICES=0" not in slurm["python_bash_script"]
    monkeypatch.setenv("FRED_SEED_FAMILY", "dfine")
    settings = runpy.run_path(str(root / "src/cfg/constants_fred_yolo11.py"))
    assert settings["EVAL_RUNLINE"].endswith("--device cuda")


def test_dfine_unusable_cuda_fails_before_loading_data(monkeypatch, tmp_path, capsys):
    def fail_cuda(*args, **kwargs):
        raise RuntimeError("CUDA device busy")

    monkeypatch.setattr(candidate_runner.torch, "empty", fail_cuda)
    monkeypatch.setattr(candidate_runner, "challenging_search",
                        lambda *a: pytest.fail("loaded FRED data before GPU check"))
    with pytest.raises(SystemExit) as exc:
        candidate_runner.run("bad-node", "network", tmp_path, "dfine",
                             {"families": {"dfine": {}}}, tmp_path, "cuda")
    assert exc.value.code == 1
    assert "ERROR infrastructure: CUDA unavailable" in capsys.readouterr().out
    assert not (tmp_path / "results" / "bad-node_results.csv").exists()


def test_dfine_runtime_cuda_busy_is_not_scored_as_bad_model(tmp_path, capsys):
    with pytest.raises(SystemExit):
        candidate_runner._record_run_failure(
            tmp_path, "busy", time.time(), "dfine", "cuda",
            RuntimeError("CUDA devices are busy or unavailable"),
        )
    assert "ERROR infrastructure:" in capsys.readouterr().out
    assert not (tmp_path / "results" / "busy_results.csv").exists()


@pytest.mark.parametrize("output", [
    "ERROR infrastructure: CUDA devices are busy or unavailable",
    "slurmstepd: error: JOB 101 CANCELLED AT 01:00 DUE TO TIME LIMIT",
])
def test_dfine_infrastructure_retries_same_gene_then_accepts_result(
    dfine_state, monkeypatch, output,
):
    record = controller.GLOBAL_DATA["gene"] = {
        "sub_flag": True, "status": "running eval", "fitness": None,
        "results_job": "101", "local_output": output, "start_time": time.time(),
    }
    submitted = []

    def submit(gene_id):
        submitted.append(gene_id)
        record.update(status="running eval", results_job="102", local_output="job done",
                      eval_start_time=time.time())

    monkeypatch.setattr(controller, "submit_run", submit)
    controller.check4results("gene")
    assert submitted == ["gene"]
    assert record["infrastructure_retries"] == 1
    assert record["infrastructure_failures"][0]["job_id"] == "101"

    (dfine_state / "gene_results.csv").write_text("0.544,0.167,850040\n")
    controller.check4results("gene")
    assert record["status"] == "completed"
    assert record["fitness"] == (0.544, 0.167, 850040.0)


def test_dfine_infrastructure_retry_is_bounded(dfine_state, monkeypatch):
    monkeypatch.setattr(controller, "submit_run", lambda _: pytest.fail("extra retry"))
    controller.GLOBAL_DATA["gene"] = {
        "status": "running eval", "fitness": None, "results_job": "102",
        "local_output": "ERROR infrastructure: GPU still unavailable",
        "infrastructure_retries": 1, "infrastructure_failures": [
            {"job_id": "101", "reason": "GPU unavailable"}],
    }
    controller.check4results("gene")
    record = controller.GLOBAL_DATA["gene"]
    assert record["status"] == "FAILED: INFRASTRUCTURE"
    assert [attempt["job_id"] for attempt in record["infrastructure_failures"]] == ["101", "102"]
    assert record["fitness"] == controller.INVALID_FITNESS_MAX


def test_dfine_submission_failure_is_not_left_running(dfine_state, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "GENERATION", 0)
    monkeypatch.setattr(controller, "LOCAL", False)
    monkeypatch.setattr(controller, "RUN_COMMAND", "sbatch")
    monkeypatch.setattr(controller, "load_yaml", lambda: {"python_bash_script": "{}"})
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="scheduler unavailable"))
    (tmp_path / "0").mkdir()
    controller.GLOBAL_DATA["gene"] = {"status": "subbed file", "fitness": None}

    controller.submit_run("gene")

    record = controller.GLOBAL_DATA["gene"]
    assert record["status"] == "FAILED: INFRASTRUCTURE"
    assert "scheduler unavailable" in record["failure_reason"]
