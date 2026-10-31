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
    monkeypatch.setattr(controller, "SEED_NETWORK",
                        str(candidate_runner.FRED_DIR / "seeds" / "seed_dfine.py"))
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
        "unchanged": seed.replace("class EventBackbone", "\nclass EventBackbone", 1),
        "good": seed.replace(
            "self.max_memory_tokens = max(256, 4 * num_queries)",
            "self.max_memory_tokens = max(384, 4 * num_queries)", 1,
        ),
    }
    for gene_id, source in candidates.items():
        controller.GLOBAL_DATA[gene_id] = {"status": "subbed file", "fitness": None}
        (tmp_path / f"network_{gene_id}.py").write_text(source)
        controller.check4model2run(gene_id)

    assert submitted == ["good"]
    assert controller.GLOBAL_DATA["unchanged"]["status"] == "FAILED: PROPOSAL"
    assert "unchanged" in controller.GLOBAL_DATA["unchanged"]["failure_reason"]
    for gene_id, cause in (("bad_constructor", "kernel_size"),
                           ("bad_training", "training-only failure")):
        assert controller.GLOBAL_DATA[gene_id]["status"] == "FAILED: PREFLIGHT"
        assert cause in controller.GLOBAL_DATA[gene_id]["failure_reason"]
    assert controller.GLOBAL_DATA["good"]["status"] == "subbed file"

    controller.GLOBAL_DATA_HIST["good"] = controller.GLOBAL_DATA.pop("good")
    controller.GLOBAL_DATA["duplicate"] = {"status": "subbed file", "fitness": None}
    (tmp_path / "network_duplicate.py").write_text(candidates["good"])
    controller.check4model2run("duplicate")
    assert submitted == ["good"]
    assert controller.GLOBAL_DATA["duplicate"]["status"] == "FAILED: DUPLICATE ARCHITECTURE"
    assert "good" in controller.GLOBAL_DATA["duplicate"]["failure_reason"]


def test_failed_dfine_generation_records_proposal_reason(dfine_state, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "DELAYED_CHECK", True)
    monkeypatch.setattr(controller, "SLURM_OUTPUT_PATH", str(tmp_path) + "/")
    log = tmp_path / "evolution" / "slurm-123.out"
    log.parent.mkdir()
    log.write_text("ERROR proposal: proposal leaves the architecture class unchanged\n")
    controller.GLOBAL_DATA["bad"] = {
        "sub_flag": True, "job_id": "123", "status": "DELAYED_CHECK",
        "fitness": None, "start_time": time.time(),
    }
    monkeypatch.setattr(controller, "check4job_completion", lambda *args, **kwargs: False)

    controller.delayed_creation_check([_candidate("bad")])

    record = controller.GLOBAL_DATA["bad"]
    assert record["status"] == "FAILED: PROPOSAL"
    assert "leaves the architecture class unchanged" in record["failure_reason"]
    assert str(log) in record["failure_reason"]


def test_exhausted_dfine_llm_timeout_is_recorded_as_infrastructure(
    dfine_state, monkeypatch, tmp_path,
):
    monkeypatch.setattr(controller, 'SLURM_OUTPUT_PATH', str(tmp_path) + '/')
    log = tmp_path / 'evolution' / 'slurm-124.out'
    log.parent.mkdir()
    log.write_text('RuntimeError: Local LLM request failed after 2 attempt(s) '
                   '(connect timeout 10s, read timeout 600s)\n')
    controller.GLOBAL_DATA['gene'] = {'job_id': '124', 'fitness': None}

    controller.record_dfine_proposal_failure('gene')

    record = controller.GLOBAL_DATA['gene']
    assert record['status'] == 'FAILED: INFRASTRUCTURE'
    assert 'failed after 2 attempt' in record['failure_reason']
    assert str(log) in record['failure_reason']


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


def test_dfine_queue_wait_does_not_consume_runtime(dfine_state, monkeypatch):
    from src import fred_runtime

    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'EVAL_QUEUE_TIMEOUT_SECONDS', 0)
    record = controller.GLOBAL_DATA['queued'] = {
        'sub_flag': True, 'status': 'running eval', 'fitness': None,
        'results_job': '42', 'local_output': None,
        'start_time': 1.0, 'eval_start_time': 1.0, 'eval_submitted_at': 1.0,
    }
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: {
        'state': 'PENDING', 'runtime_seconds': 0})
    assert controller.evaluation_wait_decision('queued', 10, 2000.0) is None
    assert record['eval_phase'] == 'PENDING'
    assert record['eval_queue_seconds'] == 1999.0
    assert record.get('eval_running_at') is None

    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: {
        'state': 'RUNNING', 'runtime_seconds': 9})
    assert controller.evaluation_wait_decision('queued', 10, 2009.0) is None
    assert record['eval_running_at'] == 2000.0
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: {
        'state': 'RUNNING', 'runtime_seconds': 11})
    assert controller.evaluation_wait_decision('queued', 10, 2011.0) == (
        'FAILED: TIMEOUT', 'Evaluation ran for 11s (limit 10s)', True)


def test_dfine_queue_and_terminal_have_separate_limits(dfine_state, monkeypatch):
    from src import fred_runtime

    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'EVAL_QUEUE_TIMEOUT_SECONDS', 100)
    monkeypatch.setattr(controller, 'EVAL_TERMINAL_GRACE_SECONDS', 30)
    controller.GLOBAL_DATA['gene'] = {
        'results_job': '42', 'start_time': 0.0,
        'eval_start_time': 0.0, 'eval_submitted_at': 0.0,
    }
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: {
        'state': 'PENDING', 'runtime_seconds': 0})
    assert controller.evaluation_wait_decision('gene', 10, 101.0)[0] == 'FAILED: QUEUE TIMEOUT'

    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: {
        'state': 'COMPLETED', 'runtime_seconds': 8})
    assert controller.evaluation_wait_decision('gene', 10, 110.0) is None
    decision = controller.evaluation_wait_decision('gene', 10, 141.0)
    assert decision[0] == 'FAILED: EVALUATION JOB' and decision[2] is False


def test_dfine_requeue_starts_a_new_queue_clock(dfine_state, monkeypatch):
    from src import fred_runtime

    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'EVAL_QUEUE_TIMEOUT_SECONDS', 20)
    record = controller.GLOBAL_DATA['gene'] = {
        'results_job': '42', 'start_time': 0.0,
        'eval_start_time': 0.0, 'eval_submitted_at': 0.0,
    }
    state = {'state': 'RUNNING', 'runtime_seconds': 50}
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: state)
    assert controller.evaluation_wait_decision('gene', 100, 100.0) is None
    state.update(state='PENDING', runtime_seconds=0)
    assert controller.evaluation_wait_decision('gene', 100, 110.0) is None
    assert record['eval_queue_seconds'] == 0
    assert record['eval_requeues'] == 1
    assert controller.evaluation_wait_decision('gene', 100, 120.0) is None
    assert record['eval_queue_seconds'] == 10


def test_dfine_scheduler_error_never_cancels_unverified_job(dfine_state, monkeypatch):
    from src import fred_runtime

    monkeypatch.setattr(controller, 'LOCAL', False)
    controller.GLOBAL_DATA['gene'] = {
        'results_job': '42', 'start_time': 0.0, 'eval_start_time': 0.0}
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing',
                        lambda job_id: (_ for _ in ()).throw(RuntimeError('scheduler offline')))
    assert controller.evaluation_wait_decision('gene', 10, 5000.0) is None
    assert controller.GLOBAL_DATA['gene']['eval_scheduler_error'] == 'scheduler offline'


def test_controller_waits_through_queue_then_cancels_running_timeout(dfine_state, monkeypatch):
    from src import fred_runtime

    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'EVAL_QUEUE_TIMEOUT_SECONDS', 0)
    controller.GLOBAL_DATA['gene'] = {
        'sub_flag': True, 'status': 'running eval', 'fitness': None,
        'results_job': '42', 'local_output': None,
        'start_time': time.time() - 1000,
        'eval_start_time': time.time() - 1000,
    }
    states = iter(({'state': 'PENDING', 'runtime_seconds': 0},
                   {'state': 'RUNNING', 'runtime_seconds': 11}))
    monkeypatch.setattr(fred_runtime, 'slurm_job_timing', lambda job_id: next(states))
    monkeypatch.setattr(controller, 'check4results', lambda gene_id: None)
    monkeypatch.setattr(controller, 'persist_initial_population_checkpoint', lambda *a: None)
    monkeypatch.setattr(controller, 'print_job_info', lambda record: None)
    cancelled = []
    monkeypatch.setattr(controller, 'cancel_eval_job', lambda gene_id, reason: cancelled.append(gene_id))
    individual = _candidate('gene')

    controller.check_and_update_fitness([individual], timeout=10, loop_delay=0)

    assert cancelled == ['gene']
    assert controller.GLOBAL_DATA['gene']['status'] == 'FAILED: TIMEOUT'
    assert individual.fitness.values == controller.INVALID_FITNESS_MAX
