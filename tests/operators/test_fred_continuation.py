"""Bounded server renewal never turns unknown failures into apparent success."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "sota/FRED_LLM_GE/scripts/continue_run.py"
spec = importlib.util.spec_from_file_location("fred_continuation", PATH)
renew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renew)


@pytest.mark.parametrize("value,expected", [
    ("40:57", 2457), ("00:00", 0), ("1:00:00", 3600), ("08:00:00", 28800),
    ("2-03:04:05", 183845),
])
def test_slurm_duration_forms(value, expected):
    assert renew.seconds(value) == expected


@pytest.mark.parametrize("value", ["N/A", "UNLIMITED", "1:99", "1:00:99", "1-40:57", "-1:00"])
def test_invalid_duration_is_not_guessed(value):
    with pytest.raises(ValueError):
        renew.seconds(value)


def test_known_short_time_parser_error_is_a_refresh():
    log = "RuntimeError: Unrecognized server time remaining from squeue: '40:57'\n"
    assert renew.restart_reason(dict(state="FAILED"), dict(state="RUNNING"), log, True) == "known_MM_SS_server_window_parser_error"


@pytest.mark.parametrize("controller,server,log,progress", [
    ("FAILED", "RUNNING", "RuntimeError: CUDA out of memory", True),
    ("CANCELLED", "RUNNING", "", True),
    ("TIMEOUT", "FAILED", "", True),
    ("TIMEOUT", "TIMEOUT", "", False),
    ("COMPLETED", "RUNNING", "", True),
    ("FAILED", "RUNNING", "RuntimeError: Unrecognized server time remaining from squeue: '40:57'\nValueError: another failure", True),
])
def test_unexpected_failure_or_no_progress_stops(controller, server, log, progress):
    with pytest.raises(RuntimeError):
        renew.restart_reason(dict(state=controller), dict(state=server), log, progress)


def test_clean_pause_and_controller_timeout_can_renew():
    assert renew.restart_reason(dict(state="COMPLETED"), dict(state="RUNNING"),
                                "-- Paused for server refresh --", True) == "clean_server_window_pause"
    assert renew.restart_reason(dict(state="TIMEOUT"), dict(state="TIMEOUT"), "", True)


@pytest.mark.parametrize("log", [
    "ServerWindowClosed: LLM server job 111 is no longer in squeue",
    "RuntimeError: Cannot check LLM server job 111: slurm_load_jobs error: Invalid job id specified",
])
def test_expired_owned_server_can_renew_after_progress(log):
    assert renew.restart_reason(dict(state="FAILED"), dict(state="TIMEOUT"), log, True, server_id="111")
    with pytest.raises(RuntimeError):
        renew.restart_reason(dict(state="FAILED"), dict(state="TIMEOUT"), log, True, server_id="999")
    with pytest.raises(RuntimeError):
        renew.restart_reason(dict(state="FAILED"), dict(state="FAILED"), log, True, server_id="111")


def test_cycle_is_dependent_on_server_start_and_preserves_launcher(monkeypatch, tmp_path):
    config = dict(cycles=[], environment={}, run_root=str(tmp_path))
    monkeypatch.setattr(renew, "verify_inputs", lambda c: dict(generation=2))
    calls = []
    def submit(argv, env):
        calls.append((argv, env))
        return str(100 + len(calls))
    monkeypatch.setattr(renew, "submit", submit)
    renew.submit_cycle(tmp_path / "state.json", config)
    assert calls[0][0] == [str(renew.ROOT / "server.sh")]
    assert "--dependency=after:101" in calls[1][0]
    assert "--requeue" in calls[1][0]
    assert "--open-mode=append" in calls[1][0]
    assert config["cycles"][0]["start_generation"] == 2
    assert config["cycles"][0]["server"] == "101"
    assert config["cycles"][0]["supervisor"] == "102"


def test_supervisor_submission_failure_releases_only_its_server(monkeypatch, tmp_path):
    config = dict(cycles=[], environment={})
    monkeypatch.setattr(renew, "verify_inputs", lambda c: dict(generation=2))
    submitted = []
    def submit(argv, env):
        submitted.append(argv)
        if len(submitted) == 2:
            raise RuntimeError("submission rejected")
        return "123"
    monkeypatch.setattr(renew, "submit", submit)
    cancelled = []
    monkeypatch.setattr(renew, "cancel_owned", cancelled.append)
    with pytest.raises(RuntimeError, match="submission rejected"):
        renew.submit_cycle(tmp_path / "state.json", config)
    assert cancelled == ["123"]


@pytest.mark.parametrize("mode", ["renew", "limit", "complete", "unknown", "not_ready"])
def test_supervisor_lifecycle(monkeypatch, tmp_path, mode):
    config = dict(cycles=[dict(number=1, server="111", controller=None, start_generation=2)],
                  generations=21, max_cycles=1 if mode == "limit" else 3, port=8479,
                  environment={}, run_root=str(tmp_path), status="waiting_for_server_allocation")
    state = tmp_path / "state.json"
    state.write_text(json.dumps(config))
    logs = tmp_path / "run_job_outputs/islands"
    logs.mkdir(parents=True)
    log = "RuntimeError: Unrecognized server time remaining from squeue: '40:57'\n"
    if mode == "complete":
        log = "-- End of Era --"
    if mode == "unknown":
        log = "RuntimeError: unrelated failure"
    (logs / "slurm-222.out").write_text(log)
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setattr(renew, "verify_inputs", lambda c: dict(generation=20 if mode == "complete" else 3))
    monkeypatch.setattr(renew, "job", lambda j: dict(state="TIMEOUT"))
    def ready(*args):
        if mode == "not_ready":
            raise RuntimeError("Server failed before readiness")
        return True
    monkeypatch.setattr(renew, "ready", ready)
    monkeypatch.setattr(renew, "wait_until", lambda predicate, *args: predicate())
    submitted = []
    def submit(argv, env):
        submitted.append((argv, env))
        return "222"
    monkeypatch.setattr(renew, "submit", submit)
    monkeypatch.setattr(renew, "wait_controller", lambda c: dict(state="COMPLETED" if mode == "complete" else "FAILED"))
    cancelled, chained, drained = [], [], []
    monkeypatch.setattr(renew, "cancel_owned", lambda j: cancelled.append(j) if j else None)
    monkeypatch.setattr(renew, "drain_children", drained.append)
    monkeypatch.setattr(renew, "submit_cycle", lambda p, c: chained.append(c.copy()))
    if mode in ("unknown", "not_ready"):
        with pytest.raises(RuntimeError):
            renew.supervise(state, 1)
        assert json.loads(state.read_text())["status"] == "stopped_on_error"
        assert not chained
    else:
        renew.supervise(state, 1)
        assert len(chained) == int(mode == "renew")
        if mode in ("limit", "complete"):
            assert json.loads(state.read_text())["status"] == ("complete" if mode == "complete" else "cycle_limit_reached")
        assert cancelled == ["111"]
    if mode == "not_ready":
        assert not submitted
    else:
        assert submitted[0][0] == [str(tmp_path / "run.sh")]
        assert submitted[0][1]["LLMGE_SERVER_JOB_ID"] == "111"
    if mode == "renew":
        assert drained


def test_active_manual_run_prevents_a_second_chain(monkeypatch):
    monkeypatch.setattr(renew, "queued_jobs", lambda: [("123", "llm_opt")])
    monkeypatch.setattr(renew, "command", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=f"Command={renew.ROOT}/run.sh WorkDir={renew.ROOT}"))
    with pytest.raises(RuntimeError, match="do not overlap"):
        renew.assert_no_other_run()


def test_unrelated_checkout_does_not_block_continuation(monkeypatch):
    monkeypatch.setattr(renew, "queued_jobs", lambda: [("123", "LLMGE01_Server")])
    monkeypatch.setattr(renew, "command", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="Command=/tmp/other-project/server.sh WorkDir=/tmp/other-project"))
    renew.assert_no_other_run()


def test_start_is_read_only_without_submit(monkeypatch, tmp_path, capsys):
    for name in renew.SHARED:
        (tmp_path / name).write_text("unchanged")
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    monkeypatch.setattr(renew, "snapshot", lambda env: dict(generation=2, next_generation=3,
                                                           population=6, fingerprint={"sha256":"fixed"}))
    monkeypatch.setattr(renew, "submit_cycle", lambda *a: pytest.fail("dry run submitted jobs"))
    args = SimpleNamespace(family="spikeyolo", run_tag="spikeyolo_20261101_eot0_trial1",
                           generations=21, cycles=3, port=8479, submit=False)
    renew.start(args)
    assert "Dry run only" in capsys.readouterr().out
    assert not (tmp_path / "fred_runs").exists()


@pytest.mark.parametrize("mode", ["healthy", "loading", "failed", "stale_host", "short_window"])
def test_readiness_checks_the_new_server_and_model(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    log = tmp_path / "run_job_outputs/server/slurm-111.out"
    log.parent.mkdir(parents=True)
    log.write_text("Waiting for application startup." if mode == "loading" else "Application startup complete.")
    (tmp_path / "hostname.log").write_text("wrong-node" if mode == "stale_host" else "gpu-node.pace.gatech.edu")
    monkeypatch.setattr(renew, "job", lambda j: dict(state="FAILED" if mode == "failed" else "RUNNING",
                                                   host="gpu-node", remaining="40:57" if mode == "short_window" else "07:00:00"))
    requested = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def open_url(url, timeout):
        requested.append(url)
        return Response()
    monkeypatch.setattr(renew.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_url))
    if mode in ("failed", "stale_host", "short_window"):
        with pytest.raises(RuntimeError):
            renew.ready("111", 8479)
    else:
        assert renew.ready("111", 8479) == (mode == "healthy")
    if mode == "healthy":
        assert requested == ["http://gpu-node:8479/"]
    if mode == "loading":
        assert not requested


@pytest.mark.parametrize("change", ["shared", "supervisor", "fingerprint"])
def test_code_and_protocol_drift_stop_renewal(monkeypatch, tmp_path, change):
    for name in renew.SHARED:
        (tmp_path / name).write_text("original")
    script = tmp_path / "continue_run.py"
    script.write_text("original supervisor")
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    monkeypatch.setattr(renew, "SCRIPT", script)
    config = dict(shared_hashes={name:renew.digest(tmp_path/name) for name in renew.SHARED},
                  supervisor_sha256=renew.digest(script), environment={}, fingerprint={"sha256":"original"})
    monkeypatch.setattr(renew, "snapshot", lambda env: dict(fingerprint={"sha256":"changed" if change == "fingerprint" else "original"}))
    if change == "shared":
        (tmp_path / "server.sh").write_text("changed")
    if change == "supervisor":
        script.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        renew.verify_inputs(config)


def test_job_completion_uses_root_accounting_record(monkeypatch):
    results = iter([
        SimpleNamespace(stdout="", stderr="slurm_load_jobs error: Invalid job id specified", returncode=1),
        SimpleNamespace(stdout="111|CANCELLED by 123|0:15|\n111.batch|COMPLETED|0:0|\n", returncode=0),
    ])
    monkeypatch.setattr(renew, "command", lambda *args, **kwargs: next(results))
    assert renew.job("111")["state"] == "CANCELLED"


@pytest.mark.parametrize("phase", ["controller_running_or_queued", "releasing_previous_server", "draining_previous_children"])
def test_requeued_supervisor_resumes_existing_controller_without_duplicate(monkeypatch, tmp_path, phase):
    cycle = dict(number=1, server="111", controller="222", supervisor="333", start_generation=2)
    if phase != "controller_running_or_queued":
        cycle["renewal_reason"] = "known_MM_SS_server_window_parser_error"
    config = dict(cycles=[cycle], generations=21, max_cycles=3, status=phase)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(config))
    log = tmp_path / "run_job_outputs/islands/slurm-222.out"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: Unrecognized server time remaining from squeue: '40:57'\n")
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setattr(renew, "verify_inputs", lambda c: dict(generation=4))
    monkeypatch.setattr(renew, "submit", lambda *a: pytest.fail("duplicate controller submitted"))
    monkeypatch.setattr(renew, "ready", lambda *a: pytest.fail("existing controller must not require server readiness"))
    monkeypatch.setattr(renew, "wait_controller", lambda c: dict(state="FAILED"))
    # A prior release may already have cancelled the server before the node failed.
    monkeypatch.setattr(renew, "job", lambda j: dict(state="TIMEOUT" if phase == "controller_running_or_queued" else "CANCELLED"))
    monkeypatch.setattr(renew, "wait_until", lambda predicate, *args: predicate())
    cancelled, chained, drained = [], [], []
    monkeypatch.setattr(renew, "cancel_owned", cancelled.append)
    monkeypatch.setattr(renew, "drain_children", drained.append)
    monkeypatch.setattr(renew, "submit_cycle", lambda p, c: chained.append(c.copy()))
    renew.supervise(state, 1)
    assert cancelled == ["111"]
    assert len(chained) == 1 and len(drained) == 1
    assert chained[0]["cycles"][0]["supervisor_restarts"] == 1
    assert chained[0]["cycles"][0]["last_completed_generation"] == 4


@pytest.mark.parametrize("phase", ["submitting_controller", "submitting_server", "submitting_supervisor"])
def test_interrupted_submission_requires_review_not_a_duplicate(monkeypatch, tmp_path, phase):
    config = dict(cycles=[dict(number=1, server="111", controller=None, start_generation=2)], status=phase)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(config))
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setattr(renew, "verify_inputs", lambda c: {})
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setattr(renew, "submit", lambda *a: pytest.fail("ambiguous submission repeated"))
    cancelled = []
    monkeypatch.setattr(renew, "cancel_owned", cancelled.append)
    with pytest.raises(RuntimeError, match="Interrupted controller submission"):
        renew.supervise(state, 1)
    assert json.loads(state.read_text())["status"] == "stopped_on_error"
    assert "111" in cancelled


def test_excess_supervisor_restarts_stop_the_chain(monkeypatch, tmp_path):
    config = dict(cycles=[dict(number=1, server="111", controller="222")], status="controller_running_or_queued")
    state = tmp_path / "state.json"
    state.write_text(json.dumps(config))
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setattr(renew, "verify_inputs", lambda c: {})
    monkeypatch.setenv("SLURM_RESTART_COUNT", str(renew.MAX_SUPERVISOR_RESTARTS + 1))
    monkeypatch.setattr(renew, "submit", lambda *a: pytest.fail("restart limit ignored"))
    cancelled = []
    monkeypatch.setattr(renew, "cancel_owned", cancelled.append)
    with pytest.raises(RuntimeError, match="restart limit exceeded"):
        renew.supervise(state, 1)
    assert cancelled == ["222", "111"]
    assert json.loads(state.read_text())["status"] == "stopped_on_error"


@pytest.mark.parametrize("status", sorted(renew.STOPPED_STATUSES))
def test_finished_supervisor_cannot_repeat_a_cycle(monkeypatch, tmp_path, status):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(dict(cycles=[dict(number=1)], status=status)))
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setattr(renew, "verify_inputs", lambda *a: pytest.fail("stopped chain restarted"))
    renew.supervise(state, 1)


@pytest.mark.parametrize("successor_recorded", [True, False])
def test_old_supervisor_never_submits_another_successor(monkeypatch, tmp_path, successor_recorded):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(dict(cycles=[dict(number=1), dict(number=2, supervisor="444" if successor_recorded else None)])))
    monkeypatch.setattr(renew, "lock", nullcontext)
    monkeypatch.setattr(renew, "verify_inputs", lambda *a: pytest.fail("old cycle continued"))
    if successor_recorded:
        renew.supervise(state, 1)
    else:
        with pytest.raises(RuntimeError, match="Interrupted next-cycle submission"):
            renew.supervise(state, 1)


def test_submitted_children_do_not_inherit_parent_restart_count(monkeypatch):
    calls = []
    def command(argv, env):
        calls.append(env)
        return SimpleNamespace(stdout="123\n")
    monkeypatch.setattr(renew, "command", command)
    original = dict(SLURM_RESTART_COUNT="2", FRED_RUN_TAG="existing")
    assert renew.submit(["--wrap", "true"], original) == "123"
    assert calls == [dict(FRED_RUN_TAG="existing")]
    assert original["SLURM_RESTART_COUNT"] == "2"


def test_status_surfaces_failed_supervisor_instead_of_stale_running_record(monkeypatch, tmp_path, capsys):
    state = tmp_path / "fred_runs/test/continuation/20261105/state.json"
    state.parent.mkdir(parents=True)
    config = dict(status="controller_running_or_queued", max_cycles=3,
                  cycles=[dict(server="111", controller="222", supervisor="333")])
    state.write_text(json.dumps(config))
    monkeypatch.setattr(renew, "ROOT", tmp_path)
    monkeypatch.setattr(renew, "job", lambda j: dict(state="NODE_FAIL" if j == "333" else "TIMEOUT"))
    renew.show_status("test")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "supervisor_stopped"
    assert result["recorded_status"] == "controller_running_or_queued"
    assert result["jobs"]["supervisor"]["state"] == "NODE_FAIL"
    assert json.loads(state.read_text()) == config
