"""Automatic search renewal without real Slurm submissions or training."""
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_yolo11_option_b as chain


@pytest.fixture
def runroot(tmp_path):
    (tmp_path / 'AUTO-ENABLED').touch()
    (tmp_path / 'controller-job.txt').write_text('200')
    (tmp_path / 'run_config.json').write_text(json.dumps({'LLMGE_GENERATIONS': '1001'}))
    return tmp_path


@pytest.fixture
def scheduler(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='300;cluster\n')
    monkeypatch.setattr(chain.subprocess, 'run', run)
    monkeypatch.setattr(chain, 'active', lambda job: False)
    monkeypatch.setattr(chain, 'state', lambda job: 'TIMEOUT')
    monkeypatch.setattr(chain, 'progress', lambda root: (4, 4))
    monkeypatch.setattr(chain, 'pending_children', lambda *a: [])
    return calls


def test_readiness_follows_controller_instead_of_starting_second_server(runroot, scheduler, monkeypatch):
    monkeypatch.setattr(chain, 'active', lambda job: job == '200')
    chain.continue_search(runroot, '100')
    assert len(scheduler) == 1
    assert '--dependency=afterany:200' in scheduler[0]
    assert (runroot / 'chain-job.txt').read_text().strip() == '300'


@pytest.mark.parametrize('controller_state', ['TIMEOUT', 'FAILED'])
def test_expected_timeout_resubmits_saved_run(runroot, scheduler, monkeypatch, controller_state):
    monkeypatch.setattr(chain, 'state', lambda job: controller_state if job == '200' else 'TIMEOUT')
    (runroot / 'evolution-200.log').write_text('RuntimeError: LLM server unavailable; restart the server\n')
    chain.continue_search(runroot, '100')
    assert scheduler == [['bash', 'scripts/resume_yolo11_option_b.sh', '--auto', str(runroot)]]
    assert json.loads((runroot / 'auto-progress.json').read_text())['progress'] == [4, 4]


def test_queued_children_delay_renewal_without_holding_cpu(runroot, scheduler, monkeypatch):
    monkeypatch.setattr(chain, 'pending_children', lambda *a: ['333', '334'])
    chain.continue_search(runroot, '100')
    assert len(scheduler) == 1
    assert '--dependency=afterany:333:334' in scheduler[0]
    assert not (runroot / 'auto-progress.json').exists()


def test_old_server_is_released_before_submitting_replacement(runroot, scheduler, monkeypatch):
    monkeypatch.setattr(chain, 'active', lambda job: job == '100' and ['scancel', '100'] not in scheduler)
    monkeypatch.setattr(chain, 'state', lambda job: 'TIMEOUT' if job == '200' else 'RUNNING')
    chain.continue_search(runroot, '100')
    assert scheduler == [['scancel', '100'], ['bash', 'scripts/resume_yolo11_option_b.sh', '--auto', str(runroot)]]


@pytest.mark.parametrize('controller_state,server_state', [
    ('FAILED', 'TIMEOUT'), ('CANCELLED', 'RUNNING'), ('OUT_OF_MEMORY', 'TIMEOUT'),
    ('TIMEOUT', 'NODE_FAIL'), ('COMPLETED', 'TIMEOUT')])
def test_real_failure_or_manual_cancel_does_not_loop(runroot, scheduler, monkeypatch, controller_state, server_state):
    monkeypatch.setattr(chain, 'state', lambda job: controller_state if job == '200' else server_state)
    with pytest.raises(RuntimeError):
        chain.continue_search(runroot, '100')
    assert scheduler == []


def test_target_and_stop_marker_end_renewal(runroot, scheduler, monkeypatch):
    monkeypatch.setattr(chain, 'progress', lambda root: (1000, 4))
    monkeypatch.setattr(chain, 'active', lambda job: job == '100')
    chain.continue_search(runroot, '100')
    assert scheduler == [['scancel', '100']]
    assert not (runroot / 'AUTO-ENABLED').exists()
    (runroot / 'AUTO-ENABLED').touch()
    (runroot / 'STOP-AUTO').touch()
    scheduler.clear()
    chain.continue_search(runroot, '100')
    assert scheduler == []


def test_no_progress_budget_stops_instead_of_consuming_more_allocations(runroot, scheduler):
    (runroot / 'auto-progress.json').write_text(json.dumps({'progress': [4, 4], 'stalled_allocations': 2}))
    with pytest.raises(RuntimeError, match='Three allocations without checkpoint progress'):
        chain.continue_search(runroot, '100')
    assert scheduler == []


def test_checkpoint_progress_reads_actual_deap_population(tmp_path):
    import run_improved as evolution
    parent = evolution.creator.Individual(['valid'])
    parent.fitness.values = (.8, .4, 1000)
    invalid = evolution.creator.Individual(['invalid'])
    invalid.fitness.values = evolution.INVALID_FITNESS_MAX
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    (checkpoints / 'checkpoint_gen_0.pkl').write_bytes(pickle.dumps({'population': [parent, invalid]}))
    assert chain.progress(tmp_path) == (0, 1)
    # The continuation starts in a fresh process without importing the controller.
    result = subprocess.run([sys.executable, '-c',
        'from pathlib import Path; from scripts.continue_yolo11_option_b import progress; '
        'import sys; print(progress(Path(sys.argv[1])))', str(tmp_path)],
        cwd=Path(chain.__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '(0, 1)'


def test_child_tracking_and_pending_lookup_ignore_other_runs(tmp_path, monkeypatch):
    import run_improved as evolution
    monkeypatch.setenv('YOLO11_PILOT_ROOT', str(tmp_path))
    evolution.record_child_job('333')
    evolution.record_child_job(None)
    evolution.record_child_job('not-a-job')
    (tmp_path / 'evolution-200.log').write_text('Submitted batch job 334\n')
    monkeypatch.setattr(chain.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='333\n334\n999\n'))
    assert chain.pending_children(tmp_path, '200') == ['333', '334']


def test_accounting_retries_and_parses_only_allocation_row(monkeypatch):
    responses = iter(['', '123.batch|FAILED|1:0\n123|TIMEOUT|0:0\n'])
    monkeypatch.setattr(chain.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(responses)))
    monkeypatch.setattr(chain.time, 'sleep', lambda *a: None)
    assert chain.state('123') == 'TIMEOUT'
