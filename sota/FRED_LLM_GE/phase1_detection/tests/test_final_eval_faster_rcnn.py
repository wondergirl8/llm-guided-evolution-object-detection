import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase1_detection import final_eval
from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import final_eval_supervisor as routing


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    task = 'final_000_s01'
    genes = {'img_width': 1280}
    p = {'final_seeds': list(range(1, 31))}
    digest = hashlib.sha256(json.dumps(p, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write(tmp_path / 'protocol.json', p)
    write(tmp_path / 'manifest.json', {'protocol_sha256': digest, 'created_local_date': '2026-11-15'})
    write(tmp_path / 'state.json', {'approved': True, 'generation': 20, 'final_tasks': [task]})
    write(tmp_path / 'tasks' / task / 'task.json', {'kind': 'final', 'genes': genes, 'seed': 1})
    (tmp_path / 'tasks' / task / 'source.py').write_text('frozen source\n')
    source = hashlib.sha256(b'frozen source\n').hexdigest()
    write(tmp_path / 'finalists.json', {'generation': 20, 'protocol_sha256': digest,
          'finalists': [{'genes': genes, 'source_sha256': source}]})
    (tmp_path / 'code').mkdir()
    return tmp_path, task


def test_dispatch_uses_original_frozen_backend_and_seed(run):
    root, task = run
    command = final_eval.faster_rcnn_command(root, task)
    assert command[0] == str(root / 'code/.venv-faster-rcnn/bin/python')
    assert command[-2:] == ['--task', task]
    assert command[2] == 'sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.worker'


@pytest.mark.parametrize('change', ['search_task', 'not_approved', 'generation', 'source', 'protocol', 'seed', 'stop'])
def test_dispatch_rejects_unfrozen_or_incompatible_final_tasks(run, change):
    root, task = run
    if change in ('search_task', 'seed'):
        p = root / 'tasks' / task / 'task.json'
        data = json.loads(p.read_text())
        data.update({'kind': 'search'} if change == 'search_task' else {'seed': 31})
        write(p, data)
    elif change in ('not_approved', 'generation'):
        p = root / 'state.json'
        data = json.loads(p.read_text())
        data.update({'approved': False} if change == 'not_approved' else {'generation': 19})
        write(p, data)
    elif change == 'source':
        (root / 'tasks' / task / 'source.py').write_text('changed')
    elif change == 'protocol':
        write(root / 'protocol.json', {'final_seeds': [1]})
    else:
        (root / 'STOP').touch()
    with pytest.raises(ValueError):
        final_eval.faster_rcnn_command(root, task)


def test_standalone_audited_copy_rejects_early_test_without_package_imports(run, tmp_path):
    root, task = run
    write(root / 'state.json', {'approved': False, 'generation': 0})
    copy = tmp_path / 'standalone/final_eval.py'
    copy.parent.mkdir()
    copy.write_text(Path(final_eval.__file__).read_text())
    result = subprocess.run([sys.executable, str(copy), '--family', 'faster_rcnn_scratch',
                             '--run', str(root), '--task', task], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'approved frozen generation-20' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_routed_final_job_keeps_gpu_budget_resume_task_and_signal(run, monkeypatch):
    root, task = run
    scheduler = SimpleNamespace(run=root, token=lambda label: 'route_' + label, active=lambda: [])
    submissions = []
    def output(command, **kwargs):
        if command[0] == 'sacct': return ''
        submissions.append(command)
        return '123\n'
    monkeypatch.setattr(routing.subprocess, 'check_output', output)
    assert routing.routed_submit(scheduler, 'final_a0', 'worker', task) == '123'
    args, = submissions
    assert '--time=08:00:00' in args and '--signal=B:USR1@180' in args
    assert '--gres=gpu:1' in args
    assert 'control/final_eval.py --family faster_rcnn_scratch' in args[-1]
    assert '--task ' + task in args[-1]


def test_routed_successor_is_prearmed_and_existing_jobs_adopted(run, monkeypatch):
    root, task = run
    scheduler = SimpleNamespace(run=root, token=lambda label: 'route_' + label, active=lambda: [])
    calls = []
    def output(command, **kwargs):
        if command[0] == 'sacct': return ''
        calls.append(command)
        return '124\n'
    monkeypatch.setattr(routing.subprocess, 'check_output', output)
    assert routing.routed_submit(scheduler, 'next', 'supervisor', after='122') == '124'
    assert '--dependency=afterany:122' in calls[0]
    assert 'final_eval_supervisor.py' in calls[0][-1]
    assert not any(a.startswith('--gres') for a in calls[0])
    scheduler.active = lambda: [['125', 'route_final', 'RUNNING', 'node']]
    assert routing.routed_submit(scheduler, 'final', 'worker', task) == '125'
    assert len(calls) == 1


def test_new_frozen_runs_submit_final_eval_module(run, monkeypatch):
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.slurm import Slurm
    root, task = run
    commands = []
    def output(command, **kwargs):
        if command[0] in ('squeue', 'sacct'): return ''
        commands.append(command)
        return '126\n'
    monkeypatch.setattr(subprocess, 'check_output', output)
    assert Slurm(root).submit('final', 'worker', task) == '126'
    assert '-m sota.FRED_LLM_GE.phase1_detection.final_eval --family faster_rcnn_scratch' in commands[0][-1]


@pytest.mark.parametrize('role', ['worker', 'server'])
def test_gpu_submissions_explicitly_exclude_reviewed_unhealthy_nodes(run, monkeypatch, role):
    root, task = run
    write(root / 'tasks' / task / 'task.json', {'kind': 'search'})
    write(root / 'protocol.json', {'server_port': 8500, 'model_path': '/local/llama'})
    scheduler = SimpleNamespace(run=root, token=lambda label: 'route_' + label,
                                active=lambda: [], excluded_nodes=['unhealthy-node'])
    submitted = []
    def output(command, **kwargs):
        if command[0] == 'sacct': return ''
        submitted.append(command)
        return '127\n'
    monkeypatch.setattr(routing.subprocess, 'check_output', output)
    assert routing.routed_submit(scheduler, 'replacement', role, task if role == 'worker' else None) == '127'
    args, = submitted
    assert '--exclude=unhealthy-node' in args
    assert '--time=08:00:00' in args
    if role == 'server':
        assert '--gres=gpu:2' in args and '--mem=160G' in args
        assert '-m uvicorn server:app' in args[-1]
    else:
        assert '--gres=gpu:1' in args
        assert 'v2.worker' in args[-1]
        assert '--task ' + task in args[-1]
