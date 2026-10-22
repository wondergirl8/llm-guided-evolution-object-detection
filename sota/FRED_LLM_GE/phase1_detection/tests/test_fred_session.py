"""Offline regressions for paired checkpoints, bounded workers and owned cleanup."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import run_improved as controller
from src import fred_session as session
from src import fred_runtime


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_population_and_global_resume_as_one_committed_pair(tmp_path, monkeypatch, family):
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'SEED_FAMILY', family)
    individual = controller.creator.Individual(['candidate'])
    individual.fitness.values = (.7, .4, 1000)
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {'candidate': {'fitness': (.7, .4, 1000), 'status': 'completed'}})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_HIST', {'old': {'status': 'completed'}})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_ANCESTRY', {'candidate': {'GENES': ['seed', 'candidate']}})
    for generation in (0, 1):
        controller.save_checkpoint(generation, tmp_path, tmp_path,
            {'population': [individual], 'hof': controller.tools.ParetoFront()})
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    # A crash writes only one half of the next generation, with no commit marker.
    (tmp_path / 'checkpoint_gen_2.pkl').write_bytes(b'INTERRUPTED SYNTHETIC CHECKPOINT')
    data, next_generation, state = controller.load_checkpoint(tmp_path, global_path=tmp_path)
    assert next_generation == 2 and data['population'][0].fitness.values == (.7, .4, 1000)
    assert state['GLOBAL_DATA_HIST']['old']['status'] == 'completed'
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == value for name, value in before.items())
    assert json.loads((tmp_path / 'resume_gen_0.json').read_text())['stage'] == 'initial_population_pending_evaluation'
    if family == 'yolo26_p2':
        assert data['selection_version'] == controller.YOLO26_SELECTION_VERSION
        assert data['coverage_version'] == controller.YOLO26_COVERAGE_VERSION
    (tmp_path / 'global_gen_1.pkl').write_bytes(b'CORRUPTED SYNTHETIC DATA')
    with pytest.raises(RuntimeError, match='artifacts preserved'):
        controller.load_checkpoint(tmp_path, global_path=tmp_path)



def test_uncommitted_initial_data_is_never_treated_as_a_new_run(tmp_path):
    (tmp_path / 'global_gen_0.pkl').write_bytes(pickle.dumps({}))
    with pytest.raises(RuntimeError, match='No valid committed'):
        session.latest_pair(tmp_path, tmp_path, 'dfine')



def test_empty_run_can_start_and_committed_pairs_cannot_be_overwritten(tmp_path):
    assert session.latest_pair(tmp_path, tmp_path, 'dfine') == (None, None, None)
    pop = {'population': [], 'hof': []}
    state = {'GLOBAL_DATA': {}, 'GLOBAL_DATA_HIST': {}, 'GLOBAL_DATA_ANCESTRY': {}}
    session.save_pair(tmp_path, tmp_path, 0, pop, state, 'dfine')
    with pytest.raises(RuntimeError, match='overwrite committed'):
        session.save_pair(tmp_path, tmp_path, 0, pop, state, 'dfine')
    with pytest.raises(RuntimeError, match='family mismatch'):
        session.latest_pair(tmp_path, tmp_path, 'yolo26_p2')


def test_dfine_resume_reuses_recorded_evaluation_job(tmp_path, monkeypatch):
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'dfine')
    individual = controller.creator.Individual(['candidate'])
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {
        'candidate': {'sub_flag': True, 'status': 'subbed file',
                      'fitness': None, 'start_time': time.time()}})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_ANCESTRY', {})
    controller.save_checkpoint(0, tmp_path, tmp_path,
                               {'population': [individual], 'hof': controller.tools.ParetoFront()})
    session.eval_submission(tmp_path, 'candidate', state='intent')
    session.eval_submission(tmp_path, 'candidate', state='submitted', job_id='42')

    data, next_generation, state = controller.load_checkpoint(tmp_path, global_path=tmp_path)

    assert next_generation == 1
    assert data['population'][0][0] == 'candidate'
    assert state['GLOBAL_DATA']['candidate']['status'] == 'running eval'
    assert state['GLOBAL_DATA']['candidate']['results_job'] == '42'
    with pytest.raises(RuntimeError, match='refusing duplicate'):
        session.eval_submission(tmp_path, 'candidate', state='intent')
    monkeypatch.setattr(controller, 'GLOBAL_DATA', state['GLOBAL_DATA'])
    monkeypatch.setattr(controller, 'submit_run',
                        lambda gene_id: pytest.fail('resubmitted a recorded evaluation'))
    def finish(gene_id):
        controller.GLOBAL_DATA[gene_id].update(status='completed', fitness=(.5, .1, 1000))
    monkeypatch.setattr(controller, 'check4results', finish)
    individual.fitness.values = controller.PLACEHOLDER_FITNESS
    controller.check_and_update_fitness([individual], loop_delay=0)
    assert individual.fitness.values == (.5, .1, 1000)


def test_dfine_submission_records_job_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'RUN_COMMAND', 'sbatch')
    monkeypatch.setattr(controller, 'CHECKPOINT_FOLDER', tmp_path)
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(controller, 'GENERATION', 0)
    monkeypatch.setattr(controller, 'EVAL_RUNLINE', 'echo {} {}')
    monkeypatch.setattr(controller, 'load_yaml', lambda: {'python_bash_script': '{}'})
    (tmp_path / '0').mkdir()
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {
        'candidate': {'sub_flag': True, 'status': 'subbed file', 'fitness': None}})
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='Submitted batch job 42', stderr=''))

    controller.submit_run('candidate')

    assert session.eval_submission(tmp_path, 'candidate')['job_id'] == '42'
    assert controller.GLOBAL_DATA['candidate']['results_job'] == '42'


def test_dfine_resume_resubmits_only_after_confirmed_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'dfine')
    individual = controller.creator.Individual(['candidate'])
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {
        'candidate': {'sub_flag': True, 'status': 'subbed file',
                      'fitness': None, 'start_time': time.time()}})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_ANCESTRY', {})
    controller.save_checkpoint(0, tmp_path, tmp_path,
                               {'population': [individual], 'hof': controller.tools.ParetoFront()})
    session.eval_submission(tmp_path, 'candidate', state='intent')
    session.eval_submission(tmp_path, 'candidate', state='submitted', job_id='42')
    session.mark_cancelled_evaluations(tmp_path, ['42'])

    _, _, state = controller.load_checkpoint(tmp_path, global_path=tmp_path)

    assert state['GLOBAL_DATA']['candidate']['status'] == 'subbed file'
    session.eval_submission(tmp_path, 'candidate', state='intent')
    assert session.eval_submission(tmp_path, 'candidate')['state'] == 'intent'


def test_unknown_submission_identity_fails_closed_on_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'dfine')
    individual = controller.creator.Individual(['candidate'])
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {
        'candidate': {'sub_flag': True, 'status': 'subbed file',
                      'fitness': None, 'start_time': time.time()}})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_ANCESTRY', {})
    controller.save_checkpoint(0, tmp_path, tmp_path,
                               {'population': [individual], 'hof': controller.tools.ParetoFront()})
    session.eval_submission(tmp_path, 'candidate', state='intent')
    session.mark_cancelled_evaluations(tmp_path, [])

    _, _, state = controller.load_checkpoint(tmp_path, global_path=tmp_path)

    assert state['GLOBAL_DATA']['candidate']['status'] == 'FAILED: INFRASTRUCTURE'
    with pytest.raises(RuntimeError, match='refusing duplicate'):
        session.eval_submission(tmp_path, 'candidate', state='intent')


def test_slurm_status_retries_timeout_but_not_terminal_job_state(monkeypatch):
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setattr(fred_runtime.time, 'sleep', lambda seconds: None)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(args, 10)
        return SimpleNamespace(returncode=0, stdout='JobState=RUNNING', stderr='')
    monkeypatch.setattr(fred_runtime.subprocess, 'run', run)
    fred_runtime.assert_llm_job_active()
    assert len(calls) == 2

    calls.clear()
    monkeypatch.setattr(fred_runtime.subprocess, 'run', lambda *a, **k: (
        calls.append(a) or SimpleNamespace(returncode=0, stdout='JobState=FAILED', stderr='')))
    with pytest.raises(RuntimeError, match='is FAILED'):
        fred_runtime.assert_llm_job_active()
    assert len(calls) == 1

    calls.clear()
    def timed_out(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 10)
    monkeypatch.setattr(fred_runtime.subprocess, 'run', timed_out)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        fred_runtime.assert_llm_job_active()
    assert len(calls) == 3


def test_unverified_cleanup_blocks_a_new_controller(tmp_path, monkeypatch):
    from src import fred_recovery_controller as recovery
    from src.cfg import constants as cfg
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(cfg, 'SEED_FAMILY', 'dfine')
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    (tmp_path / 'five-hour-deadline-123.txt').write_text(str(int(time.time() + 3600)))
    folder = tmp_path / 'fred_dfine_checkpoints'
    folder.mkdir()
    pending = folder / '.cleanup_pending'
    pending.write_text('previous session')
    monkeypatch.setattr(recovery.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(controller, 'wait_for_llm_server', lambda **kw: True)
    def fail_cleanup(*args, **kwargs):
        raise RuntimeError('squeue unavailable')
    monkeypatch.setattr(session, 'cancel_owned_jobs', fail_cleanup)
    monkeypatch.setattr(recovery.subprocess, 'Popen',
                        lambda *args, **kwargs: pytest.fail('started before cleanup was verified'))

    with pytest.raises(RuntimeError, match='squeue unavailable'):
        recovery.main()

    assert pending.exists()



def test_deadline_is_bound_to_the_actual_server_allocation(tmp_path):
    (tmp_path / 'five-hour-deadline-123.txt').write_text('18000\n')
    assert session.server_deadline(tmp_path, '123') == 18000
    with pytest.raises(ValueError, match='server job ID'):
        session.server_deadline(tmp_path, '../123')



def test_worker_stops_its_process_at_the_session_deadline(monkeypatch):
    monkeypatch.setenv('LLMGE_SESSION_DEADLINE', str(time.time() + .15))
    start = time.monotonic()
    assert session.run_worker([sys.executable, '-c', 'import time; time.sleep(30)']) == 75
    assert time.monotonic() - start < 5



def test_worker_never_starts_after_its_deadline(monkeypatch):
    monkeypatch.setenv('LLMGE_SESSION_DEADLINE', '0')
    monkeypatch.setattr(session.subprocess, 'Popen', lambda *a, **kw: pytest.fail('started expired work'))
    with pytest.raises(RuntimeError, match='window ended before worker startup'):
        session.run_worker(['fixture'])



def test_cleanup_cancels_only_owned_jobs_in_the_exact_run(tmp_path, monkeypatch):
    monkeypatch.setenv('USER', 'fixture-user')
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    calls = []
    def run(args, **kw):
        calls.append(args)
        if args[0] == 'squeue':
            jobs = '1\n3\n4\n' if ['scancel', '2'] in calls else '1\n2\n3\n4\n'
            return SimpleNamespace(returncode=0, stdout=jobs)
        if args[0] == 'scontrol':
            job = args[3]
            directory = tmp_path if job in ('2', '4') else tmp_path / 'another-run'
            user = 'another-user' if job == '4' else 'fixture-user'
            return SimpleNamespace(returncode=0, stdout=f'JobId={job} UserId={user}(1) WorkDir={directory}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(session.subprocess, 'run', run)
    assert session.cancel_owned_jobs(tmp_path) == ['2']
    assert ['scancel', '2'] in calls



def test_controller_pause_keeps_checkpoint_and_cleans_up_its_run(tmp_path, monkeypatch):
    from src import fred_recovery_controller as recovery, fred_runtime
    from src.cfg import constants as cfg
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(cfg, 'SEED_FAMILY', 'dfine')
    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setenv('LLMGE_SESSION_DEADLINE', '0')
    monkeypatch.setattr(recovery.signal, 'signal', lambda *args: None)
    (tmp_path / 'five-hour-deadline-123.txt').write_text('600')
    folder = tmp_path / 'fred_dfine_checkpoints'
    session.save_pair(folder, folder, 0, {'population': [], 'hof': []},
        {'GLOBAL_DATA': {}, 'GLOBAL_DATA_HIST': {}, 'GLOBAL_DATA_ANCESTRY': {}}, 'dfine')
    monkeypatch.setattr(controller, 'wait_for_llm_server', lambda **kw: True)
    monkeypatch.setattr(recovery, 'smoke_generation', lambda: pytest.fail('Repeated smoke on a checkpoint resume'))
    monkeypatch.setattr(fred_runtime, 'assert_llm_job_active', lambda: None)
    clock, events = [0], []
    monkeypatch.setattr(recovery.time, 'time', lambda: clock[0])
    class Child:
        pid, returncode = 4242, None
        def poll(self): return self.returncode
        def wait(self, timeout):
            if timeout == 30:
                clock[0] = 700
                raise subprocess.TimeoutExpired('synthetic-child', 30)
            self.returncode = 0
            return 0
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: Child())
    monkeypatch.setattr(recovery.os, 'killpg', lambda pid, sig: events.append(('stop', pid)))
    monkeypatch.setattr(session, 'cancel_owned_jobs', lambda root: events.append(('cleanup', root)))
    recovery.main()
    assert events == [('stop', 4242), ('cleanup', tmp_path)]
    assert session.latest_pair(folder, folder, 'dfine')[1] == 1
