import subprocess

import pytest

from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import common, pilot_supervisor as pilot


def test_priority_seed_zero_at_all_widths_before_repeats():
    items = [(f'{width}_{seed}', {'spec': {'seed': seed, 'genes': {'img_width': width}}})
             for width in (1280, 640, 960) for seed in (2, 1, 0)]
    assert [name for name, _ in sorted(items, key=pilot.priority)][:3] == ['640_0', '960_0', '1280_0']


def test_report_known_power_and_incomplete_pilot():
    state = {'tasks': {}}
    for width in (640, 960, 1280):
        for seed, score in enumerate((.683646, .728494, .693210)):
            state['tasks'][f'{width}_{seed}'] = {
                'spec': {'genes': {'img_width': width}, 'seed': seed},
                'status': 'complete', 'result': {'fitness': [score, score, 41348181]}}
    report = pilot.pilot_report(state, {'alpha': .05, 'power': .8})
    assert report['widths']['640']['mAP50']['seeds_per_architecture']['3'] == 11
    state['tasks'].pop('640_0')
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.pilot_report(state, {'alpha': .05, 'power': .8})


def test_query_failure_retries_without_cancelling_workers(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import workflow, slurm
    run = tmp_path
    (run / 'control').mkdir()
    common.write(run / 'control/pilot_recovery.json', {'controller_sha256': common.sha(pilot.__file__)})
    common.write(run / 'control/seed_policy.json', {'alpha': .05, 'power': .8})
    common.write(run / 'state.json', {'stage': 'baselines', 'generation': 0, 'tasks': {
        'seed': {'status': 'running', 'job': 'worker', 'spec': {'seed': 0}}}})
    monkeypatch.setattr(common, 'verify_run', lambda _: None)
    cancelled = []

    class Scheduler:
        def __init__(self, _): pass
        def token(self, _): return 'next'
        def active(self): return [['successor', 'next', 'PENDING', '']]
        def cancel(self, job): cancelled.append(job)

    monkeypatch.setattr(slurm, 'Slurm', Scheduler)
    calls = []

    def reconcile(_, state, scheduler):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ['squeue'])
        state['tasks']['seed']['status'] = 'complete'

    monkeypatch.setattr(workflow, 'reconcile', reconcile)
    monkeypatch.setattr(pilot, 'pilot_report', lambda *_: {'complete': True})
    monkeypatch.setattr(pilot.time, 'sleep', lambda _: None)
    monkeypatch.setattr(pilot.sys, 'argv', ['pilot', '--run', str(run)])
    monkeypatch.setenv('SLURM_JOB_ID', 'controller')
    pilot.main()
    assert len(calls) == 2
    assert cancelled == ['successor']
    assert common.read(run / 'state.json')['stage'] == 'awaiting_seed_budget_review'
