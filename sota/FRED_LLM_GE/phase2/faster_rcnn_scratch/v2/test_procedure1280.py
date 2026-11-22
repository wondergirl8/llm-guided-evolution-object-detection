import itertools
import json
import subprocess

import pytest

from . import common, genome, procedure1280 as procedure, workflow


def state(stage):
    return dict(stage=stage, generation=0, tasks={}, seen={}, slots={}, population=[])


def test_protocol_is_opt_in_and_fixed_resolution(tmp_path):
    original = {'widths': [640, 960, 1280], 'train_frame_step': 5}
    p = procedure.configure(original, tmp_path)
    assert original['widths'] == [640, 960, 1280]
    assert p['widths'] == [1280]
    assert p['train_frame_step'] == 50 and p['final_train_frame_step'] == 5
    assert p['final_seeds'] == list(range(1, 31))
    assert p['statistics']['alternative'] == 'two-sided'


def test_fixed_width_initial_proposal_and_no_input_mutation(tmp_path, monkeypatch):
    s = state('initial_proposals')
    calls = []
    def propose(run, current, scheduler, slot, parents, width=None):
        calls.append(width)
        current['proposal_attempts'] = {slot: 1}
        child = dict(parents[0], fpn_channels=128)
        return child
    monkeypatch.setattr(workflow, 'reconcile', lambda *_: None)
    monkeypatch.setattr(procedure.proposals, 'propose', propose)
    procedure.advance(tmp_path, s, None)
    assert calls == [1280]
    assert next(iter(s['tasks'].values()))['spec']['genes']['img_width'] == 1280


def test_initial_population_has_eight_then_twenty_generations(tmp_path, monkeypatch):
    s = state('initial_training')
    stock = procedure.STOCK
    workflow.add_task(tmp_path, s, stock, 'search', genome.stock(1280))
    s['tasks'][stock]['status'] = 'complete'
    for i in range(7):
        name = str(i)
        # Only state-machine sizing is exercised here; source deduplication has
        # separate real-genome tests and remains enforced by add_task.
        s['tasks'][name] = {'status': 'complete'}
        s['slots'][name] = name
    monkeypatch.setattr(workflow, 'reconcile', lambda *_: None)
    monkeypatch.setattr(workflow, 'report', lambda *_: None)
    procedure.advance(tmp_path, s, None)
    assert len(s['population']) == 8
    assert s['stage'] == 'evolution_proposals' and s['generation'] == 1
    assert not any(t.get('spec', {}).get('kind') == 'final' for t in s['tasks'].values())


def search_archive(tmp_path):
    s = state('evolution_training')
    s['generation'] = 20
    fields = {k: v for group, values in genome.GROUPS.items()
              if group != 'InputGenes' for k, v in values.items()}
    stock = procedure.STOCK
    genes = [genome.stock(1280)]
    genes += [dict(zip(fields, values), img_width=1280)
              for values in itertools.product(*fields.values())
              if dict(zip(fields, values), img_width=1280) != genes[0]][:127]
    for i, g in enumerate(genes):
        name = stock if i == 0 else f'candidate_{i}'
        workflow.add_task(tmp_path, s, name, 'search', g)
        s['tasks'][name].update(status='complete', result={
            'fitness': [.8, .4, 41000000], 'candidate_hash': genome.candidate_identity(g)})
    common.write(tmp_path / 'protocol.json', {'experiment': 'procedure1280'})
    return s


def test_all_archive_front_and_thirty_seeds_resume_idempotently(tmp_path, monkeypatch):
    s = search_archive(tmp_path)
    seen = []
    def front(current, names):
        seen.extend(names)
        return ['candidate_1']
    monkeypatch.setattr(workflow, 'front', front)
    procedure.freeze_finalists(tmp_path, s)
    assert len(seen) == 128
    assert len(s['final_tasks']) == 60  # finalist plus stock, 30 repeats each
    assert {s['tasks'][n]['spec']['seed'] for n in s['final_tasks']} == set(range(1, 31))
    assert all(s['tasks'][n]['spec']['genes']['img_width'] == 1280 for n in s['final_tasks'])
    procedure.freeze_finalists(tmp_path, s)
    assert len(s['tasks']) == 188
    selection = common.read(tmp_path / 'finalists.json')
    assert selection['finalists'][1]['stock_control']


def test_incomplete_archive_cannot_trigger_final_testing(tmp_path):
    s = state('evolution_training')
    with pytest.raises(ValueError, match='128 unique'):
        procedure.freeze_finalists(tmp_path, s)


def test_holm_known_values():
    assert procedure.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_thirty_seed_statistics_and_missing_repeats(tmp_path):
    finalists = [{'search_id': 'evolved', 'stock_control': False},
                 {'search_id': 'stock', 'stock_control': True}]
    common.write(tmp_path / 'finalists.json', {'finalists': finalists})
    s = state('final_training')
    for index in range(2):
        for seed in range(1, 31):
            score = .4 + seed * .0001 + (0.03 if index == 0 else 0.)
            s['tasks'][f'final_{index:03d}_s{seed:02d}'] = {
                'status': 'complete', 'result': {
                    'fitness': [.8, score, 41000000], 'checkpoint_sha256': f'{index}-{seed}',
                    'report': f'{index}-{seed}.json'}}
    procedure.summarize_samples(tmp_path, s)
    report = common.read(tmp_path / 'reports/final_seed_statistics.json')
    row, = report['comparisons']
    assert row['gain_pp'] == pytest.approx(3.)
    assert row['statistically_better']
    assert report['samples'][0]['metrics']['mAP50_95']['sample_sd_pp'] > 0
    s['tasks']['final_000_s30']['status'] = 'pending'
    with pytest.raises(ValueError, match='30'):
        procedure.summarize_samples(tmp_path, s)


@pytest.mark.parametrize('command,expected,calls_expected', [
    (['squeue'], 'complete', 2), (['python', 'plotter.py'], 'blocked', 1)])
def test_supervisor_retries_only_scheduler_errors(tmp_path, monkeypatch, command, expected, calls_expected):
    from . import cli
    common.write(tmp_path / 'state.json', state('smoke'))
    monkeypatch.setattr(common, 'verify_run', lambda *_: {})
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    monkeypatch.setattr(cli, 'EXIT_REQUESTED', False)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    cancelled = []
    class Scheduler:
        def submit(self, *args, **kwargs): return '101'
        def cancel(self, job): cancelled.append(job)
    monkeypatch.setattr(cli, 'Slurm', lambda _: Scheduler())
    calls = []
    def advance(run, current, scheduler):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        current['stage'] = 'complete'
    monkeypatch.setattr(workflow, 'advance', advance)
    cli.supervise(tmp_path)
    assert common.read(tmp_path / 'state.json')['stage'] == expected
    assert len(calls) == calls_expected
    assert cancelled == ['101']
