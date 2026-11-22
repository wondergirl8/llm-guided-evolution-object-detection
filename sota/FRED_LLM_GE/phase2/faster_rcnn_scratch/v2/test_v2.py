"""V2 contract tests: no GPU, downloads, or scheduler side effects."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import common, genome, proposals, training, workflow


def state(stage='smoke'):
    return {'stage': stage, 'generation': 0, 'approved': False,
            'tasks': {}, 'seen': {}, 'population': []}


class Scheduler:
    def __init__(self):
        self.states = {}
        self.submissions = []
        self.names = {}

    def state(self, job):
        return self.states.get(job, 'UNKNOWN')

    def submit(self, label, role, task=None, after=None):
        if label in self.names:
            return self.names[label]
        job = str(len(self.states) + 1)
        self.names[label] = job
        self.states[job] = 'PENDING'
        self.submissions.append((label, role, task, after))
        return job

    def cancel(self, job):
        self.states[job] = 'CANCELLED'


@pytest.mark.parametrize('width', [640, 960, 1280])
def test_resolution_roundtrip_and_legacy_validation(width, tmp_path):
    genes = genome.stock(width)
    assert genome.parse(genome.render(genes)) == genes
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.validator import validate_genes
    assert validate_genes(genome.model_genes(genes))
    assert len(genes) == 9


def test_resolution_part_of_candidate_not_model_identity():
    a, b = genome.stock(640), genome.stock(1280)
    assert genome.model_genes(a) == genome.model_genes(b)
    assert genome.candidate_identity(a) != genome.candidate_identity(b)


@pytest.mark.parametrize('key,value', [('img_width', 800), ('img_width', 640.),
                                      ('rpn_convs', True), ('anchor_scales', [2])])
def test_illegal_genes(key, value):
    genes = genome.stock()
    genes[key] = value
    with pytest.raises(ValueError):
        genome.validate(genes)


@pytest.mark.parametrize('change', ['preamble', 'call', 'extra', 'duplicate', 'wrong_class'])
def test_protected_source(change):
    text = genome.render(genome.stock())
    if change == 'preamble':
        text = 'import os\n' + text
    elif change == 'call':
        text = text.replace("{'img_width': 640}", 'dict(img_width=640)')
    elif change == 'extra':
        text += '\nprint(1)\n'
    elif change == 'duplicate':
        text = text.replace("{'img_width': 640}", "{'img_width': 640, 'img_width': 640}")
    else:
        text = text.replace('class InputGenes:', 'class Other:')
    with pytest.raises((ValueError, SyntaxError)):
        genome.parse(text)


def test_proposal_python_and_noop():
    parent = genome.stock()
    reply = "```python\nclass InputGenes:\n    values = {'img_width': 960}\n```"
    assert proposals.response_child(parent, 'InputGenes', reply)['img_width'] == 960
    with pytest.raises(ValueError, match='unchanged_parent'):
        proposals.response_child(parent, 'InputGenes', reply.replace('960', '640'))


def test_structural_freeze_accepts_different_json_format(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('{"x": 1}')
    common.freeze(path, {'x': 1})
    with pytest.raises(ValueError):
        common.freeze(path, {'x': 2})


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.5]))

    def forward(self, indices, mode):
        target = torch.rand(1) + random.random() + float(np.random.rand())
        return ((self.weight - target) ** 2).sum()

    def parse_losses(self, loss):
        return loss, {}


def seed():
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)


@pytest.mark.parametrize('interrupt_step', [1, 2, 3, 5, 9])
def test_exact_mid_epoch_resume_all_rng_and_momentum(interrupt_step):
    args = SimpleNamespace(epochs=3, batch_size=2, lr=.01, momentum=.9, weight_decay=.0001)
    data = {'images': [0] * 5}
    seed()
    full = Toy()
    visited_full, full_saved = [], []
    def prepare_full(model, data, indices, training_mode):
        visited_full.extend(indices)
        return {'indices': indices}
    training.train(full, data, args, prepare=prepare_full,
                   progress_callback=lambda s: full_saved.append(copy.deepcopy(s)), checkpoint_steps=1)
    seed()
    partial = Toy()
    visited_partial, saved = [], []
    def prepare(model, data, indices, training_mode):
        visited_partial.extend(indices)
        return {'indices': indices}
    def interrupt(s):
        saved.append(copy.deepcopy(s))
        if s['steps'] == interrupt_step:
            raise training.Paused('simulated timeout')
    with pytest.raises(training.Paused):
        training.train(partial, data, args, prepare=prepare,
                       progress_callback=interrupt, checkpoint_steps=1)
    seed()
    random.random()
    torch.rand(50)
    np.random.rand(9)
    resumed = Toy()
    resumed_saved = []
    training.train(resumed, data, args, prepare=prepare, resume_state=saved[-1],
                   progress_callback=lambda s: resumed_saved.append(copy.deepcopy(s)), checkpoint_steps=1)
    assert torch.equal(full.weight, resumed.weight)
    assert visited_full == visited_partial
    a, b = full_saved[-1]['optimizer'], resumed_saved[-1]['optimizer']
    assert torch.equal(a['state'][0]['momentum_buffer'], b['state'][0]['momentum_buffer'])
    assert full_saved[-1]['steps'] == resumed_saved[-1]['steps'] == 9


def test_signal_checkpoint_and_empty_data():
    args = SimpleNamespace(epochs=1, batch_size=2, lr=.01, momentum=.9, weight_decay=.0001)
    saved = []
    with pytest.raises(training.Paused):
        training.train(Toy(), {'images': [0, 1]}, args, should_stop=lambda: True,
                       progress_callback=lambda s: saved.append(copy.deepcopy(s)))
    assert saved[-1]['steps'] == 0
    with pytest.raises(ValueError, match='Empty'):
        training.train(Toy(), {'images': []}, args, progress_callback=lambda _: None)


def test_duplicate_reservations_and_parallel_cap(tmp_path):
    s, scheduler = state(), Scheduler()
    for index, width in enumerate((640, 960, 1280)):
        workflow.add_task(tmp_path, s, f'task{index}', 'search', genome.stock(width))
    with pytest.raises(ValueError, match='repeated'):
        workflow.add_task(tmp_path, s, 'duplicate', 'search', genome.stock(640))
    # Rejected duplicates must not be submitted in this fixture.
    s['tasks'].pop('duplicate', None)
    workflow.launch_pending(tmp_path, s, scheduler)
    assert len(scheduler.submissions) == 2
    workflow.launch_pending(tmp_path, s, scheduler)
    assert len(scheduler.submissions) == 2


@pytest.mark.parametrize('status', ['TIMEOUT', 'PREEMPTED', 'COMPLETED'])
def test_checkpoint_continuation(tmp_path, status):
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    workflow.launch_pending(tmp_path, s, scheduler)
    job = s['tasks']['task']['job']
    scheduler.states[job] = status
    common.write(tmp_path / 'tasks/task/allocation.json', {'state': 'paused'})
    common.write(tmp_path / 'tasks/task/progress.json', {'steps': 200})
    workflow.reconcile(tmp_path, s, scheduler)
    assert s['tasks']['task']['status'] == 'pending'
    workflow.launch_pending(tmp_path, s, scheduler)
    assert len(scheduler.submissions) == 2


def test_stop_and_no_progress_limits(tmp_path):
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    for _ in range(2):
        workflow.launch_pending(tmp_path, s, scheduler)
        scheduler.states[s['tasks']['task']['job']] = 'TIMEOUT'
        workflow.reconcile(tmp_path, s, scheduler)
    assert s['stage'] == 'blocked'
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    workflow.launch_pending(tmp_path, s, scheduler)
    scheduler.states[s['tasks']['task']['job']] = 'CANCELLED'
    workflow.reconcile(tmp_path, s, scheduler)
    assert s['stage'] == 'stopped'


def test_infrastructure_retry_limit(tmp_path):
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    for _ in range(4):
        workflow.launch_pending(tmp_path, s, scheduler)
        scheduler.states[s['tasks']['task']['job']] = 'NODE_FAIL'
        workflow.reconcile(tmp_path, s, scheduler)
    assert s['stage'] == 'blocked'
    assert len(scheduler.submissions) == 4  # first attempt plus three retries


def test_approval_gate_no_final_test_or_generation_one(tmp_path, monkeypatch):
    s, scheduler = state('initial_training'), Scheduler()
    s['slots'] = {}
    for width in (640, 960, 1280):
        name = workflow.stock_id(width)
        workflow.add_task(tmp_path, s, name, 'search', genome.stock(width))
        s['tasks'][name].update(status='complete', result={'fitness': [.7, .3, 40_000_000]})
    for i in range(5):
        g = genome.stock((640, 960, 960, 1280, 1280)[i])
        g['fpn_channels'] = 128
        g['rpn_channels'] = 128 if i % 2 else 256
        name = f'initial_{i}'
        workflow.add_task(tmp_path, s, name, 'search', g)
        s['slots'][str(i)] = name
        s['tasks'][name].update(status='complete', result={'fitness': [.75, .32, 30_000_000]})
    monkeypatch.setattr(workflow, 'report', lambda *_: None)
    workflow.advance(tmp_path, s, scheduler)
    assert s['stage'] == 'awaiting_budget_approval'
    assert len(s['population']) == 8
    workflow.advance(tmp_path, s, scheduler)
    assert not scheduler.submissions


def test_proposal_exhaustion_is_not_random_search(tmp_path, monkeypatch):
    s = state('initial_proposals')
    s['proposal_attempts'] = {'slot': 24}
    scheduler = Scheduler()
    # No server means a wait first; a healthy endpoint is mocked separately in integration.
    scheduler.states['server'] = 'RUNNING'
    s['server_job'] = 'server'
    scheduler.node = lambda _: 'node'
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(proposals.urllib.request, 'urlopen', lambda *_a, **_k: Response())
    common.write(tmp_path / 'protocol.json', {'server_port': 9000})
    assert proposals.propose(tmp_path, s, scheduler, 'slot', [genome.stock()]) is None
    assert s['proposal_attempts']['slot'] == 24
    assert s['seen'] == {}
    assert s['stage'] == 'blocked'


def test_official_membership_and_validation_holdout():
    from .cli import partition
    train = [(str(i), 'train' if i % 2 else 'test') for i in range(172)]
    test = [(str(i), 'train') for i in range(172, 231)]
    search, val = partition(train, test, {str(i) for i in range(20)})
    assert (len(search), len(val)) == (152, 20)
    assert not {s for s, _ in search} & {s for s, _ in val}
    assert not {s for s, _ in search + val} & {s for s, _ in test}
    with pytest.raises(ValueError):
        partition(train, test[:-1] + [train[0]], {str(i) for i in range(20)})


def test_scheduler_submission_gap_adopts_same_attempt(tmp_path):
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    original = scheduler.submit
    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('Controller died after sbatch')
    scheduler.submit = interrupt
    with pytest.raises(OSError):
        workflow.launch_pending(tmp_path, s, scheduler)
    assert s['tasks']['task']['status'] == 'submitting'
    scheduler.submit = original
    recovered = common.read(tmp_path / 'state.json')
    workflow.launch_pending(tmp_path, recovered, scheduler)
    assert len(scheduler.submissions) == 1
    assert len(recovered['tasks']['task']['attempts']) == 1


def test_supervisor_prearms_before_work_and_cancels_on_gate(tmp_path, monkeypatch):
    from . import cli
    s, scheduler = state(), Scheduler()
    common.write(tmp_path / 'state.json', s)
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    monkeypatch.setattr(cli, 'Slurm', lambda _: scheduler)
    monkeypatch.setattr(common, 'verify_run', lambda *_: {})
    def advance(run, current, sched):
        assert sched.submissions[0] == ('supervisor_100', 'supervisor', None, '100')
        current['stage'] = 'awaiting_budget_approval'
    monkeypatch.setattr(workflow, 'advance', advance)
    cli.supervise(tmp_path)
    assert common.read(tmp_path / 'state.json')['stage'] == 'awaiting_budget_approval'
    assert scheduler.states['1'] == 'CANCELLED'


def test_supervisor_cancelled_predecessor_does_not_relaunch(tmp_path, monkeypatch):
    from . import cli
    common.write(tmp_path / 'state.json', state())
    common.write(tmp_path / 'supervisor.json', {'job': '99', 'successor': '100'})
    scheduler = Scheduler()
    scheduler.states['99'] = 'CANCELLED'
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    monkeypatch.setattr(cli, 'Slurm', lambda _: scheduler)
    monkeypatch.setattr(common, 'verify_run', lambda *_: {})
    cli.supervise(tmp_path)
    assert common.read(tmp_path / 'state.json')['stage'] == 'stopped'
    assert scheduler.submissions == []


def test_sufficient_population_and_nsga_parent_selection(tmp_path):
    s = state()
    for index in range(8):
        g = genome.stock((640, 960, 1280)[index % 3])
        g['fpn_channels'] = 128 if index > 2 else 256
        g['rpn_convs'] = 2 if index > 5 else 1
        name = f't{index}'
        workflow.add_task(tmp_path, s, name, 'search', g)
        s['tasks'][name].update(status='complete', result={'fitness': [.6+.01*index, .2+.01*index, 40_000_000-index*1000]})
    assert len(workflow.select(s, list(s['tasks']), 2)) == 2
    assert len(workflow.select(s, list(s['tasks']), 6, parents=True)) == 6


def test_all_resolution_coordinates():
    from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import resized_data
    box = torch.tensor([[200., 100., 400., 300.]])
    for width in (640, 960, 1280):
        data = {'images': torch.zeros(1, 3, 720, 1280, dtype=torch.uint8),
                'boxes': [box], 'orig_boxes': [box]}
        resized = resized_data(data, width)
        assert resized['images'][0].shape[-1] == width
        scale = torch.tensor([width / 1280, resized['images'].shape[-2] / 720] * 2)
        assert torch.equal(resized['boxes'][0] / scale, box)
        assert torch.equal(resized['orig_boxes'][0], box)


def test_resumable_prediction_chunks_and_tampering(tmp_path):
    path = tmp_path / 'predictions/4.json'
    calls = []
    def predict():
        calls.append(1)
        return [{'sample_id': 'train/4:0'}], ['train/4:0']
    first = common.cached_predictions(path, 'protocol', 'weights', predict, 'train/4:')
    assert common.cached_predictions(path, 'protocol', 'weights', predict, 'train/4:') == first
    assert calls == [1]
    with pytest.raises(ValueError, match='provenance'):
        common.cached_predictions(path, 'protocol', 'different', predict, 'train/4:')
    bad = tmp_path / 'predictions/invalid.json'
    with pytest.raises(ValueError, match='Incomplete'):
        common.cached_predictions(bad, 'protocol', 'weights', lambda: ([], ['train/4:0']), 'train/4:')
    assert not bad.exists()


def test_frozen_source_dependencies_and_semantic_protocol(tmp_path, monkeypatch):
    code = tmp_path / 'code'
    code.mkdir()
    source = code / 'source.py'
    source.write_text('VALUE = 1\n')
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{"epochs":20}')
    marker = tmp_path / 'data.json'
    marker.write_text('{"frames": 5}')
    manifest = {'source_hashes': {'source.py': common.sha(source)},
                'protocol_sha256': common.json_sha(protocol),
                'data_hashes': {str(marker): common.sha(marker)},
                'detector_versions': {'torch': 'pinned'}, 'controller_versions': {'python': 'pinned'}}
    common.write(tmp_path / 'manifest.json', manifest)
    monkeypatch.setattr(common, 'versions', lambda *_: {'torch': 'pinned'})
    monkeypatch.setattr(common, 'controller_versions', lambda *_: {'python': 'pinned'})
    protocol.write_text('{\n  "epochs": 20\n}\n')
    (tmp_path / 'live_source.py').write_text('VALUE = 2\n')
    assert common.verify_run(tmp_path) == manifest
    source.write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='source'):
        common.verify_run(tmp_path)
    source.write_text('VALUE = 1\n')
    monkeypatch.setattr(common, 'versions', lambda *_: {'torch': 'changed'})
    with pytest.raises(ValueError, match='dependencies'):
        common.verify_run(tmp_path)


@pytest.mark.parametrize('role', ['worker', 'supervisor', 'server'])
def test_slurm_limits_and_dependencies(tmp_path, monkeypatch, role):
    from .slurm import Slurm
    from . import slurm
    (tmp_path / 'code').mkdir()
    common.write(tmp_path / 'protocol.json', {'server_port': 9001, 'model_path': '/trusted/model'})
    submitted = []
    def output(command, **kwargs):
        if command[0] == 'sbatch':
            submitted.append(command)
            return '123\n'
        return ''
    monkeypatch.setattr(slurm.subprocess, 'check_output', output)
    scheduler = Slurm(tmp_path)
    assert scheduler.submit('stage', role, after='99') == '123'
    command, = submitted
    assert '--time=08:00:00' in command
    assert '--signal=B:USR1@180' in command
    assert '--dependency=afterany:99' in command
    if role == 'worker':
        assert '--gres=gpu:1' in command
    elif role == 'server':
        assert '--gres=gpu:2' in command
        assert 'LLMGE_MODEL_PATH=/trusted/model' in command[-1]
    else:
        assert not any(c.startswith('--gres') for c in command)


def test_submission_adopts_already_running_job(tmp_path, monkeypatch):
    from .slurm import Slurm
    from . import slurm
    scheduler = Slurm(tmp_path)
    name = scheduler.token('task_a0')
    monkeypatch.setattr(slurm.subprocess, 'check_output',
                        lambda command, **kwargs: '33|' + name + '|RUNNING|node\n')
    assert scheduler.submit('task_a0', 'worker') == '33'


def test_server_refresh_uses_a_new_job_not_old_cancelled_name(tmp_path):
    s, scheduler = state('initial_proposals'), Scheduler()
    assert proposals.propose(tmp_path, s, scheduler, 'slot', [genome.stock()]) is None
    first = s['server_job']
    workflow.stop_server(s, scheduler)
    assert proposals.propose(tmp_path, s, scheduler, 'slot', [genome.stock()]) is None
    assert s['server_job'] != first
    assert len(scheduler.submissions) == 2


def test_unknown_accounting_does_not_duplicate_workers(tmp_path):
    s, scheduler = state(), Scheduler()
    workflow.add_task(tmp_path, s, 'task', 'search', genome.stock())
    workflow.launch_pending(tmp_path, s, scheduler)
    scheduler.states[s['tasks']['task']['job']] = 'UNKNOWN'
    for _ in range(7):
        workflow.reconcile(tmp_path, s, scheduler)
    assert s['stage'] == 'blocked'
    assert len(scheduler.submissions) == 1
