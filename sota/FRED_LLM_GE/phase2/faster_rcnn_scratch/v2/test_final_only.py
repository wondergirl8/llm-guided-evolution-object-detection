import pickle

import pytest

from . import common, genome, final_only, workflow
from .worker import task_epochs
from sota.FRED_LLM_GE.phase1_detection import final_eval


@pytest.fixture
def configured(tmp_path):
    source, run = tmp_path / 'source', tmp_path / 'final'
    genes = dict(genome.stock(1280), fpn_channels=128)
    model = 'g05_slot4_p01'
    fitness = [.8508911053325305, .42789081355376557, 28957013]
    common.write(source / 'tasks' / model / 'result.json', {
        'fitness': fitness, 'status': 'complete', 'candidate_hash': genome.candidate_identity(genes)})
    (source / 'tasks' / model / 'source.py').write_text(genome.render(genes))
    common.write(source / 'state.json', {'stage': 'stopped', 'tasks': {
        model: {'status': 'complete', 'spec': {'genes': genes}}}})
    (source / 'STOP').touch()
    common.write(source / 'manifest.json', {'source': 'unchanged'})
    membership = {'final_train': [['1', 'train'], ['2', 'test']], 'test': [['3', 'test']]}
    common.write(source / 'protocol.json', membership)
    (source / 'checkpoints').mkdir()
    (source / 'checkpoints/global_gen_8.pkl').write_bytes(pickle.dumps({
        'GLOBAL_DATA': {model: {'fitness': fitness}}}))
    protocol = dict(membership, train_frame_step=5, train_root='/verified/final_train_step5', epochs=20)
    p, _ = final_only.configure(protocol, source, model, 50, 8)
    common.write(run / 'protocol.json', p)
    common.write(run / 'manifest.json', {'protocol_sha256': common.json_sha(run / 'protocol.json')})
    final_only.initialize(run)
    return source, run, p


def test_final_only_fifty_epochs_thirty_fresh_seeds_and_cutoff_authorization(configured):
    source, run, p = configured
    state = common.read(run / 'state.json')
    assert p['final_epochs'] == 50 and p['epochs'] == 20
    assert p['widths'] == [1280] and p['final_train_frame_step'] == 5
    assert state['generation'] == 8 and len(state['final_tasks']) == 30
    assert {t['spec']['seed'] for t in state['tasks'].values()} == set(range(1, 31))
    for name, task in state['tasks'].items():
        assert task_epochs(p, task['spec']) == 50
        assert final_eval.faster_rcnn_command(run, name)[-1] == name
    assert task_epochs(p, {'kind': 'search'}) == 20
    assert not list(run.rglob('*.pt'))
    assert common.read(source / 'state.json')['stage'] == 'stopped'


@pytest.mark.parametrize('field,value', [('final_epochs', 20), ('selection_generation', 20), ('final_seeds', [0])])
def test_final_only_authorization_cannot_change_recipe(configured, field, value):
    _, run, _ = configured
    path = run / 'authorization.json'
    data = common.read(path)
    data[field] = value
    common.write(path, data)
    with pytest.raises(ValueError, match='authorization'):
        final_eval.faster_rcnn_command(run, 'final_000_s01')


def test_final_selection_model_and_source_are_protected(configured):
    _, run, _ = configured
    path = run / 'finalists.json'
    selected = common.read(path)
    selected['finalists'][0]['search_id'] = 'another_model'
    common.write(path, selected)
    with pytest.raises(ValueError, match='architecture'):
        final_eval.faster_rcnn_command(run, 'final_000_s01')


def test_final_only_launches_two_workers_and_never_proposes(configured):
    from .test_v2 import Scheduler
    _, run, _ = configured
    scheduler = Scheduler()
    state = common.read(run / 'state.json')
    workflow.advance(run, state, scheduler)
    assert len(scheduler.submissions) == 2
    assert {r[1] for r in scheduler.submissions} == {'worker'}
    assert state['stage'] == 'final_training'


def test_fifty_epoch_budget_survives_exact_batch_resume():
    import torch
    from types import SimpleNamespace
    from .test_v2 import Toy
    from .training import train, Paused
    args = SimpleNamespace(epochs=50, batch_size=2, lr=.0025, momentum=.9, weight_decay=.0001)
    data = {'images': [0, 1]}
    saved = []
    model = Toy()
    prepare = lambda model, data, indices, training: {'indices': indices}
    with pytest.raises(Paused):
        train(model, data, args, prepare=prepare, progress_callback=lambda s: saved.append(s),
              should_stop=lambda: len(saved) >= 2, checkpoint_steps=1)
    restored = Toy()
    train(restored, data, args, prepare=prepare, resume_state=saved[-1],
          progress_callback=lambda s: saved.append(s))
    assert saved[-1]['epoch'] == 50
    assert saved[-1]['steps'] == 50


def test_only_identical_previously_verified_exports_can_skip_duplicate_setup_scan(tmp_path):
    root, source = tmp_path / 'data', tmp_path / 'source'
    marker = root / 'done/1.json'
    common.write(marker, {'sequence': '1', 'hf_folder': 'train', 'frame_step': 5,
                          'frames': 1, 'records': [{'name': '1_1'}], 'size': [1280, 720]})
    common.write(source / 'manifest.json', {'data_hashes': {str(marker): common.sha(marker)}})
    exports = [(root, [('1', 'train')], 5)]
    final_only.verify_reused_exports(exports, source)
    with pytest.raises(ValueError, match='identity'):
        final_only.verify_reused_exports([(root, [('1', 'train')], 50)], source)
    common.write(marker, {'changed': True})
    with pytest.raises(ValueError, match='previously verified'):
        final_only.verify_reused_exports(exports, source)
