"""CPU-only contracts; no HF downloads, LLM server, or SLURM submissions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sota.FRED_LLM_GE.phase1_detection import bn_snn_data as data_io
from sota.FRED_LLM_GE.phase1_detection import bn_snn_search_space as policy
from sota.FRED_LLM_GE.phase1_detection import candidate_runner as runner
from sota.FRED_LLM_GE.phase1_detection.adapters.bn_snn import BNSNNAdapter
from sota.FRED_LLM_GE.phase1_detection.seeds.bn_snn.detection import anchors_for, multibox_loss, decode, xyxy
from sota.FRED_LLM_GE.phase1_detection.seeds.bn_snn.network import surrogate_spike, BIFNeuron


@pytest.fixture
def setup():
    protocol = runner.load_protocol(runner.DEFAULT_PROTOCOL)
    return protocol, BNSNNAdapter(protocol['families']['bn_snn'], runner.FRED_DIR)


def variant(tmp_path, old, new):
    path = tmp_path / 'network_test.py'
    path.write_text(policy.SEED.read_text().replace(old, new))
    return path


@pytest.mark.parametrize('old,new', [
    ('BASE_CHANNELS = 32', 'BASE_CHANNELS = 17'),
    ('BASE_CHANNELS = 32', 'BASE_CHANNELS = True'),
    ('BASE_CHANNELS = 32', 'BASE_CHANNELS = __import__("os").getpid()'),
    ('STAGE_DEPTHS = (2, 2, 2, 2)', 'STAGE_DEPTHS = (1, 2, 3)'),
    ('STAGE_DEPTHS = (2, 2, 2, 2)', 'STAGE_DEPTHS = (1, 2, 3, 4)'),
    ('num_classes + 1', 'num_classes'),
    ('BASE_CHANNELS = 32', 'BASE_CHANNELS = 32\nprint("injected")'),
])
def test_rejects_invalid_and_protected_code(tmp_path, old, new):
    with pytest.raises((ValueError, SyntaxError)):
        policy.build(variant(tmp_path, old, new))


def test_hash_noops_and_rng_preserved(tmp_path):
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first, count = policy.built_architecture(policy.SEED)
    assert count > 0
    assert torch.equal(before, torch.get_rng_state())
    clone = tmp_path / 'clone.py'
    clone.write_text(policy.SEED.read_text() + '\n# harmless comment\n')
    assert policy.built_architecture(clone)[0] == first
    assert not policy.differs_from_parents(clone, policy.SEED)
    changed = variant(tmp_path, 'BASE_CHANNELS = 32', 'BASE_CHANNELS = 24')
    assert policy.differs_from_parents(changed, policy.SEED)
    assert policy.evaluation_cache_key(first, {'seed': 0}) != policy.evaluation_cache_key(first, {'seed': 1})


def test_temporal_bins_preserve_silence_polarity_and_open_endpoints(tmp_path):
    events = np.array([(1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 1, 2), (1, 2, 1, 9), (1, 2, 0, 10)],
        dtype=[('x', 'i2'), ('y', 'i2'), ('p', 'i2'), ('t', 'i8')])
    data_io.validate_events(events)
    tensor = data_io.voxelize(events, 10, 10, 4, 4, 4)
    assert tensor.shape == (4, 2, 4, 4)
    assert tensor.sum() == 3
    assert tensor[0, 0, 2, 1] == tensor[0, 1, 2, 1] == tensor[3, 1, 2, 1] == 1
    assert tensor[1:3].sum() == 0
    np.save(tmp_path / '4.npy', events)
    lazy = data_io.LazyEvents([{'sequence': '4', 'end_us': 10}], tmp_path,
                             {'window_us': 10, 'time_steps': 4}, (4, 4))
    assert torch.equal(lazy[torch.tensor([0])][0], tensor)


def test_surrogate_gradient_and_state_reset(setup):
    x = torch.tensor([-2., -0.5, 0.5, 2.], requires_grad=True)
    surrogate_spike(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0., 0.5, 0.5, 0.]))
    _, adapter = setup
    model = adapter.build(policy.SEED, 1, 'cpu').eval()
    events = torch.rand(1, 4, 2, 32, 48)
    with torch.no_grad():
        first = model(events)
        model(events * 3)
        second = model(events)
    assert torch.equal(first['locations'], second['locations'])
    model.reset_state()
    assert all(m.mem is None for m in model.modules() if isinstance(m, BIFNeuron))


def test_ssd_decode_and_loss_for_positive_and_empty_targets():
    anchors = anchors_for(32, 48, [8, 16, 32], 'cpu')
    assert torch.allclose(decode(torch.zeros_like(anchors), anchors), xyxy(anchors).clamp(0, 1))
    loc = torch.zeros(2, len(anchors), 4, requires_grad=True)
    cls = torch.zeros(2, len(anchors), 2, requires_grad=True)
    loss = multibox_loss({'locations': loc, 'confidences': cls}, anchors,
                        [torch.tensor([[0.2, 0.2, 0.3, 0.3]]), torch.empty(0, 4)])
    assert torch.isfinite(loss)
    loss.backward()
    assert loc.grad.abs().sum() > 0 and cls.grad[1].abs().sum() > 0


def tiny_data():
    boxes = [torch.tensor([[10., 8., 16., 14.]]), torch.empty(0, 4)]
    return dict(images=torch.rand(2, 4, 2, 32, 48), boxes=boxes, orig_boxes=boxes,
                orig_size=(48, 32), sample_ids=['train/4:0', 'train/4:1'], manifest_sha256='synthetic')


def test_train_predict_and_shared_evaluator(setup, tmp_path, monkeypatch):
    protocol, adapter = setup
    proxy = dict(protocol['proxy'], epochs=1, batch_size=2, eval_conf=0.05)
    data = tiny_data()
    model = adapter.build(policy.SEED, 1, 'cpu')
    before = model.backbone.stem[0].weight.detach().clone()
    adapter.train(model, data, proxy, 'cpu')
    assert not torch.equal(before, model.backbone.stem[0].weight)
    frames = runner.predict_frames(adapter, model, data, proxy, 'cpu')
    assert len(frames) == 2
    assert all(label == 0 for f in frames for label in f['prediction']['labels'])
    assert all(0 <= x0 < x1 <= 48 and 0 <= y0 < y1 <= 32
               for f in frames for x0, y0, x1, y1 in f['prediction']['boxes'])
    monkeypatch.setattr(data_io, 'load_search', lambda _: (proxy, data, lambda: data))
    runner.run('seed', 'network', tmp_path, 'bn_snn', protocol, tmp_path, 'cpu')
    result = (tmp_path / 'results/seed_results.csv').read_text()
    assert 'inf' not in result
    assert (tmp_path / 'results/seed_evaluation.json').is_file()
    assert (tmp_path / 'trained_models/seed.pt').is_file()
    record = json.loads((tmp_path / 'training_args/seed/run.json').read_text())
    assert record['wall_time_seconds'] > 0 and record['params'] > 0
    from sota.FRED_LLM_GE.phase1_detection import bn_snn_baseline as baseline
    import yaml
    baseline_protocol = copy.deepcopy(protocol)
    baseline_protocol['proxy'] = proxy
    baseline_protocol['challenging']['search']['epochs'] = 1
    protocol_path = tmp_path / 'baseline_protocol.yaml'
    protocol_path.write_text(yaml.safe_dump(baseline_protocol))
    monkeypatch.setattr(baseline, 'PROTOCOL', protocol_path)
    assert baseline.validate(tmp_path, 0)['params'] == record['params']
    with pytest.raises(ValueError, match='provenance'):
        baseline.validate(tmp_path, 1)
    (tmp_path / 'results/seed_results.csv').write_text('-inf,-inf,inf,failed\n')
    with pytest.raises(ValueError, match='failed baseline'):
        baseline.validate(tmp_path, 0)


def test_bad_genome_uses_failure_contract(setup, tmp_path, monkeypatch):
    protocol, _ = setup
    data = tiny_data()
    monkeypatch.setattr(data_io, 'load_search', lambda _: (protocol['proxy'], data, lambda: data))
    path = variant(tmp_path, 'BASE_CHANNELS = 32', 'BASE_CHANNELS = 0')
    runner.run('test', path.stem, tmp_path, 'bn_snn', protocol, tmp_path, 'cpu')
    result = (tmp_path / 'results/test_results.csv').read_text()
    assert result.startswith('-inf,-inf,inf')
    assert not (tmp_path / 'trained_models/test.pt').exists()


def test_raw_loader_rejects_test_split(setup):
    protocol, _ = setup
    protocol = copy.deepcopy(protocol)
    protocol['proxy']['official_split'] = 'challenging_test'
    with pytest.raises(ValueError, match='challenging_train'):
        data_io.search_parts(protocol)


def test_prepare_reuses_labels_exact_timestamps_and_detects_corruption(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import zipfile

    source, cache = tmp_path / 'export', tmp_path / 'raw'
    (source / 'done').mkdir(parents=True)
    marker = source / 'done/4.json'
    marker.write_text(json.dumps(dict(sequence='4', hf_folder='train', size=[1280, 720],
        records=[dict(name='4_000001', frame_index=1, boxes=[[1, 2, 3, 4]])])))
    archive = tmp_path / '4.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('4/Event/events.hdf5', b'synthetic; decoder mocked below')
        stream.writestr('4/Event/Frames/Video_4_frame_33333.png', b'not decoded')
    @contextmanager
    def downloaded(*args):
        yield archive
    events = np.array([(2, 3, 1, 100)], dtype=[('x','i2'),('y','i2'),('p','i2'),('t','i8')])
    monkeypatch.setattr(data_io, 'downloaded_sequence', downloaded)
    monkeypatch.setattr(data_io, 'load_events', lambda _: events)
    data_io.prepare_sequence(source, '4', 'train', cache)
    index = json.loads((cache / '4.json').read_text())
    assert index['records'][0]['end_us'] == 33333
    assert index['records'][0]['boxes'] == [[1, 2, 3, 4]]
    # Rerun succeeds without another download.
    monkeypatch.setattr(data_io, 'downloaded_sequence', lambda *a: pytest.fail('unexpected download'))
    data_io.prepare_sequence(source, '4', 'train', cache)
    with (cache / '4.npy').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='corrupt'):
        data_io.prepare_sequence(source, '4', 'train', cache)


def test_bn_baseline_summarizes_and_rejects_stale_records(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection import bn_snn_baseline as baseline
    monkeypatch.setattr(baseline, 'identity', lambda: {'sha256': 'frozen'})
    def validate(root, seed):
        return dict(seed=seed, params=100, mAP50=0.5 + seed * 0.01,
                    mAP50_95=0.2, wall_time_seconds=10 + seed)
    monkeypatch.setattr(baseline, 'validate', validate)
    for seed in (0, 1, 2):
        root = tmp_path / f'seed_{seed}'
        root.mkdir()
        (root / 'baseline.json').write_text(json.dumps(dict(validate(root, seed), identity=baseline.identity())))
    baseline.summarize(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['aggregate']['mAP50']['mean'] == pytest.approx(0.51)
    assert summary['aggregate']['mAP50']['sample_sd'] == pytest.approx(0.01)
    monkeypatch.setattr(baseline, 'identity', lambda: {'sha256': 'changed'})
    with pytest.raises(ValueError, match='stale'):
        baseline.summarize(tmp_path)


def test_bn_fingerprint_tracks_raw_manifest_and_protected_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import shutil
    import yaml
    from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint as fp
    fred = tmp_path / 'sota/FRED_LLM_GE'
    protocol = runner.load_protocol(runner.DEFAULT_PROTOCOL)
    protocol_path = fred / 'configs/phase1/protocol.yaml'
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(yaml.safe_dump(protocol))
    seed = fred / 'seeds/seed_bn_snn.py'
    seed.parent.mkdir()
    shutil.copy(policy.SEED, seed)
    # Copy only identity inputs, never data artifacts.
    for name in ('candidate_runner.py', 'bn_snn_data.py', 'bn_snn_search_space.py',
                 'adapters/bn_snn.py', 'seeds/bn_snn/network.py', 'seeds/bn_snn/detection.py',
                 'bn_snn_baseline.py', 'map_reporter.py', 'fitness.py', 'result_contract.py',
                 'proxy_data.py', 'fred_splits.py', 'export_split.py'):
        destination = fred / 'phase1_detection' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(runner.FRED_DIR / 'phase1_detection' / name, destination)
    for name in ('data/fred_events.py', 'data/fred_stream.py', 'configs/phase0/default.yaml'):
        destination = fred / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(runner.FRED_DIR / name, destination)
    cache = fred / protocol['families']['bn_snn']['raw_cache']
    cache.mkdir(parents=True, exist_ok=True)
    for name in ('train', 'val'):
        (cache / f'{name}_manifest.json').write_text('{}')
    monkeypatch.setattr(fp.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='fixed-commit'))
    first = fp.compute_fingerprint(tmp_path, 'bn_snn', 'tag', seed)
    (cache / 'train_manifest.json').write_text('{"4":"changed"}')
    second = fp.compute_fingerprint(tmp_path, 'bn_snn', 'tag', seed)
    assert first['sha256'] != second['sha256']
    core = fred / 'phase1_detection/seeds/bn_snn/network.py'
    core.write_text(core.read_text() + '\n# changed\n')
    assert second['sha256'] != fp.compute_fingerprint(tmp_path, 'bn_snn', 'tag', seed)['sha256']
