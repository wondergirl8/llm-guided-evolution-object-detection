"""One GPU task: smoke or independently initialized protocol-keyed train/evaluate."""
import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from . import common, genome
from .training import Paused

STOP = False


def task_epochs(protocol, task):
    epochs = protocol.get('final_epochs', protocol.get('epochs', 20)) if task['kind'] == 'final' else protocol.get('epochs', 20)
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Invalid epoch budget')
    return epochs


def stop_signal(*_):
    global STOP
    STOP = True


def resources(task_dir, start=False):
    path = task_dir / 'resources.json'
    old = common.read(path) if path.exists() else {}
    job = os.getenv('SLURM_JOB_ID', 'local')
    if start and old.get('allocation_job') != job:
        old.update(allocation_job=job,
                   prior_elapsed_gpu_seconds=old.get('elapsed_gpu_seconds', 0),
                   allocation_start=float(os.getenv('SLURM_JOB_START_TIME', time.time())))
    if old.get('allocation_job') == job:
        old['elapsed_gpu_seconds'] = old.get('prior_elapsed_gpu_seconds', 0) + time.time() - old['allocation_start']
        old['peak_memory'] = max(old.get('peak_memory', 0), torch.cuda.max_memory_allocated())
        common.write(path, old)
    return old


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        with temp.open('wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def build(genes, seed, device='cuda'):
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
    from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import preserving_preflight
    model, initialization = build_detector(genome.config_from_genes(genes), seed=seed)
    model.to(device)
    preserving_preflight(model)
    return model, initialization


def adapter_for(model, genes, family, initialization):
    from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import FasterRCNNScratchAdapter
    adapter = FasterRCNNScratchAdapter(family, Path(__file__).resolve().parents[3])
    adapter.genes, adapter.initialization = genes, initialization
    return adapter


def smoke(run, task_dir, protocol, identity):
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import prepare_batch
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import state_sha256
    from sota.FRED_LLM_GE.phase1_detection.candidate_runner import predict_frames
    cases = [genome.stock(w) for w in protocol.get('widths', (640, 960, 1280))]
    largest = {k: values[-1] for group in genome.GROUPS.values() for k, values in group.items()}
    largest['anchor_ratios'] = [0.5, 1.0, 2.0]
    cases.append(largest)
    records = []
    for index, genes in enumerate(cases):
        if STOP:
            raise Paused('Smoke interrupted; rerun checks')
        width, height = genes['img_width'], round(720 * genes['img_width'] / 1280)
        model, init = build(genes, 0)
        assert state_sha256(model) == init['initial_state_sha256']
        from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
        duplicate, repeated = build_detector(genome.config_from_genes(genes), seed=0)
        assert repeated == init
        del duplicate
        torch.cuda.reset_peak_memory_stats()
        boxes = torch.tensor([[width*.3, height*.3, width*.5, height*.5]])
        data = {'images': torch.randint(0, 256, (2, 3, height, width), dtype=torch.uint8),
                'boxes': [boxes, torch.empty(0, 4)], 'orig_boxes': [boxes, torch.empty(0, 4)],
                'orig_size': (width, height), 'sample_ids': ['positive', 'empty']}
        model.train()
        loss, _ = model.parse_losses(model(**prepare_batch(model, data, [0, 1], True), mode='loss'))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        for name in ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'):
            assert any(p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
                       for p in getattr(model.backbone, name).parameters()), name
        optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0001)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        updated = state_sha256(model)
        assert updated != init['initial_state_sha256']
        path = task_dir / f'smoke-{index}.pt'
        atomic_torch(path, model.state_dict())
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        assert state_sha256(model) == updated
        path.unlink()
        settings = dict(protocol['evaluation'], batch_size=2, img_width=width)
        adapter = adapter_for(model, genes, protocol['optimizer'], init)
        frames = predict_frames(adapter, model, data, settings, 'cuda')
        assert len(frames) == 2 and frames[1]['target']['boxes'] == []
        for frame in frames:
            for x1, y1, x2, y2 in frame['prediction']['boxes']:
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        records.append({'width': width, 'largest': index == len(cases) - 1,
                        'peak_memory': torch.cuda.max_memory_allocated(), 'loss': loss.item()})
        common.write(task_dir / 'progress.json', {'smoke_cases': len(records)})
        del model, adapter, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()
    common.write(task_dir / 'result.json', {'status': 'complete', 'identity': identity, 'cases': records})


def task_identity(run, task, manifest):
    return genome.digest({'protocol': manifest['protocol_sha256'],
                          'sources': manifest['source_hashes'], 'data': manifest['data_hashes'],
                          'dependencies': manifest['detector_versions'], 'task': task})


def execute(run, task_id):
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import load_exported_split, REVISION
    from sota.FRED_LLM_GE.phase1_detection.candidate_runner import predict_frames
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import train
    from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import resized_data
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.train_eval import set_random_seed
    from sota.FRED_LLM_GE.phase1_detection.final_eval import predict_test_split, faster_rcnn_command
    from sota.FRED_LLM_GE.phase1_detection import map_reporter
    run = Path(run)
    task_dir = run / 'tasks' / task_id
    with common.locked(task_dir / '.worker.lock'):
        manifest = common.verify_run(run, detector=True)
        protocol = common.read(run / 'protocol.json')
        task = common.read(task_dir / 'task.json')
        if task['kind'] == 'final':
            faster_rcnn_command(run, task_id)  # Same gate as the shared CLI, before training.
        epochs = task_epochs(protocol, task)
        identity = task_identity(run, task, manifest)
        result_path = task_dir / 'result.json'
        if result_path.exists():
            if common.read(result_path)['identity'] != identity:
                raise ValueError('Completed task identity mismatch')
            return
        if not torch.cuda.is_available():
            raise RuntimeError('gpu_unavailable: CUDA allocation is required')
        resources(task_dir, start=True)
        if task['kind'] == 'smoke':
            smoke(run, task_dir, protocol, identity)
            return
        genes = genome.parse((task_dir / 'source.py').read_text())
        if genes['img_width'] not in protocol['widths']:
            raise ValueError('Candidate resolution is outside the frozen protocol')
        if genes != task['genes']:
            raise ValueError('Candidate source differs from task')
        set_random_seed(task['seed'])
        model, initialization = build(genes, task['seed'])
        architecture, candidate = genome.resolved_identity(genes, model)
        selected = protocol.get('selected_architecture')
        if selected is not None and candidate != selected['candidate_hash']:
            raise ValueError('Resolved architecture differs from selected search model')
        common.freeze(task_dir / 'resolved.json', {
            'architecture_hash': architecture, 'candidate_hash': candidate,
            'identity': identity, 'initialization': initialization,
            'params': sum(p.numel() for p in model.parameters())})
        params = sum(p.numel() for p in model.parameters())
        if protocol.get('expected_params') is not None and params != protocol['expected_params']:
            raise ValueError('Selected architecture parameter count differs from source')
        settings = dict(protocol['evaluation'], batch_size=2, img_width=genes['img_width'],
                        epochs=epochs, seed=task['seed'], test_frame_step=1,
                        events=protocol['events'])
        weights, progress = task_dir / 'model.pt', task_dir / 'training.pt'
        torch.cuda.reset_peak_memory_stats()
        training_seconds = 0.0
        allocation_stats = task_dir / 'resources.json'
        previous_peak = common.read(allocation_stats)['peak_memory'] if allocation_stats.exists() else 0
        training_records = protocol['train'] if task['kind'] != 'final' else protocol['final_train']
        training_root = protocol.get('final_train_root', protocol['train_root']) if task['kind'] == 'final' else protocol['train_root']
        training_step = protocol.get('final_train_frame_step', 5) if task['kind'] == 'final' else protocol['train_frame_step']
        if weights.exists():
            saved = torch.load(weights, map_location='cpu', weights_only=False)
            if saved['identity'] != identity or saved['initialization'] != initialization:
                raise ValueError('Completed checkpoint provenance mismatch')
            model.load_state_dict(saved['state_dict'], strict=True)
            training_seconds = saved['training_seconds']
            del saved
        else:
            saved = torch.load(progress, map_location='cpu', weights_only=False) if progress.exists() else None
            if saved is not None and (saved['identity'] != identity or saved['initialization'] != initialization):
                raise ValueError('Training checkpoint provenance mismatch')
            data = load_exported_split(training_root, training_records, expected_step=training_step)

            def save(state):
                atomic_torch(progress, dict(state, identity=identity, initialization=initialization))
                common.write(task_dir / 'progress.json', {
                    'epoch': state['epoch'], 'steps': state['steps'],
                    'next_position': state['next_position'], 'stage': 'training'})
                resources(task_dir)

            args = SimpleNamespace(epochs=epochs, batch_size=2, **protocol['optimizer'])
            stats = train(model, resized_data(data, genes['img_width']), args,
                          resume_state=saved, progress_callback=save, should_stop=lambda: STOP)
            training_seconds = stats['training_seconds']
            atomic_torch(weights, {'state_dict': model.state_dict(), 'identity': identity,
                                  'initialization': initialization, 'training_seconds': training_seconds})
            del data, saved
        adapter = adapter_for(model, genes, protocol['optimizer'], initialization)
        frames, expected = [], []
        is_final = task['kind'] == 'final'
        if is_final and not common.read(run / 'state.json').get('approved'):
            raise ValueError('Final testing requires approved completed evolution')
        if is_final:
            selection = common.read(run / 'finalists.json')
            expected_generation = protocol.get('selection_generation', 20) if protocol.get('experiment') == 'final_only' else 20
            if selection['generation'] != expected_generation or not any(f['genes'] == genes for f in selection['finalists']):
                raise ValueError('Task is not in frozen final selection')
        sequences = protocol['test'] if is_final else protocol['validation']
        checkpoint_sha = common.sha(weights)
        for number, (sequence, folder) in enumerate(sequences, 1):
            if STOP:
                raise Paused('Continue sequence evaluation in next allocation')
            path = task_dir / 'predictions' / f'{sequence}.json'
            def predict():
                if is_final:
                    predicted, ids = predict_test_split(adapter, model, [(sequence, folder)],
                                                        settings, 'cuda', task_dir)
                else:
                    data = load_exported_split(protocol['validation_root'], [(sequence, folder)], expected_step=10)
                    predicted = predict_frames(adapter, model, data, settings, 'cuda')
                    ids = data['sample_ids']
                return predicted, ids
            chunk = common.cached_predictions(path, identity, checkpoint_sha, predict,
                                               f'{folder}/{sequence}:')
            ids = chunk['expected']
            expected.extend(ids)
            frames.extend(chunk['frames'])
            common.write(task_dir / 'progress.json', {'epoch': epochs, 'sequences': number,
                                                      'stage': 'test' if is_final else 'validation'})
        metadata = {
            'dataset_revision': REVISION, 'manifest_identity': manifest['protocol_sha256'],
            'split_identity': 'official_challenging_152_train_20_val' if not is_final else 'official_challenging_172_train_59_test',
            'annotation_policy': 'fred_coordinates_txt_clipped_to_image',
            'model_identity': common.sha(task_dir / 'source.py'), 'checkpoint_identity': checkpoint_sha,
            'code_revision': manifest['git_commit'], 'postprocessing': adapter.postprocessing(settings),
            'training_config_identity': identity, 'modality': 'event', 'benchmark_split': 'challenging',
            'project_split': 'held_out_test' if is_final else 'validation',
            'official_split': 'challenging_test' if is_final else 'challenging_train',
            'img_width': genes['img_width'], 'completed_epochs': epochs, 'seed': task['seed'],
            'train_frame_step': training_step,
            'initialization': initialization, 'candidate_identity': candidate,
            'architecture_identity': architecture}
        result, report = map_reporter.report({'metadata': metadata, 'frames': frames,
                                              'expected_sample_ids': expected, 'params': params},
                                             sota_root=task_dir, gene_id=task_id,
                                             mode='held-out' if is_final else 'validation')
        common.verify_run(run, detector=True)
        if common.sha(weights) != checkpoint_sha:
            raise ValueError('Model checkpoint changed during evaluation')
        resource_record = resources(task_dir)
        common.write(result_path, {
            'status': 'complete', 'identity': identity,
            'fitness': [result.map50, result.map50_95, result.params],
            'candidate_hash': candidate, 'architecture_hash': architecture,
            'checkpoint_sha256': common.sha(weights), 'report': str(report),
            'training_seconds': training_seconds, 'img_width': genes['img_width'],
            'elapsed_gpu_seconds': resource_record['elapsed_gpu_seconds'],
            'peak_memory': max(previous_peak,
                               common.read(allocation_stats)['peak_memory'] if allocation_stats.exists() else 0,
                               torch.cuda.max_memory_allocated()),
            'images_per_second': len(training_records) and sum(
                common.read(Path(training_root) / 'done' / f'{s}.json')['frames']
                for s, _ in training_records) * epochs / training_seconds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop_signal)
    out = args.run / 'tasks' / args.task
    try:
        execute(args.run, args.task)
        common.write(out / 'allocation.json', {'state': 'complete'})
    except Paused as exc:
        common.write(out / 'allocation.json', {'state': 'paused', 'reason': str(exc)})
    except Exception as exc:
        message = str(exc)
        infra = any(s in message.lower() for s in ('gpu_unavailable', 'input/output error', 'stale file handle', 'connection', 'timed out'))
        common.write(out / 'allocation.json', {'state': 'infrastructure_failure' if infra else 'failed',
                                               'reason': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if (out / 'resources.json').exists() and torch.cuda.is_available():
            resources(out)


if __name__ == '__main__':
    main()
