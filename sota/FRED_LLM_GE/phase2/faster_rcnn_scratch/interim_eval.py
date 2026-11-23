"""Evaluate a frozen epoch-8 finalist snapshot, without training or final outputs."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from sota.FRED_LLM_GE.phase1_detection import map_reporter
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import FasterRCNNScratchAdapter
from sota.FRED_LLM_GE.phase1_detection.candidate_runner import DEFAULT_PROTOCOL, FRED_DIR, load_protocol
from sota.FRED_LLM_GE.phase1_detection.export_split import dataset_parts
from sota.FRED_LLM_GE.phase1_detection.faster_rcnn_baseline import identity
from sota.FRED_LLM_GE.phase1_detection.final_eval import predict_test_split
from sota.FRED_LLM_GE.phase1_detection.fred_splits import challenging_split
from sota.FRED_LLM_GE.phase1_detection.proxy_data import REVISION
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.finalize import checkpoint_matches
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import file_sha256, freeze_manifest
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.train_eval import set_random_seed, write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def verify(run_root, output):
    """Validate snapshot, frozen finalist, and unchanged training/data protocol."""
    request = load_json(output / 'request.json')
    if (request['completed_epochs'] != 8 or request['purpose'] != 'interim_official_test'
            or request['use_for_selection'] is not False):
        raise ValueError('Invalid interim evaluation request')
    if request['runner_sha256'] != file_sha256(__file__):
        raise ValueError('Evaluation runner changed since submission')
    selection_path = run_root / 'final/finalists.json'
    if file_sha256(selection_path) != request['selection_sha256']:
        raise ValueError('Frozen finalist selection changed')
    selection = load_json(selection_path)
    if selection['identity'] != identity(run_root):
        raise ValueError('Run code, dependencies, data, or protocol changed')
    finalist, = [f for f in selection['finalists'] if f['gene_id'] == request['gene_id']]
    source = run_root / finalist['source']
    if file_sha256(source) != finalist['source_sha256']:
        raise ValueError('Frozen model source changed')
    checkpoint = output / 'checkpoint.pt'
    if file_sha256(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('Interim checkpoint snapshot changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['completed_epochs'] != 8:
        raise ValueError('Snapshot is not epoch eight')
    protocol = load_protocol(DEFAULT_PROTOCOL)
    family = dict(protocol['families']['faster_rcnn_scratch'], search=selection['training'])
    settings = dict(protocol['proxy'], **selection['training'])
    settings.update(train_frame_step=protocol['challenging']['final']['train_frame_step'],
                    test_frame_step=protocol['challenging']['final']['test_frame_step'],
                    events=protocol['challenging']['events'])
    if settings['test_frame_step'] != 1:
        raise ValueError('Official interim test must include every frame')
    data_root = FRED_DIR / protocol['challenging']['data_root']
    (train_root, train, _), = dataset_parts(protocol['challenging'], data_root, 'final')
    if len(train) != 172:
        raise ValueError('Expected 172 official training sequences')
    markers = {s: file_sha256(train_root / 'done' / f'{s}.json') for s, _ in train}
    provenance = {
        'selection_sha256': file_sha256(selection_path),
        'architecture_sha256': finalist['architecture_sha256'],
        'source_sha256': finalist['source_sha256'], 'settings': settings, 'family': family,
        'final_export_sha256': hashlib.sha256(json.dumps(markers, sort_keys=True).encode()).hexdigest()}
    checkpoint_matches(saved, provenance)
    return request, selection, finalist, saved, settings, family, data_root


def evaluate(run_root, output, verify_only=False):
    with (output / '.evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request, selection, finalist, saved, settings, family, data_root = verify(run_root, output)
        if verify_only:
            print('Verified: exact epoch-8 snapshot, source, initialization metadata, and training provenance', flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('Interim evaluation requires a CUDA allocation')
        set_random_seed(settings['seed'])
        adapter = FasterRCNNScratchAdapter(family, FRED_DIR)
        model = adapter.build(run_root / finalist['source'], 1, 'cuda')
        if adapter.initialization != saved['initialization']:
            raise ValueError('Checkpoint fresh initialization identity differs')
        if adapter.param_count(model) != finalist['params']:
            raise ValueError('Parameter count differs from frozen finalist')
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        model.eval()
        _, test = challenging_split(data_root)
        if len(test) != 59 or len(set(test)) != 59:
            raise ValueError('Expected 59 distinct official test sequences')
        evaluation_identity = dict(request=request, test_sequences=test)
        # JSON normalizes tuples to lists; use the same representation on resume.
        evaluation_identity = json.loads(json.dumps(evaluation_identity))
        frames, expected = [], []
        for number, (sequence, folder) in enumerate(test, 1):
            chunk = output / 'predictions' / f'{sequence}.json'
            chunk_identity = dict(evaluation=evaluation_identity, sequence=sequence, folder=folder)
            if chunk.exists():
                record = load_json(chunk)
                if record['identity'] != chunk_identity:
                    raise ValueError(f'Cached predictions differ: {sequence}')
            else:
                sequence_frames, ids = predict_test_split(
                    adapter, model, [(sequence, folder)], settings, 'cuda', output)
                record = {'identity': chunk_identity, 'frames': sequence_frames, 'expected': ids}
            ids = record['expected']
            actual = [f['sample_id'] for f in record['frames']]
            if (not ids or len(set(ids)) != len(ids) or actual != ids
                    or any(not s.startswith(f'{folder}/{sequence}:') for s in ids)):
                raise ValueError(f'Incomplete/invalid test sequence: {sequence}')
            if not chunk.exists():
                write_json(chunk, record)
            frames.extend(record['frames'])
            expected.extend(ids)
            write_json(output / 'status.json', {
                'status': 'evaluating', 'completed_epochs': 8, 'completed_sequences': number,
                'total_sequences': 59, 'frames_so_far': len(expected),
                'checkpoint_sha256': request['checkpoint_sha256'], 'use_for_selection': False})
            print(f'Interim epoch 8: {number}/59 sequences; {len(expected)} frames', flush=True)
        metadata = {
            'dataset_revision': REVISION, 'manifest_identity': selection['identity']['sha256'],
            'split_identity': 'official_challenging_train:172;official_challenging_test:59',
            'annotation_policy': 'fred_coordinates_txt_clipped_to_image',
            'model_identity': finalist['source_sha256'],
            'checkpoint_identity': request['checkpoint_sha256'],
            'code_revision': selection['identity']['git_commit'],
            'postprocessing': adapter.postprocessing(settings),
            'training_config_identity': hashlib.sha256(
                json.dumps(evaluation_identity, sort_keys=True).encode()).hexdigest(),
            'modality': 'event', 'benchmark_split': 'challenging',
            'project_split': 'held_out_test', 'official_split': 'challenging_test',
            'initialization_policy': 'fresh_framework_random',
            'initial_state_sha256': adapter.initialization['initial_state_sha256'],
            'evaluation_stage': 'interim_epoch8_not_final', 'completed_training_epochs': 8,
            'planned_training_epochs': 20, 'use_for_selection': False,
            'runner_sha256': request['runner_sha256']}
        name = f"{finalist['gene_id']}_epoch8_interim"
        report_path = output / 'held_out_reports' / f'{name}_evaluation.json'
        if report_path.exists():
            report = load_json(report_path)
            if report['metadata'] != metadata or report['num_frames'] != len(expected):
                raise ValueError('Existing interim report differs')
        else:
            result, report_path = map_reporter.report({
                'metadata': metadata, 'frames': frames, 'expected_sample_ids': expected,
                'params': finalist['params']}, sota_root=output, gene_id=name, mode='held-out')
            report = load_json(report_path)
        verify(run_root, output)
        write_json(output / 'status.json', {
            'status': 'complete', 'completed_epochs': 8, 'completed_sequences': 59,
            'test_frames': len(expected), 'metrics': report['metrics'],
            'report': str(report_path), 'use_for_selection': False,
            'checkpoint_sha256': request['checkpoint_sha256']})
        print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    evaluate(args.run_root.resolve(), args.output.resolve(), args.verify_only)
