"""Separately authorized final retraining after an explicitly stopped search."""
import csv
import json
from pathlib import Path
import pickle
import statistics

from . import common, genome


def verify_reused_exports(exports, source_run):
    """Reuse pinned export verification; workers still stat every image before SGD."""
    expected = common.read(Path(source_run).resolve() / 'manifest.json')['data_hashes']
    for root, members, step in exports:
        size = None
        for sequence, folder in members:
            path = Path(root) / 'done' / f'{sequence}.json'
            if expected.get(str(path)) != common.sha(path):
                raise ValueError('Export differs from previously verified source: ' + str(path))
            marker = common.read(path)
            if (str(marker.get('sequence')) != str(sequence) or marker.get('hf_folder') != folder
                    or marker.get('frame_step') != step
                    or marker.get('frames', 0) <= 0
                    or marker['frames'] != len(marker.get('records', []))):
                raise ValueError('Final export marker identity mismatch')
            size = size or marker['size']
            if marker['size'] != size:
                raise ValueError('Final export dimensions differ')


def configure(protocol, source_run, model_id, epochs, cutoff):
    source_run = Path(source_run).resolve()
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Final epochs must be a positive integer')
    if protocol['train_frame_step'] != 5:
        raise ValueError('Final-only preparation requires the verified step-5 training export')
    state = common.read(source_run / 'state.json')
    if state['stage'] != 'stopped' or not (source_run / 'STOP').exists():
        raise ValueError('Final-only evaluation requires a stopped source search')
    checkpoint = source_run / 'checkpoints' / f'global_gen_{cutoff}.pkl'
    archive = pickle.loads(checkpoint.read_bytes())['GLOBAL_DATA']
    if model_id not in archive:
        raise ValueError('Selected model was not scored by the approved generation cutoff')
    task = state['tasks'][model_id]
    result_path = source_run / 'tasks' / model_id / 'result.json'
    result = common.read(result_path)
    if task['status'] != 'complete' or result.get('status') != 'complete' or list(archive[model_id]['fitness']) != result['fitness']:
        raise ValueError('Selected validation fitness differs from successful cutoff archive')
    source = source_run / 'tasks' / model_id / 'source.py'
    genes = genome.parse(source.read_text())
    if genes != task['spec']['genes'] or genes['img_width'] != 1280:
        raise ValueError('Selected model source or resolution differs')
    original = common.read(source_run / 'protocol.json')
    if (list(map(tuple, protocol['final_train'])) != list(map(tuple, original['final_train']))
            or list(map(tuple, protocol['test'])) != list(map(tuple, original['test']))):
        raise ValueError('Official final train/test membership changed')
    selected = {'search_id': model_id, 'genes': genes,
                'candidate_hash': result['candidate_hash'], 'source_sha256': common.sha(source),
                'validation_fitness': result['fitness'], 'stock_control': False,
                'source_run': str(source_run), 'source_checkpoint': str(checkpoint),
                'source_checkpoint_sha256': common.sha(checkpoint),
                'source_manifest_identity': common.json_sha(source_run / 'manifest.json')}
    protocol = dict(protocol, experiment='final_only', widths=[1280],
                    selection_generation=cutoff, final_epochs=epochs,
                    final_seeds=list(range(1, 31)), selected_architecture=selected,
                    expected_params=int(result['fitness'][2]),
                    final_train_root=protocol['train_root'], final_train_frame_step=5,
                    scheduler_excluded_nodes=['atl1-1-03-014-16-0'],
                    approval_required_after_initial=False,
                    final_initialization='Fresh deterministic random weights for each seed; no search checkpoint loading',
                    statistics={'primary_metric': 'mAP50_95',
                                'comparison': 'Descriptive single-model seed distribution; no stock comparison yet'})
    return protocol, [source, checkpoint]


def initialize(run):
    from . import workflow
    protocol = common.read(run / 'protocol.json')
    selected = protocol['selected_architecture']
    common.freeze(run / 'authorization.json', {
        'type': 'final_only_after_search_cutoff',
        'protocol_sha256': common.json_sha(run / 'protocol.json'),
        'selection_generation': protocol['selection_generation'],
        'final_epochs': protocol['final_epochs'], 'final_seeds': protocol['final_seeds'],
        'models': [selected['search_id']],
        'basis': f'User selected {selected["search_id"]} with {protocol["final_epochs"]} epochs after cutoff {protocol["selection_generation"]}'})
    common.freeze(run / 'finalists.json', {
        'generation': protocol['selection_generation'],
        'protocol_sha256': common.json_sha(run / 'protocol.json'),
        'selection_policy': 'User-selected architecture frozen before official testing',
        'finalists': [selected], 'seeds': protocol['final_seeds']})
    state = dict(stage='final_training', generation=protocol['selection_generation'],
                 approved=True, tasks={}, seen={}, population=[], final_tasks=[])
    for seed in protocol['final_seeds']:
        name = f'final_000_s{seed:02d}'
        workflow.add_task(run, state, name, 'final', selected['genes'], seed=seed)
        state['final_tasks'].append(name)
    common.write(run / 'state.json', state)
    common.write(run / 'status.json', {'stage': 'final_training', 'generation': state['generation'],
                                      'completed_tasks': 0, 'total_tasks': len(state['tasks']), 'reason': None})


def summarize(run, state):
    from scipy.stats import t
    protocol = common.read(run / 'protocol.json')
    rows = []
    for name in state['final_tasks']:
        record = state['tasks'][name]
        if record['status'] != 'complete':
            raise ValueError('Every scheduled final seed must finish before publishing summary')
        result = record['result']
        rows.append(dict(task=name, seed=record['spec']['seed'],
                         test_mAP50=result['fitness'][0], test_mAP50_95=result['fitness'][1],
                         parameters=result['fitness'][2], checkpoint_sha256=result['checkpoint_sha256'],
                         report=result['report']))
    metrics = {}
    for key in ('test_mAP50', 'test_mAP50_95'):
        values = [100 * row[key] for row in rows]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        margin = float(t.ppf(.975, len(values)-1)) * sd / len(values)**.5
        metrics[key] = dict(mean_pp=mean, sample_sd_pp=sd,
                            mean_ci95_pp=[mean-margin, mean+margin], values_pp=values)
    common.write(run / 'reports/final_seed_statistics.json', {
        'model': protocol['selected_architecture']['search_id'],
        'final_epochs': protocol['final_epochs'], 'seeds': protocol['final_seeds'],
        'metrics': metrics, 'samples': rows,
        'comparison': 'Single architecture only; no significance claim against stock or paper'})
    with (run / 'reports/final_seed_samples.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def advance(run, state, scheduler):
    from . import workflow
    workflow.reconcile(run, state, scheduler)
    if state['stage'] in workflow.TERMINAL:
        return
    if state['stage'] != 'final_training':
        raise ValueError('Final-only run cannot perform evolution')
    if workflow.check_required(state, state['final_tasks']):
        summarize(run, state)
        state['stage'] = 'complete'
    if state['stage'] not in workflow.TERMINAL:
        workflow.launch_pending(run, state, scheduler)
