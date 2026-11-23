"""Persistent generation state machine; safe to advance after any allocation."""
import copy
import csv
import pickle
import random
from pathlib import Path

from . import common, genome, proposals
from .slurm import ACTIVE

TERMINAL = {'awaiting_budget_approval', 'complete', 'blocked', 'stopped'}


def stock_id(width, seed=0):
    return 'seed' if (width, seed) == (640, 0) else f'stock{width}_s{seed}'


def add_task(run, state, name, kind, genes=None, seed=0, slot=None):
    key = None
    if genes is not None and seed == 0 and kind != 'final':
        key = genome.candidate_identity(genes)
        if state['seen'].get(key) not in (None, name):
            raise ValueError('Globally repeated candidate')
    spec = {'kind': kind, 'seed': seed}
    if genes is not None:
        spec['genes'] = genome.validate(genes)
    common.freeze(run / 'tasks' / name / 'task.json', spec)
    if genes is not None:
        path = run / 'tasks' / name / 'source.py'
        text = genome.render(genes)
        if path.exists() and path.read_text() != text:
            raise ValueError('Candidate source changed')
        path.write_text(text)
        models = run / 'models'
        models.mkdir(exist_ok=True)
        (models / f'network_{name}.py').write_text(text)
    if name not in state['tasks']:
        state['tasks'][name] = {'spec': spec, 'status': 'pending', 'attempts': [],
                                'infra_failures': 0, 'no_progress': 0, 'slot': slot}
    if key is not None:
        state['seen'][key] = name


def progress_token(run, name):
    path = run / 'tasks' / name / 'progress.json'
    return common.read(path) if path.exists() else None


def reconcile(run, state, scheduler):
    for name, record in state['tasks'].items():
        if record['status'] in ('complete', 'failed'):
            continue
        path = run / 'tasks' / name / 'result.json'
        if path.exists():
            result = common.read(path)
            manifest = common.read(run / 'manifest.json')
            expected = genome.digest({'protocol': manifest['protocol_sha256'],
                                      'sources': manifest['source_hashes'], 'data': manifest['data_hashes'],
                                      'dependencies': manifest['detector_versions'], 'task': record['spec']})
            if result.get('status') != 'complete' or result.get('identity') != expected:
                raise ValueError('Task result identity mismatch: ' + name)
            if record['spec']['kind'] != 'smoke':
                from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import valid_fitness
                if not valid_fitness(result.get('fitness')):
                    raise ValueError('Invalid completed task fitness: ' + name)
                # Reject two literal genomes that resolve to the same effective candidate.
                same = [r for n, r in state['tasks'].items() if n != name
                        and r.get('result', {}).get('candidate_hash') == result['candidate_hash']
                        and r['spec']['seed'] == record['spec']['seed']
                        and r['spec']['kind'] == record['spec']['kind']]
                if same:
                    raise ValueError('Resolved duplicate escaped literal deduplication')
            record.update(status='complete', result=result)
            continue
        job = record.get('job')
        if not job:
            continue
        status = scheduler.state(job)
        if status == 'UNKNOWN':
            record['unknown_checks'] = record.get('unknown_checks', 0) + 1
            if record['unknown_checks'] > 6:
                state.update(stage='blocked', reason=f'Unable to reconcile scheduler job {job}; no duplicate submitted')
                return
            continue
        record['unknown_checks'] = 0
        if status in ACTIVE:
            continue
        path = run / 'tasks' / name / 'allocation.json'
        allocation = common.read(path) if path.exists() else {}
        current = progress_token(run, name)
        previous = record['attempts'][-1]['start_progress']
        if status == 'CANCELLED':
            state.update(stage='stopped', reason=f'Explicit cancellation of {name} ({job})')
            return
        if allocation.get('state') == 'failed':
            record.update(status='failed', reason=allocation.get('reason', 'Model failure'))
        elif status in ('TIMEOUT', 'PREEMPTED') or allocation.get('state') == 'paused':
            record['no_progress'] = record['no_progress'] + 1 if current == previous else 0
            if record['no_progress'] >= 2:
                state.update(stage='blocked', reason=f'Two allocations without durable progress: {name}')
                return
            record['status'] = 'pending'
        elif (allocation.get('state') == 'infrastructure_failure'
              or status in ('NODE_FAIL', 'BOOT_FAIL', 'REVOKED')):
            record['infra_failures'] += 1
            if record['infra_failures'] > 3:
                state.update(stage='blocked', reason=f'Infrastructure retry limit: {name}')
                return
            record['status'] = 'pending'
        else:
            record.update(status='failed', reason=f'Unexpected worker termination: {status}; {allocation}')
        record['last_job_state'] = status
        record.pop('job', None)


def launch_pending(run, state, scheduler):
    occupied = sum(bool(r.get('job')) and scheduler.state(r['job']) in ACTIVE
                   for r in state['tasks'].values())
    for name, record in state['tasks'].items():
        if occupied >= 2:
            break
        if record['status'] not in ('pending', 'submitting'):
            continue
        if record['status'] == 'pending':
            attempt = len(record['attempts'])
            allocation = run / 'tasks' / name / 'allocation.json'
            if allocation.exists():
                allocation.rename(allocation.with_name(f'allocation-{attempt-1}.json'))
            record['submission_label'] = f'{name}_a{attempt}'
            record['attempts'].append({'job': None, 'start_progress': progress_token(run, name)})
            record['status'] = 'submitting'
            common.write(run / 'state.json', state)
        job = scheduler.submit(record['submission_label'], 'worker', task=name)
        record['job'] = job
        record['status'] = 'running'
        record['attempts'][-1]['job'] = job
        common.write(run / 'state.json', state)
        occupied += 1


def scored(state, names):
    return {name: {'fitness': state['tasks'][name]['result']['fitness']}
            for name in names if state['tasks'][name]['status'] == 'complete'}


def front(state, names):
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import pareto_front
    return [name for name, _ in pareto_front(scored(state, names))]


def select(state, names, k, parents=False):
    from deap import base, creator, tools
    if not hasattr(creator, 'FREDV2Fitness'):
        creator.create('FREDV2Fitness', base.Fitness, weights=(1., 1., -1.))
        creator.create('FREDV2Individual', list, fitness=creator.FREDV2Fitness)
    pop = []
    for name in names:
        individual = creator.FREDV2Individual([name])
        individual.fitness.values = state['tasks'][name]['result']['fitness']
        pop.append(individual)
    if not parents:
        return [i[0] for i in tools.selSPEA2(pop, k)]
    ranked = tools.selNSGA2(pop, len(pop))
    rng = proposals.state_rng(state)
    previous = random.getstate()
    try:
        random.setstate(rng.getstate())
        chosen = tools.selTournamentDCD(ranked, len(ranked))[:k]
        rng.setstate(random.getstate())
    finally:
        random.setstate(previous)
    proposals.save_rng(state, rng)
    return [i[0] for i in chosen]


def report(run, state, generation):
    names = [n for n, r in state['tasks'].items() if r['status'] == 'complete'
             and r['spec']['kind'] == 'search' and r['spec']['seed'] == 0]
    archive = {'GLOBAL_DATA': scored(state, names), 'GLOBAL_DATA_HIST': {},
               'GLOBAL_DATA_ANCESTRY': state.get('ancestry', {})}
    directory = run / 'checkpoints'
    directory.mkdir(exist_ok=True)
    temp = directory / f'global_gen_{generation}.tmp'
    temp.write_bytes(pickle.dumps(archive))
    temp.replace(directory / f'global_gen_{generation}.pkl')
    result = {'generation': generation, 'front': front(state, names),
              'unique_successful': len(names), 'population': state['population']}
    common.write(run / 'reports' / f'pareto_gen_{generation}.json', result)
    if generation == 0 and (not (run / 'protocol.json').exists() or common.read(run / 'protocol.json').get('experiment') != 'procedure1280'):
        import statistics
        estimates = {}
        for width in (640, 960, 1280):
            records = [r['result'] for r in state['tasks'].values() if r['status'] == 'complete'
                       and r['spec'].get('genes', {}).get('img_width') == width]
            mean_hours = sum(r.get('elapsed_gpu_seconds', r['training_seconds']) for r in records) / len(records) / 3600
            estimates[str(width)] = {'mean_20_epoch_training_and_validation_gpu_hours': mean_hours,
                                      'remaining_120_candidates_gpu_hours': mean_hours * 120,
                                      'two_gpu_calendar_days_excluding_queue': mean_hours * 120 / 48}
        baseline = {str(width): [state['tasks'][stock_id(width, seed)]['result']['fitness']
                                 for seed in (0, 1, 2)] for width in (640, 960, 1280)}
        variability = {}
        for width, scores in baseline.items():
            if len({s[2] for s in scores}) != 1:
                raise ValueError('Stock parameter counts differ across seeds')
            variability[width] = {metric: {'mean': statistics.mean(s[i] for s in scores),
                                           'sample_sd': statistics.stdev(s[i] for s in scores)}
                                   for i, metric in enumerate(('mAP50', 'mAP50_95'))}
        common.write(run / 'reports/initial_review.json', dict(result, baseline_seeds=baseline,
                                                              baseline_variability=variability,
                                                              compute_estimates=estimates))
    models = [(n, r) for n, r in state['tasks'].items()
              if r['status'] == 'complete' and r['spec']['kind'] != 'smoke']
    path = run / 'reports/candidates.csv'
    with path.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'kind', 'seed', 'width', 'mAP50', 'mAP50_95', 'params', 'training_seconds', 'elapsed_gpu_seconds', 'peak_memory'])
        for name, record in models:
            writer.writerow([name, record['spec']['kind'], record['spec']['seed'],
                             record['spec']['genes']['img_width'], *record['result']['fitness'],
                             record['result']['training_seconds'], record['result'].get('elapsed_gpu_seconds'),
                             record['result']['peak_memory']])
    if generation in (0, 20) and (run / 'code/analysis/scripts/fred_pareto_graphs.py').exists():
        import subprocess
        subprocess.run([str(run / 'code/.venv/bin/python'),
                        str(run / 'code/analysis/scripts/fred_pareto_graphs.py'),
                        '--family', 'faster_rcnn_scratch', '--checkpoints', str(directory),
                        '--models_dir', str(run / 'models'), '--seed_file', str(
                            run / 'tasks/seed/source.py' if common.read(run / 'protocol.json').get('experiment') == 'procedure1280' else genome.SEED),
                        '--title', 'FRED Faster R-CNN V2: 20-epoch search validation',
                        '--output', str(run / 'plots')], check=True)


def stop_server(state, scheduler):
    if state.get('server_job'):
        scheduler.cancel(state.pop('server_job'))


def check_required(state, names):
    failed = [n for n in names if state['tasks'][n]['status'] == 'failed']
    if failed:
        state.update(stage='blocked', reason='Required stock/smoke/final tasks failed: ' + ', '.join(failed))
        return False
    return all(state['tasks'][n]['status'] == 'complete' for n in names)


def advance(run, state, scheduler):
    if (run / 'protocol.json').exists() and common.read(run / 'protocol.json').get('experiment') == 'final_only':
        from .final_only import advance as advance_final
        return advance_final(run, state, scheduler)
    if (run / 'protocol.json').exists() and common.read(run / 'protocol.json').get('experiment') == 'procedure1280':
        from .procedure1280 import advance as advance_procedure
        return advance_procedure(run, state, scheduler)
    reconcile(run, state, scheduler)
    if state['stage'] in TERMINAL:
        return
    stage = state['stage']
    if stage == 'smoke':
        add_task(run, state, 'smoke', 'smoke')
        if check_required(state, ['smoke']):
            for width in (640, 960, 1280):
                for seed in (0, 1, 2):
                    add_task(run, state, stock_id(width, seed), 'search', genome.stock(width), seed)
            state['stage'] = 'baselines'
    elif stage == 'baselines':
        names = [stock_id(w, s) for w in (640, 960, 1280) for s in (0, 1, 2)]
        if check_required(state, names):
            state['stage'] = 'initial_proposals'
            state['slots'] = {}
    elif stage in ('initial_proposals', 'evolution_proposals'):
        initial = stage == 'initial_proposals'
        count = 5 if initial else 6
        if not initial and 'parents' not in state:
            state['elites'] = select(state, state['population'], 2)
            state['parents'] = select(state, state['population'], 6, parents=True)
        for index in range(count):
            slot = f'g{state["generation"]:02d}_slot{index}'
            previous = state['slots'].get(slot)
            if previous and state['tasks'][previous]['status'] != 'failed':
                continue
            if initial:
                width = (640, 960, 960, 1280, 1280)[index]
                parents = [genome.stock(width)]
            else:
                width = None
                ids = state['parents']
                parents = [state['tasks'][ids[index]]['spec']['genes'],
                           state['tasks'][ids[index ^ 1]]['spec']['genes']]
            child = proposals.propose(run, state, scheduler, slot, parents, width)
            if child is not None:
                attempt = state['proposal_attempts'][slot]
                name = slot + f'_p{attempt:02d}'
                add_task(run, state, name, 'search', child, slot=slot)
                state['slots'][slot] = name
                state.setdefault('ancestry', {})[name] = {'parents': parents, 'slot': slot}
            break
        else:
            stop_server(state, scheduler)
            state['stage'] = 'initial_training' if initial else 'evolution_training'
    elif stage in ('initial_training', 'evolution_training'):
        names = list(state['slots'].values())
        if any(state['tasks'][n]['status'] == 'failed' for n in names):
            state['stage'] = 'initial_proposals' if stage == 'initial_training' else 'evolution_proposals'
            return
        if check_required(state, names):
            if stage == 'initial_training':
                state['population'] = [stock_id(w) for w in (640, 960, 1280)] + names
            else:
                state['population'] = state.pop('elites') + names
                state.pop('parents', None)
            report(run, state, state['generation'])
            if state['generation'] == 0:
                state['stage'] = 'awaiting_budget_approval'
            elif state['generation'] < 20:
                state['generation'] += 1
                state['slots'] = {}
                state['stage'] = 'evolution_proposals'
            else:
                candidates = [n for n, r in state['tasks'].items() if r['status'] == 'complete'
                              and r['spec']['kind'] == 'search' and r['spec']['seed'] == 0]
                if len(candidates) != 128:
                    raise ValueError('Completed evolution must have 128 unique successful candidates')
                selected = front(state, candidates)
                widths = {state['tasks'][n]['spec']['genes']['img_width'] for n in selected}
                selected += [stock_id(w) for w in sorted(widths) if stock_id(w) not in selected]
                finalists = [{'search_id': n, 'genes': state['tasks'][n]['spec']['genes'],
                               'validation_fitness': state['tasks'][n]['result']['fitness'],
                               'candidate_hash': state['tasks'][n]['result']['candidate_hash'],
                               'source_sha256': common.sha(run / 'tasks' / n / 'source.py'),
                               'stock_control': state['tasks'][n]['spec']['genes'] == genome.stock(
                                   state['tasks'][n]['spec']['genes']['img_width'])}
                              for n in selected]
                common.freeze(run / 'finalists.json', {'generation': 20, 'finalists': finalists,
                                                        'protocol_sha256': common.json_sha(run / 'protocol.json')})
                state['final_tasks'] = []
                for index, finalist in enumerate(finalists):
                    name = f'final_{index:03d}'
                    add_task(run, state, name, 'final', finalist['genes'])
                    state['final_tasks'].append(name)
                state['stage'] = 'final_training'
    elif stage == 'final_training':
        if check_required(state, state['final_tasks']):
            report(run, state, 20)
            common.write(run / 'reports/final_comparison.json', {
                name: state['tasks'][name]['result'] for name in state['final_tasks']})
            with (run / 'reports/final_comparison.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'width', 'test_mAP50', 'test_mAP50_95', 'parameters', 'checkpoint_sha256'])
                for name in state['final_tasks']:
                    result = state['tasks'][name]['result']
                    writer.writerow([name, result['img_width'], *result['fitness'], result['checkpoint_sha256']])
            state['stage'] = 'complete'
    else:
        raise ValueError('Unknown workflow stage: ' + stage)
    # Proposal batches must not run simultaneously with long candidate jobs.
    if state['stage'] not in TERMINAL and state['stage'] not in ('initial_proposals', 'evolution_proposals'):
        launch_pending(run, state, scheduler)
