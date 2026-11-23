"""Explicit opt-in fixed-resolution experiment, isolated from the V2 pilot."""
import csv
import math
import statistics

from . import common, genome, proposals

STOCK = 'seed'  # Existing Pareto plotter's stock-control identifier.


def configure(protocol, final_root):
    return dict(protocol, experiment='procedure1280', widths=[1280],
                train_frame_step=50, final_train_frame_step=5,
                final_train_root=str(final_root),
                approval_required_after_initial=False,
                search_seed=0, final_seeds=list(range(1, 31)),
                statistics={'primary_metric': 'mAP50_95', 'alpha': .05,
                            'alternative': 'two-sided', 'test': 'Welch',
                            'multiplicity': 'Holm across all evolved finalists',
                            'minimum_relevant_gain_pp': 2,
                            'planned_power': .8,
                            'sample_size': 30,
                            'note': '30 fixed by supplied procedure; power is not guaranteed'},
                search_variability_policy='Single search seed 0; final repeats assess frozen architectures, not search reproducibility')


def freeze_finalists(run, state):
    from . import workflow as w
    candidates = [name for name, task in state['tasks'].items()
                  if task['status'] == 'complete' and task['spec']['kind'] == 'search']
    if len(candidates) != 128 or len({genome.candidate_identity(state['tasks'][n]['spec']['genes'])
                                     for n in candidates}) != 128:
        raise ValueError('Expected 128 unique successful search candidates')
    selected = w.front(state, candidates)
    stock = STOCK
    if stock not in selected:
        selected.append(stock)
    finalists = []
    for name in selected:
        record = state['tasks'][name]
        finalists.append({'search_id': name, 'genes': record['spec']['genes'],
                          'candidate_hash': record['result']['candidate_hash'],
                          'validation_fitness': record['result']['fitness'],
                          'source_sha256': common.sha(run / 'tasks' / name / 'source.py'),
                          'stock_control': name == stock})
    common.freeze(run / 'finalists.json', {
        'generation': 20, 'protocol_sha256': common.json_sha(run / 'protocol.json'),
        'finalists': finalists, 'selection_uses': 'All successful search validation scores only',
        'seeds': list(range(1, 31))})
    state['final_tasks'] = []
    for index, finalist in enumerate(finalists):
        for seed in range(1, 31):
            name = f'final_{index:03d}_s{seed:02d}'
            w.add_task(run, state, name, 'final', finalist['genes'], seed=seed)
            state['final_tasks'].append(name)


def holm(pvalues):
    result = [0.] * len(pvalues)
    previous = 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[index]))
        result[index] = previous
    return result


def summarize_samples(run, state):
    """Never used by the search controller for fitness, rankings, or training."""
    from scipy.stats import ttest_ind, t
    selection = common.read(run / 'finalists.json')['finalists']
    summaries = []
    samples = []
    for index, finalist in enumerate(selection):
        rows = [state['tasks'][f'final_{index:03d}_s{seed:02d}'] for seed in range(1, 31)]
        if any(r['status'] != 'complete' for r in rows):
            raise ValueError('All 30 provenance-checked seed results are required')
        metrics = {}
        for k, metric in enumerate(('mAP50', 'mAP50_95')):
            values = [100 * r['result']['fitness'][k] for r in rows]
            sd = statistics.stdev(values)
            mean = statistics.mean(values)
            margin = t.ppf(.975, 29) * sd / math.sqrt(30)
            metrics[metric] = {'mean_pp': mean, 'sample_sd_pp': sd,
                               'mean_ci95_pp': [mean - margin, mean + margin],
                               'values_pp': values}
        summaries.append(dict(finalist, metrics=metrics))
        for seed, r in enumerate(rows, 1):
            samples.append({'search_id': finalist['search_id'], 'seed': seed,
                            'test_mAP50': r['result']['fitness'][0],
                            'test_mAP50_95': r['result']['fitness'][1],
                            'checkpoint_sha256': r['result']['checkpoint_sha256'],
                            'report': r['result']['report']})
    baseline, = [s for s in summaries if s['stock_control']]
    comparisons = []
    for summary in summaries:
        if summary['stock_control']:
            continue
        a = summary['metrics']['mAP50_95']['values_pp']
        b = baseline['metrics']['mAP50_95']['values_pp']
        result = ttest_ind(a, b, equal_var=False, alternative='two-sided')
        if not math.isfinite(float(result.pvalue)):
            raise ValueError('Degenerate/nonfinite Welch comparison requires review')
        ci = result.confidence_interval(.95)
        delta = statistics.mean(a) - statistics.mean(b)
        comparisons.append({'search_id': summary['search_id'], 'gain_pp': delta,
                            't': float(result.statistic), 'df': float(result.df),
                            'p_raw': float(result.pvalue),
                            'gain_ci95_pp': [float(ci.low), float(ci.high)],
                            'meets_2pp_point_estimate': delta >= 2})
    for row, adjusted in zip(comparisons, holm([c['p_raw'] for c in comparisons])):
        row.update(p_holm=adjusted, statistically_better=adjusted < .05 and row['gain_pp'] > 0)
    common.write(run / 'reports/final_seed_statistics.json', {
        'primary_metric': 'test_mAP50_95', 'alternative': 'two-sided',
        'test': 'Welch', 'alpha': .05, 'multiplicity': 'Holm', 'samples': summaries,
        'comparisons': comparisons,
        'limitations': ['One search run; not an estimate of between-search variability',
                       'Conditional on fixed official test sequences',
                       'Same integer seed labels used across architectures; independence is an analysis assumption',
                       'Unadjusted 95% intervals are descriptive, not simultaneous intervals',
                       'Nonsignificance does not establish equivalence or prove an effect is chance']})
    path = run / 'reports/final_seed_samples.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['search_id', 'seed', 'test_mAP50',
                                                   'test_mAP50_95', 'checkpoint_sha256', 'report'])
        writer.writeheader()
        writer.writerows(samples)
    table = ['# Frozen finalist evaluation (30 seeds each)', '',
             'Official challenging test, every released frame. No test scores enter search.', '',
             '| Model | Control | mAP50 mean ± SD | mAP50:95 mean ± SD | Parameters |',
             '|---|---|---:|---:|---:|']
    for item in summaries:
        a, b = item['metrics']['mAP50'], item['metrics']['mAP50_95']
        table.append(f'| {item["search_id"]} | {item["stock_control"]} | '
                     f'{a["mean_pp"]:.2f} ± {a["sample_sd_pp"]:.2f} | '
                     f'{b["mean_pp"]:.2f} ± {b["sample_sd_pp"]:.2f} | '
                     f'{item.get("validation_fitness", [None, None, None])[2]} |')
    (run / 'reports/final_comparison.md').write_text('\n'.join(table) + '\n')


def advance(run, state, scheduler):
    from . import workflow as w
    w.reconcile(run, state, scheduler)
    if state['stage'] in w.TERMINAL:
        return
    stage = state['stage']
    stock = STOCK
    if stage == 'smoke':
        w.add_task(run, state, 'smoke', 'smoke')
        if w.check_required(state, ['smoke']):
            w.add_task(run, state, stock, 'search', genome.stock(1280))
            state['stage'] = 'procedure_baseline'
    elif stage == 'procedure_baseline':
        if w.check_required(state, [stock]):
            state.update(stage='initial_proposals', slots={}, approved=True)
    elif stage in ('initial_proposals', 'evolution_proposals'):
        initial = stage == 'initial_proposals'
        if not initial and 'parents' not in state:
            state['elites'] = w.select(state, state['population'], 2)
            state['parents'] = w.select(state, state['population'], 6, parents=True)
        for index in range(7 if initial else 6):
            slot = f'g{state["generation"]:02d}_slot{index}'
            old = state['slots'].get(slot)
            if old and state['tasks'][old]['status'] != 'failed':
                continue
            parents = [genome.stock(1280)] if initial else [
                state['tasks'][state['parents'][i]]['spec']['genes'] for i in (index, index ^ 1)]
            child = proposals.propose(run, state, scheduler, slot, parents, width=1280)
            if child is not None:
                if child['img_width'] != 1280:
                    raise ValueError('Fixed 1280 width changed')
                name = slot + f'_p{state["proposal_attempts"][slot]:02d}'
                w.add_task(run, state, name, 'search', child, slot=slot)
                state['slots'][slot] = name
                state.setdefault('ancestry', {})[name] = {'parents': parents, 'slot': slot}
            break
        else:
            w.stop_server(state, scheduler)
            state['stage'] = 'initial_training' if initial else 'evolution_training'
    elif stage in ('initial_training', 'evolution_training'):
        names = list(state['slots'].values())
        if any(state['tasks'][n]['status'] == 'failed' for n in names):
            state['stage'] = 'initial_proposals' if stage == 'initial_training' else 'evolution_proposals'
            return
        if w.check_required(state, names):
            if stage == 'initial_training':
                state['population'] = [stock] + names
            else:
                state['population'] = state.pop('elites') + names
                state.pop('parents', None)
            w.report(run, state, state['generation'])
            if state['generation'] < 20:
                state.update(generation=state['generation'] + 1, slots={}, stage='evolution_proposals')
            else:
                freeze_finalists(run, state)
                state['stage'] = 'final_training'
    elif stage == 'final_training':
        if w.check_required(state, state['final_tasks']):
            summarize_samples(run, state)
            state['stage'] = 'complete'
    else:
        raise ValueError('Unexpected fixed-1280 stage: ' + stage)
    if state['stage'] not in w.TERMINAL and state['stage'] not in ('initial_proposals', 'evolution_proposals'):
        w.launch_pending(run, state, scheduler)
