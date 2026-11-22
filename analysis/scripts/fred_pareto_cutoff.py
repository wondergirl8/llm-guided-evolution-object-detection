"""Two validation/size Pareto fronts at a completed-generation cutoff.

Reuse the shared FRED plotter's fitness reader and exact two-objective dominance
implementation. Never read a later-generation checkpoint into the selected data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fred_pareto_graphs import discover_generations, load_candidates, front_2d, pareto_mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_3d(data, plots, reports, cutoff, checkpoint):
    mask = pareto_mask(data)
    front = data[mask].sort_values('params')
    other = data[~mask & (data.gene_id != 'seed')]
    stock = data[data.gene_id == 'seed']
    fig = plt.figure(figsize=(15, 7.8))
    ax = fig.add_axes([.02, .13, .59, .73], projection='3d')
    ax.set_proj_type('ortho')
    ax.scatter(other.params / 1e6, other.mAP50 * 100, other.mAP50_95 * 100,
               color='#bac3cd', s=30, alpha=.65, depthshade=False, label='Other architectures')
    ax.scatter(front.params / 1e6, front.mAP50 * 100, front.mAP50_95 * 100,
               color='#cf3744', s=85, edgecolor='white', linewidth=.7,
               depthshade=False, label='Three-objective Pareto front')
    if len(stock):
        ax.scatter(stock.params / 1e6, stock.mAP50 * 100, stock.mAP50_95 * 100,
                   color='#172b4d', marker='X', s=130, depthshade=False, label='Stock baseline')
    for i, (_, row) in enumerate(front.iterrows()):
        ax.text(row.params/1e6 + .6, row.mAP50*100 + (.15 if i%2 else -.15),
                row.mAP50_95*100 + .2, f'P{i+1}', fontsize=8, color='#962532')
    ax.set_xlabel('Parameters (millions; minimize)', labelpad=13)
    ax.set_ylabel('Validation mAP50 (%; maximize)', labelpad=13)
    ax.set_zlabel('Validation mAP50:95 (%; maximize)', labelpad=13)
    ax.view_init(elev=23, azim=-57)
    ax.tick_params(labelsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(0, .99), fontsize=9)
    details = fig.add_axes([.65, .19, .34, .64])
    details.axis('off')
    details.text(0, .98, f'{len(front)} nondominated architectures', fontsize=12,
                 fontweight='bold', va='top')
    cells = [[f'P{i+1}: {r.gene_id}', f'{r.params/1e6:.3f}',
              f'{r.mAP50*100:.2f}', f'{r.mAP50_95*100:.2f}'] for i, r in enumerate(front.itertuples())]
    table = details.table(cellText=cells,
                          colLabels=['Model', 'Params (M)', 'mAP50 (%)', 'mAP50:95 (%)'],
                          colWidths=[.47, .17, .17, .19], bbox=[0, .23, 1, .65], cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dfe5eb')
        cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor('#eaf0f5')
            cell.set_text_props(weight='bold')
        elif row % 2:
            cell.set_facecolor('#fafbfc')
    details.text(0, .12, 'No other evaluated model is at least as good on all\n'
                 'three objectives and strictly better on one.', fontsize=9, color='#52616b')
    fig.suptitle('Faster R-CNN: three-objective Pareto front\n'
                  f'Initial population + completed generations 1–{cutoff}',
                  x=.06, ha='left', fontsize=15, fontweight='bold')
    fig.text(.06, .035, f'{len(data)} unique architectures · validation scores · width 1280 · 20 search epochs · seed 0\n'
              f'Generations after {cutoff} and official-test results excluded. Points represent discrete architectures; no interpolated surface.',
              fontsize=9, color='#52616b')
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(plots / ('pareto_3d_params_mAP50_mAP50_95' + suffix),
                    dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    front.to_csv(reports / 'pareto_3d_models.csv', index=False)
    (reports / 'pareto_3d_manifest.json').write_text(json.dumps({
        'cutoff_generation': cutoff, 'successful_unique_candidates': len(data),
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha(checkpoint),
        'objectives': {'params': 'minimize', 'mAP50': 'maximize', 'mAP50_95': 'maximize'},
        'front': front.to_dict(orient='records')}, indent=2) + '\n')
    print('Three-objective Pareto front:', len(front), 'models')
    print(front[['gene_id', 'generation', 'params', 'mAP50', 'mAP50_95']].to_string(index=False))


def draw(data, metric, title, out, cutoff):
    front = front_2d(data, metric)
    stock = data[data.gene_id == 'seed']
    others = data[(data.gene_id != 'seed') & ~data.index.isin(front.index)]
    fig, (ax, details) = plt.subplots(figsize=(14, 6.7), ncols=2,
                                     gridspec_kw={'width_ratios': [3.3, 2]})
    ax.scatter(others.params / 1e6, others[metric] * 100, s=36,
               color='#bcc5ce', edgecolor='white', linewidth=.5,
               label='Other evaluated architectures', zorder=2)
    ax.step(front.params / 1e6, front[metric] * 100, where='post', color='#cf3744',
            linewidth=1.5, alpha=.8, zorder=3)
    ax.scatter(front.params / 1e6, front[metric] * 100, s=95,
               color='#cf3744', edgecolor='white', linewidth=1.1,
               label='Two-objective Pareto front', zorder=4)
    if len(stock):
        ax.scatter(stock.params / 1e6, stock[metric] * 100, marker='X',
                   color='#172b4d', s=125, edgecolor='white', linewidth=.8,
                   label='Stock baseline', zorder=5)
    for i, (_, row) in enumerate(front.iterrows()):
        ax.annotate(f'P{i+1}', (row.params/1e6, row[metric]*100),
                    xytext=(-22 if i % 2 else 7, 9), textcoords='offset points',
                    fontsize=8.5, color='#962532',
                    bbox=dict(boxstyle='round,pad=.25', facecolor='white', alpha=.85, edgecolor='none'))
    ax.set_xlabel('Number of parameters (millions)', fontsize=11)
    ax.set_ylabel('Validation ' + title + ' (%)', fontsize=11)
    fig.suptitle('Faster R-CNN: parameter count vs ' + title + '\n'
                 f'Initial population + completed generations 1–{cutoff}',
                 fontsize=14, fontweight='bold', x=.07, ha='left')
    ax.grid(alpha=.3, linestyle=':', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=.12, y=.22)
    ax.legend(loc='lower right', fontsize=9, framealpha=.95)
    details.axis('off')
    details.text(0, .98, 'Pareto-front models', fontsize=12, fontweight='bold', va='top')
    cells = [[f'P{i+1}: {row.gene_id}', f'{row.params/1e6:.3f}', f'{row[metric]*100:.2f}']
             for i, (_, row) in enumerate(front.iterrows())]
    height = .065 * (len(cells) + 1)
    table = details.table(cellText=cells, colLabels=['Model', 'Params (M)', title + ' (%)'],
                          colWidths=[.54, .23, .23], bbox=[0, .89-height, 1, height], cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#e1e6eb')
        cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor('#eaf0f5')
            cell.set_text_props(weight='bold')
        elif row % 2:
            cell.set_facecolor('#fafbfc')
    if len(stock):
        row = stock.iloc[0]
        details.text(0, .80-height, f'Stock baseline: {row.params/1e6:.3f}M parameters\n'
                     f'Validation {title}: {row[metric]*100:.2f}%', fontsize=10,
                     color='#172b4d', va='top', linespacing=1.5)
    fig.text(.07, .025, f'{len(data)} unique architectures · width 1280 · 20 search epochs · seed 0\n'
             f'Validation only; generations after {cutoff} and official-test results excluded.',
             fontsize=8, color='#52616b')
    fig.tight_layout(rect=[0, .07, 1, .88], w_pad=3)
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(suffix), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return front


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--generation', type=int, default=8)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--3d-only', dest='only_3d', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoints = [(g, Path(p)) for g, p in discover_generations(run / 'checkpoints')
                   if g <= args.generation]
    if not checkpoints or checkpoints[-1][0] != args.generation:
        raise ValueError('Requested cutoff generation is not complete')
    first = {}
    for generation, path in checkpoints:
        for name in load_candidates(path).gene_id:
            first.setdefault(name, generation)
    data = load_candidates(checkpoints[-1][1]).copy()
    data['generation'] = data.gene_id.map(first)
    if data.empty or data.gene_id.duplicated().any() or data.generation.isna().any():
        raise ValueError('Invalid cutoff archive')
    if not (data.generation <= args.generation).all():
        raise ValueError('Later-generation candidate leaked into cutoff')
    identities = set()
    result_hashes = {}
    for row in data.itertuples():
        path = run / 'tasks' / row.gene_id / 'result.json'
        result = json.loads(path.read_text())
        if result['status'] != 'complete' or result['img_width'] != 1280:
            raise ValueError('Unscored or incompatible candidate')
        if tuple(result['fitness']) != (row.mAP50, row.mAP50_95, row.params):
            raise ValueError('Archive/result fitness differs')
        if result['candidate_hash'] in identities:
            raise ValueError('Repeated candidate identity')
        identities.add(result['candidate_hash'])
        result_hashes[row.gene_id] = sha(path)
    plots = args.output_root / 'plots' / run.name / f'through_gen_{args.generation}'
    reports = args.output_root / 'reports' / run.name / f'through_gen_{args.generation}'
    plots.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if args.only_3d:
        draw_3d(data, plots, reports, args.generation, checkpoints[-1][1])
        return
    fronts = {}
    for metric, title in [('mAP50', 'mAP50'), ('mAP50_95', 'mAP50:95')]:
        front = draw(data, metric, title, plots / ('pareto_params_vs_' + metric), args.generation)
        front.to_csv(reports / ('pareto_' + metric + '.csv'), index=False)
        data['pareto_' + metric] = data.gene_id.isin(front.gene_id)
        fronts[metric] = front.to_dict(orient='records')
        print(title, front[['gene_id', 'generation', 'params', metric]].to_string(index=False))
    data.to_csv(reports / 'all_candidates.csv', index=False)
    manifest = dict(run=str(run), cutoff_generation=args.generation,
                    generation_range_inclusive=[0, args.generation], successful_unique_candidates=len(data),
                    objectives='Minimize parameters; maximize the AP metric separately for each plot',
                    score_split='Fixed validation sequences; no official-test scores',
                    checkpoint_sha256={str(p): sha(p) for _, p in checkpoints},
                    result_sha256=result_hashes, fronts=fronts)
    (reports / 'cutoff_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PLOTS', plots.resolve())
    print('REPORTS', reports.resolve())


if __name__ == '__main__':
    main()
