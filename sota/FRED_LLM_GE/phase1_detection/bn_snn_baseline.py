"""Independent scratch baseline: run-seed --seed 0/1/2, then summarize.

Call inside an existing GPU allocation. This module never submits jobs.
"""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from .bn_snn_data import digest
from .protocol_fingerprint import compute_fingerprint

FRED = Path(__file__).resolve().parents[1]
PROTOCOL = FRED / 'configs/phase1/protocol.yaml'
SEED = FRED / 'seeds/seed_bn_snn.py'
DEFAULT_OUTPUT = FRED / 'baseline_validation/phase1_detection/bn_snn_stock'


def identity():
    return compute_fingerprint(FRED.parents[1], 'bn_snn', 'stock_bn_snn', SEED)


def validate(root, seed):
    report_path = root / 'results/seed_evaluation.json'
    report = json.loads(report_path.read_text())
    metadata = report['metadata']
    args_path = root / 'training_args/seed/args.yaml'
    args = yaml.safe_load(args_path.read_text())
    protocol = yaml.safe_load(PROTOCOL.read_text())
    if (report['mode'] != 'validation' or report['gene_id'] != 'seed'
            or metadata['model_identity'] != digest(SEED)
            or metadata['official_split'] != 'challenging_train'
            or metadata['input_representation'] != 'raw_event_temporal_counts'
            or args['initialization'] != 'scratch' or args['proxy']['seed'] != seed
            or args['family_settings'] != protocol['families']['bn_snn']
            or args['proxy']['epochs'] != protocol['families']['bn_snn']['training']['epochs']
            or any(args['proxy'][k] != protocol['proxy'][k]
                   for k in ('batch_size', 'img_width', 'eval_conf', 'nms_iou'))):
        raise ValueError(f'baseline provenance mismatch: {root}')
    with (root / 'results/seed_results.csv').open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 1:
        raise ValueError(f'invalid result rows: {root}')
    values = list(map(float, rows[0][:3]))
    if (len(values) != 3 or not all(math.isfinite(v) for v in values)
            or not 0 <= values[1] <= values[0] <= 1 or values[2] <= 0 or not values[2].is_integer()):
        raise ValueError(f'failed baseline: {root}')
    for key, value in zip(('mAP50', 'mAP50_95', 'params'), values):
        if not math.isclose(report['metrics'][key], value, abs_tol=1e-6):
            raise ValueError(f'baseline CSV/report disagree: {root}')
    run = json.loads((root / 'training_args/seed/run.json').read_text())
    if run['wall_time_seconds'] <= 0:
        raise ValueError('missing baseline duration')
    if metadata['checkpoint_identity'] != digest(root / 'trained_models/seed.pt'):
        raise ValueError('baseline weights changed')
    return dict(seed=seed, mAP50=values[0], mAP50_95=values[1], params=int(values[2]),
                wall_time_seconds=run['wall_time_seconds'], report_sha256=digest(report_path),
                args_sha256=digest(args_path))


def run_seed(seed, output, device):
    before = identity()
    root = output / f'seed_{seed}'
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'sota.FRED_LLM_GE.phase1_detection.candidate_runner',
        '--family', 'bn_snn', '--model', 'network', '--seed', str(seed), '--device', device,
        '--protocol', str(PROTOCOL), '--sota_root', str(root)], cwd=FRED.parents[1], check=True)
    result = validate(root, seed)  # candidate failures may exit zero but have -inf fitness
    if identity() != before:
        raise ValueError('protocol/code/data changed during baseline')
    (root / 'baseline.json').write_text(json.dumps(dict(result, identity=before), indent=2) + '\n')
    print(result)


def summarize(output):
    current = identity()
    runs = []
    for seed in (0, 1, 2):
        root = output / f'seed_{seed}'
        recorded = json.loads((root / 'baseline.json').read_text())
        actual = validate(root, seed)
        if recorded != dict(actual, identity=current):
            raise ValueError(f'stale, incomplete, or changed baseline: {root}')
        runs.append(actual)
    if len({r['params'] for r in runs}) != 1:
        raise ValueError('seed parameter counts disagree')
    aggregate = {key: dict(mean=statistics.mean(r[key] for r in runs),
                          sample_sd=statistics.stdev(r[key] for r in runs))
                 for key in ('mAP50', 'mAP50_95', 'wall_time_seconds')}
    summary = dict(schema_version='fred-stock-detector-baseline-v1', family='bn_snn',
        initialization='scratch', identity=current, seeds=[0, 1, 2], params=runs[0]['params'],
        input_representation='raw_event_temporal_counts', aggregate=aggregate, runs=runs)
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(output / 'summary.json')
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run-seed')
    run.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    run.add_argument('--device', default='cuda')
    commands.add_parser('summarize')
    args = parser.parse_args()
    if args.command == 'run-seed':
        run_seed(args.seed, args.output.resolve(), args.device)
    else:
        summarize(args.output.resolve())


if __name__ == '__main__':
    main()
