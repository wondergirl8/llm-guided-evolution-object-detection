"""Recovery controller for frozen V2 baseline pilots; never proposes candidates.

Copy this file into run/control before execution. Training continues to use the
original frozen source, protocol, dependencies, and checkpoint identities.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time


def priority(item):
    name, record = item
    spec = record['spec']
    return (spec.get('seed', 0), spec.get('genes', {}).get('img_width', 0), name)


def pilot_report(state, policy):
    """Planning estimates, not hypothesis tests on selected search winners."""
    from scipy.stats import nct, t
    report = {'policy': policy, 'widths': {}, 'assumptions': [
        'Independent normally distributed training-seed scores; fixed validation data',
        'Equal group sizes and equal population SD assumed for planning',
        'One preplanned comparison; no multiplicity adjustment',
        'Three-seed SD estimates are uncertain; no automatic evolution approval']}
    for width in (640, 960, 1280):
        rows = [r['result']['fitness'] for r in state['tasks'].values()
                if r['spec'].get('genes', {}).get('img_width') == width
                and r['spec'].get('seed') in (0, 1, 2)
                and r['status'] == 'complete']
        if len(rows) != 3:
            raise ValueError('Incomplete baseline repeats')
        metrics = {}
        for i, metric in enumerate(('mAP50', 'mAP50_95')):
            values = [100 * row[i] for row in rows]
            sd = statistics.stdev(values)
            estimates = {}
            for gain in (1, 2, 3, 5):
                required = None
                if sd > 0:
                    # Scan from small n: avoid numerical failures in noncentral-t
                    # tails at very large degrees of freedom during root finding.
                    for n in range(2, 2001):
                        df = 2 * n - 2
                        critical = t.ppf(1 - policy['alpha'] / 2, df)
                        nc = gain / (sd * math.sqrt(2 / n))
                        power = nct.sf(critical, df, nc) + nct.cdf(-critical, df, nc)
                        if math.isfinite(power) and power >= policy['power']:
                            required = n
                            break
                estimates[str(gain)] = required
            metrics[metric] = dict(values_pp=values, mean_pp=statistics.mean(values),
                                   sample_sd_pp=sd, seeds_per_architecture=estimates)
        report['widths'][str(width)] = metrics
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / 'code'))
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import common, workflow
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.slurm import Slurm, ACTIVE
    script = Path(__file__).resolve()
    control = run / 'control'
    record = common.read(control / 'pilot_recovery.json')
    if common.sha(script) != record['controller_sha256']:
        raise ValueError('Recovery controller identity changed')
    policy = common.read(control / 'seed_policy.json')
    scheduler = Slurm(run)
    exiting = False

    def stop_signal(*_):
        nonlocal exiting
        exiting = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop_signal)

    with common.locked(run / '.supervisor.lock'):
        common.verify_run(run)
        state = common.read(run / 'state.json')
        if (run / 'STOP').exists() or state['stage'] in ('stopped', 'complete', 'awaiting_seed_budget_review'):
            return
        if args.recover:
            if state['stage'] != 'blocked' or 'squeue' not in state.get('reason', ''):
                raise ValueError('Only the audited scheduler-query failure can be recovered')
            common.freeze(control / 'state_before_recovery.json', state)
            for name, task in state['tasks'].items():
                if task.get('job') and task['status'] != 'complete':
                    status = scheduler.state(task['job'])
                    if status == 'CANCELLED':
                        if not (run / 'tasks' / name / 'training.pt').exists():
                            raise ValueError('Cancelled task has no resumable checkpoint')
                        task['recovery_cancelled_job'] = task.pop('job')
                        task['status'] = 'pending'
            state.update(stage='baselines', reason=None)
            common.write(run / 'state.json', state)
        if state['stage'] != 'baselines':
            raise ValueError('Pilot controller only supports baseline stage')
        job = os.environ['SLURM_JOB_ID']
        successor_name = scheduler.token('pilot_after_' + job)
        # Pre-arm continuation before any training submission. On scheduler query
        # failure, exit without touching worker jobs; the existing chain recovers.
        existing = [r[0] for r in scheduler.active() if r[1] == successor_name]
        if existing:
            successor = existing[0]
        else:
            wrap = 'exec ' + shlex.join([str(run / 'code/.venv/bin/python'), str(script), '--run', str(run)])
            successor = subprocess.check_output([
                'sbatch', '--parsable', '--job-name=' + successor_name,
                '--partition=pace-cpu', '--time=08:00:00', '--signal=B:USR1@180', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=2', '--mem=8G', '--dependency=afterany:' + job,
                '--output=' + str(run / 'logs/pilot-%j.out'), '--wrap', wrap], text=True).strip().split(';')[0]
        common.write(run / 'supervisor.json', {'job': job, 'successor': successor, 'controller': str(script)})
        query_failures = 0
        while not exiting:
            try:
                if (run / 'STOP').exists():
                    state.update(stage='stopped', reason='User stop request')
                    for task in state['tasks'].values():
                        if task.get('job'):
                            scheduler.cancel(task['job'])
                    scheduler.cancel(successor)
                    common.write(run / 'state.json', state)
                    return
                workflow.reconcile(run, state, scheduler)
                if state['stage'] in ('blocked', 'stopped'):
                    common.write(run / 'state.json', state)
                    scheduler.cancel(successor)
                    return
                if any(t['status'] == 'failed' for t in state['tasks'].values()):
                    raise ValueError('Baseline failure requires review')
                tasks = {n: t for n, t in sorted(state['tasks'].items(), key=priority)}
                state['tasks'] = tasks
                if all(t['status'] == 'complete' for t in tasks.values()):
                    common.write(run / 'reports/baseline_seed_budget.json', pilot_report(state, policy))
                    state.update(stage='awaiting_seed_budget_review', reason='Pilot complete; seed policy must be frozen before proposals/evolution')
                    scheduler.cancel(successor)
                else:
                    previous_partition = os.environ.get('SBATCH_PARTITION')
                    os.environ['SBATCH_PARTITION'] = 'coe-gpu'
                    try:
                        workflow.launch_pending(run, state, scheduler)
                    finally:
                        if previous_partition is None:
                            os.environ.pop('SBATCH_PARTITION', None)
                        else:
                            os.environ['SBATCH_PARTITION'] = previous_partition
                query_failures = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                # Uncertain scheduler state is never interpreted as worker failure.
                query_failures += 1
                common.write(control / 'scheduler_warning.json', {
                    'time': time.time(), 'consecutive_failures': query_failures, 'error': repr(exc)})
                if query_failures >= 30:
                    return  # pre-armed successor retries; no cancellation
            except Exception as exc:
                state.update(stage='blocked', reason=repr(exc))
                common.write(run / 'state.json', state)
                scheduler.cancel(successor)
                raise
            common.write(run / 'state.json', state)
            common.write(run / 'status.json', {
                'stage': state['stage'], 'generation': state['generation'],
                'completed_tasks': sum(t['status'] == 'complete' for t in state['tasks'].values()),
                'total_tasks': len(state['tasks']), 'reason': state.get('reason')})
            if state['stage'] == 'awaiting_seed_budget_review':
                return
            time.sleep(20)


if __name__ == '__main__':
    main()
