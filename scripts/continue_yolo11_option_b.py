"""Renew a saved search after expected Slurm time limits, never after code errors."""
import argparse
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import time


def active(job):
    return job in subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'],
                                 capture_output=True, text=True, check=True).stdout.split()


def state(job):
    # Accounting may appear a few seconds after a dependency releases.
    for _ in range(6):
        result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                                 '--format=JobIDRaw,State,ExitCode'],
                                capture_output=True, text=True, check=True)
        for row in result.stdout.splitlines():
            fields = row.split('|')
            if fields[0] == job:
                return fields[1].split()[0].rstrip('+')
        time.sleep(10)
    raise RuntimeError(f'Accounting unavailable for job {job}; automatic renewal stopped')


def queue_continuation(root, server, dependency):
    result = subprocess.run(['sbatch', '--parsable', '--job-name=yolo11-b-chain',
        f'--dependency=afterany:{dependency}', '--kill-on-invalid-dep=yes',
        '--cpus-per-task=1', '--mem=2G', '--time=00:15:00',
        f'--output={root}/chain-%j.log', 'scripts/yolo11_scratch_job.sh',
        'continue-search', str(root), server], capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(';')[0]
    (root / 'chain-job.txt').write_text(job + '\n')
    print(f'Automatic continuation job: {job}', flush=True)
    return job


def progress(root):
    checkpoints = list((root / 'checkpoints').glob('checkpoint_gen_*.pkl'))
    if not checkpoints:
        return (-1, 0)
    latest = max(checkpoints, key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    generation = int(latest.stem.rsplit('_', 1)[1])
    # Trusted local checkpoint; DEAP recreates its fitness classes when unpickling.
    with latest.open('rb') as file:
        saved = pickle.load(file)
    import math
    valid = sum(bool(ind.fitness.values) and all(math.isfinite(v) for v in ind.fitness.values)
                for ind in saved['population'])
    return (generation, valid)


def pending_children(root, controller):
    jobs = set()
    marker = root / 'child-jobs.txt'
    if marker.is_file():
        jobs.update(line for line in marker.read_text().splitlines() if line.isdigit())
    log = root / f'evolution-{controller}.log'
    if log.is_file():
        # Also covers an allocation launched before child journaling was installed.
        jobs.update(re.findall(r'Submitted batch job (\d+)', log.read_text(errors='replace')))
    if not jobs:
        return []
    result = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'],
                            capture_output=True, text=True, check=True)
    return sorted(jobs.intersection(result.stdout.split()))


def continue_search(root, server):
    if (root / 'STOP-AUTO').exists() or not (root / 'AUTO-ENABLED').exists():
        print('Automatic renewal disabled. No further jobs submitted.', flush=True)
        return
    controller_file = root / 'controller-job.txt'
    if not controller_file.exists():
        raise RuntimeError('Resume did not submit a controller; inspect the resume log')
    controller = controller_file.read_text().strip()
    if active(controller):
        # First invoked after the readiness job, then again after its controller.
        queue_continuation(root, server, controller)
        return
    target = int(json.loads((root / 'run_config.json').read_text())['LLMGE_GENERATIONS']) - 1
    current = progress(root)
    if current[0] >= target:
        print(f'Generation target {target} completed. Automatic renewal finished.', flush=True)
        if active(server):
            subprocess.run(['scancel', server], check=True)
        (root / 'AUTO-ENABLED').unlink(missing_ok=True)
        return
    controller_state = state(controller)
    server_state = state(server)
    log = root / f'evolution-{controller}.log'
    log_text = log.read_text(errors='replace') if log.exists() else ''
    server_expired = (server_state == 'TIMEOUT' and controller_state == 'FAILED' and
        re.search(r'^RuntimeError: LLM server unavailable;', log_text, re.MULTILINE))
    if controller_state != 'TIMEOUT' and not server_expired:
        raise RuntimeError(f'Controller {controller} ended {controller_state}, server {server_state}. '
                           'Not a normal time limit; inspect logs before restarting')
    if server_state not in ('TIMEOUT', 'RUNNING', 'COMPLETING', 'COMPLETED'):
        raise RuntimeError(f'Server ended {server_state}; automatic renewal stopped')
    record = root / 'auto-progress.json'
    previous = json.loads(record.read_text()) if record.exists() else {}
    stalled = previous.get('stalled_allocations', 0) + 1 if tuple(previous.get('progress', (-2, 0))) >= current else 0
    if stalled >= 3:
        raise RuntimeError('Three allocations without checkpoint progress; inspect generation failures before restarting')
    children = pending_children(root, controller)
    if children:
        print(f'Waiting for {len(children)} outstanding candidate jobs: {", ".join(children)}', flush=True)
        queue_continuation(root, server, ':'.join(children))
        return  # Slurm waits without holding a CPU allocation.
    if active(server):
        subprocess.run(['scancel', server], check=True)
        for _ in range(30):
            if not active(server):
                break
            time.sleep(10)
        else:
            raise RuntimeError('Previous server is still ending; automatic renewal stopped')
    if (root / 'STOP-AUTO').exists():
        print('Automatic renewal disabled while waiting. No further jobs submitted.', flush=True)
        return
    record.write_text(json.dumps({'progress': current, 'stalled_allocations': stalled}, indent=2))
    subprocess.run(['bash', 'scripts/resume_yolo11_option_b.sh', '--auto', str(root)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('server')
    args = parser.parse_args()
    try:
        continue_search(args.root, args.server)
    except Exception:
        # A real failure disables chaining; preserve its traceback and artifacts.
        (args.root / 'AUTO-ENABLED').unlink(missing_ok=True)
        if active(args.server):
            subprocess.run(['scancel', args.server], check=True)
        raise
