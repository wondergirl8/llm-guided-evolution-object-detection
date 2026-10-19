"""Opt-in bounded sessions and committed checkpoint pairs for owned FRED runs."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import time


def enabled():
    return os.getenv('LLMGE_BOUNDED_SESSION') == '1'


def server_deadline(root, job):
    if not str(job).isdecimal():
        raise ValueError('Expected this session server job ID')
    path = Path(root) / f'five-hour-deadline-{job}.txt'
    for _ in range(20):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            time.sleep(1)
    raise RuntimeError(f'Server did not publish its session deadline: {path}')


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_pair(folder, global_folder, generation, population, global_data, family):
    folder, global_folder = Path(folder), Path(global_folder)
    marker = folder / f'resume_gen_{generation}.json'
    if marker.exists():
        raise RuntimeError(f'Refusing to overwrite committed generation {generation}')
    files = [(folder / f'checkpoint_gen_{generation}.pkl', population),
             (global_folder / f'global_gen_{generation}.pkl', global_data)]
    hashes = {}
    for path, value in files:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write(path, data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    atomic_write(marker, json.dumps({'generation': generation, 'family': family, 'hashes': hashes,
        'stage': 'initial_population_pending_evaluation' if generation == 0 else 'completed_generation'}).encode())
    print(f'RESUMABLE CHECKPOINT PAIR COMMITTED: generation {generation}', flush=True)


def latest_pair(folder, global_folder, family):
    folder, global_folder = Path(folder), Path(global_folder)
    markers = sorted(((int(m.group(1)), p) for p in folder.glob('resume_gen_*.json')
                      if (m := re.fullmatch(r'resume_gen_(\d+)\.json', p.name))), reverse=True)
    for generation, marker in markers:
        try:
            meta = json.loads(marker.read_text())
            if meta['generation'] != generation or meta['family'] != family:
                raise ValueError('generation or family mismatch')
            values = []
            for parent, name in ((folder, f'checkpoint_gen_{generation}.pkl'),
                                 (global_folder, f'global_gen_{generation}.pkl')):
                data = (parent / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != meta['hashes'][name]:
                    raise ValueError('checkpoint hash mismatch')
                values.append(pickle.loads(data))  # Only this user's trusted run files.
            if not {'population', 'hof'} <= values[0].keys():
                raise ValueError('population schema mismatch')
            if not {'GLOBAL_DATA', 'GLOBAL_DATA_HIST', 'GLOBAL_DATA_ANCESTRY'} <= values[1].keys():
                raise ValueError('global-state schema mismatch')
            return values[0], generation + 1, values[1]
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(f'Committed checkpoint pair {marker.name} is unusable; '
                               f'artifacts preserved: {error}') from error
    if markers or list(folder.glob('checkpoint_gen_*.pkl')) or list(global_folder.glob('global_gen_*.pkl')):
        raise RuntimeError('No valid committed checkpoint pair; existing artifacts preserved')
    return None, None, None


def cancel_owned_jobs(root):
    """Cancel only the current user's jobs whose WorkDir is this exact checkout."""
    root = Path(root).resolve()
    user, own = os.environ['USER'], os.environ.get('SLURM_JOB_ID')
    jobs = subprocess.check_output(['squeue', '-h', '-u', user, '-o', '%i'], text=True).split()
    cancel = []
    for job in jobs:
        if job == own:
            continue
        result = subprocess.run(['scontrol', 'show', 'job', job, '-o'], text=True,
                                capture_output=True, timeout=10)
        if result.returncode:
            fresh = subprocess.check_output(['squeue', '-h', '-u', user, '-o', '%i'], text=True).split()
            if job in fresh:
                raise RuntimeError(f'Cannot inspect active job {job}; refusing unverified cancellation')
            continue
        directory = re.search(r'(?:^|\s)WorkDir=(\S+)', result.stdout)
        owner = re.search(r'(?:^|\s)UserId=([^\s(]+)', result.stdout)
        if directory and owner and Path(directory[1]).resolve() == root and owner[1] == user:
            cancel.append(job)
    if cancel:
        subprocess.run(['scancel', *cancel], check=True, timeout=30)
    return cancel


def run_worker(command):
    remaining = float(os.environ['LLMGE_SESSION_DEADLINE']) - time.time()
    if remaining <= 0:
        raise RuntimeError('ERROR infrastructure: five-hour session window ended before worker startup')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        print('ERROR infrastructure: session window ended; unfinished work may repeat on resume', flush=True)
        return 75
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    raise SystemExit(run_worker(sys.argv[1:]))
