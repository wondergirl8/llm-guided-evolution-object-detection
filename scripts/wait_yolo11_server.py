"""Wait for the specific Slurm server allocation to finish model loading."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_for_server(job, port, timeout=2400):
    repo = Path(__file__).resolve().parents[1]
    marker = repo / f'run_job_outputs/server/hostname-{job}.log'
    deadline = time.monotonic() + timeout
    next_message = 0
    while time.monotonic() < deadline:
        state = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                               capture_output=True, text=True, check=True).stdout.strip()
        if not state or state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'COMPLETED', 'NODE_FAIL', 'OUT_OF_MEMORY'):
            raise RuntimeError(f'LLM server job {job} stopped before becoming ready ({state or "not queued"})')
        if marker.is_file():
            host = marker.read_text().strip()
            try:
                with urllib.request.urlopen(f'http://{host}:{port}/', timeout=10) as response:
                    ready = response.status == 200 and str(json.load(response).get('slurm_job_id')) == job
                if ready:
                    (repo / 'hostname.log').write_text(host + '\n')
                    print(f'LLM server {job} is ready at {host}:{port}', flush=True)
                    return
            except (OSError, ValueError, urllib.error.URLError):
                pass
        if time.monotonic() >= next_message:
            print(f'Waiting for LLM server {job} to load its model...', flush=True)
            next_message = time.monotonic() + 60
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError(f'LLM server {job} was not ready within {timeout} seconds; inspect its server log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.cfg.constants import PORT
    wait_for_server(args.job, PORT, args.timeout)
