"""Drain only the predecessor checkout's owned jobs before the next family."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    deadline = time.monotonic() + 4 * 3600 + 300
    while True:
        jobs = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'], text=True).split()
        remaining = []
        for job in jobs:
            status = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True, text=True, timeout=10)
            if status.returncode:
                fresh = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'], text=True).split()
                if job in fresh:
                    raise RuntimeError(f'Cannot inspect active job {job}')
                continue
            match = re.search(r'(?:^|\s)WorkDir=(\S+)', status.stdout)
            if not match:
                raise RuntimeError(f'Cannot identify active job {job}')
            if Path(match[1]).resolve() == root:
                remaining.append(job)
        if not remaining:
            print('Predecessor evaluations drained', flush=True)
            return
        print('Waiting for predecessor jobs: ' + ', '.join(remaining), flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError('Predecessor did not drain within its evaluation time allowance')
        time.sleep(60)


if __name__ == '__main__':
    main()
