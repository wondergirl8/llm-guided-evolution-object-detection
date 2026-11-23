"""Idempotent Slurm submission and accounting for one frozen experiment."""
import os
import shlex
import subprocess
from pathlib import Path

MODULE = 'sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


class Slurm:
    def __init__(self, run):
        self.run = Path(run).resolve()

    def token(self, label):
        import hashlib
        return 'frv2_' + hashlib.sha256(str(self.run).encode()).hexdigest()[:10] + '_' + label

    def active(self):
        output = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%j|%T|%N'], text=True, timeout=45)
        return [line.strip().split('|') for line in output.splitlines() if line.strip()]

    def submit(self, label, role, task=None, after=None):
        name = self.token(label)
        matches = [row[0] for row in self.active() if row[1] == name]
        if len(matches) > 1:
            raise RuntimeError(f'Duplicate active jobs: {name}')
        if matches:
            return matches[0]
        # Recover a submission that finished between sbatch and the state-file update.
        from .common import read
        manifest = self.run / 'manifest.json'
        if manifest.exists():
            history = subprocess.check_output([
                'sacct', '-X', '-n', '-P', '--name=' + name,
                '--starttime=' + read(manifest)['created_local_date'], '-o', 'JobIDRaw,JobName%128'], text=True)
            previous = [line.split('|')[0] for line in history.splitlines()
                        if len(line.split('|')) > 1 and line.split('|')[1].strip() == name]
            if previous:
                return str(max(int(j) for j in previous if j.isdigit()))
        code = self.run / 'code'
        detector = role == 'worker'
        python = code / ('.venv-faster-rcnn/bin/python' if detector else '.venv/bin/python')
        command = [str(python), '-m', MODULE + ('.worker' if detector else '.cli'),
                   *([] if detector else ['supervise']), '--run', str(self.run)]
        if task:
            command += ['--task', task]
            if detector:
                from .common import read
                if read(self.run / 'tasks' / task / 'task.json')['kind'] == 'final':
                    command = [str(python), '-m', 'sota.FRED_LLM_GE.phase1_detection.final_eval',
                               '--family', 'faster_rcnn_scratch', '--run', str(self.run), '--task', task]
        wrap = 'set -euo pipefail; cd ' + shlex.quote(str(code)) + '; '
        if detector:
            wrap += 'module load cuda; '
        if role == 'server':
            from .common import read
            port = read(self.run / 'protocol.json')['server_port']
            command = [str(python), '-m', 'uvicorn', 'server:app', '--host', '0.0.0.0',
                       '--port', str(port), '--workers', '1']
            wrap += ('module load cuda; export LLMGE_DETECTOR=faster_rcnn_scratch '
                     'FRED_SEED_FAMILY=faster_rcnn_scratch FRED_RUN_TAG=' + shlex.quote(self.run.name)
                     + f' LLMGE_PORT={port}; ')
            model_path = read(self.run / 'protocol.json')['model_path']
            wrap += 'export LLMGE_MODEL_PATH=' + shlex.quote(model_path) + '; '
        wrap += 'exec ' + shlex.join(command)
        log = self.run / 'logs' / (label + '-%j.out')
        log.parent.mkdir(parents=True, exist_ok=True)
        args = ['sbatch', '--parsable', '--job-name=' + name, '--comment=' + name,
                '--partition=' + ('coe-gpu' if detector or role == 'server' else 'pace-cpu'),
                '--time=08:00:00', '--signal=B:USR1@180', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=' + ('16' if role == 'server' else '12' if detector else '2'),
                '--mem=' + ('160G' if role == 'server' else '32G' if detector else '8G'),
                '--output=' + str(log), '--export=ALL']
        if detector or role == 'server':
            args += ['--gres=gpu:' + ('2' if role == 'server' else '1'), '--constraint=H200']
            from .common import read
            protocol_path = self.run / 'protocol.json'
            exclusions = read(protocol_path).get('scheduler_excluded_nodes', []) if protocol_path.exists() else []
            if exclusions:
                args += ['--exclude=' + ','.join(exclusions)]
        if after:
            args += ['--dependency=afterany:' + str(after)]
        args += ['--wrap', wrap]
        # Do not inherit the dashboard's submit directory or job-specific export controls.
        env = dict(os.environ)
        env['SLURM_EXPORT_ENV'] = 'ALL'
        job = subprocess.check_output(args, text=True, env=env, cwd=code).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Invalid scheduler job ID')
        return job

    def state(self, job):
        for number, _, state, _ in self.active():
            if number == str(job):
                return state
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
                                          '-o', 'JobIDRaw,State'], text=True)
        for line in output.splitlines():
            fields = line.split('|')
            if fields[0] == str(job):
                return fields[1].split()[0].rstrip('+')
        return 'UNKNOWN'

    def node(self, job):
        rows = [r for r in self.active() if r[0] == str(job)]
        return rows[0][3] if rows and rows[0][2] == 'RUNNING' else None

    def cancel(self, job):
        subprocess.run(['scancel', str(job)], check=False)
