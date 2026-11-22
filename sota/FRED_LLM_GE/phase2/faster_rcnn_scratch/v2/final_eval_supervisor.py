"""Audited final_eval.py routing adapter for an already-frozen V2 experiment.

Search workers, checkpoints and training implementations remain frozen. Final
tasks use the shared final-evaluation CLI; GPU jobs honor reviewed node exclusions.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def routed_submit(scheduler, label, role, task=None, after=None):
    run = scheduler.run
    name = scheduler.token(label)
    matches = [r[0] for r in scheduler.active() if r[1] == name]
    if len(matches) > 1:
        raise RuntimeError('Duplicate active routed jobs: ' + name)
    if matches:
        return matches[0]
    manifest = json.loads((run / 'manifest.json').read_text())
    history = subprocess.check_output([
        'sacct', '-X', '-n', '-P', '--name=' + name,
        '--starttime=' + manifest['created_local_date'], '-o', 'JobIDRaw,JobName%128'],
        text=True, timeout=45)
    previous = [line.split('|')[0] for line in history.splitlines()
                if len(line.split('|')) > 1 and line.split('|')[1].strip() == name
                and line.split('|')[0].isdigit()]
    if previous:
        return str(max(map(int, previous)))
    worker = role == 'worker'
    server = role == 'server'
    gpu = worker or server
    if worker:
        kind = json.loads((run / 'tasks' / task / 'task.json').read_text())['kind']
        if kind == 'final':
            command = [str(run / 'code/.venv-faster-rcnn/bin/python'),
                       str(run / 'control/final_eval.py'), '--family', 'faster_rcnn_scratch',
                       '--run', str(run), '--task', task]
        else:
            command = [str(run / 'code/.venv-faster-rcnn/bin/python'), '-m',
                       'sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.worker',
                       '--run', str(run), '--task', task]
    elif server:
        protocol = json.loads((run / 'protocol.json').read_text())
        command = [str(run / 'code/.venv/bin/python'), '-m', 'uvicorn', 'server:app',
                   '--host', '0.0.0.0', '--port', str(protocol['server_port']), '--workers', '1']
    else:
        command = [str(run / 'code/.venv/bin/python'), str(Path(__file__).resolve()),
                   '--run', str(run)]
    wrap = 'set -euo pipefail; cd ' + shlex.quote(str(run / 'code')) + '; '
    if gpu:
        wrap += 'module load cuda; '
    if server:
        wrap += ('export LLMGE_DETECTOR=faster_rcnn_scratch FRED_SEED_FAMILY=faster_rcnn_scratch '
                 'FRED_RUN_TAG=' + shlex.quote(run.name) + ' LLMGE_PORT=' + str(protocol['server_port'])
                 + ' LLMGE_MODEL_PATH=' + shlex.quote(protocol['model_path']) + '; ')
    wrap += 'exec ' + shlex.join(command)
    args = ['sbatch', '--parsable', '--job-name=' + name, '--comment=' + name,
            '--partition=' + ('coe-gpu' if gpu else 'pace-cpu'),
            '--time=08:00:00', '--signal=B:USR1@180', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + ('16' if server else '12' if worker else '2'),
            '--mem=' + ('160G' if server else '32G' if worker else '8G'), '--export=ALL',
            '--output=' + str(run / 'logs' / (label + '-%j.out'))]
    if gpu:
        args += ['--gres=gpu:' + ('2' if server else '1'), '--constraint=H200']
        excluded = getattr(scheduler, 'excluded_nodes', [])
        if excluded:
            args += ['--exclude=' + ','.join(excluded)]
    if after:
        args += ['--dependency=afterany:' + str(after)]
    args += ['--wrap', wrap]
    env = dict(os.environ, SLURM_EXPORT_ENV='ALL')
    job = subprocess.check_output(args, text=True, env=env, cwd=run / 'code').strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid routed scheduler job ID')
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / 'code'))
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import common, cli
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.slurm import Slurm
    record = common.read(run / 'control/final_eval_entrypoint.json')
    for filename, expected in record['files'].items():
        if common.sha(run / 'control' / filename) != expected:
            raise ValueError('Audited final-evaluation routing changed: ' + filename)
    if common.json_sha(run / 'manifest.json') != record['original_manifest_identity']:
        raise ValueError('Original run manifest changed')

    repair_root = run / 'control/proposal_repair'
    pointer = run / 'control/proposal_repair_active.json'
    if pointer.exists():
        active_repair = common.read(pointer)
        repair_root = (run / 'control' / active_repair['directory']).resolve()
        if repair_root.parent != (run / 'control').resolve():
            raise ValueError('Invalid proposal repair directory')
        if common.json_sha(repair_root / 'manifest.json') != active_repair['manifest_identity']:
            raise ValueError('Active proposal repair manifest changed')
    if (repair_root / 'manifest.json').exists():
        import importlib.util
        from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2 import proposals
        repair = common.read(repair_root / 'manifest.json')
        for filename, expected in repair['files'].items():
            if common.sha(repair_root / filename) != expected:
                raise ValueError('Proposal repair changed: ' + filename)
        for filename, expected in repair['tokenizer_files'].items():
            if common.sha(Path(filename)) != expected:
                raise ValueError('Proposal tokenizer changed: ' + filename)
        spec = importlib.util.spec_from_file_location(
            'sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.repaired_proposals',
            repair_root / 'proposals.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.TEMPLATE_ROOT = repair_root / 'templates'
        proposals.propose = module.propose
        proposals.response_child = module.response_child

    class RoutedSlurm(Slurm):
        def submit(self, label, role, task=None, after=None):
            if role in ('supervisor', 'worker', 'server'):
                return routed_submit(self, label, role, task, after)
            return super().submit(label, role, task, after)

    RoutedSlurm.excluded_nodes = repair.get('excluded_nodes', []) if (repair_root / 'manifest.json').exists() else []
    cli.Slurm = RoutedSlurm
    cli.supervise(run)


if __name__ == '__main__':
    main()
