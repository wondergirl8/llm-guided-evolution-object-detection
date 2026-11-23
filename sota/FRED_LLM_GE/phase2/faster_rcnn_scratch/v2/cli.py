"""Launch/status/resume/approve-continuation/stop for frozen Faster R-CNN V2 runs."""
import argparse
import datetime
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import common, genome, workflow
from .slurm import Slurm

PROJECT = Path(__file__).resolve().parents[5]
MODULE = 'sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2'
EXIT_REQUESTED = False


def request_exit(*_):
    global EXIT_REQUESTED
    EXIT_REQUESTED = True


def partition(train, test, validation_ids):
    train_ids, test_ids = [s for s, _ in train], [s for s, _ in test]
    if (len(train_ids) != 172 or len(set(train_ids)) != 172 or len(test_ids) != 59
            or len(set(test_ids)) != 59 or set(train_ids) & set(test_ids)):
        raise ValueError('Official membership must be unique and train/test disjoint')
    if len(validation_ids) != 20 or not validation_ids <= set(train_ids):
        raise ValueError('Validation must hold out 20 official training sequences')
    return ([(s, f) for s, f in train if s not in validation_ids],
            [(s, f) for s, f in train if s in validation_ids])


def prepare(tag, procedure_1280=False, final_options=None):
    import yaml
    from sota.FRED_LLM_GE.phase1_detection.protocol_fingerprint import scratch_code_identity
    from sota.FRED_LLM_GE.phase1_detection.fred_splits import challenging_split
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import load_exported_split, REVISION
    if not re.fullmatch(r'faster_rcnn_scratch_[0-9]{8}_eot0_v2_[a-z0-9_-]+', tag):
        raise ValueError('Use faster_rcnn_scratch_YYYYMMDD_eot0_v2_<name>')
    run = PROJECT / 'fred_runs' / tag
    if run.exists():
        raise FileExistsError('Use resume for an existing V2 run')
    original = yaml.safe_load((PROJECT / 'sota/FRED_LLM_GE/configs/phase1/protocol.yaml').read_text())
    challenging = original['challenging']
    root = (PROJECT / 'sota/FRED_LLM_GE' / challenging['data_root']).resolve()
    train, test = challenging_split(root)
    validation_ids = {str(s) for s in challenging['search']['val_sequences']}
    search, val = partition(train, test, validation_ids)
    if len(search) != 152 or len(val) != 20 or len(train) != 172 or len(test) != 59:
        raise ValueError('Official membership counts differ from agreed protocol')
    if challenging['events']['source'] != 'released':
        raise ValueError('V2 requires released event frames')
    final_root = root / 'final_train_step5'
    train_root = root / ('search_train_step50' if procedure_1280 else 'final_train_step5')
    validation_root = root / 'search_val_step10'
    # Reuse native-size exports; subset by official membership rather than copying images.
    exports = [(train_root, search if procedure_1280 else train, 50 if procedure_1280 else 5),
               (validation_root, val, 10)]
    if procedure_1280:
        exports.append((final_root, train, 5))
    if final_options:
        from .final_only import verify_reused_exports
        verify_reused_exports(exports, final_options['source_run'])
    else:
        for directory, members, step in exports:
            load_exported_split(directory, members, expected_step=step)
    data_paths = list((root / 'splits').glob('*.txt')) + list((root / 'splits').glob('hf_folders_*.json'))
    data_paths += [directory / 'done' / f'{s}.json' for directory, members, _ in exports for s, _ in members]
    data_paths += [validation_root / 'done' / f'{s}.json' for s, _ in val]
    data_hashes = {str(p): common.sha(p) for p in data_paths}
    protocol = {
        'schema_version': 'fred-frcnn-v2', 'genome_version': genome.SCHEMA['version'],
        'widths': [640, 960, 1280], 'epochs': 20, 'batch_size': 2,
        'optimizer': {'lr': .0025, 'momentum': .9, 'weight_decay': .0001},
        'backbone': {'frozen_stages': -1, 'norm_eval': True, 'norm_affine_trainable': True},
        'train': search, 'validation': val, 'final_train': train, 'test': test,
        'train_root': str(train_root), 'validation_root': str(validation_root),
        'train_frame_step': 5, 'validation_frame_step': 10, 'test_frame_step': 1,
        'events': challenging['events'], 'dataset_revision': REVISION,
        'evaluation': {'eval_conf': .05, 'nms_iou': .5},
        'population': 8, 'elites': 2, 'offspring': 6, 'generations': 20,
        'proposal_attempts': 24, 'max_training_jobs': 2,
        'approval_required_after_initial': True,
        'prior_test_exposure': 'Historical official2 interim epoch-8 test was viewed; not used as V2 fitness.',
        'server_port': 8300 + int(genome.digest(tag)[:6], 16) % 1000,
        'model_path': os.getenv('LLMGE_MODEL_PATH', '/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/'),
    }
    if procedure_1280:
        from .procedure1280 import configure
        protocol = configure(protocol, final_root)
    if final_options:
        from .final_only import configure
        protocol, source_paths = configure(protocol, **final_options)
        data_hashes.update({str(p): common.sha(p) for p in source_paths})
    run.mkdir(parents=True)
    code = run / 'code'
    sources = scratch_code_identity(PROJECT)
    plotter = 'analysis/scripts/fred_pareto_graphs.py'
    sources[plotter] = common.sha(PROJECT / plotter)
    for directory in (PROJECT / 'sota/FRED_LLM_GE/phase2/faster_rcnn_scratch/v2',
                      PROJECT / 'templates/FRED_FasterRCNNScratchV2',
                      PROJECT / 'sota/FRED_LLM_GE/seeds'):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json', '.txt') and '__pycache__' not in path.parts:
                sources[str(path.relative_to(PROJECT))] = common.sha(path)
    for relative in sources:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, target)
    for environment in ('.venv', '.venv-faster-rcnn'):
        (code / environment).symlink_to((PROJECT / environment).resolve(), target_is_directory=True)
    common.freeze(run / 'protocol.json', protocol)
    manifest = {'source_hashes': sources, 'data_hashes': data_hashes,
                'protocol_sha256': common.json_sha(run / 'protocol.json'),
                'detector_versions': common.versions(code / '.venv-faster-rcnn/bin/python'),
                'controller_versions': common.controller_versions(code / '.venv/bin/python'),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'created_local_date': datetime.date.today().isoformat()}
    common.freeze(run / 'manifest.json', manifest)
    common.write(run / 'state.json', {'stage': 'smoke', 'generation': 0, 'approved': False,
                                       'tasks': {}, 'seen': {}, 'population': []})
    if procedure_1280:
        common.write(run / 'authorization.json', {
            'scope': '1280-only generation 0 plus generations 1-20; all archived Pareto finalists and stock control, seeds 1-30',
            'procedure_sha256': common.sha(code / 'sota/FRED_LLM_GE/phase2/faster_rcnn_scratch/v2/experiment_procedure.txt'),
            'protocol_sha256': manifest['protocol_sha256']})
    if final_options:
        from .final_only import initialize
        initialize(run)
    common.verify_run(run)
    return run


def supervise(run):
    global EXIT_REQUESTED
    scheduler = Slurm(run)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_exit)
    with common.locked(run / '.supervisor.lock'):
        common.verify_run(run)
        state = common.read(run / 'state.json')
        if state['stage'] in workflow.TERMINAL or (run / 'STOP').exists():
            return
        job = os.environ.get('SLURM_JOB_ID')
        if not job:
            raise RuntimeError('supervise must run in a Slurm allocation')
        previous_path = run / 'supervisor.json'
        if previous_path.exists():
            previous_job = common.read(previous_path)['job']
            if previous_job != job and scheduler.state(previous_job) == 'CANCELLED':
                state.update(stage='stopped', reason='Previous supervisor explicitly cancelled')
                common.write(run / 'state.json', state)
                for record in state['tasks'].values():
                    if record.get('job'):
                        scheduler.cancel(record['job'])
                workflow.stop_server(state, scheduler)
                return
        # A queued CPU successor exists before any long work or worker submission.
        successor = scheduler.submit('supervisor_' + job, 'supervisor', after=job)
        common.write(run / 'supervisor.json', {'job': job, 'successor': successor})
        while not EXIT_REQUESTED:
            if (run / 'STOP').exists():
                state.update(stage='stopped', reason='User stop request')
            if state['stage'] not in workflow.TERMINAL:
                try:
                    workflow.advance(run, state, scheduler)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    # Scheduler uncertainty is not evidence of worker/model failure.
                    command = exc.cmd if isinstance(exc.cmd, (tuple, list)) else [exc.cmd]
                    if Path(str(command[0])).name in ('squeue', 'sacct', 'sbatch', 'scancel'):
                        common.write(run / 'scheduler_warning.json', {'error': repr(exc), 'time': time.time()})
                    else:
                        state.update(stage='blocked', reason=repr(exc))
                except Exception as exc:
                    # Persist failure and stop the chain, never convert it to a completed stage.
                    import traceback
                    state.update(stage='blocked', reason=repr(exc), traceback=traceback.format_exc())
            common.write(run / 'state.json', state)
            common.write(run / 'status.json', {
                'stage': state['stage'], 'generation': state['generation'],
                'completed_tasks': sum(r['status'] == 'complete' for r in state['tasks'].values()),
                'total_tasks': len(state['tasks']), 'reason': state.get('reason')})
            if state['stage'] in workflow.TERMINAL:
                workflow.stop_server(state, scheduler)
                if state['stage'] in ('blocked', 'stopped'):
                    for record in state['tasks'].values():
                        if record.get('job') and scheduler.state(record['job']) in ('RUNNING', 'PENDING', 'CONFIGURING'):
                            scheduler.cancel(record['job'])
                common.write(run / 'state.json', state)
                scheduler.cancel(successor)
                return
            time.sleep(20)
        # Signal exit releases the lock. The pre-armed successor takes over outstanding jobs.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'status', 'resume', 'approve-continuation', 'stop', 'supervise'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--procedure-1280', action='store_true',
                        help='20-generation fixed-1280 search, frame step 50; final seeds 1-30')
    parser.add_argument('--final-from', type=Path, help='Stopped search to use as final architecture source')
    parser.add_argument('--final-model')
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--selection-generation', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'launch':
        if not args.tag:
            parser.error('--tag is required')
        if not args.submit:
            if args.final_from:
                print('Final only:', args.final_model, args.final_epochs, 'epochs; seeds 1-30; all 172 step-5 train, all 59 every-frame test')
                return
            if args.procedure_1280:
                print('1280 only: smoke -> stock baseline -> initial population 8 -> generations 1-20 -> frozen Pareto + stock -> seeds 1-30, final train/test -> Welch report')
                return
            print('Dry run: smoke -> 9 stock baselines -> 5 LLM variants -> awaiting_budget_approval')
            print('Two training GPUs; 20 epochs; widths 640/960/1280; no automatic evolution approval.')
            return
        options = None
        if args.final_from:
            if not args.final_model or not args.final_epochs or args.procedure_1280:
                parser.error('Final-only launch requires --final-model and --final-epochs, without --procedure-1280')
            options = dict(source_run=args.final_from, model_id=args.final_model,
                           epochs=args.final_epochs, cutoff=args.selection_generation)
        run = prepare(args.tag, procedure_1280=args.procedure_1280, final_options=options)
        job = Slurm(run).submit('supervisor_launch', 'supervisor')
        common.write(run / 'launch.json', {'supervisor': job})
        print(f'Run: {run}\nSupervisor: {job}', flush=True)
        return
    if args.run is None:
        parser.error('--run is required')
    run = args.run.resolve()
    # Always operate existing runs with their frozen implementation, not today's working tree.
    frozen = run / 'code' / Path(__file__).resolve().relative_to(PROJECT)
    if Path(__file__).resolve() != frozen:
        os.chdir(run / 'code')
        env = dict(os.environ, PYTHONPATH=str(run / 'code'))
        os.execve(str(run / 'code/.venv/bin/python'),
                  [str(run / 'code/.venv/bin/python'), '-m', MODULE + '.cli', *sys.argv[1:]], env)
    if args.command == 'supervise':
        supervise(run)
        return
    state = common.read(run / 'state.json')
    if args.command == 'status':
        import json
        print(json.dumps(state, indent=2))
        return
    scheduler = Slurm(run)
    if args.command == 'stop':
        common.write(run / 'STOP', {'reason': 'Explicit user cancellation'})
        for job, name, _, _ in scheduler.active():
            if name.startswith(scheduler.token('')):
                scheduler.cancel(job)
        with common.locked(run / '.supervisor.lock', blocking=True):
            state = common.read(run / 'state.json')
            state.update(stage='stopped', previous_stage=state['stage'])
            common.write(run / 'state.json', state)
        print('Stopped run and automatic resubmission')
        return
    if args.command == 'resume':
        active = [job for job, name, _, _ in scheduler.active()
                  if name.startswith(scheduler.token('supervisor'))]
        if active:
            print('Supervisor already scheduled:', ', '.join(active))
            return
    with common.locked(run / '.supervisor.lock'):
        common.verify_run(run)
        state = common.read(run / 'state.json')
        if args.command == 'approve-continuation':
            if state['stage'] != 'awaiting_budget_approval':
                raise ValueError('Approval requires a completed initial review')
            common.freeze(run / 'budget_approval.json', {
                'initial_report_sha256': common.sha(run / 'reports/initial_review.json'),
                'authorized_generations': 20, 'unique_successful_target': 128})
            state.update(approved=True, generation=1, stage='evolution_proposals', slots={})
        elif state['stage'] in ('complete', 'awaiting_budget_approval', 'blocked'):
            raise ValueError('Cannot resume this stage without review/approval: ' + state['stage'])
        elif state['stage'] == 'stopped':
            raise ValueError('Explicitly stopped runs require inspection, not automatic restart')
        common.write(run / 'state.json', state)
        job = scheduler.submit('supervisor_resume_' + str(int(time.time())), 'supervisor')
        print('Supervisor:', job)


if __name__ == '__main__':
    main()
