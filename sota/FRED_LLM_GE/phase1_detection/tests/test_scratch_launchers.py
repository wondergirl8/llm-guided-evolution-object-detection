"""Exercise real launcher logic with mocked Slurm/server, without GPU submissions."""
import http.server, json, os, pathlib, shutil, subprocess, tempfile, threading

def test_option_b_launch_preparation_dependencies_and_resume():
    repo = pathlib.Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix='yolo11-launch-check-') as td:
        root = pathlib.Path(td)

        def copy(relative):
            source = repo / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', 'data_cache', 'results', 'models', 'trained_models', 'tests'), symlinks=False)
            else:
                shutil.copy2(source, target)
        for relative in ['scripts', 'src/cfg', 'sota/FRED_LLM_GE/configs', 'sota/FRED_LLM_GE/data', 'sota/FRED_LLM_GE/phase0_data', 'sota/FRED_LLM_GE/seeds/seed_ultralytics_yolo11.py']:
            copy(relative)
        for relative in ['server.sh', 'src/llm_mutation.py', 'src/llm_crossover.py', 'src/llm_utils.py',
                         'src/generated_class.py', 'src/candidate_preflight.py', 'templates/FRED/UltralyticsRules.txt']:
            copy(relative)
        phase = root / 'sota/FRED_LLM_GE/phase1_detection'
        for relative in ['seeds/yolo11', 'map_reporter.py', 'fitness.py', 'result_contract.py', 'proxy_data.py']:
            copy('sota/FRED_LLM_GE/phase1_detection/' + relative)
        (root / '.venv').symlink_to(repo / '.venv', target_is_directory=True)
        subprocess.run(['git', 'init', '-q', '-b', 'yolo11-scratch-evolution', str(root)], check=True)
        manifest = json.loads((root / 'sota/FRED_LLM_GE/configs/phase1/yolo11_option_b.json').read_text())
        data = phase / 'seeds/yolo11/data_cache/challenging'
        for kind, step in [('train', 50), ('val', 10)]:
            part = data / f'search_{kind}_step{step}'
            (part / 'done').mkdir(parents=True)
            (part / 'images').mkdir()
            (part / 'data.yaml').write_text('fixture: true\n')
            for seq in manifest[kind + '_sequences']:
                name = f'{seq}_000001'
                (part / 'images' / f'{name}.png').write_bytes(b'scheduler fixture; images are not decoded')
                (part / 'done' / f'{seq}.json').write_text(json.dumps({'dataset_revision': manifest['dataset_revision'], 'sequence': seq, 'hf_folder': manifest['archive_sources'][seq], 'frame_step': step, 'frames': 1, 'size': [1280, 720], 'records': [{'name': name, 'frame_index': 1, 'boxes': []}]}))
        bindir = root / 'fake-bin'
        bindir.mkdir()
        sbatch = bindir / 'sbatch'
        sbatch.write_text("#!/usr/bin/env python3\nimport json, os, pathlib, sys\np=pathlib.Path(os.environ['FAKE_SLURM_LOG'])\nlines=p.read_text().splitlines() if p.exists() else []\njob=31000+len(lines)\nwith p.open('a') as f:\n f.write(json.dumps({'id':job,'args':sys.argv[1:],'generations':os.getenv('LLMGE_GENERATIONS'),'profile':os.getenv('FRED_DATA_PROFILE')})+'\\n')\nprint(job)\n")
        sbatch.chmod(493)
        (bindir / 'squeue').write_text('#!/bin/sh\nprintf "%s" "${FAKE_ACTIVE_JOB:-}"\n')
        (bindir / 'squeue').chmod(493)

        class Handler(http.server.BaseHTTPRequestHandler):

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def log_message(self, *args):
                pass
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        (root / 'hostname.log').write_text('127.0.0.1\n')
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'], FAKE_SLURM_LOG=str(root / 'jobs.jsonl'), LLMGE_PORT=str(server.server_port))
        for name in list(env):
            if name.startswith('FRED_') or name in ('LLMGE_GENERATIONS', 'LLMGE_POPULATION'):
                env.pop(name)

        def run(script, *args, ok=True, extra=None):
            result = subprocess.run(['bash', 'scripts/' + script, *args], cwd=root, env=dict(env, **extra or {}), capture_output=True, text=True)
            if ok and result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            if not ok and result.returncode == 0:
                raise AssertionError('Expected rejection')
            return result
        run('prepare_yolo11_option_b.sh')
        result = run('run_yolo11_option_b.sh')
        runroot = pathlib.Path((root / 'fred_ultralytics_yolo11_output/latest-option-b.txt').read_text().strip())
        config = json.loads((runroot / 'run_config.json').read_text())
        assert (config['LLMGE_GENERATIONS'], config['training_frames'], config['validation_frames']) == ('1001', 152, 20)
        assert config['FRED_LR'] == '0.001'
        session = json.loads(next(runroot.glob('generation-session-*.json')).read_text())
        assert session['LLM_MAX_NEW_TOKENS'] == '4096'
        assert session['LLMGE_INITIALIZATION_BATCHES'] == '3'
        (runroot / 'baseline').mkdir()
        (runroot / 'baseline/seed_results.csv').write_text('fixture\n')
        (runroot / 'checkpoints').mkdir()
        (runroot / 'checkpoints/checkpoint_gen_3.pkl').write_text('fixture')
        run('run_yolo11_option_b.sh', '--resume', str(runroot))
        calls = [json.loads(line) for line in (root / 'jobs.jsonl').read_text().splitlines()]
        assert len(calls) == 5
        assert '--array=0-7%4' in calls[0]['args']
        assert '--dependency=afterok:31000' in calls[1]['args']
        assert '--dependency=afterok:31002' in calls[3]['args']
        assert calls[3]['generations'] == calls[4]['generations'] == '1001'
        assert '--time=07:45:00' in calls[3]['args']
        assert not any((arg.startswith('--dependency') for arg in calls[4]['args']))
        run('run_yolo11_option_b.sh', '--resume', str(runroot), ok=False, extra={'FAKE_ACTIVE_JOB': '31004'})
        run('resume_yolo11_option_b.sh', ok=False, extra={'FAKE_ACTIVE_JOB': '31004'})
        run('resume_yolo11_option_b.sh')
        calls = [json.loads(line) for line in (root / 'jobs.jsonl').read_text().splitlines()]
        assert len(calls) == 7
        assert calls[5]['args'] == ['--parsable', 'server.sh']
        assert '--dependency=after:31005' in calls[6]['args']
        assert calls[6]['args'][-4:] == ['scripts/yolo11_scratch_job.sh', 'resume-when-ready', str(runroot), '31005']
        assert (runroot / 'resume-job.txt').read_text().strip() == '31006'
        first = manifest['train_sequences'][0]
        (data / 'search_train_step50/images' / f'{first}_000001.png').unlink()
        run('run_yolo11_option_b.sh', ok=False)
        (data / 'search_val_step10/data.yaml').unlink()
        run('run_yolo11_option_b.sh', ok=False)
        assert len((root / 'jobs.jsonl').read_text().splitlines()) == 7
        server.shutdown()
        print('Launcher integration passed: CPU array/finalization, baseline dependency, 1000-generation target, resume, duplicate and incomplete-export rejection (mock Slurm/server).')
