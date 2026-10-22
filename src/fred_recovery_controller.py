"""Gate evolution on real LLM generation and stop when its server job dies."""
import ast
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


def stop_own_server():
    job = os.environ['LLMGE_SERVER_JOB_ID']
    result = subprocess.run(['scontrol', 'show', 'job', job, '-o'], text=True, capture_output=True, timeout=10)
    directory = re.search(r'(?:^|\s)WorkDir=(\S+)', result.stdout)
    owner = re.search(r'(?:^|\s)UserId=([^\s(]+)', result.stdout)
    if result.returncode == 0 and directory and owner:
        if Path(directory[1]).resolve() == ROOT and owner[1] == os.environ['USER']:
            subprocess.run(['scancel', job], check=False, timeout=15)


def smoke_generation():
    from src.cfg import constants as cfg
    from src import llm_mutation
    from sota.FRED_LLM_GE.phase1_detection.candidate_runner import ADAPTERS, load_protocol, DEFAULT_PROTOCOL, FRED_DIR
    folder = ROOT / 'recovery-smoke'
    folder.mkdir(exist_ok=True)
    template = ROOT / 'templates' / cfg.DEFAULT_PROMPT_GROUP / '1.txt'
    prompt = folder / 'prompt.txt'
    prompt.write_text(template.read_text() + '\n' + (ROOT / cfg.CONSTANT_RULES_PATH).read_text())
    candidate = folder / 'network_smoke.py'
    llm_mutation.augment_network(cfg.SEED_NETWORK, str(candidate), str(prompt),
                                llm_model=cfg.LLM_MODEL, apply_quality_control=False)
    protocol = load_protocol(DEFAULT_PROTOCOL)
    adapter = ADAPTERS[cfg.SEED_FAMILY](protocol['families'][cfg.SEED_FAMILY], FRED_DIR)
    if cfg.SEED_FAMILY == 'yolo26_p2':
        adapter.preflight(candidate, Path(cfg.SEED_NETWORK))
    elif cfg.SEED_FAMILY == 'dfine':
        if ast.dump(ast.parse(candidate.read_text())) == ast.dump(ast.parse(Path(cfg.SEED_NETWORK).read_text())):
            raise RuntimeError('D-FINE smoke generation returned an unchanged parent')
        adapter.preflight(candidate, protocol['proxy'])
    else:
        raise RuntimeError('Recovery controller supports only YOLO26n-P2 and D-FINE')
    (folder / 'passed.txt').write_text('Real LLM generation and CPU preflight passed. No training fitness or formal generation is claimed by this smoke check.\n')
    print('LIVE LLM GENERATION AND CPU PREFLIGHT PASSED', flush=True)


def main():
    from src.cfg import constants as cfg
    from src.fred_runtime import assert_llm_job_active
    import run_improved as controller
    from src.fred_session import (atomic_write, enabled, cancel_owned_jobs, latest_pair,
                                  mark_cancelled_evaluations, server_deadline)
    os.chdir(ROOT)
    process = None
    pending_cleanup = None
    bounded = enabled()
    stop = [False]
    if bounded:
        def on_signal(signum, frame):
            if signum == signal.SIGTERM:
                raise SystemExit(143)
            stop[0] = True
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, on_signal)
    try:
        if bounded:
            os.environ['LLMGE_SESSION_DEADLINE'] = str(server_deadline(ROOT, os.environ['LLMGE_SERVER_JOB_ID']))
            if time.time() >= float(os.environ['LLMGE_SESSION_DEADLINE']):
                return
        controller.wait_for_llm_server(timeout=1800)
        checkpoints = ('fred_yolo26_p2_map5095_coverage_v1_checkpoints' if cfg.SEED_FAMILY == 'yolo26_p2'
                       else 'fred_dfine_checkpoints')
        if bounded and cfg.SEED_FAMILY == 'dfine':
            pending_cleanup = ROOT / checkpoints / '.cleanup_pending'
            if pending_cleanup.exists():
                cancelled = cancel_owned_jobs(ROOT, exclude_jobs=(os.environ['LLMGE_SERVER_JOB_ID'],))
                mark_cancelled_evaluations(pending_cleanup.parent, cancelled or [])
                pending_cleanup.unlink()
        previous = latest_pair(ROOT / checkpoints, ROOT / checkpoints, cfg.SEED_FAMILY)[0] if bounded else None
        if previous is None:
            if bounded and (ROOT / 'recovery-smoke').exists():
                # A previous attempt can finish its smoke proposal but stop
                # before population zero. Preserve that attempt before retrying.
                (ROOT / 'recovery-smoke').rename(ROOT / f'recovery-smoke-before-resume-{time.time_ns()}')
            smoke_generation()
        if bounded and (stop[0] or time.time() >= float(os.environ['LLMGE_SESSION_DEADLINE'])):
            return
        if pending_cleanup is not None:
            atomic_write(pending_cleanup, b'cleanup required before another controller starts\n')
        process = subprocess.Popen([sys.executable, 'run_improved.py', '--checkpoints', checkpoints,
                                    '--global_path', checkpoints, '--prompt_group', cfg.DEFAULT_PROMPT_GROUP],
                                   start_new_session=bounded)
        while True:
            if bounded and (stop[0] or time.time() >= float(os.environ['LLMGE_SESSION_DEADLINE'])):
                print('SESSION PAUSED: resume from the newest committed checkpoint pair; '
                      'unfinished generation work may repeat.', flush=True)
                return
            assert_llm_job_active()
            try:
                code = process.wait(timeout=30)
                raise SystemExit(code)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process is not None and process.poll() is None:
            if bounded:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if bounded:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
        try:
            if bounded:
                cancelled = cancel_owned_jobs(ROOT)
                if pending_cleanup is not None:
                    mark_cancelled_evaluations(pending_cleanup.parent, cancelled or [])
                    pending_cleanup.unlink(missing_ok=True)
            else:
                stop_own_server()
        except Exception as error:
            print(f'Could not clean up own LLM server job: {error}', flush=True)
            if bounded:
                raise


if __name__ == '__main__':
    main()
