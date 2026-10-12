"""Real TCP collision and HTTP startup checks, without loading a model."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from scripts.launch_yolo11_server import reserve_socket


REPO = Path(__file__).resolve().parents[4]


def test_reserved_port_cannot_be_taken_during_model_loading():
    with reserve_socket('127.0.0.1', 0) as listener:
        with socket.socket() as competitor:
            with pytest.raises(OSError):
                competitor.bind(listener.getsockname())


def test_real_uvicorn_uses_reserved_alternative_and_reports_allocation(tmp_path):
    # The first port belongs to someone else. It must stay untouched.
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        preferred = occupied.getsockname()[1]
        (tmp_path / 'test_app.py').write_text('''import asyncio, json, os
async def app(scope, receive, send):
    if scope['type'] == 'lifespan':
        while True:
            message = await receive()
            if message['type'] == 'lifespan.startup':
                await asyncio.sleep(0.2)
                await send({'type': 'lifespan.startup.complete'})
            elif message['type'] == 'lifespan.shutdown':
                await send({'type': 'lifespan.shutdown.complete'})
                return
    elif scope['type'] == 'http':
        body = json.dumps({'slurm_job_id': os.environ['SLURM_JOB_ID'],
                           'port': os.environ['LLMGE_PORT']}).encode()
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': body})
''')
        env = dict(os.environ, PYTHONPATH=str(REPO), SERVER_HOSTNAME='127.0.0.1',
                   SLURM_JOB_ID='123', LLMGE_PORT=str(preferred))
        with (tmp_path / 'server.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-c',
                'from scripts.launch_yolo11_server import launch; '
                'import sys; launch(repo=sys.argv[1], app="test_app:app")', str(tmp_path)],
                cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                marker = tmp_path / 'run_job_outputs/server/connection-123.json'
                deadline = time.monotonic() + 10
                payload = None
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        pytest.fail((tmp_path / 'server.log').read_text())
                    if marker.exists():
                        connection = json.loads(marker.read_text())
                        assert connection['port'] != preferred
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{connection['port']}/", timeout=.5) as response:
                                payload = json.load(response)
                            break
                        except OSError:
                            pass
                    time.sleep(.05)
                assert payload is not None, (tmp_path / 'server.log').read_text()
                assert payload == {'slurm_job_id': '123', 'port': str(connection['port'])}
                assert occupied.getsockname()[1] == preferred
                with socket.socket() as competitor:
                    with pytest.raises(OSError):
                        competitor.bind(('127.0.0.1', connection['port']))
            finally:
                child.terminate()
                child.wait(timeout=5)


def test_invalid_bind_address_is_not_treated_as_a_port_collision():
    with pytest.raises(OSError):
        reserve_socket('192.0.2.1', 0)
