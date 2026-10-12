"""Preserve allocated GPU IDs/UUIDs when a server is renewed on another node."""
import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml


REPO = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('mask', ['2,3', 'GPU-first,GPU-second', 'MIG-first,MIG-second'])
def test_real_server_script_preserves_scheduler_mask(tmp_path, mask):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    uv = bindir / 'uv'
    uv.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                  'with open(os.environ["UV_CALL_LOG"],"a") as f:\n'
                  ' f.write(json.dumps({"args":sys.argv[1:],"mask":os.getenv("CUDA_VISIBLE_DEVICES"),'
                  '"order":os.getenv("CUDA_DEVICE_ORDER"),"expected":os.getenv("LLM_SERVER_EXPECTED_GPUS")})+"\\n")\n'
                  'if "-c" in sys.argv: print("8293")\n')
    uv.chmod(0o755)
    envfile = tmp_path / 'modules.sh'
    envfile.write_text('module() { :; }\n')
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               BASH_ENV=str(envfile), CUDA_VISIBLE_DEVICES=mask, SLURM_JOB_ID='123',
               SUBMIT_ISLAND_CONTROLLER='0', UV_CALL_LOG=str(tmp_path / 'uv.jsonl'))
    result = subprocess.run(['bash', str(REPO / 'server.sh')], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    calls = [json.loads(row) for row in (tmp_path / 'uv.jsonl').read_text().splitlines()]
    assert calls[-1]['args'][:4] == ['run', 'python', '-m', 'uvicorn']
    assert all(call['mask'] == mask for call in calls)
    assert calls[-1]['expected'] == '2' and calls[-1]['order'] == 'PCI_BUS_ID'
    assert f'Slurm CUDA_VISIBLE_DEVICES={mask}' in result.stdout


def test_candidate_gpu_script_preserves_mask_while_llm_cpu_job_hides_gpus(tmp_path):
    scripts = yaml.safe_load((REPO / 'slurm-config/slurm_config.yaml').read_text())
    envfile = tmp_path / 'modules.sh'
    envfile.write_text('module() { :; }\n')
    for name, expected in [('python_bash_script', '3'), ('llm_bash_script', '')]:
        code = scripts[name].replace('{}', 'printf "MASK=%s\\n" "$CUDA_VISIBLE_DEVICES"')
        result = subprocess.run(['bash', '-c', code], cwd=tmp_path,
            env=dict(os.environ, BASH_ENV=str(envfile), CUDA_VISIBLE_DEVICES='3', SLURM_JOB_ID='123'),
            capture_output=True, text=True, check=True)
        assert f'MASK={expected}\n' in result.stdout


def cuda_validator(monkeypatch, count=2, available=True, failed_device=None):
    # Isolate the real validation methods from FastAPI and the 70B model loader.
    # No copies of the validation implementation and no local CUDA are needed.
    tree = ast.parse((REPO / 'server.py').read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLMModel')
    model.body = [node for node in model.body if isinstance(node, ast.FunctionDef)
                  and node.name in ('_log_cuda_state', '_validate_cuda_available')]
    calls = []
    def properties(index):
        if index == failed_device:
            raise RuntimeError('CUDA initialization failed on allocated device')
        return SimpleNamespace(name=f'GPU-{index}', total_memory=141 * 2**30)
    def empty(size, device):
        calls.append(device)
        return object()
    torch = SimpleNamespace(__version__='test', empty=empty, cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count,
        get_device_properties=properties, synchronize=lambda index: None))
    namespace = {'os': os, 'torch': torch}
    exec(compile(ast.Module(body=[model], type_ignores=[]), 'server.py', 'exec'), namespace)
    monkeypatch.setenv('LLM_SERVER_EXPECTED_GPUS', '2')
    return namespace['LLMModel']._validate_cuda_available, calls


def test_cuda_probe_checks_every_allocated_device(monkeypatch):
    validate, calls = cuda_validator(monkeypatch)
    validate()
    assert calls == ['cuda:0', 'cuda:1']


@pytest.mark.parametrize('count,available,failed_device,reason', [
    (1, True, None, 'Expected 2 allocated'),
    (0, False, None, 'CUDA is not available'),
    (2, True, 1, 'CUDA initialization failed'),
])
def test_cuda_probe_fails_before_model_loading(monkeypatch, count, available, failed_device, reason):
    validate, _ = cuda_validator(monkeypatch, count, available, failed_device)
    with pytest.raises(RuntimeError, match=reason):
        validate()
