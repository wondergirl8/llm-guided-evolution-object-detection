"""Regressions for the actual rejected proposals and failed server allocation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

import run_improved as controller
from src.fred_runtime import assert_llm_job_active, gpu_memory_budget
from src.fred_yolo26_proposals import generate_validated_section

FRED = Path(__file__).resolve().parents[2]
SEED = FRED / 'seeds/seed_yolo26n_p2.yaml'


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine', 'tiny_yolo11'])
def test_generated_slurm_templates_keep_allocated_devices(tmp_path, family):
    """Execute the real generator in an isolated checkout layout before testing YAML."""
    source = FRED.parents[1]
    for relative in ('slurm.py', 'src/cfg/constants.py', 'slurm-config/pace-ice.txt'):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    env = dict(os.environ, FRED_SEED_FAMILY=family, LLMGE_CLUSTER='pace-ice',
               CUDA_VISIBLE_DEVICES='MIG-SYNTHETIC-ALLOCATION')
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, str(tmp_path / 'slurm.py')], cwd=tmp_path,
                   env=env, check=True, capture_output=True, text=True)
    config = yaml.safe_load((tmp_path / 'slurm-config/slurm_config.yaml').read_text())
    for template in ('python_bash_script', 'llm_bash_script'):
        script = config[template].replace('{}', 'printf "%s" "$CUDA_VISIBLE_DEVICES"')
        # Execute only the visibility-related lines; module/scheduler calls need ICE.
        visibility = '\n'.join(line for line in script.splitlines()
                               if 'CUDA_VISIBLE_DEVICES' in line)
        result = subprocess.run(['bash', '-c', visibility], env=env, check=True,
                                capture_output=True, text=True)
        expected = 'MIG-SYNTHETIC-ALLOCATION' if family in ('yolo26_p2', 'dfine') else '0'
        assert result.stdout == expected
        assert ('export CUDA_VISIBLE_DEVICES=0' in config[template]) == (family == 'tiny_yolo11')


def test_invalid_channels_then_valid_reply_gets_feedback_and_keeps_parent(tmp_path):
    original = SEED.read_text()
    parts = original.split('# --OPTION--')
    section = yaml.safe_load(parts[1])
    section['backbone'][0][3][0] = 96  # Actual rejected proposal from ICE.
    bad = yaml.safe_dump(section, sort_keys=False)
    section['backbone'][0][3][0] = 128
    good = yaml.safe_dump(section, sort_keys=False)
    prompts, replies = [], iter([bad, good])
    def generate(prompt):
        prompts.append(prompt)
        return next(replies)
    output = tmp_path / 'candidate.py'
    generate_validated_section(SEED, output, parts, 1, 'Mutate one layer.', generate, FRED)
    assert SEED.read_text() == original
    assert 'allowed channel arguments [64, 128]' in prompts[0]
    assert 'channels are out of bounds' in prompts[1]
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert [attempt['accepted'] for attempt in audit['attempts']] == [False, True]
    config = yaml.safe_load(output.read_text())
    assert config['backbone'][0][3][0] == 128


def test_invalid_proposals_are_not_silently_rounded_or_parent_fallback(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    section = yaml.safe_load(parts[2])
    section['head'][8][3][0] = 96
    output = tmp_path / 'bad.py'
    with pytest.raises(RuntimeError, match='No valid YOLO26 proposal after 2 attempts'):
        generate_validated_section(SEED, output, parts, 2, 'Mutate.',
            lambda _: yaml.safe_dump(section), FRED, attempts=2)
    assert not output.exists()


def test_multiple_layer_changes_are_rejected(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    section = yaml.safe_load(parts[1])
    section['backbone'][0][3][0] = 128
    section['backbone'][8][1] = 1
    with pytest.raises(RuntimeError, match='No valid YOLO26 proposal'):
        generate_validated_section(SEED, tmp_path / 'bad.py', parts, 1, 'Mutate.',
            lambda _: yaml.safe_dump(section), FRED, attempts=1)


class FakeCUDA:
    def __init__(self, bad_second=False):
        self.bad_second, self.probed = bad_second, []
    def is_available(self): return True
    def device_count(self): return 2
    def get_device_properties(self, index):
        if index == 1 and self.bad_second:
            raise RuntimeError('device=1, num_gpus=1')
        self.probed.append(index)
        return SimpleNamespace(name='test GPU')
    def synchronize(self, index): pass
    def mem_get_info(self, index): return 1000, 1200


def test_gpu_probe_initializes_every_device_and_uses_measured_memory(monkeypatch):
    monkeypatch.setenv('LLMGE_EXPECTED_GPUS', '2')
    cuda = FakeCUDA()
    torch = SimpleNamespace(cuda=cuda, empty=lambda *a, **k: object())
    assert gpu_memory_budget(torch) == {0: 900, 1: 900}
    assert cuda.probed == [0, 1]


def test_nvml_count_without_second_cuda_device_is_not_accepted(monkeypatch):
    monkeypatch.setenv('LLMGE_EXPECTED_GPUS', '2')
    torch = SimpleNamespace(cuda=FakeCUDA(bad_second=True), empty=lambda *a, **k: object())
    with pytest.raises(RuntimeError, match='Allocated GPU 1/2 cannot initialize'):
        gpu_memory_budget(torch)


@pytest.mark.parametrize('state', ['FAILED', 'TIMEOUT', 'CANCELLED', 'COMPLETED'])
def test_dead_server_stops_even_when_controller_cached_ready(monkeypatch, state):
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setattr(controller, 'LOCAL_LLM', True)
    monkeypatch.setattr(controller, 'LLM_SERVER_READY', True)
    monkeypatch.setattr('src.fred_runtime.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=f'JobId=123 JobState={state}', stderr=''))
    with pytest.raises(RuntimeError, match=f'is {state}; stopping'):
        controller.wait_for_llm_server()


def test_running_server_is_accepted_and_legacy_without_job_id_is_unchanged(monkeypatch):
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setattr('src.fred_runtime.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='JobId=123 JobState=RUNNING', stderr=''))
    assert_llm_job_active()
    monkeypatch.delenv('LLMGE_SERVER_JOB_ID')
    assert_llm_job_active()


def test_mutation_worker_routes_real_section_through_validation_retry(tmp_path, monkeypatch):
    from src import llm_mutation
    monkeypatch.setattr(llm_mutation, 'SEED_FAMILY', 'yolo26_p2')
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *a: 1)
    section = yaml.safe_load(SEED.read_text().split('# --OPTION--')[1])
    replies = []
    for channel in (96, 128):
        section['backbone'][0][3][0] = channel
        replies.append(yaml.safe_dump(section))
    def generate(*args): return replies.pop(0)
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', generate)
    template = tmp_path / 'prompt.txt'
    template.write_text('Improve this section: {}')
    output = tmp_path / 'candidate.py'
    llm_mutation.augment_network(str(SEED), str(output), str(template))
    assert not replies and output.exists()


def test_crossover_worker_preserves_first_parent_one_layer_contract(tmp_path, monkeypatch):
    from src import llm_crossover
    monkeypatch.setattr(llm_crossover, 'SEED_FAMILY', 'yolo26_p2')
    monkeypatch.setattr(llm_crossover.random, 'shuffle', lambda parts: None)
    parts = SEED.read_text().split('# --OPTION--')
    section = yaml.safe_load(parts[1])
    section['backbone'][0][3][0] = 128
    parts[1] = yaml.safe_dump(section)
    other = tmp_path / 'other.py'
    other.write_text('# --OPTION--'.join(parts))
    monkeypatch.setattr(llm_crossover, 'generate_augmented_code', lambda *a: parts[1])
    output = tmp_path / 'candidate.py'
    llm_crossover.augment_network(str(SEED), str(other), str(output))
    assert yaml.safe_load(output.read_text())['backbone'][0][3][0] == 128


def test_readiness_timeout_is_fatal_for_a_managed_server(monkeypatch):
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setenv('LLM_SERVER_READY_TIMEOUT', '0')
    monkeypatch.setattr(controller, 'LOCAL_LLM', True)
    monkeypatch.setattr(controller, 'LLM_SERVER_READY', False)
    monkeypatch.setattr('src.fred_runtime.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='JobState=RUNNING', stderr=''))
    times = iter([0, 1])
    monkeypatch.setattr(controller.time, 'time', lambda: next(times))
    with pytest.raises(RuntimeError, match='readiness timed out'):
        controller.wait_for_llm_server()


@pytest.mark.parametrize('own', [False, True])
def test_cleanup_only_cancels_same_user_and_checkout(monkeypatch, tmp_path, own):
    from src import fred_recovery_controller as runner
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setenv('USER', 'fixture_user')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        directory = tmp_path if own else tmp_path / 'unrelated'
        return SimpleNamespace(returncode=0, stdout=f'UserId=fixture_user(1000) WorkDir={directory}')
    monkeypatch.setattr(runner.subprocess, 'run', run)
    runner.stop_own_server()
    assert any(args[0] == 'scancel' for args in calls) == own


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine', 'unrelated'])
def test_http_timeouts_are_scoped_to_recovered_families(monkeypatch, family):
    from src import llm_utils
    monkeypatch.setattr(llm_utils, 'SEED_FAMILY', family)
    monkeypatch.setenv('LLMGE_LLM_HTTP_TIMEOUT_SECONDS', '600')
    monkeypatch.setattr(llm_utils, 'get_llm_server_hostname', lambda: 'fixture')
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {'generated_text': 'ok', 'response_time_sec': 0})
    monkeypatch.setattr(llm_utils.requests, 'post', post)
    assert llm_utils.submit_mixtral_local('fixture') == 'ok'
    assert llm_utils.submit_deepseek_local('fixture') == 'ok'
    expected = None if family == 'unrelated' else (10, 600)
    assert all(call['timeout'] == expected for call in calls)
