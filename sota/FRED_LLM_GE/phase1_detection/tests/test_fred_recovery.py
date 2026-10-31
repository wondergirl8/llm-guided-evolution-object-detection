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
from src.fred_yolo26_proposals import (
    INITIAL_COVERAGE_TARGETS, generate_validated_section, initial_target_for_slot,
    verify_initial_coverage,
)

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
    run_script = (tmp_path / 'run.sh').read_text()
    server_script = (tmp_path / 'server.sh').read_text()
    assert 'run_improved.py --checkpoints ' in run_script
    assert ' --global_path ' in run_script
    assert '#SBATCH -t 7:45:00' in run_script
    assert 'export CUDA_VISIBLE_DEVICES=0,1' not in server_script
    assert 'from src.cfg.constants import PORT' in server_script
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
    bad = yaml.safe_dump({'layer': 0, 'field': 'channels', 'value': 96})
    good = yaml.safe_dump({'layer': 0, 'field': 'channels', 'value': 128})
    prompts, replies = [], iter([bad, good])
    def generate(prompt):
        prompts.append(prompt)
        return next(replies)
    output = tmp_path / 'candidate.py'
    generate_validated_section(SEED, output, parts, 1, 'Mutate one layer.', generate, FRED)
    assert SEED.read_text() == original
    assert 'layer 0 Conv: channels [64, 128]' in prompts[0]
    assert 'channels are out of bounds' in prompts[1]
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert [attempt['accepted'] for attempt in audit['attempts']] == [False, True]
    config = yaml.safe_load(output.read_text())
    assert config['backbone'][0][3][0] == 128


def test_invalid_proposals_are_not_silently_rounded_or_parent_fallback(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    output = tmp_path / 'bad.py'
    with pytest.raises(RuntimeError, match='No valid YOLO26 proposal after 2 attempts'):
        generate_validated_section(SEED, output, parts, 2, 'Mutate.',
            lambda _: yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 96}),
            FRED, attempts=2)
    assert not output.exists()


def test_full_section_reply_is_rejected(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    section = yaml.safe_load(parts[1])
    section['backbone'][0][3][0] = 128
    with pytest.raises(RuntimeError, match='No valid YOLO26 proposal'):
        generate_validated_section(SEED, tmp_path / 'bad.py', parts, 1, 'Mutate.',
            lambda _: yaml.safe_dump(section), FRED, attempts=1)


def test_duplicate_edit_is_retried_before_model_preflight(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import (
        UltralyticsYolo26P2Adapter,
    )
    parts = SEED.read_text().split('# --OPTION--')
    first = yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 64})
    second = yaml.safe_dump({'layer': 16, 'field': 'channels', 'value': 128})
    preflight = UltralyticsYolo26P2Adapter.preflight
    calls = []
    def counted(self, candidate, parent):
        calls.append(candidate)
        return preflight(self, candidate, parent)
    monkeypatch.setattr(UltralyticsYolo26P2Adapter, 'preflight', counted)
    generate_validated_section(SEED, tmp_path / 'network_first.py', parts, 2,
                               'Mutate.', lambda _: first, FRED)
    replies = iter((first, second))
    output = tmp_path / 'network_second.py'
    generate_validated_section(SEED, output, parts, 2, 'Mutate.',
                               lambda _: next(replies), FRED)
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert len(calls) == 2  # Duplicate does not construct the model again.
    assert [row['accepted'] for row in audit['attempts']] == [False, True]
    assert 'duplicates network_first.py' in audit['attempts'][0]['error']
    assert yaml.safe_load(output.read_text())['head'][5][3][0] == 128


def test_compact_edit_supports_reductions_and_rejects_unchanged(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    unchanged = yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 128})
    reduction = yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 64})
    replies = iter((unchanged, reduction))
    output = tmp_path / 'network_reduced.py'
    generate_validated_section(SEED, output, parts, 2, 'Mutate.',
                               lambda _: next(replies), FRED)
    assert yaml.safe_load(output.read_text())['head'][8][3][0] == 64
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert 'unchanged' in audit['attempts'][0]['error']


def test_initial_yolo26_panel_covers_four_distinct_reductions(tmp_path, monkeypatch):
    parts = SEED.read_text().split('# --OPTION--')
    source = yaml.safe_load(SEED.read_text())
    entries = []
    for slot in range(4):
        target = initial_target_for_slot(slot)
        layer = INITIAL_COVERAGE_TARGETS[target]
        original = source['head'][layer - 11][3][0]
        output = tmp_path / f'network_{target}.py'
        prompts = []
        generate_validated_section(
            SEED, output, parts, 2, 'Mutate.',
            lambda prompt: (prompts.append(prompt) or yaml.safe_dump(
                {'layer': layer, 'field': 'channels', 'value': original // 2})),
            FRED, coverage_target=target)
        assert f'Required initial coverage target: {target}' in prompts[0]
        entries.append((target, output))
    report = verify_initial_coverage(entries)
    assert report['coverage_version'] == controller.YOLO26_COVERAGE_VERSION
    assert {item['target'] for item in report['candidates']} == set(INITIAL_COVERAGE_TARGETS)
    assert all(item['parameter_delta'] < 0 for item in report['candidates'])
    with pytest.raises(ValueError, match='incomplete initial YOLO26 coverage'):
        verify_initial_coverage(entries[:-1])
    output_dir = tmp_path / 'controller_output'
    output_dir.mkdir()
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'yolo26_p2')
    monkeypatch.setattr(controller, 'VARIANT_DIR', str(tmp_path))
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(output_dir))
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {
        target: {'coverage_target': target} for target, _ in entries})
    controller.verify_yolo26_initial_population([[target] for target, _ in entries])
    controller.verify_yolo26_initial_population([[target] for target, _ in entries])
    saved = json.loads((output_dir / 'initial_yolo26_coverage.json').read_text())
    assert saved == {**report, 'open_candidates': []}
    open_ids = ['open_1', 'open_2', 'open_3']
    controller.GLOBAL_DATA.update({gene: {'coverage_target': None} for gene in open_ids})
    seven_output = tmp_path / 'seven_controller_output'
    seven_output.mkdir()
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(seven_output))
    controller.verify_yolo26_initial_population(
        [[target] for target, _ in entries] + [[gene] for gene in open_ids])
    seven = json.loads((seven_output / 'initial_yolo26_coverage.json').read_text())
    assert seven['open_candidates'] == open_ids


def test_initial_yolo26_target_retries_wrong_layer_and_direction(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    replies = iter((
        yaml.safe_dump({'layer': 22, 'field': 'channels', 'value': 128}),
        yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 192}),
        yaml.safe_dump({'layer': 19, 'field': 'channels', 'value': 64}),
    ))
    output = tmp_path / 'network_reduced.py'
    generate_validated_section(SEED, output, parts, 2, 'Mutate.',
                               lambda _: next(replies), FRED, coverage_target='p2_down')
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert [row['accepted'] for row in audit['attempts']] == [False, False, True]
    assert audit['attempts'][-1]['parameter_delta'] < 0


def test_initial_yolo26_targets_pass_through_controller_to_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'yolo26_p2')
    requested = []
    monkeypatch.setattr(controller.toolbox, 'individual',
                        lambda **kwargs: requested.append(kwargs.get('coverage_target')) or
                        [kwargs.get('coverage_target')])
    assert len(controller.create_population(4, 'test-llm', coverage_panel=True)) == 4
    assert requested == list(INITIAL_COVERAGE_TARGETS)
    requested.clear()
    assert len(controller.create_population(7, 'test-llm', coverage_panel=True)) == 7
    assert requested == [*INITIAL_COVERAGE_TARGETS, None, None, None]
    requested.clear()
    controller.create_population(1, 'test-llm')
    assert requested == [None]
    monkeypatch.setattr(controller, 'GEN_COUNT', -1)
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr(controller, 'generate_template', lambda *args: ('Prompt {}', 'normal'))
    monkeypatch.setattr(controller, 'load_yaml', lambda: {
        'llm_bash_script': '{}', 'gpu_selection': ''})
    script = controller.write_bash_script(
        llm_model='test-llm', job_name='test', input_filename_x=str(SEED),
        output_filename=str(tmp_path / 'models/network_test.py'),
        coverage_target='p3_down')
    assert '--coverage_target p3_down' in script


def test_depth_scaled_repeat_alias_does_not_create_another_candidate(tmp_path):
    parts = SEED.read_text().split('# --OPTION--')
    first = yaml.safe_dump({'layer': 19, 'field': 'repeats', 'value': 3})
    alias = yaml.safe_dump({'layer': 19, 'field': 'repeats', 'value': 4})
    generate_validated_section(SEED, tmp_path / 'network_first.py', parts, 2,
                               'Mutate.', lambda _: first, FRED)
    output = tmp_path / 'network_alias.py'
    with pytest.raises(RuntimeError, match='No valid YOLO26 proposal'):
        generate_validated_section(SEED, output, parts, 2,
                                   'Mutate.', lambda _: alias, FRED, attempts=1)
    assert not output.exists()
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert 'duplicates network_first.py' in audit['attempts'][0]['error']


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
    replies = [yaml.safe_dump({'layer': 0, 'field': 'channels', 'value': channel})
               for channel in (96, 128)]
    def generate(*args): return replies.pop(0)
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', generate)
    template = tmp_path / 'prompt.txt'
    template.write_text('Improve this section: {}')
    output = tmp_path / 'candidate.py'
    llm_mutation.augment_network(str(SEED), str(output), str(template))
    assert not replies and output.exists()


@pytest.mark.parametrize('reply', ['ERROR', None, 'unchanged'])
def test_dfine_mutation_rejects_failed_or_unchanged_proposal(tmp_path, monkeypatch, reply):
    from src import llm_mutation
    seed = FRED / 'seeds/seed_dfine.py'
    monkeypatch.setattr(llm_mutation, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *a: 1)
    if reply == 'unchanged':
        reply = seed.read_text().split('# --OPTION--')[1].strip() + '\n'
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', lambda *a, **k: reply)
    template = tmp_path / 'prompt.txt'
    template.write_text('Improve this section: {}')
    output = tmp_path / 'candidate.py'

    with pytest.raises(RuntimeError, match='No valid D-FINE proposal after 5 attempts'):
        llm_mutation.augment_network(str(seed), str(output), str(template))

    assert not output.exists()
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert len(audit['attempts']) == 5
    assert not any(attempt['accepted'] for attempt in audit['attempts'])


def test_dfine_crossover_rejects_failed_proposal(tmp_path, monkeypatch):
    from src import llm_crossover
    seed = FRED / 'seeds/seed_dfine.py'
    monkeypatch.setattr(llm_crossover, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(llm_crossover, 'ROOT_DIR', str(tmp_path))
    monkeypatch.setattr(llm_crossover, 'CONSTANT_RULES_PATH', None)
    monkeypatch.setattr(llm_crossover, 'generate_augmented_code', lambda *a: 'ERROR')
    templates = tmp_path / 'templates/CrossOver'
    templates.mkdir(parents=True)
    for name in ('crossover.txt', 'crossover_s.txt'):
        (templates / name).write_text('Combine {} and {}')
    output = tmp_path / 'candidate.py'

    with pytest.raises(RuntimeError, match='No valid D-FINE proposal after 5 attempts'):
        llm_crossover.augment_network(str(seed), str(seed), str(output))

    assert not output.exists()


def test_dfine_retry_feedback_produces_novel_executable_candidate(tmp_path):
    from src.fred_dfine_proposals import architecture_hash, generate_validated_section

    seed = FRED / 'seeds/seed_dfine.py'
    parts = seed.read_text().split('# --OPTION--')
    valid = parts[1].replace('ConvNormAct(width * 2, width * 2),', 'nn.Identity(),')
    assert valid != parts[1]
    prompts, replies = [], iter(('ERROR', valid))
    def generate(prompt):
        prompts.append(prompt)
        return next(replies)
    output = tmp_path / 'network_valid.py'
    generate_validated_section(seed, output, parts, 1, 'Change the backbone.',
                               generate, FRED, attempts=2)
    assert output.is_file()
    assert 'Previous proposal was rejected' in prompts[1]
    assert architecture_hash(output) != architecture_hash(seed)
    audit = json.loads(output.with_suffix('.py.proposal_attempts.json').read_text())
    assert [row['accepted'] for row in audit['attempts']] == [False, True]
    duplicate = tmp_path / 'network_duplicate.py'
    with pytest.raises(RuntimeError, match='No valid D-FINE proposal after 1 attempts'):
        generate_validated_section(seed, duplicate, parts, 1, 'Change the backbone.',
                                   lambda _: valid, FRED, attempts=1)
    assert not duplicate.exists()
    duplicate_audit = json.loads(duplicate.with_suffix('.py.proposal_attempts.json').read_text())
    assert 'duplicates network_valid.py' in duplicate_audit['attempts'][0]['error']


def test_dfine_docstring_and_import_are_not_architecture_edits(tmp_path):
    from src.fred_dfine_proposals import _validate_section, architecture_hash

    seed = FRED / 'seeds/seed_dfine.py'
    parts = seed.read_text().split('# --OPTION--')
    prose = list(parts)
    prose[1] = prose[1].replace('Three-scale CNN Event backbone',
                                'Another description of this backbone')
    prose_path = tmp_path / 'network_prose.py'
    prose_path.write_text('# --OPTION--'.join(prose))
    assert architecture_hash(prose_path) == architecture_hash(seed)
    with pytest.raises(ValueError, match='may not add imports'):
        _validate_section(parts[1], prose[1] + '\n    import os\n')


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
    monkeypatch.setattr(llm_crossover, 'generate_augmented_code',
                        lambda *a: yaml.safe_dump({'layer': 0, 'field': 'channels', 'value': 128}))
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


@pytest.mark.parametrize('client_name', ['submit_mixtral_local', 'submit_deepseek_local'])
def test_dfine_local_llm_retries_transient_read_timeout(monkeypatch, client_name):
    from src import llm_utils, local_llm_request

    monkeypatch.setattr(llm_utils, 'SEED_FAMILY', 'dfine')
    monkeypatch.setenv('LLMGE_LLM_REQUEST_ATTEMPTS', '2')
    monkeypatch.setenv('LLMGE_LLM_HTTP_TIMEOUT_SECONDS', '600')
    monkeypatch.setattr(local_llm_request.time, 'sleep', lambda _: None)
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise llm_utils.requests.exceptions.ReadTimeout('temporary stall')
        return SimpleNamespace(status_code=200,
                               json=lambda: {'generated_text': 'valid', 'response_time_sec': 1})

    monkeypatch.setattr(llm_utils.requests, 'post', post)
    client = getattr(llm_utils, client_name)
    assert client('prompt', server_url='http://fixture/generate') == 'valid'
    assert len(calls) == 2
    assert all(call['timeout'] == (10, 600) for call in calls)


def test_dfine_local_llm_exhausted_timeout_fails_visibly(monkeypatch):
    from src import llm_utils, local_llm_request

    monkeypatch.setattr(llm_utils, 'SEED_FAMILY', 'dfine')
    monkeypatch.setenv('LLMGE_LLM_REQUEST_ATTEMPTS', '2')
    monkeypatch.setattr(local_llm_request.time, 'sleep', lambda _: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise llm_utils.requests.exceptions.ReadTimeout('stalled')

    monkeypatch.setattr(llm_utils.requests, 'post', fail)
    with pytest.raises(RuntimeError, match='failed after 2 attempt'):
        llm_utils.submit_mixtral_local('prompt', server_url='http://fixture/generate')
    assert len(calls) == 2
