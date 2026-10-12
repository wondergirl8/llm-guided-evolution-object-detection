"""Regressions for ICE generation failures, bounded recovery and server batching."""
import asyncio
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.generated_class import make_candidate_validator
from src.llm_batch import generate_batch, prepare_prompts


def test_bad_constructor_and_residual_are_retried_before_acceptance(tmp_path, monkeypatch, capsys):
    from src import llm_mutation
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import runner
    module = importlib.import_module(llm_mutation.generate_augmented_code.__module__)
    parts = runner.SEED_FILE.read_text().split(runner.MARKER)
    parent = parts[1].strip()
    bad_constructor = parent.replace('self.layers = nn.ModuleList(layers)',
        'self.layers = nn.ModuleList(layers)\n        self.extra = Conv(in_ch=3, out_ch=3)')
    bad_shape = parent.replace('x = layer(x)',
        'x = layer(x)\n            if index == 5:\n                x = x + torch.nn.functional.interpolate(x, scale_factor=2)')
    valid = parent.replace('self.layers = nn.ModuleList(layers)',
        'self.layers = nn.ModuleList(layers)\n        width = layers[4].cv2.conv.out_channels\n        self.extra = Conv(width, width, 3, 1)').replace(
        'features.append(x)', 'features.append(x + 0.1 * self.extra(x) if index == 4 else x)')
    responses = iter([bad_constructor, bad_shape, valid])
    prompts = []
    def provider(prompt, **kwargs):
        prompts.append(prompt)
        return next(responses)
    monkeypatch.setenv('FRED_YOLO_SCALE', 'n')
    monkeypatch.setattr(module, 'retrieve_base_code', lambda *a: parent)
    monkeypatch.setattr(module, 'get_llm_code_generator', lambda *a: (provider, None))
    monkeypatch.setattr(module, 'max_gen_attempts', 3)
    audit = tmp_path / 'candidate.generation.json'
    # The unchanged parent is rejected without launching CPU checks.
    validator = make_candidate_validator(parts, 1)
    with pytest.raises(ValueError, match='unchanged parent'):
        validator(parent)
    result = module.generate_augmented_code('Rewrite backbone', 1, False, .1, 'mixtral', .1,
                                            validator=validator, audit_path=audit)
    assert result == valid
    records = json.loads(audit.read_text())['attempts']
    assert [r['status'] for r in records] == ['rejected', 'rejected', 'accepted']
    assert 'in_ch' in records[0]['reason']
    assert 'size of tensor' in records[1]['reason']
    assert records[2]['cpu_preflight']['status'] == 'passed'
    assert 'Previous rejected proposal' in prompts[1] and 'in_ch' in prompts[1]
    assert 'size of tensor' in prompts[2]
    assert not list(tmp_path.glob('*evaluation.json'))
    import run_improved as evolution
    output = capsys.readouterr().out
    assert evolution.check_contents_for_error(output) is None
    assert evolution.check_contents_for_error(output + '\nJob Done\n') is True


def test_cpu_validation_timeout_and_failed_request_stop(monkeypatch, tmp_path):
    import src.generated_class as generated
    from src import llm_mutation
    module = importlib.import_module(llm_mutation.generate_augmented_code.__module__)
    from sota.FRED_LLM_GE.phase1_detection.tests.test_generated_class import CLASS
    def timeout(*args, **kwargs):
        raise generated.subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(generated.subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='exceeded'):
        make_candidate_validator(['', CLASS], 1, require_change=False)(CLASS)
    calls = []
    def provider(*args, **kwargs):
        calls.append(args)
        return None
    monkeypatch.setattr(module, 'retrieve_base_code', lambda *a: CLASS)
    monkeypatch.setattr(module, 'get_llm_code_generator', lambda *a: (provider, None))
    with pytest.raises(ValueError, match='after 1 attempts'):
        module.generate_augmented_code('rewrite', 1, False, .1, 'mixtral', .1,
                                       audit_path=tmp_path / 'audit.json')
    assert len(calls) == 1


def test_chat_batch_groups_settings_and_preserves_result_order():
    calls = []
    def pipeline(prompts, **settings):
        calls.append((prompts, settings))
        return [[{'generated_text': f'code:{messages[-1]["content"]}'}] for messages in prompts]
    def request(prompt, temperature):
        return dict(prompt=prompt, max_new_tokens=4096, temperature=temperature, top_p=.1)
    results = asyncio.run(generate_batch(pipeline, SimpleNamespace(chat_template='llama'),
        [request('one', .1), request('two', .2), request('three', .1)]))
    assert [r['generated_text'] for r in results] == ['code:one', 'code:two', 'code:three']
    assert [r['batch_size'] for r in results] == [2, 1, 2]
    assert [settings['temperature'] for _, settings in calls] == [.1, .2]
    assert calls[0][0][0][0]['role'] == 'system'
    assert all(settings['return_full_text'] is False for _, settings in calls)
    assert prepare_prompts(['raw'], SimpleNamespace()) == ['raw']
    with pytest.raises(ValueError, match='number of completions'):
        asyncio.run(generate_batch(lambda *a, **k: [], SimpleNamespace(), [request('one', .1)]))


def test_initialization_keeps_valid_parent_and_restores_generation(monkeypatch):
    import run_improved as evolution
    def individual(name, fitness):
        value = evolution.creator.Individual([name])
        value.fitness.values = fitness
        return value
    good = (0.2, 0.1, 100.)
    first = individual('first', good)
    batches = iter([[individual('bad', evolution.INVALID_FITNESS_MAX)], [individual('second', good)]])
    scores = {'bad': evolution.INVALID_FITNESS_MAX, 'second': good}
    states = []
    def evaluate(batch):
        for item in batch:
            item.fitness.values = scores[item[0]]
    monkeypatch.setattr(evolution, 'GEN_COUNT', 7)
    monkeypatch.setattr(evolution, 'TOP_N_GENES', ['original'])
    monkeypatch.setattr(evolution, 'num_elites', 2)
    monkeypatch.setattr(evolution, 'MAX_INITIALIZATION_BATCHES', 2)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {'old': {'status': 'completed'}})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution.toolbox, 'population', lambda **k: next(batches))
    monkeypatch.setattr(evolution, 'delayed_creation_check', lambda batch: None)
    monkeypatch.setattr(evolution, 'check_and_update_fitness', evaluate)
    survivors = evolution.replenish_population([first], 'mixtral', lambda pop: states.append([p[0] for p in pop]))
    assert [p[0] for p in survivors] == ['first', 'second']
    assert states == [['first'], ['first', 'second']]
    assert evolution.GEN_COUNT == 7 and evolution.TOP_N_GENES == ['original']
    assert 'old' in evolution.GLOBAL_DATA_HIST
    monkeypatch.setattr(evolution.toolbox, 'population', lambda **k: [individual('bad', evolution.INVALID_FITNESS_MAX)])
    states.clear()
    with pytest.raises(RuntimeError, match='after 2 retry batches'):
        evolution.replenish_population([first], 'mixtral', lambda pop: states.append([p[0] for p in pop]))
    assert states == [['first'], ['first']]
    assert evolution.GEN_COUNT == 7


def test_server_wait_checks_allocation_identity_and_exit(tmp_path, monkeypatch):
    from scripts import wait_yolo11_server as waiter
    monkeypatch.setattr(waiter, '__file__', str(tmp_path / 'scripts/wait_yolo11_server.py'))
    marker = tmp_path / 'run_job_outputs/server/hostname-123.log'
    marker.parent.mkdir(parents=True)
    marker.write_text('127.0.0.1')
    monkeypatch.setattr(waiter.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='RUNNING'))
    class Response(io.BytesIO):
        status = 200
    identities = iter(['different-server', '123'])
    monkeypatch.setattr(waiter.urllib.request, 'urlopen', lambda *a, **k:
                        Response(json.dumps({'slurm_job_id': next(identities)}).encode()))
    monkeypatch.setattr(waiter.time, 'sleep', lambda *a: None)
    waiter.wait_for_server('123', 8293, 1)
    assert (tmp_path / 'hostname.log').read_text().strip() == '127.0.0.1'
    monkeypatch.setattr(waiter.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    with pytest.raises(RuntimeError, match='stopped before becoming ready'):
        waiter.wait_for_server('123', 8293, 1)
    with pytest.raises(TimeoutError, match='not ready within'):
        waiter.wait_for_server('123', 8293, 0)


def test_fatal_job_failure_is_not_recovered_by_success_text(monkeypatch):
    import run_improved as evolution
    assert evolution.check_contents_for_error('Return the class; print("job done")') is None
    assert evolution.check_contents_for_error('Traceback (most recent call last):\nRuntimeError: failed\nJob Done') is False
    assert evolution.check_contents_for_error('slurmstepd: error: OOM\nJob Done') is False
    monkeypatch.setattr(evolution, 'DELAYED_CHECK', True)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {'bad': dict(status='DELAYED_CHECK', sub_flag=True, job_id='123')})
    monkeypatch.setattr(evolution, 'check4job_completion', lambda *a, **k: False)
    def unexpected_gpu(*args):
        raise AssertionError('A rejected generation must not submit GPU training')
    monkeypatch.setattr(evolution, 'check4model2run', unexpected_gpu)
    candidate = evolution.creator.Individual(['bad'])
    evolution.delayed_creation_check([candidate])
    candidate.fitness.values = evolution.PLACEHOLDER_FITNESS
    evolution.check_and_update_fitness([candidate])
    assert candidate.fitness.values == evolution.INVALID_FITNESS_MAX
    assert evolution.GLOBAL_DATA['bad']['status'] == 'completed'
