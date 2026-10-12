"""Real LLM response parsing and failure behavior; no live model calls."""

import importlib
from pathlib import Path

import pytest

from src.generated_class import extract_generated_class

CLASS = '''class YOLO11Neck(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, features):
        return features
'''


@pytest.mark.parametrize('response', [CLASS, f'```python\n{CLASS}```',
                                      f'Here is the code:\n```\n{CLASS}```',
                                      f'```Python\n{CLASS}```'])
def test_complete_plain_or_fenced_class(response):
    assert extract_generated_class(response, 'YOLO11Neck') == CLASS.strip()


@pytest.mark.parametrize('response', [None, '', '```python\n```',
    f'```python\n{CLASS}', 'Improve this backbone or neck for tiny drones.',
    f'```python\n{CLASS}```\n```python\n{CLASS}```', 'ERROR',
    CLASS.replace('YOLO11Neck', 'OtherClass'), CLASS + '\nprint("demo")',
    CLASS.replace('return features', 'import os\n        return features'),
    CLASS.replace('def forward(self, features):', 'def forward(self, features)'),
    CLASS.replace('self, layers', 'self, channels'),
    CLASS.replace('    def forward(self, features):\n        return features\n', '')])
def test_invalid_output_is_rejected(response):
    with pytest.raises(ValueError):
        extract_generated_class(response, 'YOLO11Neck')


def test_malformed_provider_responses_have_bounded_retries(monkeypatch):
    from src import llm_mutation
    module = importlib.import_module(llm_mutation.generate_augmented_code.__module__)
    calls = []
    def provider(prompt, **kwargs):
        calls.append(prompt)
        return '```python\n```'
    monkeypatch.setattr(module, 'get_llm_code_generator', lambda *a: (provider, None))
    monkeypatch.setattr(module, 'max_gen_attempts', 2)
    monkeypatch.setattr(module, 'retrieve_base_code', lambda *a: CLASS)
    with pytest.raises(ValueError, match='after 2 attempts'):
        module.generate_augmented_code('rewrite class', 1, False, 0.1, 'mixtral', 0.1)
    assert len(calls) == 2
    assert 'Validation rejected the proposal' in calls[1]


@pytest.mark.parametrize('response', ['', None, 'ERROR', 'Propose an alternative feature-processing architecture.'])
def test_mutation_never_writes_prompt_as_candidate(tmp_path, monkeypatch, response):
    from src import llm_mutation
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import runner
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *a, **k: 2)
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', lambda *a, **k: response)
    template = tmp_path / 'prompt.txt'
    template.write_text('Propose an alternative feature-processing architecture.\n```python\n{}\n```')
    output = tmp_path / 'network_bad.py'
    with pytest.raises(ValueError):
        llm_mutation.augment_network(str(runner.SEED_FILE), str(output), str(template), llm_model='mixtral')
    assert not output.exists()


def test_mutation_rejects_unchanged_parent(tmp_path, monkeypatch):
    from src import llm_mutation
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import runner
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *a, **k: 2)
    original = runner.SEED_FILE.read_text().split(runner.MARKER)[2]
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', lambda *a, **k: original)
    template = tmp_path / 'prompt.txt'
    template.write_text('Rewrite {}')
    output = tmp_path / 'network_unchanged.py'
    with pytest.raises(ValueError, match='unchanged parent'):
        llm_mutation.augment_network(str(runner.SEED_FILE), str(output), str(template), llm_model='mixtral')
    assert not output.exists()


def test_empty_crossover_is_failed_instead_of_successful_parent_copy(tmp_path, monkeypatch):
    from src import llm_crossover
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import runner
    monkeypatch.setattr(llm_crossover, 'generate_augmented_code', lambda *a, **k: '')
    output = tmp_path / 'network_empty.py'
    with pytest.raises(ValueError, match='no candidate code'):
        llm_crossover.augment_network(str(runner.SEED_FILE), str(runner.SEED_FILE), str(output), llm_model='mixtral')
    assert not output.exists()
