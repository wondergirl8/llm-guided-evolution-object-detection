"""Regression tests for the YOLO11 response that echoed an unchanged YAML block."""

from copy import deepcopy

import pytest
import yaml

from src import llm_mutation, llm_utils
from sota.FRED_LLM_GE.phase1_detection import yolo11_search_space as policy


def _section(*, widths=()):
    section = deepcopy(yaml.safe_load(policy.SEED_YAML.read_text())['backbone'])
    for index, width in widths:
        section[index][3][0] = width
    return yaml.safe_dump({'backbone': section}, sort_keys=False)


def _reply(*sections):
    return ''.join(f'```yaml\n{section}```\n' for section in sections)


def _run_mutation(tmp_path, monkeypatch, replies):
    parent = tmp_path / 'parent.yaml'
    child = tmp_path / 'child.yaml'
    parent.write_text(policy.SEED_YAML.read_text())
    monkeypatch.setattr(llm_mutation, 'load_candidate_policy', lambda _: policy)
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *_: 1)
    monkeypatch.setattr(policy, 'differs_from_parents', lambda *_: True)
    responses = iter(replies)
    calls = []

    def generate(prompt, section, qc, top_p, model, temperature, *, return_raw):
        calls.append((prompt, section, return_raw))
        return next(responses)

    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', generate)
    llm_mutation.augment_network(str(parent), str(child),
                                 'templates/FRED_YOLO11/Normal/2.txt')
    return child, calls


def test_mutation_skips_unchanged_first_block_and_uses_changed_block(tmp_path, monkeypatch):
    child, calls = _run_mutation(tmp_path, monkeypatch, [
        _reply(_section(), _section(widths=((0, 128),)))])
    assert len(calls) == 1
    assert calls[0][2] is True
    assert yaml.safe_load(child.read_text())['backbone'][0][3][0] == 128


def test_first_attempt_rejects_two_edits_and_retry_keeps_section(tmp_path, monkeypatch):
    child, calls = _run_mutation(tmp_path, monkeypatch, [
        _reply(_section(widths=((0, 128), (1, 64)))),
        _reply(_section(widths=((0, 128),))),
    ])
    assert len(calls) == 2
    assert [call[1] for call in calls] == [0, 0]
    assert 'changed 2' in calls[1][0]
    assert yaml.safe_load(child.read_text())['backbone'][0][3][0] == 128


def test_mutation_retries_when_repair_collapses_width_to_parent(tmp_path, monkeypatch):
    child, calls = _run_mutation(tmp_path, monkeypatch, [
        _reply(_section(widths=((0, 65),))),
        _reply(_section(widths=((0, 128),))),
    ])
    assert len(calls) == 2
    assert 'changed 0' in calls[1][0]
    assert yaml.safe_load(child.read_text())['backbone'][0][3][0] == 128


def test_only_complete_yaml_blocks_are_considered():
    reply = _reply(_section()) + '```yaml\nbackbone:\n  - [incomplete'
    assert policy.candidate_chunks(reply) == [_section().strip()]


def test_raw_response_preserves_later_blocks(monkeypatch):
    reply = _reply(_section(), _section(widths=((0, 128),)))
    monkeypatch.setattr(llm_utils, 'get_llm_code_generator',
                        lambda _: (lambda *args, **kwargs: reply, None))
    assert llm_utils.generate_augmented_code('prompt', 0, False, .1, 'mixtral', .1,
                                             return_raw=True) == reply
    assert llm_utils.generate_augmented_code('prompt', 0, False, .1, 'mixtral', .1) == (
        _section().strip())


def test_missing_complete_block_fails_without_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mutation, 'max_gen_attempts', 2, raising=False)
    with pytest.raises(RuntimeError, match='no complete fenced YAML section'):
        _run_mutation(tmp_path, monkeypatch, ['```yaml\nhead: [', 'no YAML block'])
    assert not (tmp_path / 'child.yaml').exists()


def test_local_yolo_request_uses_served_models_chat_template(monkeypatch):
    class Tokenizer:
        bos_token = '<bos>'

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{'role': 'user', 'content': 'edit this YAML'}]
            assert tokenize is False and add_generation_prompt is True
            return '<bos><user>edit this YAML</user><assistant>'

    sent = []

    class Response:
        status_code = 200

        def json(self):
            return {'generated_text': 'done'}

    monkeypatch.setattr(llm_utils, 'CANDIDATE_POLICY_MODULE', policy.__name__, raising=False)
    monkeypatch.setattr(llm_utils, '_local_chat_tokenizer', lambda _: Tokenizer())
    monkeypatch.setattr(llm_utils, 'post_with_retry',
                        lambda url, **kwargs: sent.append(kwargs['payload']) or Response())
    assert llm_utils.submit_mixtral_local('edit this YAML', server_url='http://test/generate') == 'done'
    assert sent[0]['prompt'] == '<user>edit this YAML</user><assistant>'


def test_other_problems_keep_plain_local_prompt(monkeypatch):
    monkeypatch.setattr(llm_utils, 'CANDIDATE_POLICY_MODULE', None, raising=False)
    assert llm_utils._format_local_prompt('legacy prompt') == 'legacy prompt'
