import io
import json
from pathlib import Path
import shutil

import pytest

from . import common, genome, proposals


@pytest.mark.parametrize('reply', [
    "```python\nclass FeatureGenes:\n    values = {'fpn_channels': 128}\n```",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128}\n```\nExplanation follows.",
    "```python\n```class FeatureGenes:\n    genes = {'fpn_channels': 128}\n```",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128}\n",
])
def test_complete_fenced_or_bare_literal_classes(reply):
    result, fixes = proposals.parse_response(genome.stock(1280), 'FeatureGenes', reply)
    assert result['fpn_channels'] == 128 and result['img_width'] == 1280
    assert fixes == (['genes_to_values'] if 'genes =' in reply else [])


@pytest.mark.parametrize('reply', [
    "class FeatureGenes:\n    genes = {'fpn_channels': 192}",
    "class FeatureGenes:\n    genes = {'fpn_channels': 256}",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128, 'extra': 1}",
    "import os\nclass FeatureGenes:\n    genes = {'fpn_channels': 128}",
    "class FeatureGenes:\n    genes = dict(fpn_channels=128)",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128}\n    def run(self): pass",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128}\n    values = {'fpn_channels': 128}",
    "class FeatureGenes:\n    genes = {'fpn_channels': 128",
])
def test_alias_does_not_relax_architecture_or_executable_code_checks(reply):
    with pytest.raises((ValueError, SyntaxError)):
        proposals.response_child(genome.stock(1280), 'FeatureGenes', reply)


def test_chat_template_is_applied_once_without_duplicate_bos(monkeypatch):
    class Tokenizer:
        bos_token = '<BOS>'
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'request'}]
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return '<BOS><USER>request<ASSISTANT>'
    monkeypatch.setattr(proposals, 'chat_tokenizer', lambda _: Tokenizer())
    assert proposals.format_prompt('request', 'local') == '<USER>request<ASSISTANT>'


def test_exhausted_classes_are_not_proposed_and_forbidden_values_are_listed():
    parent = genome.stock(1280)
    child = dict(parent, fpn_channels=128)
    seen = {genome.candidate_identity(parent): 'stock', genome.candidate_identity(child): 'pending'}
    available = proposals.available_groups(parent, ['FeatureGenes', 'ProposalGenes', 'RoIGenes'], [parent], seen)
    assert 'FeatureGenes' not in available
    assert {'rpn_channels': 256, 'rpn_convs': 1, 'anchor_scales': [8],
            'anchor_ratios': [0.5, 1.0, 2.0]} in available['ProposalGenes']


def test_accepted_reply_recovery_does_not_require_llm_server(tmp_path):
    genes = dict(genome.stock(1280), fpn_channels=128)
    common.write(tmp_path / 'proposals/slot/attempt-00.json', {
        'status': 'accepted', 'candidate': genome.candidate_identity(genes), 'genes': genes})
    state = {'proposal_attempts': {'slot': 1}, 'seen': {}}
    assert proposals.propose(tmp_path, state, None, 'slot', [genome.stock(1280)], 1280) == genes
    with pytest.raises(ValueError):
        proposals.propose(tmp_path, state, None, 'slot', [genome.stock(640)], 640)


@pytest.mark.parametrize('duplicate', [False, True])
def test_request_uses_real_class_and_chat_format_and_rejects_global_duplicates(tmp_path, monkeypatch, duplicate):
    templates = tmp_path / 'templates'
    templates.mkdir()
    root = Path(__file__).resolve().parents[5] / 'templates/FRED_FasterRCNNScratchV2'
    for name in ('mutation.txt', 'crossover.txt', 'rules.txt'):
        shutil.copyfile(root / name, templates / name)
    monkeypatch.setattr(proposals, 'TEMPLATE_ROOT', templates)
    monkeypatch.setattr(proposals, 'format_prompt', lambda text, _: '<CHAT>' + text)
    common.write(tmp_path / 'protocol.json', {'server_port': 9000, 'model_path': 'local'})
    genes = dict(genome.stock(1280), rpn_convs=2)
    state = {'server_job': '1', 'seen': {genome.candidate_identity(genes): 'old'} if duplicate else {}}
    class Scheduler:
        def state(self, _): return 'RUNNING'
        def node(self, _): return 'node'
    requests = []
    class Reply(io.BytesIO):
        status = 200
    def urlopen(request, **kwargs):
        if isinstance(request, str): return Reply(b'{}')
        payload = json.loads(request.data)
        requests.append(payload)
        return Reply(json.dumps({'generated_text': "class ProposalGenes:\n    genes = {'rpn_channels': 256, 'rpn_convs': 2, 'anchor_scales': [8], 'anchor_ratios': [0.5, 1.0, 2.0]}"}).encode())
    monkeypatch.setattr(proposals.urllib.request, 'urlopen', urlopen)
    result = proposals.propose(tmp_path, state, Scheduler(), 'slot', [genome.stock(1280)], 1280)
    assert result == (None if duplicate else genes)
    assert requests[0]['prompt'].startswith('<CHAT>')
    assert 'class ProposalGenes:\n    values =' in requests[0]['prompt']
    trace = common.read(tmp_path / 'proposals/slot/attempt-00.json')
    assert trace['normalizations'] == ['genes_to_values']
    assert trace['status'] == ('rejected' if duplicate else 'accepted')
