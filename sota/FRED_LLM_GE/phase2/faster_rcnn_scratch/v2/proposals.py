"""Bounded LLM proposals, persisted before training and never silently randomized."""
import ast
from functools import lru_cache
import itertools
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from . import common, genome

TEMPLATE_ROOT = None  # Explicit audited overlay for already-frozen runs only.


@lru_cache(maxsize=1)
def chat_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def format_prompt(prompt, model_path):
    tokenizer = chat_tokenizer(model_path)
    formatted = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    # TextGenerationPipeline adds BOS when it tokenizes our string.
    bos = tokenizer.bos_token
    return formatted[len(bos):] if bos and formatted.startswith(bos) else formatted


def available_groups(base, groups, parents, seen):
    """Check finite class feasibility; the LLM still proposes every child value."""
    available = {}
    for group in groups:
        fields = genome.GROUPS[group]
        forbidden = []
        possible = False
        for values in itertools.product(*fields.values()):
            replacement = dict(zip(fields, values))
            candidate = dict(base, **replacement)
            if genome.candidate_identity(candidate) in seen or any(candidate == p for p in parents):
                forbidden.append(replacement)
            else:
                possible = True
        if possible:
            available[group] = forbidden
    return available


def response_child(parent, group, response):
    return parse_response(parent, group, response)[0]


def parse_response(parent, group, response):
    if not isinstance(response, str):
        raise ValueError('Expected a text proposal')
    response = re.sub(r'```[^\n`]+\n```(?=class\s)', '```python\n', response)
    chunks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    # A missing/stray opening fence must not hide an otherwise complete class.
    # Parse whole segments, never extract a class from executable Python code.
    chunks += re.split(r'^\s*```[^\n]*$', response, flags=re.MULTILINE)
    reason = 'No complete literal gene class'
    parts = genome.render(parent).split('# --OPTION--')
    index = list(genome.GROUPS).index(group) + 1
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            tree = ast.parse(chunk.strip())
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
                raise ValueError('Expected one literal gene class')
            normalizations = []
            for statement in tree.body[0].body:
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == 'genes'):
                    statement.targets[0].id = 'values'
                    normalizations.append('genes_to_values')
            changed = list(parts)
            changed[index] = '\n' + ast.unparse(tree) + '\n'
            result = genome.parse('# --OPTION--'.join(changed))
            if result == parent:
                raise ValueError('unchanged_parent')
            return result, normalizations
        except (ValueError, SyntaxError, TypeError) as exc:
            reason = str(exc)
    raise ValueError(reason)


def propose(run, state, scheduler, slot, parents, width=None):
    """One durable attempt per tick; None means wait, rejection, or blocked."""
    attempts = state.setdefault('proposal_attempts', {})
    attempt = attempts.get(slot, 0)
    if attempt:
        previous_trace = run / 'proposals' / slot / f'attempt-{attempt-1:02d}.json'
        if previous_trace.exists():
            record = common.read(previous_trace)
            if record.get('status') == 'accepted' and record['candidate'] not in state['seen']:
                genes = genome.validate(record['genes'])
                if (genome.candidate_identity(genes) != record['candidate']
                        or any(genes == p for p in parents)
                        or (width is not None and genes['img_width'] != width)):
                    raise ValueError('Recovered proposal identity or width differs')
                return genes
    if attempt >= 24:
        state.update(stage='blocked', reason=f'Proposal budget exhausted for {slot}')
        return None
    if state.get('server_submission'):
        state['server_job'] = scheduler.submit(state['server_submission'], 'server')
        state.pop('server_submission')
        return None
    server = state.get('server_job')
    server_state = scheduler.state(server) if server else None
    if server_state == 'UNKNOWN':
        return None
    if not server or server_state not in scheduler_active():
        if server_state == 'CANCELLED':
            state.update(stage='stopped', reason='LLM server explicitly cancelled')
            return None
        if server:
            state['server_failures'] = state.get('server_failures', 0) + 1
            if state['server_failures'] > 3:
                state.update(stage='blocked', reason='LLM server infrastructure retry limit')
                return None
        state['server_sequence'] = state.get('server_sequence', 0) + 1
        state['server_submission'] = f'server_{state["server_sequence"]}'
        common.write(run / 'state.json', state)
        state['server_job'] = scheduler.submit(state['server_submission'], 'server')
        state.pop('server_submission')
        return None
    node = scheduler.node(state['server_job'])
    if not node:
        return None
    port = common.read(run / 'protocol.json')['server_port']
    try:
        with urllib.request.urlopen(f'http://{node}:{port}/', timeout=5) as response:
            if response.status != 200:
                return None
    except (OSError, urllib.error.URLError):
        return None
    rng = state_rng(state)
    groups = list(genome.GROUPS)
    if width is not None:
        groups.remove('InputGenes')
    base = dict(parents[0])
    if width is not None:
        base['img_width'] = width
    available = available_groups(base, groups, parents, state['seen'])
    if not available:
        state.update(stage='blocked', reason=f'No unseen legal class replacements for {slot}')
        return None
    group = rng.choice(list(available))
    current = {k: base[k] for k in genome.GROUPS[group]}
    other = {k: parents[-1][k] for k in genome.GROUPS[group]}
    operation = 'crossover' if len(parents) > 1 and rng.random() < .5 else 'mutation'
    templates = Path(TEMPLATE_ROOT) if TEMPLATE_ROOT else run / 'code/templates/FRED_FasterRCNNScratchV2'
    template = (templates / (operation + '.txt')).read_text()
    rules = (templates / 'rules.txt').read_text()
    prompt = template.replace('{group}', group).replace('{current}', repr(current)).replace('{other}', repr(other))
    prompt += '\n' + rules + '\nLegal values for this class: ' + json.dumps(genome.GROUPS[group])
    prompt += f'\nCurrent parent class (change at least one legal value):\n```python\nclass {group}:\n    values = {current!r}\n```\n'
    prompt += '\nDo not return ANY of these exact dictionaries: they reproduce a parent or a reserved, failed, or evaluated candidate with these other classes. Choose a different legal dictionary:\n' + json.dumps(available[group])
    if width is not None:
        prompt += f'\nInput resolution is protected for this proposal: img_width={width}. Do not change InputGenes.'
    previous = state.get('proposal_errors', {}).get(slot)
    if previous:
        prompt += '\nPrevious rejection: ' + previous
    plain_prompt = prompt
    request = {'prompt': format_prompt(prompt, common.read(run / 'protocol.json')['model_path']), 'max_new_tokens': 512,
               'temperature': min(.8, .1 + .1 * attempt), 'top_p': min(.9, .15 + .1 * attempt)}
    trace = run / 'proposals' / slot / f'attempt-{attempt:02d}.json'
    payload = {'operation': operation, 'group': group, 'request': request, 'plain_prompt': plain_prompt,
               'parents': parents, 'status': 'requested'}
    common.write(trace, payload)
    attempts[slot] = attempt + 1
    save_rng(state, rng)
    common.write(run / 'state.json', state)
    try:
        message = urllib.request.Request(f'http://{node}:{port}/generate',
                                         data=json.dumps(request).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(message, timeout=140) as reply:
            result = json.load(reply)
        payload['response'] = result['generated_text']
        child, payload['normalizations'] = parse_response(base, group, payload['response'])
        if width is not None and child['img_width'] != width:
            raise ValueError('Initial resolution assignment changed')
        if operation == 'crossover':
            # Both parents contribute a real changed configuration, not just new formatting.
            if any(child == p for p in parents):
                raise ValueError('unchanged_parent')
        key = genome.candidate_identity(child)
        if key in state['seen'] or any(child == p for p in parents):
            raise ValueError('duplicate_or_parent_candidate')
        payload.update(status='accepted', candidate=key, genes=child)
        common.write(trace, payload)
        return child
    except (ValueError, KeyError, OSError, urllib.error.URLError) as exc:
        payload.update(status='rejected', reason=str(exc))
        common.write(trace, payload)
        state.setdefault('proposal_errors', {})[slot] = str(exc)
        return None


def scheduler_active():
    from .slurm import ACTIVE
    return ACTIVE


def state_rng(state):
    import base64
    import pickle
    import random
    rng = random.Random(0)
    if state.get('rng'):
        rng.setstate(pickle.loads(base64.b64decode(state['rng'])))
    return rng


def save_rng(state, rng):
    import base64
    import pickle
    state['rng'] = base64.b64encode(pickle.dumps(rng.getstate())).decode()
