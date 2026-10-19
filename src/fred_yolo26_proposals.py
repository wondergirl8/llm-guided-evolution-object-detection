"""Validate complete YOLO26 sections and retry with concrete error feedback.

The model still proposes each architecture. No channel rounding, automatic
architecture edits, or unchanged-parent fallback is performed here.
"""
import json
from pathlib import Path
import tempfile

import yaml


def generate_validated_section(parent, output, parts, index, prompt, generate,
                               fred_dir, attempts=3):
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter

    parent, output, fred_dir = Path(parent), Path(output), Path(fred_dir)
    if attempts < 1:
        raise ValueError('Proposal attempts must be positive')
    if index not in (1, 2) or len(parts) != 3:
        raise ValueError('YOLO26 requires its protected preamble and two mutable sections')
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing candidate: {output}')
    protocol = yaml.safe_load((fred_dir / 'configs/phase1/protocol.yaml').read_text())
    adapter = UltralyticsYolo26P2Adapter(protocol['families']['yolo26_p2'], fred_dir)
    seed = yaml.safe_load(adapter.variant_path('network', None).read_text())
    section = ('backbone', 'head')[index - 1]
    choices = []
    for layer_index, (_, _, module, args) in enumerate(seed[section]):
        if module in ('Conv', 'C3k2'):
            allowed = list(range(max(64, args[0] // 2), args[0] * 2 + 1, 64))
            choices.append(f'{section} layer {layer_index}: allowed channel arguments {allowed}')
    prompt += ('\n\nValidator-derived channel choices (raw YAML values, before scaling):\n'
               + '\n'.join(choices)
               + '\nIn particular, 32 and 96 are NEVER valid channel arguments. '
                 'Change exactly one layer relative to the supplied parent. Copy every '
                 'other layer exactly, including repeats. Return only the complete '
                 f'{section} section in one YAML code block.')
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {'parent': str(parent), 'section': section, 'attempts': []}
    audit_path = output.with_suffix(output.suffix + '.proposal_attempts.json')
    feedback = ''
    for attempt in range(1, attempts + 1):
        reply = generate(prompt + feedback)
        entry = {'attempt': attempt, 'proposal': reply, 'accepted': False}
        try:
            if not reply or reply == 'ERROR' or '# --OPTION--' in reply:
                raise ValueError('A complete YAML section without option markers is required')
            replacement = yaml.safe_load(reply)
            if not isinstance(replacement, dict) or set(replacement) != {section}:
                raise ValueError(f'Return only the complete {section} section')
            merged = list(parts)
            merged[index] = '\n' + reply.strip() + '\n'
            with tempfile.TemporaryDirectory(prefix='proposal-', dir=output.parent) as temp:
                candidate = Path(temp) / 'candidate.py'
                candidate.write_text('# --OPTION--'.join(merged))
                digest = adapter.preflight(candidate, parent)
                candidate.replace(output)
            entry.update(accepted=True, architecture_sha256=digest)
        except Exception as error:
            entry['error'] = f'{type(error).__name__}: {error}'
            print(f'YOLO26 proposal attempt {attempt}/{attempts} rejected: {entry["error"]}', flush=True)
            feedback = (f'\n\nYour previous proposal was rejected: {entry["error"]}\n'
                        f'Previous replacement section:\n```yaml\n{reply}\n```\n'
                        'Generate a corrected replacement relative to the ORIGINAL parent. '
                        'Use only the listed channel arguments and change exactly one layer. '
                        'Do not repeat the rejected proposal.')
        audit['attempts'].append(entry)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        if entry['accepted']:
            print(f'YOLO26 proposal accepted on attempt {attempt}: {digest}', flush=True)
            return
    raise RuntimeError(f'No valid YOLO26 proposal after {attempts} attempts; see {audit_path}')
