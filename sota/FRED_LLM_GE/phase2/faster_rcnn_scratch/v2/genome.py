"""Literal-only V2 genomes; model identity is distinct from input width."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / 'genome_spec.json').read_text())
SEED = HERE / 'seed.py'
GROUPS = SCHEMA['groups']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate(genes):
    allowed = {k: v for group in GROUPS.values() for k, v in group.items()}
    if not isinstance(genes, dict) or set(genes) != set(allowed):
        raise ValueError('V2 requires exactly nine approved fields')
    for key, value in genes.items():
        if not any(type(value) is type(choice) and repr(value) == repr(choice)
                   for choice in allowed[key]):
            raise ValueError(f'Illegal V2 gene {key}: {value!r}')
    return genes


def parse(source):
    parts = source.split('# --OPTION--')
    if len(parts) != 5 or parts[0] != SEED.read_text().split('# --OPTION--')[0]:
        raise ValueError('Protected V2 preamble/markers changed')
    genes = {}
    for chunk, (name, fields) in zip(parts[1:], GROUPS.items()):
        tree = ast.parse(chunk)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
            raise ValueError('Expected one literal gene class')
        cls = tree.body[0]
        if cls.name != name or cls.bases or cls.keywords or cls.decorator_list:
            raise ValueError('Protected class interface changed')
        body = cls.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Assign):
            raise ValueError('Only a values assignment is permitted')
        node = body[0]
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != 'values':
            raise ValueError('Expected values dictionary')
        # literal_eval alone loses duplicate keys: check the AST first.
        if not isinstance(node.value, ast.Dict):
            raise ValueError('Expected literal dictionary')
        keys = [ast.literal_eval(k) for k in node.value.keys]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate gene keys')
        values = ast.literal_eval(node.value)
        if set(values) != set(fields):
            raise ValueError('Gene fields crossed class boundaries')
        genes.update(values)
    return validate(genes)


def render(genes):
    validate(genes)
    head = SEED.read_text().split('# --OPTION--')[0]
    return head + ''.join('# --OPTION--\nclass ' + name + ':\n    values = '
                          + repr({k: genes[k] for k in fields}) + '\n\n'
                          for name, fields in GROUPS.items())


def stock(width=640):
    genes = parse(SEED.read_text())
    genes['img_width'] = width
    return validate(genes)


def model_genes(genes):
    validate(genes)
    return {k: v for k, v in genes.items() if k != 'img_width'}


def config_from_genes(genes):
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_config
    return build_config(model_genes(genes))


def candidate_identity(genes):
    # The pinned builder maps these literals to one resolved configuration.
    return digest({'version': SCHEMA['version'], 'model': model_genes(genes),
                   'img_width': genes['img_width']})


def resolved_identity(genes, model):
    architecture = digest({'config': config_from_genes(genes).to_dict(),
                           'modules': [(n, type(m).__module__, type(m).__qualname__)
                                       for n, m in model.named_modules()],
                           'tensors': [(n, list(t.shape), str(t.dtype))
                                       for n, t in model.state_dict().items()]})
    return architecture, digest({'architecture': architecture, 'img_width': genes['img_width']})
