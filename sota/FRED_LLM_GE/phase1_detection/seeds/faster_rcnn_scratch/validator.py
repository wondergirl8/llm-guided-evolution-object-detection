"""Validate bounded literal scratch genomes before importing candidates."""
import ast
import json
from pathlib import Path

SEED = Path(__file__).resolve().parents[3] / "seeds/seed_faster_rcnn_scratch.py"
SCHEMA = json.loads(Path(__file__).with_name("genome_spec.yaml").read_text())

def validate_genes(genes):
    allowed = {key: choices for group in SCHEMA["groups"].values() for key, choices in group.items()}
    if not isinstance(genes, dict) or set(genes) != set(allowed):
        raise ValueError("Genome must contain exactly the approved gene fields")
    for key, value in genes.items():
        if not any(type(value) is type(choice) and value == choice and repr(value) == repr(choice)
                   for choice in allowed[key]):
            raise ValueError(f"Illegal prototype gene {key}: {value!r}")
    return genes

def validate_source(path):
    parts = Path(path).read_text().split("# --OPTION--")
    reference = SEED.read_text().split("# --OPTION--")
    if len(parts) != 4 or parts[0] != reference[0]:
        raise ValueError("Protected preamble or option marker count changed")
    genes = {}
    for chunk, (name, fields) in zip(parts[1:], SCHEMA["groups"].items()):
        tree = ast.parse(chunk)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
            raise ValueError("Mutable block must contain exactly one gene class")
        cls = tree.body[0]
        if cls.name != name or cls.bases or cls.keywords or cls.decorator_list:
            raise ValueError("Gene class interface changed")
        body = list(cls.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
        if len(body) != 1 or not isinstance(body[0], ast.Assign):
            raise ValueError("Gene classes may only assign a literal values dictionary")
        assignment = body[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "values":
            raise ValueError("Expected values assignment")
        values = ast.literal_eval(assignment.value)
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError("Gene class fields changed")
        genes.update(values)
    return validate_genes(genes)
