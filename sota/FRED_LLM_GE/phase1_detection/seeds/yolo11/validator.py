"""Validate generated YOLO11 source before importing it."""

import ast
import importlib.util
import re
from pathlib import Path


SEED_PATH = Path(__file__).with_name("network.py")
MODEL_NAME = re.compile(r"network(?:_([A-Za-z0-9]+))?\Z")


def validate_source(candidate_path: Path) -> dict:
    source = candidate_path.read_text(encoding="utf-8")
    seed = SEED_PATH.read_text(encoding="utf-8")
    if source.count("# --OPTION--") != 1:
        raise ValueError("candidate must have exactly one mutation boundary")
    candidate_ast = ast.parse(source, filename=str(candidate_path))
    seed_ast = ast.parse(seed, filename=str(SEED_PATH))
    if len(candidate_ast.body) != len(seed_ast.body):
        raise ValueError("candidate changed protected YOLO11 code")
    for actual, expected in zip(candidate_ast.body[:-1], seed_ast.body[:-1]):
        if ast.dump(actual) != ast.dump(expected):
            raise ValueError("candidate changed protected YOLO11 code")
    final = candidate_ast.body[-1]
    if not isinstance(final, ast.Assign) or len(final.targets) != 1:
        raise ValueError("mutation must be one literal GENOME assignment")
    if not isinstance(final.targets[0], ast.Name) or final.targets[0].id != "GENOME":
        raise ValueError("mutation must assign GENOME")
    genome = ast.literal_eval(final.value)
    if not isinstance(genome, dict):
        raise ValueError("GENOME must be a literal dict")
    # Validate allowed keys, values and layer contract without importing code.
    spec = importlib.util.spec_from_file_location("fred_yolo11_seed_validation", SEED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if set(genome) != set(module._LAYERS):
        raise ValueError("GENOME contains missing or unsupported genes")
    if any(type(value) is not int or value not in module._ALLOWED_REPEATS
           for value in genome.values()):
        raise ValueError("GENOME contains an unsupported repeat")
    return genome


def load_model_module(model_name: str, variant_dir: Path):
    if not MODEL_NAME.fullmatch(model_name):
        raise ValueError("--model must be network or network_<alphanumeric gene ID>")
    path = SEED_PATH if model_name == "network" else (variant_dir / f"{model_name}.py").resolve()
    if not path.is_file() or path.parent != (SEED_PATH.parent if model_name == "network" else variant_dir.resolve()):
        raise FileNotFoundError(path)
    validate_source(path)
    spec = importlib.util.spec_from_file_location(f"fred_yolo11_{model_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
