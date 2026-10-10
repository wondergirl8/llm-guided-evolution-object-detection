"""Verify frozen code, permitting only the documented prediction batching repair."""
import ast
import hashlib
import subprocess

LEGACY_REVISION = "f2980bf092279082f6dc929d24f3e887b21b8999"
LEGACY_TRAINER_SHA256 = "2a2f494f5d2025c58f4a5b56b9455485f447d1dc8b6fafbd8112b2a5e51112e5"
TRAINER_PATH = "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py"


def without_prediction_body(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_frames"]
    if len(functions) != 1:
        raise ValueError("expected exactly one prediction helper")
    functions[0].body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def verify_code_files(root, metadata):
    changes = {}
    for name, identity in metadata["code_files_sha256"].items():
        path = (root / name if name in ("train_eval.py", "network.py", "validator.py") else
                root.parent.parent / "adapters" / name if name == "yolo11.py" else root.parent.parent / name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == identity:
            continue
        if name != "train_eval.py" or identity != LEGACY_TRAINER_SHA256:
            raise ValueError(f"original model/evaluator code changed: {name}")
        original = subprocess.check_output(["git", "show", f"{LEGACY_REVISION}:{TRAINER_PATH}"])
        if (hashlib.sha256(original).hexdigest() != identity or
                without_prediction_body(original) != without_prediction_body(path.read_bytes())):
            raise ValueError("original training/evaluation code changed beyond prediction batching")
        changes[name] = {"original_sha256": identity, "current_sha256": actual,
                         "change": "bounded prediction source lists; other executable code unchanged"}
    return changes
