"""Deterministic validation of LLM-written YOLO11 class responses."""

import ast
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap


def extract_generated_class(response, expected_name):
    """Accept one complete Python class, fenced or plain; reject empty/prose output."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("LLM returned no code")
    code = response.strip()
    if "```" in code:
        blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", code,
                            flags=re.DOTALL | re.IGNORECASE)
        if code.count("```") != 2 or len(blocks) != 1:
            raise ValueError("Expected one complete Python code block")
        code = blocks[0]
    code = textwrap.dedent(code).strip()
    if not code or "# --OPTION--" in code:
        raise ValueError("Generated class is empty or contains option markers")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"Generated class has invalid Python syntax: {error.msg}") from error
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef) or
            tree.body[0].name != expected_name):
        raise ValueError(f"Expected only class {expected_name}")
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError("Generated classes cannot add imports")
    methods = {node.name: node for node in tree.body[0].body if isinstance(node, ast.FunctionDef)}
    for name in ("__init__", "forward"):
        if name not in methods:
            raise ValueError(f"Generated class must define {name}")
    constructor = methods["__init__"].args
    if ([arg.arg for arg in constructor.posonlyargs + constructor.args] != ["self", "layers"] or
            constructor.vararg or constructor.kwarg or constructor.kwonlyargs):
        raise ValueError("Generated class must preserve __init__(self, layers)")
    return code


def make_candidate_validator(parent_parts, section_index, *, require_change=True):
    """Validate the complete proposed model in a bounded, isolated CPU process."""
    reference = ast.dump(ast.parse(parent_parts[section_index]))
    repo = Path(__file__).resolve().parents[1]

    def validate(code):
        if require_change and ast.dump(ast.parse(code)) == reference:
            raise ValueError("LLM returned the unchanged parent class")
        parts = list(parent_parts)
        parts[section_index] = "\n" + code + "\n"
        with tempfile.TemporaryDirectory(prefix="fred-generated-preflight-") as temporary:
            directory = Path(temporary)
            (directory / "network_check.py").write_text("# --OPTION--".join(parts))
            command = [sys.executable, str(repo / "src/candidate_preflight.py"), str(directory),
                       "--scale", os.getenv("FRED_YOLO_SCALE", "n")]
            environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            timeout = int(os.getenv("LLMGE_GENERATION_PREFLIGHT_TIMEOUT_SECONDS", "60"))
            try:
                result = subprocess.run(command, cwd=repo, env=environment,
                                        capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise ValueError(f"CPU model preflight exceeded {timeout} seconds") from error
            if result.returncode:
                details = result.stderr.strip() or result.stdout.strip()
                # Retry feedback is printed in job logs; retain useful frames/errors
                # without emitting a standalone fatal Python traceback header.
                lines = [line for line in details.splitlines()
                         if not line.lstrip().startswith('Traceback (most recent call last):')]
                raise ValueError("CPU model preflight failed: " + "\n".join(lines[-12:]))
            import json
            return json.loads(result.stdout.splitlines()[-1])
    return validate
