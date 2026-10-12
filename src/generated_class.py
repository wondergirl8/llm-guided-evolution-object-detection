"""Deterministic validation of LLM-written YOLO11 class responses."""

import ast
import re
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
