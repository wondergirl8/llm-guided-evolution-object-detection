"""Validate and retry D-FINE architecture proposals before GPU evaluation."""

import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


def architecture_hash(path):
    """Identify editable class ASTs, ignoring formatting, comments, and prose."""
    from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import _source_parts

    parts = _source_parts(path)
    classes = ast.parse("\n".join(parts[1:]))
    for node in ast.walk(classes):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                    node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)  # prose changes are not architecture changes
    return hashlib.sha256(ast.dump(classes, include_attributes=False).encode()).hexdigest()


def _validate_section(parent, proposal):
    if not proposal or proposal == "ERROR":
        raise ValueError("LLM returned no valid code")
    try:
        original = ast.parse(parent).body
        proposed = ast.parse(proposal).body
    except SyntaxError as exc:
        raise ValueError(f"invalid Python: {exc.msg}") from exc
    if (len(original) != 1 or len(proposed) != 1
            or not isinstance(original[0], ast.ClassDef)
            or not isinstance(proposed[0], ast.ClassDef)
            or proposed[0].name != original[0].name):
        raise ValueError("proposal must contain only the selected architecture class")
    for node in ast.walk(proposed[0]):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("architecture classes may not add imports")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"open", "eval", "exec", "compile", "__import__", "input"}):
            raise ValueError(f"architecture classes may not call {node.func.id}")
    if ast.dump(original[0], include_attributes=False) == ast.dump(
            proposed[0], include_attributes=False):
        raise ValueError("proposal leaves the architecture class unchanged")


def generate_validated_section(parent, output, parts, index, prompt, generate,
                               fred_dir, attempts=5, other_parent=None):
    """Publish one novel, executable candidate; retain every rejected attempt."""
    from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import DFINEAdapter

    parent, output, fred_dir = Path(parent), Path(output), Path(fred_dir)
    if attempts < 1 or index not in (1, 2, 3) or len(parts) != 4:
        raise ValueError("D-FINE needs three editable sections and positive attempts")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing candidate: {output}")
    protocol = yaml.safe_load((fred_dir / "configs/phase1/protocol.yaml").read_text())
    adapter = DFINEAdapter(protocol["families"]["dfine"], fred_dir)
    seed = adapter.variant_path("network", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_suffix(output.suffix + ".proposal_attempts.json")
    audit = {"parent": str(parent), "other_parent": str(other_parent) if other_parent else None,
             "section": index, "attempts": []}
    feedback = ""
    for attempt in range(1, attempts + 1):
        full_prompt = prompt + feedback
        reply = generate(full_prompt)
        entry = {"attempt": attempt, "prompt": full_prompt, "proposal": reply, "accepted": False}
        fatal = None
        try:
            _validate_section(parts[index], reply)
            candidate_parts = list(parts)
            candidate_parts[index] = "\n" + reply.strip() + "\n"
            with tempfile.TemporaryDirectory(prefix="dfine-proposal-", dir=output.parent) as temp:
                candidate = Path(temp) / "candidate.py"
                candidate.write_text("# --OPTION--".join(candidate_parts))
                adapter.preflight(candidate, protocol["proxy"])
                digest = architecture_hash(candidate)
                with (output.parent / ".dfine-proposals.lock").open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if output.exists():
                        raise FileExistsError(f"Refusing to replace an existing candidate: {output}")
                    prior_paths = dict.fromkeys((seed, parent, other_parent,
                                                  *sorted(output.parent.glob("network_*.py"))))
                    for prior in prior_paths:
                        if prior is None or Path(prior) == output or not Path(prior).is_file():
                            continue
                        if architecture_hash(prior) == digest:
                            raise ValueError(f"Architecture duplicates {Path(prior).name} ({digest})")
                    candidate.replace(output)
            entry.update(accepted=True, architecture_sha256=digest)
        except ImportError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            fatal = exc
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"D-FINE proposal attempt {attempt}/{attempts} rejected: {entry['error']}", flush=True)
            feedback = (f"\n\nPrevious proposal was rejected: {entry['error']}\n"
                        f"Previous response:\n```python\n{str(reply)[:1500]}\n```\n"
                        "Return one different, complete architecture class from the ORIGINAL parent. "
                        "Keep its class name and constructor interface; no imports or explanation.")
        audit["attempts"].append(entry)
        temp_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temp_audit.write_text(json.dumps(audit, indent=2) + "\n")
        os.replace(temp_audit, audit_path)
        if fatal is not None:
            raise RuntimeError(f"D-FINE proposal preflight cannot run: {fatal}; see {audit_path}") from fatal
        if entry["accepted"]:
            print(f"D-FINE proposal accepted on attempt {attempt}: {entry['architecture_sha256']}",
                  flush=True)
            return
    raise RuntimeError(f"No valid D-FINE proposal after {attempts} attempts; see {audit_path}")
