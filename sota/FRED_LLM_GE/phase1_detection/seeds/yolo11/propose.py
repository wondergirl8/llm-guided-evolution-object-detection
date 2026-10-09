"""Propose one unscored YOLO11 genome with a local GPU language model.

This is an engineering candidate-generation check, not an evolution run. The
generated text may only replace the seed's literal GENOME assignment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .validator import SEED_PATH, validate_source


ROOT = Path(__file__).resolve().parents[5]
PROMPT_PATH = ROOT / "templates" / "FRED" / "Event" / "Normal" / "1.txt"
RULES_PATH = ROOT / "templates" / "FRED" / "Event" / "ConstantRules.txt"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
FENCE = re.compile(r"\A```(?:python)?\s*\n(.*?)\n```\s*\Z", re.DOTALL)


def parse_genome(response: str, original: dict[str, int]) -> dict[str, int]:
    """Accept exactly one literal assignment changing one gene by one step."""
    candidate = response.strip()
    fence = FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1)
    tree = ast.parse(candidate)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("response must contain only one GENOME assignment")
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise ValueError("response must assign GENOME once")
    if assignment.targets[0].id != "GENOME":
        raise ValueError("response must assign GENOME")
    if not isinstance(assignment.value, ast.Dict):
        raise ValueError("GENOME must be a literal dictionary")
    keys = [ast.literal_eval(key) for key in assignment.value.keys]
    if any(type(key) is not str for key in keys):
        raise ValueError("GENOME keys must be strings")
    if len(keys) != len(set(keys)):
        raise ValueError("GENOME has duplicate keys")
    genome = ast.literal_eval(assignment.value)
    if not isinstance(genome, dict) or set(genome) != set(original):
        raise ValueError("genome keys differ from the seed")
    if any(type(value) is not int or value not in (2, 3, 4)
           for value in genome.values()):
        raise ValueError("genome values must be integers 2, 3, or 4")
    changed = [key for key in original if genome[key] != original[key]]
    if len(changed) != 1 or abs(genome[changed[0]] - original[changed[0]]) != 1:
        raise ValueError("genome must change exactly one value by one step")
    return {key: genome[key] for key in original}


def render_candidate(seed: str, genome: dict[str, int]) -> str:
    if seed.count("# --OPTION--") != 1:
        raise ValueError("seed mutation boundary is missing or ambiguous")
    prefix = seed.split("# --OPTION--", 1)[0]
    genes = "\n".join(f'    "{key}": {value},' for key, value in genome.items())
    return prefix + "# --OPTION--\nGENOME = {\n" + genes + "\n}\n"


def build_prompt(original: dict[str, int]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    if template.count("{}") != 1:
        raise ValueError("FRED prompt must contain exactly one mutation slot")
    current = "GENOME = " + repr(original)
    return RULES_PATH.read_text(encoding="utf-8") + "\n\n" + template.format(current)


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def local_responses(model_id: str, revision: str, prompt: str, attempts: int):
    """Load a pinned public model on the assigned GPU and yield bounded replies."""
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("candidate generation requires an allocated CUDA GPU")
    pinned_revision = model_info(model_id, revision=revision).sha
    if not pinned_revision:
        raise RuntimeError("could not resolve the language model revision")
    print(f"LLM model: {model_id}@{pinned_revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=pinned_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=pinned_revision, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=False,
    ).eval()
    messages = [
        {"role": "system", "content": "Return only the requested Python assignment."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    for attempt in range(attempts):
        seed = 20260930 + attempt
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                temperature=0.25, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        yield pinned_revision, seed, reply


def propose(candidate_id: str, output_dir: Path, model_id: str,
            revision: str = "main", attempts: int = 3, responses=None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9]+", candidate_id):
        raise ValueError("candidate ID must be alphanumeric")
    if attempts < 1 or attempts > 5:
        raise ValueError("attempts must be between 1 and 5")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse candidate directory: {output_dir}")
    seed = SEED_PATH.read_text(encoding="utf-8")
    original = validate_source(SEED_PATH)
    prompt = build_prompt(original)
    attempts_record = []
    if responses is None:
        responses = local_responses(model_id, revision, prompt, attempts)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fred-proposal-", dir=output_dir.parent) as temp_name:
        temp = Path(temp_name)
        candidate = temp / f"network_{candidate_id}.py"
        accepted = None
        for pinned_revision, generation_seed, reply in responses:
            record = {"model_revision": pinned_revision,
                      "generation_seed": generation_seed, "response": reply}
            attempts_record.append(record)
            try:
                genome = parse_genome(reply, original)
                candidate.write_text(render_candidate(seed, genome), encoding="utf-8")
                validate_source(candidate)
            except (SyntaxError, ValueError) as error:
                candidate.unlink(missing_ok=True)
                record["error"] = f"{type(error).__name__}: {error}"
                continue
            accepted = {"model_revision": pinned_revision, "genome": genome}
            break
        metadata = {
            "kind": "fred_yolo11_unscored_candidate_proposal_v1",
            "status": "candidate_generated_unscored" if accepted else "invalid_llm_responses",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate_id,
            "model_id": model_id,
            "requested_revision": revision,
            "generation_parameters": {
                "max_new_tokens": 256, "temperature": 0.25, "top_p": 0.9,
                "attempt_limit": attempts,
            },
            "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "runtime": runtime_versions(),
            "attempts": attempts_record,
            **(accepted or {}),
        }
        (temp / "prompt.txt").write_text(prompt, encoding="utf-8")
        (temp / "proposal.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        temp.rename(output_dir)
    if not accepted:
        raise RuntimeError(f"LLM produced no valid candidate; see {output_dir / 'proposal.json'}")
    print(f"Candidate source: {output_dir / candidate.name}", flush=True)
    print(f"Proposal record: {output_dir / 'proposal.json'}", flush=True)
    return output_dir / candidate.name


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args(argv)
    propose(args.candidate_id, args.output_dir, args.model_id,
            args.model_revision, args.attempts)


if __name__ == "__main__":
    main()
