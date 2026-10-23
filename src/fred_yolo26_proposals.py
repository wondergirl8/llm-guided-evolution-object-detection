"""Turn one LLM-proposed YOLO26 layer edit into a validated, novel model YAML.

The worker applies only the requested value; it never rounds, invents, or
silently falls back to the parent architecture.
"""
import copy
import fcntl
import json
from pathlib import Path
import tempfile

import yaml


# The first four candidates deliberately probe narrower detection paths that
# the archived search never evaluated. Later mutation/crossover stays open.
INITIAL_COVERAGE_TARGETS = {
    "p2_down": 19,
    "p3_down": 22,
    "p4_down": 25,
    "p5_down": 28,
}


def initial_target_for_slot(slot):
    if type(slot) is not int or slot < 0:
        raise ValueError("initial candidate slot must be nonnegative")
    # Each required reduction occupies one slot. Repeating P2 cannot make a
    # distinct model: its only allowed reduction is 128 -> 64 channels.
    targets = tuple(INITIAL_COVERAGE_TARGETS)
    return targets[slot] if slot < len(targets) else None


def _model_parameters(variant):
    """Count the constructed model, including width/depth scaling."""
    from ultralytics import YOLO

    config = yaml.safe_load(Path(variant).read_text())
    with tempfile.TemporaryDirectory(prefix="yolo26_proposal_params_") as temp:
        path = Path(temp) / "yolo26n-p2.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        model = YOLO(str(path), task="detect", verbose=False)
        return sum(parameter.numel() for parameter in model.model.parameters())


def verify_initial_coverage(entries):
    """Refuse GPU evaluation unless the new seed panel covers every target."""
    expected = set(INITIAL_COVERAGE_TARGETS)
    seen, hashes, candidates = set(), set(), []
    for target, model_path in entries:
        model_path = Path(model_path)
        if target not in expected or not model_path.is_file():
            raise ValueError(f"missing initial YOLO26 candidate for {target}: {model_path}")
        audit_path = model_path.with_suffix(model_path.suffix + ".proposal_attempts.json")
        if not audit_path.is_file():
            raise ValueError(f"missing YOLO26 proposal audit: {audit_path}")
        audit = json.loads(audit_path.read_text())
        accepted = [attempt for attempt in audit["attempts"] if attempt.get("accepted")]
        if len(accepted) != 1 or audit.get("coverage_target") != target:
            raise ValueError(f"initial YOLO26 target was not fulfilled: {target}")
        result = accepted[0]
        if (result["edit"].get("layer") != INITIAL_COVERAGE_TARGETS[target]
                or result["edit"].get("field") != "channels"
                or result.get("parameter_delta", 0) >= 0):
            raise ValueError(f"initial YOLO26 candidate did not reduce {target}")
        from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import (
            UltralyticsYolo26P2Adapter,
        )
        digest = UltralyticsYolo26P2Adapter.architecture_hash(model_path)
        if result.get("architecture_sha256") != digest or digest in hashes:
            raise ValueError(f"initial YOLO26 architecture duplicate or changed: {model_path}")
        seen.add(target)
        hashes.add(digest)
        candidates.append({"target": target, "model": str(model_path),
                           "architecture_sha256": digest, "edit": result["edit"],
                           "parameters_before": result["parameters_before"],
                           "parameters_after": result["parameters_after"],
                           "parameter_delta": result["parameter_delta"]})
    if seen != expected:
        raise ValueError(f"incomplete initial YOLO26 coverage: missing {sorted(expected - seen)}")
    return {"coverage_version": "initial_p2_p5_reductions_v1",
            "candidates": candidates}


def _effective_repeats(repeats, depth):
    return max(round(repeats * depth), 1) if repeats > 1 else repeats


def _effective_config(path):
    """Normalize C3k2 depth scaling used by Ultralytics' model builder."""
    config = yaml.safe_load(Path(path).read_text())
    depth = config["scales"][config["scale"]][0]
    for section in ("backbone", "head"):
        for layer in config[section]:
            if layer[2] == "C3k2":
                layer[1] = _effective_repeats(layer[1], depth)
    return config


def _edit_section(section, layers, proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"layer", "field", "value"}:
        raise ValueError("Return only layer, field, and value in one YAML mapping")
    layer, field, value = proposal["layer"], proposal["field"], proposal["value"]
    offset = 0 if section == "backbone" else 11
    if type(layer) is not int or not offset <= layer < offset + len(layers):
        raise ValueError(f"layer must be an absolute {section} index")
    if type(value) is not int:
        raise ValueError("value must be an integer")
    updated = copy.deepcopy(layers)
    selected = updated[layer - offset]
    module = selected[2]
    if field == "channels" and module in ("Conv", "C3k2"):
        target, position = selected[3], 0
    elif field == "kernel" and module == "Conv":
        target, position = selected[3], 1
    elif field == "repeats" and module == "C3k2":
        target, position = selected, 1
    else:
        raise ValueError(f"{field!r} cannot be edited on {module} layer {layer}")
    if target[position] == value:
        raise ValueError(f"layer {layer} {field} is unchanged; choose a different value")
    target[position] = value
    return updated


def _existing_architecture(digest, effective, adapter, seed_path, parent, output):
    for prior in dict.fromkeys((seed_path, parent, *sorted(output.parent.glob("network_*.py")))):
        if prior == output or not prior.is_file():
            continue
        try:
            if (adapter.architecture_hash(prior) == digest
                    or _effective_config(prior) == effective):
                return prior
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            # An incomplete or rejected file has no usable architecture identity.
            continue
    return None


def generate_validated_section(parent, output, parts, index, prompt, generate,
                               fred_dir, attempts=3, coverage_target=None):
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter

    parent, output, fred_dir = Path(parent), Path(output), Path(fred_dir)
    if attempts < 1:
        raise ValueError("Proposal attempts must be positive")
    if index not in (1, 2) or len(parts) != 3:
        raise ValueError("YOLO26 requires its protected preamble and two mutable sections")
    if coverage_target is not None and coverage_target not in INITIAL_COVERAGE_TARGETS:
        raise ValueError(f"unknown YOLO26 coverage target: {coverage_target}")
    if coverage_target is not None and index != 2:
        raise ValueError("initial YOLO26 coverage targets require the head section")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing candidate: {output}")
    protocol = yaml.safe_load((fred_dir / "configs/phase1/protocol.yaml").read_text())
    adapter = UltralyticsYolo26P2Adapter(protocol["families"]["yolo26_p2"], fred_dir)
    seed_path = adapter.variant_path("network", None)
    seed = yaml.safe_load(seed_path.read_text())
    section = ("backbone", "head")[index - 1]
    current = yaml.safe_load(parts[index])[section]
    offset = 0 if section == "backbone" else 11
    depth = seed["scales"][seed["scale"]][0]
    choices = []
    for relative, (_, _, module, args) in enumerate(seed[section]):
        absolute = offset + relative
        if coverage_target is not None and absolute != INITIAL_COVERAGE_TARGETS[coverage_target]:
            continue
        if module not in ("Conv", "C3k2"):
            continue
        allowed = list(range(max(64, args[0] // 2), args[0] * 2 + 1, 64))
        current_layer = current[relative]
        if coverage_target is not None:
            allowed = [value for value in allowed if value < current_layer[3][0]]
            if not allowed:
                raise ValueError(f"{coverage_target} has no valid channel reduction")
        if coverage_target is not None:
            extra = "channels only for this initial candidate"
        elif module == "Conv":
            extra = f"kernel [1, 3, 5] (current {current_layer[3][1]})"
        else:
            current_effective = _effective_repeats(current_layer[1], depth)
            distinct = {}
            for value in range(1, 5):
                effective = _effective_repeats(value, depth)
                if effective != current_effective:
                    distinct.setdefault(effective, value)
            extra = f"repeats {list(distinct.values())} (current {current_layer[1]})"
        choices.append(f"layer {absolute} {module}: channels {allowed}; {extra}; "
                       f"current channels {current_layer[3][0]}")
    prompt += ("\n\nReturn one compact YAML edit, not a replacement section. "
               "Use this schema with actual allowed integers:\n"
               "```yaml\nlayer: ABSOLUTE_INDEX\nfield: channels\nvalue: ALLOWED_INTEGER\n```\n"
               "Use exactly the keys layer, field, value. Layer is the absolute index "
               "(backbone 0-10; head 11-29). Field is channels, kernel, or repeats. "
               "Choose a value different from the parent. Favor a smaller P2/P3 width "
               "or fewer repeats when useful; valid increases are also allowed. "
               "Do not return the full section or explanatory text.\n"
               "Allowed values for this section:\n" + "\n".join(choices))
    if coverage_target is not None:
        prompt += (f"\n\nRequired initial coverage target: {coverage_target}. "
                   f"Return layer: {INITIAL_COVERAGE_TARGETS[coverage_target]}, "
                   "field: channels, and one listed value below the parent. "
                   "A proposal outside this target will be rejected.")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {"parent": str(parent), "section": section,
             "coverage_target": coverage_target, "attempts": []}
    audit_path = output.with_suffix(output.suffix + ".proposal_attempts.json")
    feedback = ""
    for attempt in range(1, attempts + 1):
        reply = generate(prompt + feedback)
        entry = {"attempt": attempt, "proposal": reply, "accepted": False}
        try:
            if not reply or reply == "ERROR" or "# --OPTION--" in reply:
                raise ValueError("A single YAML layer edit is required")
            proposal = yaml.safe_load(reply)
            updated = _edit_section(section, current, proposal)
            if coverage_target is not None and (
                    proposal["layer"] != INITIAL_COVERAGE_TARGETS[coverage_target]
                    or proposal["field"] != "channels"
                    or proposal["value"] >= current[proposal["layer"] - offset][3][0]):
                raise ValueError(f"{coverage_target} requires a channel reduction on "
                                 f"layer {INITIAL_COVERAGE_TARGETS[coverage_target]}")
            merged = list(parts)
            merged[index] = "\n" + yaml.safe_dump({section: updated}, sort_keys=False).strip() + "\n"
            with tempfile.TemporaryDirectory(prefix="proposal-", dir=output.parent) as temp:
                candidate = Path(temp) / "candidate.py"
                candidate.write_text("# --OPTION--".join(merged))
                digest = adapter.architecture_hash(candidate)
                effective = _effective_config(candidate)
                # Serialize the novelty check with publication across LLM workers.
                with (output.parent / ".yolo26-proposals.lock").open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if output.exists():
                        raise FileExistsError(f"Refusing to replace an existing candidate: {output}")
                    prior = _existing_architecture(
                        digest, effective, adapter, seed_path, parent, output)
                    if prior is not None:
                        raise ValueError(f"Architecture duplicates {prior.name} ({digest})")
                    adapter.preflight(candidate, parent)
                    before, after = _model_parameters(parent), _model_parameters(candidate)
                    if coverage_target is not None and after >= before:
                        raise ValueError(f"{coverage_target} did not reduce model parameters")
                    candidate.replace(output)
            entry.update(accepted=True, edit=proposal, architecture_sha256=digest,
                         parameters_before=before, parameters_after=after,
                         parameter_delta=after - before)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            print(f"YOLO26 proposal attempt {attempt}/{attempts} rejected: {entry['error']}", flush=True)
            feedback = (f"\n\nYour previous edit was rejected: {entry['error']}\n"
                        f"Previous edit:\n```yaml\n{reply}\n```\n"
                        "Choose a different, valid layer/field/value relative to the ORIGINAL parent. "
                        "Do not repeat the rejected edit or return a full section.")
        audit["attempts"].append(entry)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        if entry["accepted"]:
            print(f"YOLO26 proposal accepted on attempt {attempt}: {digest}", flush=True)
            return
    raise RuntimeError(f"No valid YOLO26 proposal after {attempts} attempts; see {audit_path}")
