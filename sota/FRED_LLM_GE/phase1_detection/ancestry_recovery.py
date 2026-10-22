"""Recover generation-zero FRED refill lineage from saved operator inputs."""

import copy
import glob
import shlex
from pathlib import Path


def recover_population_ancestry(population, ancestry, root, operator_dir,
                                variant_dir, model, prompt_glob, rules_path):
    """Return complete lineage, or fail before changing checkpointed state.

    Older generation-zero checkpoints omitted lineage for refill mutations.
    Their saved mutation commands identify each parent, and their saved prompt
    files identify the exact normal mutation template used.
    """
    missing = [individual[0] for individual in population if individual[0] not in ancestry]
    if not missing:
        return ancestry, []

    root = Path(root)
    operator_dir = Path(operator_dir) / "0"
    variant_dir = Path(variant_dir)
    rules_path = Path(rules_path)
    if not rules_path.is_absolute():
        rules_path = root / rules_path
    rules = rules_path.read_text()
    templates = sorted(Path(path) for path in glob.glob(str(root / prompt_glob), recursive=True))
    recovered = copy.deepcopy(ancestry)
    repaired = []
    visiting = set()

    def recover(gene):
        if gene in recovered:
            return
        if gene in visiting:
            raise RuntimeError(f"Ancestry cycle while recovering {gene}")
        visiting.add(gene)
        script = operator_dir / f"{gene}.sh"
        if not script.is_file():
            raise RuntimeError(f"Missing ancestry and saved mutation script for {gene}: {script}")

        commands = []
        for line in script.read_text().splitlines():
            if line.startswith("uv run python src/llm_mutation.py "):
                commands.append(shlex.split(line))
        if len(commands) != 1 or len(commands[0]) < 7:
            raise RuntimeError(f"Cannot verify one mutation command for {gene}: {script}")
        command = commands[0]
        parent_path, child_path, prompt_path = map(Path, command[4:7])
        expected_child = variant_dir / f"{model}_{gene}.py"
        expected_prompt = operator_dir / f"{gene}_model.txt"
        if (child_path.resolve() != expected_child.resolve() or
                prompt_path.resolve() != expected_prompt.resolve() or
                parent_path.parent.resolve() != variant_dir.resolve() or
                not parent_path.name.startswith(f"{model}_") or
                parent_path.suffix != ".py"):
            raise RuntimeError(f"Saved mutation inputs do not match candidate {gene}: {script}")

        parent = parent_path.stem[len(model) + 1:]
        if not parent:
            raise RuntimeError(f"Saved mutation has no parent ID for {gene}: {script}")
        recover(parent)
        saved_prompt = prompt_path.read_text()
        matching_types = [template.name.split(".")[0] for template in templates
                          if template.read_text() + "\n" + rules == saved_prompt]
        if len(matching_types) != 1:
            raise RuntimeError(f"Cannot identify one saved mutation template for {gene}: "
                               f"{expected_prompt}")

        parent_lineage = recovered[parent]
        recovered[gene] = copy.deepcopy(parent_lineage)
        recovered[gene]["GENES"] = [*parent_lineage["GENES"], gene]
        recovered[gene]["MUTATE_TYPE"] = [*parent_lineage["MUTATE_TYPE"], matching_types[0]]
        visiting.remove(gene)
        repaired.append(gene)

    for gene in missing:
        recover(gene)
    return recovered, repaired
