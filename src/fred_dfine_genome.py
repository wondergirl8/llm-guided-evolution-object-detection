"""Prototype, versioned D-FINE motif genome for bounded architecture search.

Candidates are rendered from the pinned seed. The LLM chooses one value for
one gene; it never writes executable Python in this search space.
"""

import fcntl
import hashlib
import json
from pathlib import Path
import random
import tempfile


VERSION = "dfine_motifs_v1"
SEED_SOURCE_SHA256 = "e22243c708fd01f9c8875c08c3d1aefc96e507f9c72d0c5dd2bd0763fe5eadd9"
GENES = ("backbone", "encoder", "decoder")
CHOICES = {
    "backbone": ("none", *(f"p4_dw_k{k}_s{s}" for k in (3, 5)
                            for s in ("025", "050", "100"))),
    "encoder": ("none", *(f"bidir_td{td}_bu{bu}" for td in ("050", "100")
                           for bu in ("025", "050", "100"))),
    "decoder": ("none", "p4_quota_25", "p4_quota_50", "p4_quota_75"),
}
INITIAL_COVERAGE_TARGETS = GENES
_SCALE = {"025": "0.25", "050": "0.5", "100": "1.0"}


def initial_target_for_slot(slot):
    if type(slot) is not int or slot < 0:
        raise ValueError("initial candidate slot must be nonnegative")
    return GENES[slot] if slot < len(GENES) else None


def seed_genome():
    return {"version": VERSION, "genes": {gene: "none" for gene in GENES}}


def validate_genome(genome):
    if (not isinstance(genome, dict) or set(genome) != {"version", "genes"}
            or genome["version"] != VERSION or not isinstance(genome["genes"], dict)
            or set(genome["genes"]) != set(GENES)):
        raise ValueError(f"expected complete {VERSION} genome")
    for gene in GENES:
        if genome["genes"][gene] not in CHOICES[gene]:
            raise ValueError(f"invalid {gene} value: {genome['genes'][gene]}")
    return genome


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("D-FINE seed changed; review the motif renderer before use")
    return source.replace(old, new, 1)


def render_source(seed_path, genome):
    """Build the exact Python candidate from a validated three-gene genome."""
    validate_genome(genome)
    source = Path(seed_path).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != SEED_SOURCE_SHA256:
        raise ValueError("D-FINE seed source changed; version and review the motif genome")
    backbone = genome["genes"]["backbone"]
    if backbone != "none":
        _, _, kernel, scale = backbone.split("_")
        kernel = int(kernel[1:])
        source = _replace_once(
            source, "        self.out_channels = (width, width * 2, width * 4)",
            "        self.p4_refine = nn.Sequential(\n"
            f"            nn.Conv2d(width, width, {kernel}, padding={kernel // 2}, "
            "groups=width, bias=False),\n"
            "            nn.BatchNorm2d(width),\n"
            "            nn.GELU(),\n"
            "        )\n"
            "        self.out_channels = (width, width * 2, width * 4)",
        )
        source = _replace_once(source, "        p4 = self.stem(event)\n",
                               "        p4 = self.stem(event)\n"
                               f"        p4 = p4 + {_SCALE[scale[1:]]} * self.p4_refine(p4)\n")
    encoder = genome["genes"]["encoder"]
    if encoder != "none":
        _, td, bu = encoder.split("_")
        top_down, bottom_up = _SCALE[td[2:]], _SCALE[bu[2:]]
        source = _replace_once(
            source,
            '        p8 = self.refine[1](p8 + F.interpolate(p16, size=p8.shape[-2:], mode="nearest"))',
            f'        p8 = self.refine[1](p8 + {top_down} * F.interpolate(p16, size=p8.shape[-2:], mode="nearest"))',
        )
        source = _replace_once(
            source,
            '        p4 = self.refine[0](p4 + F.interpolate(p8, size=p4.shape[-2:], mode="nearest"))',
            f'        p4 = self.refine[0](p4 + {top_down} * F.interpolate(p8, size=p4.shape[-2:], mode="nearest"))\n'
            f'        p8 = p8 + {bottom_up} * F.adaptive_avg_pool2d(p4, p8.shape[-2:])\n'
            f'        p16 = p16 + {bottom_up} * F.adaptive_avg_pool2d(p8, p16.shape[-2:])',
        )
    decoder = genome["genes"]["decoder"]
    if decoder != "none":
        percentage = int(decoder.rsplit("_", 1)[1])
        old = ("        memory_indices = objectness.topk(min(token_count, self.max_memory_tokens), dim=1).indices\n"
               "        query_indices = memory_indices[:, :self.num_queries]")
        new = (
            "        p4_tokens = features[0].shape[-2] * features[0].shape[-1]\n"
            f"        p4_queries = min(p4_tokens, max(1, self.num_queries * {percentage} // 100))\n"
            "        p4_indices = objectness[:, :p4_tokens].topk(p4_queries, dim=1).indices\n"
            "        reserved = torch.zeros_like(objectness, dtype=torch.bool)\n"
            "        reserved.scatter_(1, p4_indices, True)\n"
            "        remaining = objectness.masked_fill(reserved, float('-inf')).topk(\n"
            "            min(token_count, self.max_memory_tokens) - p4_queries, dim=1).indices\n"
            "        memory_indices = torch.cat((p4_indices, remaining), dim=1)\n"
            "        query_indices = memory_indices[:, :self.num_queries]"
        )
        source = _replace_once(source, old, new)
    return source


def genome_path(source_path):
    return Path(str(source_path) + ".genome.json")


def load_genome(source_path, seed_path):
    """Reject legacy or edited candidates rather than mixing genome versions."""
    source_path, seed_path = Path(source_path), Path(seed_path)
    sidecar = genome_path(source_path)
    if not sidecar.is_file():
        if source_path.read_bytes() == seed_path.read_bytes():
            return seed_genome()
        raise ValueError(f"D-FINE candidate has no {VERSION} genome: {sidecar}")
    genome = validate_genome(json.loads(sidecar.read_text()))
    if source_path.read_text() != render_source(seed_path, genome):
        raise ValueError(f"D-FINE source and {VERSION} genome disagree: {source_path}")
    return genome


def _proposal_value(reply, target):
    if not isinstance(reply, str):
        raise ValueError("LLM returned no JSON proposal")
    raw = reply.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    proposal = json.loads(raw)
    if not isinstance(proposal, dict) or set(proposal) != {"value"}:
        raise ValueError('return only JSON shaped as {"value":"..."}')
    value = proposal["value"]
    if value not in CHOICES[target]:
        raise ValueError(f"invalid {target} value: {value}")
    return value


def _prompt(template, genome, target, feedback=""):
    context = json.dumps({"version": VERSION, "current_genes": genome["genes"],
                          "target_gene": target, "allowed_values": CHOICES[target]},
                         sort_keys=True)
    if "{}" not in template:
        raise ValueError("structured D-FINE prompt needs one {} context placeholder")
    return template.replace("{}", context, 1) + feedback


def _duplicate_path(seed_path, parent_paths, output, genome):
    prior_paths = dict.fromkeys((seed_path, *parent_paths,
                                 *sorted(output.parent.glob("network_*.py"))))
    for prior in prior_paths:
        if (prior is not None and Path(prior).is_file()
                and load_genome(prior, seed_path) == genome):
            return Path(prior)
    return None


def _publish(seed_path, parent_paths, output, genome):
    """Validate canonical source and refuse an AST duplicate before publishing."""
    from src.fred_dfine_proposals import architecture_hash
    from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import DFINEAdapter
    import yaml

    output, seed_path = Path(output), Path(seed_path)
    fred_dir = seed_path.parent.parent
    protocol = yaml.safe_load((fred_dir / "configs/phase1/protocol.yaml").read_text())
    adapter = DFINEAdapter(protocol["families"]["dfine"], fred_dir)
    source = render_source(seed_path, genome)
    output.parent.mkdir(parents=True, exist_ok=True)
    duplicate = _duplicate_path(seed_path, parent_paths, output, genome)
    if duplicate is not None:
        raise ValueError(f"Architecture duplicates {duplicate.name}")
    with tempfile.TemporaryDirectory(prefix="dfine-motif-", dir=output.parent) as temp:
        candidate = Path(temp) / "candidate.py"
        candidate.write_text(source)
        adapter.preflight(candidate, protocol["proxy"])
        digest = architecture_hash(candidate)
        with (output.parent / ".dfine-proposals.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.exists() or genome_path(output).exists():
                raise FileExistsError(f"Refusing to replace D-FINE candidate: {output}")
            duplicate = _duplicate_path(seed_path, parent_paths, output, genome)
            if duplicate is not None:
                raise ValueError(f"Architecture duplicates {duplicate.name} ({digest})")
            sidecar = genome_path(output)
            sidecar.write_text(json.dumps(genome, indent=2) + "\n")
            try:
                candidate.replace(output)
            except Exception:
                sidecar.unlink(missing_ok=True)
                raise
    return digest


def generate_mutation(parent, output, seed_path, template, generate, coverage_target=None,
                      attempts=5, target=None):
    """Ask for one bounded value; build, preflight, and audit the result."""
    from src.fred_dfine_proposals import architecture_hash

    parent, output = Path(parent), Path(output)
    if attempts < 1 or output.exists():
        raise ValueError("invalid attempt count or existing D-FINE output")
    genome = load_genome(parent, seed_path)
    if coverage_target is not None and coverage_target not in GENES:
        raise ValueError(f"invalid D-FINE coverage target: {coverage_target}")
    target = coverage_target or target or random.choice(GENES)
    if target not in GENES:
        raise ValueError(f"invalid D-FINE target: {target}")
    audit_path = Path(str(output) + ".proposal_attempts.json")
    audit = {"search_space_version": VERSION, "parent": str(parent),
             "parent_architecture_sha256": architecture_hash(parent),
             "coverage_target": coverage_target, "target_gene": target, "attempts": []}
    feedback = ""
    for attempt in range(1, attempts + 1):
        prompt = _prompt(template, genome, target, feedback)
        reply = generate(prompt)
        entry = {"attempt": attempt, "prompt": prompt, "proposal": reply, "accepted": False}
        try:
            value = _proposal_value(reply, target)
            if value == genome["genes"][target] or (coverage_target and value == "none"):
                raise ValueError("proposal does not change the targeted mechanism")
            child = {"version": VERSION, "genes": dict(genome["genes"], **{target: value})}
            digest = _publish(seed_path, (parent,), output, child)
            entry.update(accepted=True, architecture_sha256=digest, genome=child)
        except ImportError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            audit["attempts"].append(entry)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            raise RuntimeError(f"D-FINE preflight unavailable: {exc}") from exc
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            feedback = (f"\nPrevious value was rejected: {entry['error']}. "
                        "Choose another allowed value for the same gene. Return JSON only.\n")
        audit["attempts"].append(entry)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        if entry["accepted"]:
            return
    raise RuntimeError(f"No valid D-FINE {VERSION} proposal after {attempts} attempts; see {audit_path}")


def generate_crossover(first, second, output, seed_path, rng=None):
    """Recombine whole semantic genes; incompatible sources fail visibly."""
    from src.fred_dfine_proposals import architecture_hash

    first_genome = load_genome(first, seed_path)
    second_genome = load_genome(second, seed_path)
    differing = [gene for gene in GENES
                 if first_genome["genes"][gene] != second_genome["genes"][gene]]
    if len(differing) < 2:
        raise ValueError("D-FINE parents have no novel one-gene recombinant")
    rng = rng or random
    rng.shuffle(differing)
    for target in differing:
        child = {"version": VERSION, "genes": dict(first_genome["genes"])}
        child["genes"][target] = second_genome["genes"][target]
        if child == seed_genome():
            continue
        try:
            digest = _publish(seed_path, (first, second), output, child)
            break
        except ValueError as exc:
            if "Architecture duplicates" not in str(exc):
                raise
    else:
        raise ValueError("D-FINE parents have no novel one-gene recombinant")
    audit = {"search_space_version": VERSION, "operator": "semantic_crossover",
             "parents": [str(first), str(second)], "target_gene": target,
             "genome": child, "architecture_sha256": digest,
             "parent_architecture_sha256": [architecture_hash(first), architecture_hash(second)]}
    Path(str(output) + ".proposal_attempts.json").write_text(json.dumps(audit, indent=2) + "\n")


def verify_initial_coverage(entries, seed_path):
    """Require one distinct, audited candidate for each initial mechanism."""
    from src.fred_dfine_proposals import architecture_hash

    seen, digests, candidates = set(), set(), []
    for target, path in entries:
        path = Path(path)
        if target not in GENES or not path.is_file():
            raise ValueError(f"missing D-FINE initial candidate: {target} {path}")
        genome = load_genome(path, seed_path)
        audit = json.loads(Path(str(path) + ".proposal_attempts.json").read_text())
        accepted = [item for item in audit.get("attempts", ()) if item.get("accepted")]
        digest = architecture_hash(path)
        if (audit.get("search_space_version") != VERSION
                or audit.get("coverage_target") != target
                or audit.get("target_gene") != target or len(accepted) != 1
                or accepted[0].get("architecture_sha256") != digest
                or genome["genes"][target] == "none" or digest in digests
                or any(genome["genes"][gene] != "none" for gene in GENES if gene != target)):
            raise ValueError(f"invalid D-FINE initial coverage candidate: {target} {path}")
        seen.add(target)
        digests.add(digest)
        candidates.append({"target": target, "model": str(path), "genome": genome,
                           "architecture_sha256": digest})
    if seen != set(GENES) or len(entries) != len(GENES):
        raise ValueError(f"incomplete D-FINE initial coverage: {sorted(set(GENES) - seen)}")
    return {"search_space_version": VERSION, "candidates": candidates}
