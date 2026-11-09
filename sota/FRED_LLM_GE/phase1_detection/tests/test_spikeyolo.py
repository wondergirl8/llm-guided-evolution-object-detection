"""Official-source, genome, adapter and standalone-baseline contracts."""
import importlib.util
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection import spikeyolo_search_space as policy
from sota.FRED_LLM_GE.phase1_detection import spikeyolo_baseline as baseline
from sota.FRED_LLM_GE.phase1_detection import spikeyolo_source as source

ROOT = policy.SEED.parents[3]


def write_genome(path, config):
    header = {key: value for key, value in config.items() if key not in policy.SECTIONS}
    source_text = yaml.safe_dump(header, sort_keys=False)
    for section in policy.SECTIONS:
        source_text += "\n# --OPTION--\n" + yaml.safe_dump({section: config[section]}, sort_keys=False)
    path.write_text(source_text)
    return path


def seed_config():
    return copy.deepcopy(policy.load_and_validate(policy.SEED))


def test_official_seed_config_and_source():
    upstream = source.verify_source()
    assert upstream["commit"] == "dc51f35d0b6e970cd157080c98157a41af4392f5"
    official = yaml.safe_load((source.SOURCE / upstream["model_config"]).read_text())
    assert policy.model_config(policy.SEED) == dict(official, nc=1, scale="s")
    assert seed_config()["genome_version"] == "spikeyolo-graph-v2"
    assert seed_config()["backbone"][0] == [-1, 1, "MS_GetT", [3, 1]]


@pytest.mark.parametrize("text", [
    "STAGE_REPEATS = (3, 6, 9, 1)",
    "!!python/object/apply:os.system ['false']",
    "nc: 1\nnc: 2",
    "# --OPTION--\nbackbone: []",
])
def test_reject_unsafe_or_legacy_genomes(tmp_path, text):
    path = tmp_path / "network_invalid.py"
    path.write_text(text)
    with pytest.raises(ValueError):
        policy.load_and_validate(path)


@pytest.mark.parametrize("section,index,field,value", [
    ("backbone", 0, 3, [3, 2]),                  # timestep
    ("backbone", 1, 0, 0),                      # routing
    ("backbone", 1, 1, 6),                      # downsampling repeats
    ("backbone", 1, 2, "Conv"),                 # ANN substitution
    ("backbone", 1, 3, [128, 7, 2, 2, True]),   # stride
    ("backbone", 1, 3, [128, 3, 4, 2, True]),   # padding for changed kernel
    ("backbone", 1, 3, [128, 7, 4, 2, False]),  # first-layer neuron
    ("backbone", 1, 3, [True, 7, 4, 2, True]),  # bool as channel count
    ("backbone", 1, 3, [96, 7, 4, 2, True]),    # channel granularity
    ("backbone", 1, 3, [320, 7, 4, 2, True]),   # seed-relative width bound
    ("backbone", 2, 1, True),                   # bool repeat
    ("backbone", 2, 1, 15),                     # depth bound
    ("backbone", 2, 2, "__import__('os')"),     # constructor string injection
    ("backbone", 2, 3, [5, 7]),                 # expansion bound
    ("backbone", 2, 3, [4, 4]),                 # even kernel
    ("backbone", 2, 3, [4, 7, True]),           # optional upstream flags
    ("head", 1, 3, [None, '(1,4,4)', 'nearest']), # upsample
    ("head", 3, 3, [1]),                        # Concat axis
    ("head", 10, 3, [256, 3, 1]),               # neck stride
    ("head", 16, 3, [2]),                       # class head
])
def test_reject_invalid_graph_edits(tmp_path, section, index, field, value):
    config = seed_config()
    config[section][index][field] = value
    path = write_genome(tmp_path / "network_invalid.py", config)
    with pytest.raises(ValueError):
        policy.load_and_validate(path)


@pytest.mark.parametrize("key,value", [("nc", True), ("scale", "m"),
                                         ("genome_version", "spikeyolo-depth-v1")])
def test_reject_protected_preamble(tmp_path, key, value):
    config = seed_config()
    config[key] = value
    with pytest.raises(ValueError, match="protected"):
        policy.load_and_validate(write_genome(tmp_path / "bad.py", config))


def test_reject_marker_section_and_layer_corruption(tmp_path):
    original = policy.SEED.read_text()
    invalid = [original + "\n# --OPTION--\nhead: []",
               original.replace("backbone:", "head:", 1),
               original + "\noptimizer: AdamW\n"]
    config = seed_config()
    config["head"].pop(0)
    invalid.append(write_genome(tmp_path / "removed.py", config).read_text())
    invalid.append(original.replace("nc: 1", "nc: 1\nnc: 1", 1))
    for text in invalid:
        path = tmp_path / "bad.py"
        path.write_text(text)
        with pytest.raises(ValueError):
            policy.load_and_validate(path)


def test_all_mutable_backbone_and_neck_positions_are_exposed(tmp_path):
    original = seed_config()
    changes = 0
    for section in policy.SECTIONS:
        for index, layer in enumerate(original[section]):
            if layer[2] in policy.FROZEN_MODULES:
                continue
            config = seed_config()
            changed = config[section][index]
            if layer[2] in policy.BLOCKS:
                changed[2] = "MS_AllConvBlock" if layer[2] == "MS_ConvBlock" else "MS_ConvBlock"
            else:
                changed[3][0] -= 64
            child = write_genome(tmp_path / "child.py", config)
            policy.load_and_validate(child)
            assert policy.changed_layer_count(child, policy.SEED) == 1
            changes += 1
    assert changes == 19


def test_mutation_counts_effective_layers_and_allows_coherent_fields(tmp_path):
    config = seed_config()
    config["head"][2] = [-1, 6, "MS_AllConvBlock", [2, 5]]
    child = write_genome(tmp_path / "child.py", config)
    assert policy.changed_layer_count(child, policy.SEED) == 1
    config["backbone"][2][1] = 1  # a stock repeat=3 alias
    assert policy.changed_layer_count(write_genome(child, config), policy.SEED) == 1
    config["backbone"][3][3][0] = 192
    with pytest.raises(ValueError, match="exactly one"):
        policy.differs_from_parents(write_genome(child, config), policy.SEED)


def test_source_tampering_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("original")
    import hashlib
    (tmp_path / "UPSTREAM.json").write_text(json.dumps({"files": {
        "module.py": hashlib.sha256(b"original").hexdigest()}}))
    monkeypatch.setattr(source, "SOURCE", tmp_path)
    (tmp_path / "module.py").write_text("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        source.verify_source()


def test_import_conflicts_are_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(__file__="/tmp/pypi/ultralytics/__init__.py"))
    with pytest.raises(RuntimeError, match="fresh SpikeYOLO process"):
        source.activate()


def test_registration_and_family_settings(monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.candidate_runner import ADAPTERS
    assert ADAPTERS["spikeyolo"].name == "spikeyolo"
    monkeypatch.setenv("FRED_SEED_FAMILY", "spikeyolo")
    monkeypatch.setenv("FRED_RUN_TAG", "spikeyolo_20261101_eot0_contract")
    monkeypatch.delenv("FRED_EVENTS", raising=False)
    spec = importlib.util.spec_from_file_location("spike_constants", ROOT / "src/cfg/constants_fred_yolo11.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    assert constants.SEED_NETWORK == str(policy.SEED)
    assert constants.CANDIDATE_POLICY_MODULE.endswith("spikeyolo_search_space")
    assert "--family spikeyolo" in constants.EVAL_RUNLINE
    assert "--extra spikeyolo" in constants.EVAL_RUNLINE


def test_official_build_loss_backward_and_candidate(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "sota.FRED_LLM_GE.phase1_detection.spikeyolo_smoke", "--backward"],
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=180)
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["backward"] and record["params"] == 23_122_451
    assert record["checked_shapes"] == [[1, 5, 84], [2, 5, 126]]
    config = seed_config()
    config["head"][2][3][1] = 5
    child = write_genome(tmp_path / "child.py", config)
    assert policy.differs_from_parents(child, policy.SEED)
    config = seed_config()
    config["backbone"][2][1] = 1
    alias = write_genome(tmp_path / "alias.py", config)
    assert not policy.differs_from_parents(alias, policy.SEED)


def _baseline_artifacts(root, seed=0):
    protocol = yaml.safe_load(baseline.PROTOCOL.read_text())
    family = protocol["families"]["spikeyolo"]
    args = root / "training_args/seed"
    args.mkdir(parents=True)
    (root / "trained_models").mkdir()
    (root / "trained_models/seed.pt").write_bytes(b"test weights")
    (root / "results").mkdir()
    (root / "results/seed_results.csv").write_text("0.4,0.2,23122451\n")
    proxy = dict(protocol["proxy"], epochs=protocol["challenging"]["search"]["epochs"], seed=seed)
    (args / "args.yaml").write_text(yaml.safe_dump(dict(
        family_settings=family, proxy=proxy, initialization="scratch")))
    official = dict(family["training"], pretrained=False, seed=seed, epochs=proxy["epochs"],
                    imgsz=proxy["img_width"], batch=proxy["batch_size"])
    (args / "official_args.yaml").write_text(yaml.safe_dump(official))
    (args / "model.json").write_text(json.dumps(policy.model_config(policy.SEED)))
    (args / "runtime.json").write_text(json.dumps({"actual_batch_shape": [proxy["img_width"]] * 2, "actual_train_frames": 10053}))
    (args / "run.json").write_text(json.dumps({"wall_time_seconds": 10}))
    report = dict(mode="validation", gene_id="seed", num_frames=6733,
                  metrics=dict(mAP50=.4, mAP50_95=.2, params=23_122_451),
                  metadata=dict(model_identity=baseline.digest(policy.SEED),
                                code_revision="training-commit",
                                official_split="challenging_train",
                                input_representation="released_event_frame_MS_GetT_T1",
                                upstream=family["upstream"], checkpoint_policy=family["checkpoint_policy"],
                                genome_version=family["genome_version"],
                                checkpoint_identity=baseline.digest(root / "trained_models/seed.pt")))
    (root / "results/seed_evaluation.json").write_text(json.dumps(report))


def test_baseline_provenance_and_tampering(tmp_path):
    _baseline_artifacts(tmp_path)
    assert baseline.validate(tmp_path, 0)["params"] == 23_122_451
    with pytest.raises(ValueError, match="provenance"):
        baseline.validate(tmp_path, 1)
    (tmp_path / "trained_models/seed.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights changed"):
        baseline.validate(tmp_path, 0)


def test_baseline_accepts_real_csv_rounding(tmp_path):
    from sota.FRED_LLM_GE.phase1_detection.result_contract import DetectionResult, write_results
    _baseline_artifacts(tmp_path)
    metrics = dict(mAP50=0.8402510086117155, mAP50_95=0.42908857953944, params=23_122_451)
    path = tmp_path / "results/seed_evaluation.json"
    report = json.loads(path.read_text())
    report["metrics"] = metrics
    path.write_text(json.dumps(report))
    write_results(tmp_path, "seed", DetectionResult(metrics["mAP50"], metrics["mAP50_95"], metrics["params"]))
    result = baseline.validate(tmp_path, 0)
    assert (result["mAP50"], result["mAP50_95"]) == (0.840251, 0.429089)
    # One unit in the CSV's final decimal place is a real disagreement.
    (tmp_path / "results/seed_results.csv").write_text("0.840252,0.429089,23122451,\n")
    with pytest.raises(ValueError, match="metrics disagree"):
        baseline.validate(tmp_path, 0)


@pytest.mark.parametrize("change,error", [
    ({"metrics": {"mAP50": float("nan"), "mAP50_95": .2, "params": 23_122_451}}, "invalid.*metrics"),
    ({"metrics": {"mAP50": .4, "mAP50_95": .2, "params": 23_122_452}}, "metrics disagree"),
    ({"num_frames": 6732}, "coverage"),
])
def test_baseline_still_rejects_invalid_report(tmp_path, change, error):
    _baseline_artifacts(tmp_path)
    path = tmp_path / "results/seed_evaluation.json"
    report = json.loads(path.read_text())
    report.update(change)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=error):
        baseline.validate(tmp_path, 0)


def _baseline_identity():
    return dict(files={"spikeyolo_baseline.py": "old-validator", "adapters/spikeyolo.py": "trainer"},
                git_commit="training-commit", dependencies={"torch": "fixed"}, sha256="original")


@pytest.mark.parametrize("legacy", [False, True])
def test_baseline_recovery_preserves_artifacts_and_summary_audit(tmp_path, monkeypatch, legacy):
    root = tmp_path / "seed_0"
    _baseline_artifacts(root)
    original = _baseline_identity()
    artifacts = {p: baseline.digest(root / p) for p in baseline.ARTIFACTS}
    observation = dict(schema_version="fred-spikeyolo-baseline-start-v1", seed=0, identity=original)
    name = "baseline_start.json"
    if legacy:
        name = "baseline_recovery_observation.json"
        observation.update(schema_version="fred-spikeyolo-post-training-observation-v1",
                           basis="User confirmed unchanged since submission; observed after training",
                           artifacts=artifacts)
    (root / name).write_text(json.dumps(observation))
    current = copy.deepcopy(original)
    current["files"]["spikeyolo_baseline.py"] = "fixed-validator"
    current.update(git_commit="fix-commit", sha256="fixed")
    monkeypatch.setattr(baseline, "verify_export", lambda: None)
    monkeypatch.setattr(baseline, "identity", lambda: current)
    monkeypatch.setattr(baseline.subprocess, "run", lambda *a, **kw: pytest.fail("recovery launched training"))
    baseline.recover_seed(0, tmp_path, name)
    assert artifacts == {p: baseline.digest(root / p) for p in baseline.ARTIFACTS}
    with pytest.raises(FileExistsError):
        baseline.recover_seed(0, tmp_path, name)
    for seed in (1, 2):
        other = tmp_path / f"seed_{seed}"
        _baseline_artifacts(other, seed)
        (other / "baseline.json").write_text(json.dumps(dict(baseline.validate(other, seed), identity=current)))
    baseline.summarize(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs"][0]["recovery"]["observation_sha256"] == baseline.digest(root / name)
    # The recovered record must not let later model changes through the gate.
    current["files"]["adapters/spikeyolo.py"] = "changed-trainer"
    with pytest.raises(ValueError, match="execution inputs changed"):
        baseline.summarize(tmp_path)


def test_baseline_recovery_refuses_changed_training_inputs(tmp_path, monkeypatch):
    root = tmp_path / "seed_0"
    _baseline_artifacts(root)
    original = _baseline_identity()
    (root / "baseline_start.json").write_text(json.dumps(dict(
        schema_version="fred-spikeyolo-baseline-start-v1", seed=0, identity=original)))
    current = copy.deepcopy(original)
    current["dependencies"]["torch"] = "changed"
    monkeypatch.setattr(baseline, "verify_export", lambda: None)
    monkeypatch.setattr(baseline, "identity", lambda: current)
    with pytest.raises(ValueError, match="execution inputs changed"):
        baseline.recover_seed(0, tmp_path, "baseline_start.json")
    assert not (root / "baseline.json").exists()


@pytest.mark.parametrize("change,error", [
    ("artifact", "observation or artifacts changed"),
    ("revision", "training revision"),
    ("missing", None),
])
def test_baseline_recovery_requires_unchanged_evidence(tmp_path, monkeypatch, change, error):
    root = tmp_path / "seed_0"
    _baseline_artifacts(root)
    current = _baseline_identity()
    observation = dict(schema_version="fred-spikeyolo-post-training-observation-v1", seed=0,
                       identity=current, basis="User confirmed unchanged since submission",
                       artifacts={p: baseline.digest(root / p) for p in baseline.ARTIFACTS})
    if change == "revision":
        observation["identity"] = dict(current, git_commit="different-commit")
    if change != "missing":
        (root / "baseline_recovery_observation.json").write_text(json.dumps(observation))
    if change == "artifact":
        # A valid-looking replacement CSV still violates the frozen observation.
        (root / "results/seed_results.csv").write_text("0.400000,0.200000,23122451,\n")
    monkeypatch.setattr(baseline, "verify_export", lambda: None)
    monkeypatch.setattr(baseline, "identity", lambda: current)
    with pytest.raises(FileNotFoundError if change == "missing" else ValueError, match=error):
        baseline.recover_seed(0, tmp_path, "baseline_recovery_observation.json")
    assert not (root / "baseline.json").exists()


def test_baseline_saves_start_identity_before_launch(tmp_path, monkeypatch):
    current = _baseline_identity()
    monkeypatch.setattr(baseline, "verify_export", lambda: None)
    monkeypatch.setattr(baseline, "identity", lambda: current)
    def failed_launch(*args, **kwargs):
        start = json.loads((tmp_path / "seed_0/baseline_start.json").read_text())
        assert start["identity"] == current and start["seed"] == 0
        raise subprocess.CalledProcessError(1, "candidate_runner")
    monkeypatch.setattr(baseline.subprocess, "run", failed_launch)
    with pytest.raises(subprocess.CalledProcessError):
        baseline.run_seed(0, tmp_path, "cpu")
    assert not (tmp_path / "seed_0/baseline.json").exists()


def test_baseline_refuses_overwrite_and_stale_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "verify_export", lambda: None)
    monkeypatch.setattr(baseline, "identity", lambda: {"sha256": "current"})
    for seed in range(3):
        root = tmp_path / f"seed_{seed}"
        _baseline_artifacts(root, seed)
        record = dict(baseline.validate(root, seed), identity={"sha256": "current"})
        (root / "baseline.json").write_text(json.dumps(record))
    baseline.summarize(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["seeds"] == [0, 1, 2]
    with pytest.raises(FileExistsError):
        baseline.run_seed(0, tmp_path, "cpu")
    monkeypatch.setattr(baseline, "identity", lambda: {"sha256": "new"})
    with pytest.raises(ValueError, match="stale"):
        baseline.summarize(tmp_path)


@pytest.mark.parametrize("artifact,value,error", [
    ("official_args.yaml", {"pretrained": True}, "trainer arguments"),
    ("runtime.json", {"actual_batch_shape": [320, 320], "actual_train_frames": 10053}, "resolution"),
    ("runtime.json", {"actual_batch_shape": [1280, 1280], "actual_train_frames": 10052}, "coverage"),
])
def test_baseline_rejects_wrong_training_execution(tmp_path, artifact, value, error):
    _baseline_artifacts(tmp_path)
    path = tmp_path / "training_args/seed" / artifact
    if path.suffix == ".yaml":
        recorded = yaml.safe_load(path.read_text())
        recorded.update(value)
        path.write_text(yaml.safe_dump(recorded))
    else:
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=error):
        baseline.validate(tmp_path, 0)


def test_mutation_template_reaches_operator_and_retries_unchanged_reply(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from src import llm_mutation
    original = seed_config()
    changed = seed_config()
    changed["backbone"][1][3][0] = 192
    replies = iter([original, changed])
    calls = []

    def generate(prompt, *args, **kwargs):
        calls.append(prompt)
        return "```yaml\n" + yaml.safe_dump({"backbone": next(replies)["backbone"]}) + "```"

    monkeypatch.setattr(llm_mutation, "ROOT_DIR", str(ROOT))
    monkeypatch.setattr(llm_mutation, "load_candidate_policy", lambda _: policy)
    monkeypatch.setattr(llm_mutation, "generate_augmented_code", generate)
    monkeypatch.setattr(llm_mutation.np.random, "randint", lambda *_: 1)
    monkeypatch.setattr(policy, "built_architecture", lambda path: (json.dumps(policy.load_and_validate(path)), 1))
    child = tmp_path / "network_child.py"
    llm_mutation.augment_network(str(policy.SEED), str(child),
                                 "templates/FRED_SPIKEYOLO/Normal/architecture.txt")
    assert len(calls) == 2
    assert "backbone:" in calls[0] and "Previous proposal rejected" in calls[1]
    assert policy.load_and_validate(child) == changed


@pytest.mark.parametrize("section", policy.SECTIONS)
def test_real_crossover_combines_layers_in_each_section(tmp_path, monkeypatch, section):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from src import llm_crossover
    first, second, child_config = seed_config(), seed_config(), seed_config()
    indices = (1, 3) if section == "backbone" else (0, 4)
    first[section][indices[0]][3][0] -= 64
    second[section][indices[1]][3][0] -= 64
    child_config[section][indices[0]] = first[section][indices[0]]
    child_config[section][indices[1]] = second[section][indices[1]]
    a = write_genome(tmp_path / "a.py", first)
    b = write_genome(tmp_path / "b.py", second)
    child = tmp_path / "child.py"
    monkeypatch.setattr(llm_crossover, "ROOT_DIR", str(ROOT))
    monkeypatch.setattr(llm_crossover, "CROSSOVER_TEMPLATE_DIR", "templates/FRED_SPIKEYOLO/CrossOver", raising=False)
    monkeypatch.setattr(llm_crossover, "CONSTANT_RULES_PATH", "templates/FRED_SPIKEYOLO/ConstantRules.txt", raising=False)
    monkeypatch.setattr(llm_crossover, "load_candidate_policy", lambda _: policy)
    monkeypatch.setattr(llm_crossover, "generate_augmented_code", lambda *a, **kw: yaml.safe_dump({section: child_config[section]}))
    monkeypatch.setattr(policy, "built_architecture", lambda path: (json.dumps(policy.load_and_validate(path)), 1))
    llm_crossover.augment_network(str(a), str(b), str(child))
    assert policy.load_and_validate(child) == child_config
    assert policy.changed_layer_count(child, a) == 1
    assert policy.changed_layer_count(child, b) == 1


def test_response_parsing_ignores_prose_and_rejects_partial_sections(tmp_path):
    chunks = policy.candidate_chunks("explanation\n```yaml\nhead: []\n```\n```python\nwrong\n```")
    assert chunks == ["head: []"]
    parts = policy.SEED.read_text().split("# --OPTION--")
    parts[2] = chunks[0]
    bad = tmp_path / "partial.py"
    bad.write_text("# --OPTION--".join(parts))
    with pytest.raises(ValueError, match="positions"):
        policy.load_and_validate(bad)


def test_representative_wider_graphs_forward_backward_and_operation_hashes():
    stock = policy.model_config(policy.SEED)
    cases = {"stock": copy.deepcopy(stock)}
    edits = [
        ("backbone_width", "backbone", 3, 3, [192, 3, 2, 1, False]),
        ("neck_width", "head", 4, 3, [192, 1, 1]),
        ("backbone_type", "backbone", 2, 2, "MS_ConvBlock"),
        ("neck_type", "head", 2, 2, "MS_AllConvBlock"),
        ("neck_depth", "head", 6, 1, 6),
        ("expansion", "backbone", 4, 3, [2, 7]),
        ("separable_kernel", "head", 2, 3, [3, 5]),
        ("pool_kernel", "backbone", 9, 3, [1024, 3]),
        ("downsample_kernel", "backbone", 1, 3, [128, 5, 4, 1, True]),
        ("neck_projection_kernel", "head", 10, 3, [256, 5, 2]),
        ("alias", "backbone", 2, 1, 1),
    ]
    for name, section, index, field, value in edits:
        config = copy.deepcopy(stock)
        config[section][index][field] = value
        cases[name] = config
    combined = copy.deepcopy(stock)
    combined["backbone"][3][3][0] = 192
    combined["head"][2] = [-1, 6, "MS_AllConvBlock", [2, 5]]
    combined["head"][4][3][0] = 192
    cases["combined_backbone_neck"] = combined
    script = (
        "import json,sys; from sota.FRED_LLM_GE.phase1_detection.spikeyolo_smoke import check; "
        "cases=json.loads(sys.argv[1]); "
        "records={name:check(config, backward=name not in ('stock','alias')) for name,config in cases.items()}; "
        "print(json.dumps(records))"
    )
    result = subprocess.run([sys.executable, "-c", script, json.dumps(cases)],
                            cwd=ROOT, text=True, capture_output=True, check=True, timeout=300)
    records = json.loads(result.stdout.splitlines()[-1])
    stock_record = records["stock"]
    assert records["alias"]["architecture_hash"] == stock_record["architecture_hash"]
    assert records["pool_kernel"]["params"] == stock_record["params"]
    for name, record in records.items():
        assert record["checked_shapes"] == [[1, 5, 84], [2, 5, 126]]
        if name not in ("stock", "alias"):
            assert record["backward"]
            assert record["architecture_hash"] != stock_record["architecture_hash"]


def test_spike_fingerprint_tracks_prompt_files_and_genome_version(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint as fingerprints
    fred = tmp_path / "sota/FRED_LLM_GE"
    protocol = yaml.safe_load(baseline.PROTOCOL.read_text())
    protocol["challenging"]["data_root"] = "cache"
    names = ["configs/phase1/protocol.yaml", "seeds/seed_spikeyolo.yaml",
             "data/fred_stream.py"]
    names += ["phase1_detection/" + name for name in (
        "candidate_runner.py", "spikeyolo_source.py", "spikeyolo_search_space.py",
        "spikeyolo_smoke.py", "spikeyolo_training_smoke.py", "adapters/spikeyolo.py",
        "spikeyolo_baseline.py", "map_reporter.py", "fitness.py", "result_contract.py",
        "proxy_data.py", "fred_splits.py", "export_split.py")]
    names += ["cache/" + name for name in (
        "splits/challenging_train_split.txt", "search_train_step50/train.txt",
        "search_val_step10/train.txt", "search_train_step50/data.yaml",
        "search_val_step10/data.yaml", "search_train_step50/done/5.json",
        "search_val_step10/done/4.json")]
    for name in names:
        path = fred / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test input")
    (fred / "configs/phase1/protocol.yaml").write_text(yaml.safe_dump(protocol))
    vendor = fred / "third_party/spikeyolo"
    vendor.mkdir(parents=True)
    (vendor / "UPSTREAM.json").write_text("pinned source")
    rules = tmp_path / "templates/FRED_SPIKEYOLO/ConstantRules.txt"
    rules.parent.mkdir(parents=True)
    rules.write_text("v2 rules")
    constants = tmp_path / "src/cfg/constants_fred_yolo11.py"
    constants.parent.mkdir(parents=True)
    constants.write_text("constants")
    (tmp_path / "pyproject.toml").write_text("dependencies")
    monkeypatch.setattr(source, "SOURCE", vendor)
    monkeypatch.setattr(source, "verify_source", lambda: {})
    monkeypatch.setattr(fingerprints.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(fingerprints.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="commit\n"))
    seed = fred / "seeds/seed_spikeyolo.yaml"
    first = fingerprints.compute_fingerprint(tmp_path, "spikeyolo", "v2_test", seed)
    assert first["genome_version"] == policy.GENOME_VERSION
    assert "prompts/ConstantRules.txt" in first["files"]
    rules.write_text("changed rules")
    assert fingerprints.compute_fingerprint(tmp_path, "spikeyolo", "v2_test", seed)["sha256"] != first["sha256"]
