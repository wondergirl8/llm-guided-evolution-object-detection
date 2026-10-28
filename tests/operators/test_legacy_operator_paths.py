"""Exercise the shared operators with no architecture policy configured."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_operator(monkeypatch, filename, root):
    constants = types.ModuleType("cfg.constants")
    constants.ROOT_DIR = str(root)
    constants.LLM_DEEPSEEK = "deepseek"
    constants.LLM_QWEN = "qwen"
    constants.FORBIDDEN_PATTERNS = []
    config_package = types.ModuleType("cfg")
    config_package.__path__ = []
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = []
    print_utils = types.ModuleType("utils.print_utils")
    print_utils.box_print = lambda *args, **kwargs: None
    llm_utils = types.ModuleType("llm_utils")
    llm_utils.split_file = lambda path: Path(path).read_text().split("# --OPTION--")
    llm_utils.generate_augmented_code = lambda *args, **kwargs: "edited: true"
    llm_utils.extract_note = lambda text: ""
    for name in ("submit_mixtral", "submit_mixtral_hf", "llm_code_qc", "str2bool",
                 "clean_code_from_llm", "retrieve_base_code", "strip_leading_imports",
                 "is_renderer_gene", "RENDERER_SAMPLING"):
        setattr(llm_utils, name, lambda *args, **kwargs: None)
    stubs = {"cfg": config_package, "cfg.constants": constants,
             "utils": utils_package, "utils.print_utils": print_utils,
             "llm_utils": llm_utils, "transformers": types.ModuleType("transformers")}
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(f"legacy_test_{filename}", ROOT / "src" / filename)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    assert operator.load_candidate_policy(operator.__dict__) is None
    return operator


@pytest.fixture
def legacy_root(tmp_path):
    (tmp_path / "templates" / "CrossOver").mkdir(parents=True)
    (tmp_path / "templates" / "CrossOver" / "crossover.txt").write_text("{}\n{}")
    (tmp_path / "templates" / "CrossOver" / "crossover_s.txt").write_text("{}\n{}")
    (tmp_path / "mutation.txt").write_text("Edit this: {}")
    return tmp_path


def test_mutation_without_policy_uses_one_generation(legacy_root, monkeypatch):
    operator = _load_operator(monkeypatch, "llm_mutation.py", legacy_root)
    source = legacy_root / "parent.py"
    target = legacy_root / "child.py"
    source.write_text("protected\n# --OPTION--\noriginal: true\n")
    calls = []
    operator.generate_augmented_code = lambda *args, **kwargs: calls.append(1) or "edited: true"
    operator.augment_network(str(source), str(target), "mutation.txt")
    assert len(calls) == 1
    assert target.read_text().split("# --OPTION--")[1].strip() == "edited: true"


def test_crossover_without_policy_uses_legacy_template_once(legacy_root, monkeypatch):
    operator = _load_operator(monkeypatch, "llm_crossover.py", legacy_root)
    first, second, target = (legacy_root / name for name in ("first.py", "second.py", "child.py"))
    first.write_text("protected\n# --OPTION--\nfirst: true\n")
    second.write_text("protected\n# --OPTION--\nsecond: true\n")
    calls = []
    operator.generate_augmented_code = lambda *args, **kwargs: calls.append(1) or "edited: true"
    operator.augment_network(str(first), str(second), str(target))
    assert len(calls) == 1
    assert target.read_text().split("# --OPTION--")[1].strip() == "edited: true"


def test_policy_mutation_retry_uses_feedback_and_higher_sampling(legacy_root, monkeypatch):
    operator = _load_operator(monkeypatch, "llm_mutation.py", legacy_root)
    source = legacy_root / "parent.py"
    target = legacy_root / "child.py"
    source.write_text("protected\n# --OPTION--\noriginal: true\n")
    responses = []
    policy = types.SimpleNamespace(
        retry_prompt=lambda prompt, error, attempt: prompt if attempt == 0 else f'{prompt}\nERROR: {error}',
        repair_channel_values=lambda chunk: chunk,
        changed_layer_count=lambda *_: 1,
        differs_from_parents=lambda *_: len(responses) == 2,
    )
    monkeypatch.setattr(operator, 'load_candidate_policy', lambda _: policy)
    monkeypatch.setattr(operator, 'max_gen_attempts', 2, raising=False)
    operator.generate_augmented_code = lambda prompt, _, qc, top_p, model, temperature: (
        responses.append((prompt, top_p, temperature)) or 'edited: true')
    operator.augment_network(str(source), str(target), "mutation.txt", top_p=.1,
                             temperature=.2)
    assert len(responses) == 2
    assert 'unchanged_parent' in responses[1][0]
    assert responses[1][1] > responses[0][1]
    assert responses[1][2] > responses[0][2]
