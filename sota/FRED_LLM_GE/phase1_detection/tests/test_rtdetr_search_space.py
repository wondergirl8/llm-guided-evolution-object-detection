import pytest

from src.candidate_policy import load_candidate_policy
from sota.FRED_LLM_GE.phase1_detection import rtdetr_search_space as policy


_SMALL_MODEL = {
    "depth": 18, "num_queries": 16, "num_denoising": 8, "hidden_dim": 32,
    "dim_feedforward": 64, "decoder_dim": 32, "num_decoder_layers": 2,
}


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    monkeypatch.setattr(policy, "_model_settings", lambda: dict(_SMALL_MODEL))
    policy._seed_params.cache_clear()
    yield
    policy._seed_params.cache_clear()


def _variant(tmp_path, name, old=None, new=None):
    source = policy.SEED_PY.read_text()
    if old is not None:
        assert source.count(old) == 1
        source = source.replace(old, new)
    path = tmp_path / f"network_{name}.py"
    path.write_text(source)
    return path


def test_rtdetr_policy_identity_is_stable(tmp_path):
    first = policy.built_architecture(policy.SEED_PY)
    assert policy.built_architecture(_variant(tmp_path, "copy")) == first


def test_rtdetr_policy_rejects_cosmetic_rewrites(tmp_path):
    # The first R50 run's "best" candidate only regrouped the stem like this.
    regrouped = _variant(
        tmp_path, "regrouped",
        "self.conv1 = nn.Sequential(OrderedDict([\n"
        "            (name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, name in conv_def\n"
        "        ]))",
        "self.conv1 = nn.Sequential(*[\n"
        "            ConvNormLayer(c_in, c_out, k, s, act=act) for c_in, c_out, k, s, _ in conv_def\n"
        "        ])")
    docstring = _variant(
        tmp_path, "docstring",
        '"""Query selection from encoder tokens plus a deformable-attention decoder (official)."""',
        '"""Query selection from encoder tokens plus a deformable-attention decoder (improved)."""')
    assert not policy.differs_from_parents(regrouped, policy.SEED_PY)
    assert not policy.differs_from_parents(docstring, policy.SEED_PY)


def test_rtdetr_policy_accepts_real_changes(tmp_path):
    fewer_points = _variant(tmp_path, "points", "num_decoder_points: int = 4",
                            "num_decoder_points: int = 2")
    _, params = policy.built_architecture(fewer_points)
    assert params < policy.built_architecture(policy.SEED_PY)[1]
    assert policy.differs_from_parents(fewer_points, policy.SEED_PY)


def test_rtdetr_policy_rejects_protected_and_oversized_changes(tmp_path, monkeypatch):
    changed = _variant(tmp_path, "preamble", "Original RT-DETR (v1) detector",
                       "Changed RT-DETR (v1) detector")
    with pytest.raises(ValueError, match="protected preamble"):
        policy.built_architecture(changed)
    monkeypatch.setattr(policy, "MAX_PARAMS_RATIO", 0.5)
    with pytest.raises(ValueError, match="parameter count"):
        policy.built_architecture(policy.SEED_PY)


def test_rtdetr_policy_retry_prompt_and_wiring():
    assert policy.retry_prompt("prompt", "unchanged_parent", 0) == "prompt"
    retry = policy.retry_prompt("prompt", "unchanged_parent: same graph", 1)
    assert retry.startswith("prompt") and "unchanged_parent: same graph" in retry
    config = {"CANDIDATE_POLICY_MODULE": "sota.FRED_LLM_GE.phase1_detection.rtdetr_search_space"}
    assert load_candidate_policy(config) is policy
    # Python-class genomes use the operators' generic code path.
    assert not hasattr(policy, "candidate_chunks")
    assert not hasattr(policy, "changed_layer_count")
