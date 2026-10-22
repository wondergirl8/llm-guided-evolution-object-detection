"""The shared operators must not activate a detector policy by default."""

from src.candidate_policy import load_candidate_policy


def test_legacy_configuration_has_no_candidate_policy():
    assert load_candidate_policy({}) is None
    assert load_candidate_policy({"CANDIDATE_POLICY_MODULE": None}) is None


def test_configured_policy_is_imported():
    policy = load_candidate_policy({
        "CANDIDATE_POLICY_MODULE": "sota.FRED_LLM_GE.phase1_detection.yolo11_search_space"
    })
    assert callable(policy.load_and_validate)
    assert callable(policy.differs_from_parents)
    assert callable(policy.evaluation_cache_key)
