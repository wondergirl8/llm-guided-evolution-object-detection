"""Exercise the shared operators with the Python BN-SNN policy, without an LLM."""
from src import llm_mutation, llm_crossover
from sota.FRED_LLM_GE.phase1_detection import bn_snn_search_space as policy


def test_mutation_retries_noop_and_invalid_python(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mutation, 'load_candidate_policy', lambda _: policy)
    monkeypatch.setattr(llm_mutation.np.random, 'randint', lambda *_: 1)
    monkeypatch.setattr(llm_mutation, 'max_gen_attempts', 3)
    replies = iter(['BASE_CHANNELS = 32', 'BASE_CHANNELS = 17', 'BASE_CHANNELS = 24'])
    prompts = []
    def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        return next(replies)
    monkeypatch.setattr(llm_mutation, 'generate_augmented_code', generate)
    child = tmp_path / 'child.py'
    llm_mutation.augment_network(str(policy.SEED), str(child),
                                'templates/FRED_BNSNN/Normal/architecture.txt')
    assert policy.load_and_validate(child)['BASE_CHANNELS'] == 24
    assert 'unchanged_parent' in prompts[1]
    assert 'BASE_CHANNELS must be' in prompts[2]


def test_crossover_accepts_bounded_child(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_crossover, 'load_candidate_policy', lambda _: policy)
    monkeypatch.setattr(llm_crossover, 'CROSSOVER_TEMPLATE_DIR', 'templates/FRED_BNSNN/CrossOver')
    monkeypatch.setattr(llm_crossover, 'CONSTANT_RULES_PATH', 'templates/FRED_BNSNN/ConstantRules.txt')
    monkeypatch.setattr(llm_crossover, 'generate_augmented_code', lambda *a, **k: 'BASE_CHANNELS = 40')
    parent = tmp_path / 'parent.py'
    parent.write_text(policy.SEED.read_text().replace('BASE_CHANNELS = 32', 'BASE_CHANNELS = 24'))
    child = tmp_path / 'child.py'
    llm_crossover.augment_network(str(policy.SEED), str(parent), str(child))
    assert policy.differs_from_parents(child, policy.SEED, parent)
