"""Controller regressions for shared parents and YOLO11 graph identity."""

from types import SimpleNamespace

import pytest

import run_improved as evolution
from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint


def _individual(gene, fitness=(0.8, 0.4, 2_000_000)):
    individual = evolution.creator.Individual([gene])
    individual.fitness.values = fitness
    return individual


@pytest.mark.parametrize('process_type', ['Mutation', 'Mating'])
def test_changing_one_of_two_selected_copies_preserves_successful_parent(monkeypatch,
                                                                          process_type):
    monkeypatch.setattr(evolution, 'load_candidate_policy', lambda _: object())
    successful = (0.8, 0.4, 2_000_000)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'parent': {'sub_flag': True, 'status': 'completed', 'fitness': successful,
                   'failure_reason': None},
        'child': {'sub_flag': True, 'status': 'subbed file', 'fitness': None},
    })
    selected = [_individual('parent'), _individual('parent')]
    changed = evolution.update_individual(selected[0], 'child', old_gene_id='parent',
                                          process_type=process_type)
    assert changed[0] == 'child'
    assert selected[1][0] == 'parent'
    assert evolution.GLOBAL_DATA['parent']['fitness'] == successful
    with pytest.raises(RuntimeError, match='Refusing to replace successful result'):
        evolution.record_failure('parent', 'submission_failed: no job was submitted')


def test_pending_graph_is_reserved_before_second_evaluation(tmp_path, monkeypatch):
    policy = SimpleNamespace(
        built_architecture=lambda _: ('same-graph', 2_000_000),
        evaluation_cache_key=lambda *_: 'same-protocol-seed-graph',
    )
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        name: {'sub_flag': True, 'status': 'subbed file', 'fitness': None}
        for name in ('first', 'second')})
    monkeypatch.setattr(evolution, 'EVAL_CACHE', {})
    monkeypatch.setattr(evolution, 'EVAL_PENDING', {})
    monkeypatch.setattr(evolution, 'run_fingerprint', lambda: {'sha256': 'protocol'})
    monkeypatch.setattr(evolution, 'SOTA_ROOT', str(tmp_path))
    assert evolution.candidate_preflight('first', policy) is True
    assert evolution.candidate_preflight('second', policy) is False
    assert evolution.EVAL_PENDING == {'same-protocol-seed-graph': 'first'}
    assert evolution.GLOBAL_DATA['second']['failure_category'] == 'duplicate_graph'


def test_refill_stops_when_distinct_target_cannot_be_met(monkeypatch):
    monkeypatch.setattr(evolution, 'load_candidate_policy', lambda _: object())
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'seed': {'fitness': (0.8, 0.4, 2_000_000), 'failure_reason': None,
                 'architecture_hash': 'seed-graph'}})
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 0, raising=False)
    monkeypatch.setattr(evolution, 'RUN_ROOT', None, raising=False)
    with pytest.raises(RuntimeError, match='depleted: generation 0 has 1/8'):
        evolution.refill_initial_population([_individual('seed')], 8, 'mixtral')


def test_initial_refill_reaches_eight_distinct_built_graphs(monkeypatch):
    monkeypatch.setattr(evolution, 'load_candidate_policy', lambda _: object())
    successful = (0.8, 0.4, 2_000_000)
    records = {'seed': {'fitness': successful, 'failure_reason': None,
                        'architecture_hash': 'seed-graph'}}
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 7, raising=False)
    monkeypatch.setattr(evolution, 'RUN_ROOT', None, raising=False)
    monkeypatch.setattr(evolution, 'delayed_mutate_check', lambda fresh: fresh)
    monkeypatch.setattr(evolution, 'log_failure_counts', lambda *_: None)
    counter = iter(range(7))

    def mutate(individual, llm_model):
        gene = f'new-{next(counter)}'
        individual[0] = gene
        records[gene] = {'fitness': None, 'failure_reason': None,
                         'architecture_hash': gene}
        return individual

    def evaluate(fresh):
        for child in fresh:
            records[child[0]]['fitness'] = successful
            child.fitness.values = successful

    monkeypatch.setattr(evolution.toolbox, 'mutate', mutate)
    monkeypatch.setattr(evolution, 'check_and_update_fitness', evaluate)
    result = evolution.refill_initial_population([_individual('seed')], 8, 'mixtral')
    assert len(result) == len({records[child[0]]['architecture_hash'] for child in result}) == 8


def test_evaluation_submission_requests_supported_gpu(tmp_path, monkeypatch):
    (tmp_path / '0').mkdir()
    monkeypatch.setattr(evolution, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'LOCAL', False)
    monkeypatch.setattr(evolution, 'RUN_COMMAND', 'sbatch')
    monkeypatch.setattr(evolution, 'EVAL_GPU_CONSTRAINT', 'H200', raising=False)
    monkeypatch.setattr(evolution, 'load_yaml', lambda: {'python_bash_script': '{}'})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {'gene': {'sub_flag': True}})
    calls = []
    monkeypatch.setattr(evolution.subprocess, 'run', lambda args, **kwargs:
                        calls.append(args) or SimpleNamespace(
                            returncode=0, stdout='Submitted batch job 123', stderr=''))
    evolution.submit_run('gene')
    assert calls[0][:3] == ['sbatch', '-C', 'H200']
    assert evolution.GLOBAL_DATA['gene']['results_job'] == '123'


def test_server_window_stops_before_allocation_expires(monkeypatch):
    monkeypatch.setenv('LLMGE_SERVER_JOB_ID', '123')
    monkeypatch.setenv('LLMGE_MIN_SERVER_REMAINING_SECONDS', '3600')
    monkeypatch.setattr(evolution.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout='00:30:00\n', stderr=''))
    with pytest.raises(evolution.ServerWindowClosed, match='1800s left'):
        evolution.check_server_window()


@pytest.mark.parametrize(('log', 'category'), [
    ('RuntimeError: retry_exhausted: last=invalid_architecture', 'invalid_architecture'),
    ('RuntimeError: retry_exhausted: last=unchanged_parent', 'unchanged_parent'),
    ('Connection refused', 'llm_unavailable'),
    ('not compatible with the current PyTorch installation', 'gpu_unsupported'),
    ('RuntimeError: retry_exhausted', 'retry_exhausted'),
])
def test_operator_failure_retains_cause_and_operator(tmp_path, monkeypatch, log, category):
    output_dir = tmp_path / 'evolution'
    output_dir.mkdir()
    (output_dir / 'slurm-123.out').write_text(log)
    monkeypatch.setattr(evolution, 'SLURM_OUTPUT_PATH', f'{tmp_path}/')
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'gene': {'job_id': '123', 'sub_flag': True, 'operator_type': 'crossover'}})
    assert evolution.operator_failure_reason('gene', 'crossover').startswith(
        f'{category}: crossover:')


def test_blackwell_kernel_failure_is_environment_failure():
    assert evolution.is_unsupported_gpu_error(
        'CUDA error: no kernel image is available for execution on the device')
    assert not evolution.is_unsupported_gpu_error('invalid architecture: bad from link')


def test_resume_identity_records_generation_limit(monkeypatch):
    monkeypatch.setattr(protocol_fingerprint, 'compute_fingerprint',
                        lambda *_: {'sha256': 'same-training-protocol'})
    monkeypatch.setattr(evolution, '_ACTIVE_FINGERPRINT', None)
    monkeypatch.setattr(evolution, 'num_generations', 11)
    first = evolution.run_fingerprint()
    assert first['evolution_settings']['num_generations'] == 11
    monkeypatch.setattr(evolution, 'num_generations', 21)
    with pytest.raises(RuntimeError, match='changed during this run'):
        evolution.run_fingerprint()
