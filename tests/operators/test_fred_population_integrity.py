"""Controller regressions for shared parents and YOLO11 graph identity."""

import json
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
    evolution.submit_run('gene', exclude_nodes=('bad-node-1', 'bad-node-2'))
    assert calls[0][:5] == [
        'sbatch', '-C', 'H200', '--exclude', 'bad-node-1,bad-node-2']
    assert evolution.GLOBAL_DATA['gene']['results_job'] == '123'


@pytest.mark.parametrize('preserve_slurm_binding', [True, False])
def test_evaluation_script_preserves_slurm_gpu_assignment_when_requested(
        tmp_path, monkeypatch, preserve_slurm_binding):
    (tmp_path / '0').mkdir()
    monkeypatch.setattr(evolution, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'LOCAL', False)
    monkeypatch.setattr(evolution, 'RUN_COMMAND', 'sbatch')
    monkeypatch.setattr(evolution, 'PRESERVE_SLURM_CUDA_VISIBLE_DEVICES',
                        preserve_slurm_binding, raising=False)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {'gene': {'sub_flag': True}})
    monkeypatch.setattr(evolution, 'load_yaml', lambda: {
        'python_bash_script': (
            '#!/bin/bash\n#SBATCH --gres=gpu:1\n'
            'export CUDA_VISIBLE_DEVICES=0\n{}\n')})
    monkeypatch.setattr(evolution.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout='Submitted batch job 123',
                                        stderr=''))

    evolution.submit_run('gene')

    script = (tmp_path / '0' / 'gene_model.sh').read_text()
    assert '#SBATCH --gres=gpu:1' in script
    assert 'candidate_runner.py' in script
    if preserve_slurm_binding:
        assert 'export CUDA_VISIBLE_DEVICES=0' not in script
        assert 'SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}' in script
        assert 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}' in script
    else:
        assert 'export CUDA_VISIBLE_DEVICES=0' in script


def test_gpu_unavailable_retry_preserves_failure_and_recovers(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    results.mkdir()
    logs = tmp_path / 'evaluation'
    logs.mkdir()
    (logs / 'slurm-111.out').write_text('job done\n')
    (results / 'seed_results.csv').write_text(
        '-inf,-inf,inf,gpu_unavailable: CUDA unavailable\n')
    monkeypatch.setattr(evolution, 'LOCAL', False)
    monkeypatch.setattr(evolution, 'SOTA_ROOT', str(tmp_path))
    monkeypatch.setattr(evolution, 'SLURM_OUTPUT_PATH', f'{tmp_path}/')
    monkeypatch.setattr(evolution, 'EVAL_GPU_RETRY_LIMIT', 2, raising=False)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'seed': {'results_job': '111', 'local_output': None,
                 'status': 'running eval', 'fitness': None,
                 'evaluation_cache_key': 'seed-key'}})
    monkeypatch.setattr(evolution, 'EVAL_PENDING', {'seed-key': 'seed'})
    monkeypatch.setattr(evolution, 'EVAL_CACHE', {})
    monkeypatch.setattr(evolution.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout='bad-node-1\n'))
    submitted = []

    def resubmit(gene_id, exclude_nodes=()):
        submitted.append((gene_id, exclude_nodes))
        evolution.GLOBAL_DATA[gene_id]['results_job'] = '222'

    monkeypatch.setattr(evolution, 'submit_run', resubmit)
    evolution.check4results('seed')
    assert submitted == [('seed', ['bad-node-1'])]
    assert evolution.GLOBAL_DATA['seed']['gpu_retry_count'] == 1
    assert evolution.GLOBAL_DATA['seed']['gpu_retry_history'][0]['job_id'] == '111'
    assert (results / 'seed_gpu_retry_1.csv').exists()
    assert not (results / 'seed_results.csv').exists()
    assert evolution.EVAL_PENDING == {'seed-key': 'seed'}

    (logs / 'slurm-222.out').write_text('job done\n')
    (results / 'seed_results.csv').write_text('0.7,0.3,2500000\n')
    evolution.check4results('seed')
    assert evolution.GLOBAL_DATA['seed']['fitness'] == (0.7, 0.3, 2_500_000)
    assert evolution.EVAL_CACHE['seed-key']['gene_id'] == 'seed'
    assert evolution.EVAL_PENDING == {}


def test_gpu_unavailable_retry_limit_records_failure(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results' / 'seed_results.csv').write_text(
        '-inf,-inf,inf,gpu_unavailable: CUDA unavailable\n')
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'evaluation' / 'slurm-333.out').write_text('job done\n')
    monkeypatch.setattr(evolution, 'LOCAL', False)
    monkeypatch.setattr(evolution, 'SOTA_ROOT', str(tmp_path))
    monkeypatch.setattr(evolution, 'SLURM_OUTPUT_PATH', f'{tmp_path}/')
    monkeypatch.setattr(evolution, 'EVAL_GPU_RETRY_LIMIT', 2, raising=False)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'seed': {'results_job': '333', 'local_output': None,
                 'status': 'running eval', 'fitness': None, 'gpu_retry_count': 2}})
    evolution.check4results('seed')
    assert evolution.GLOBAL_DATA['seed']['fitness'] == evolution.INVALID_FITNESS_MAX
    assert evolution.GLOBAL_DATA['seed']['failure_category'] == 'gpu_unavailable'
    assert 'after 2 GPU retries' in evolution.GLOBAL_DATA['seed']['failure_reason']


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


def test_resume_identity_records_gpu_retry_policy(monkeypatch):
    monkeypatch.setattr(protocol_fingerprint, 'compute_fingerprint',
                        lambda *_: {'sha256': 'same-training-protocol'})
    monkeypatch.setattr(evolution, '_ACTIVE_FINGERPRINT', None)
    monkeypatch.setattr(evolution, 'EVAL_GPU_RETRY_LIMIT', 2, raising=False)
    first = evolution.run_fingerprint()
    assert first['evolution_settings']['eval_gpu_retry_limit'] == 2
    monkeypatch.setattr(evolution, 'EVAL_GPU_RETRY_LIMIT', 3)
    with pytest.raises(RuntimeError, match='changed during this run'):
        evolution.run_fingerprint()


def test_dfine_initial_refill_restores_distinct_population(monkeypatch, tmp_path):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(evolution, 'RUN_ROOT', None)
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 2)
    monkeypatch.setattr(evolution, 'GEN_COUNT', 0)
    monkeypatch.setattr(evolution, 'log_failure_counts', lambda *_: None)
    monkeypatch.setattr(evolution, 'delayed_creation_check', lambda fresh: fresh)
    fitness = (0.8, 0.4, 2_000_000)
    records = {
        'seed': {'fitness': fitness, 'architecture_sha256': 'seed'},
        'first': {'fitness': fitness, 'architecture_sha256': 'first'},
    }
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    counter = iter(('second',))
    def make_population(n, llm_model):
        assert n == 1 and llm_model == 'mixtral'
        gene = next(counter)
        records[gene] = {'fitness': None}
        return [_individual(gene)]
    def evaluate(fresh):
        for individual in fresh:
            records[individual[0]].update(fitness=fitness,
                                          architecture_sha256=individual[0])
            individual.fitness.values = fitness
    monkeypatch.setattr(evolution.toolbox, 'population', make_population)
    monkeypatch.setattr(evolution, 'check_and_update_fitness', evaluate)
    result = evolution.refill_detection_initial_population(
        [_individual('seed'), _individual('first')], 3, 'mixtral')
    assert [individual[0] for individual in result] == ['seed', 'first', 'second']
    assert evolution.GEN_COUNT == 0


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_custom_detector_initial_refill_reaches_eight_distinct_candidates(
    monkeypatch, family,
):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_ROOT', None)
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 3)
    monkeypatch.setattr(evolution, 'GEN_COUNT', 0)
    monkeypatch.setattr(evolution, 'log_failure_counts', lambda *_: None)
    monkeypatch.setattr(evolution, 'delayed_creation_check', lambda fresh: fresh)
    fitness = (0.8, 0.4, 2_000_000)
    genes = ['seed', 'p2', 'p3', 'p4', 'p5']
    records = {gene: {'fitness': fitness, 'architecture_sha256': gene,
                      'coverage_target': f'{gene}_down' if gene != 'seed' else None}
               for gene in genes}
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    def make_population(n, llm_model):
        assert n == 3 and llm_model == 'mixtral'
        fresh = ['fresh1', 'fresh2', 'fresh3']
        records.update({gene: {'fitness': None, 'coverage_target': None}
                        for gene in fresh})
        return [_individual(gene) for gene in fresh]
    def evaluate(fresh):
        for individual in fresh:
            records[individual[0]].update(fitness=fitness,
                                          architecture_sha256=individual[0])
            individual.fitness.values = fitness
    monkeypatch.setattr(evolution.toolbox, 'population', make_population)
    monkeypatch.setattr(evolution, 'check_and_update_fitness', evaluate)

    result = evolution.refill_detection_initial_population(
        [_individual(gene) for gene in genes], 8, 'mixtral')

    assert [individual[0] for individual in result] == [
        *genes, 'fresh1', 'fresh2', 'fresh3']
    assert len({records[individual[0]]['architecture_sha256'] for individual in result}) == 8
    assert evolution.GEN_COUNT == 0


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_invalid_initial_population_writes_depleted_status(monkeypatch, tmp_path, family):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_ROOT', str(tmp_path))
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 0)
    monkeypatch.setattr(evolution, 'log_failure_counts', lambda *_: None)
    genes = ['seed', 'p2', 'p3', 'p4', 'p5', 'bad1', 'bad2', 'bad3']
    records = {gene: {'fitness': (0.8, 0.4, 2_000_000),
                      'architecture_sha256': gene,
                      'coverage_target': f'{gene}_down' if gene in genes[1:5] else None}
               for gene in genes}
    for gene in genes[5:]:
        records[gene].update(fitness=evolution.INVALID_FITNESS_MAX,
                             failure_reason='invalid proposal')
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    with pytest.raises(RuntimeError, match='generation 0 has 5/8'):
        evolution.refill_detection_initial_population(
            [_individual(gene) for gene in genes], 8, 'mixtral')
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['status'] == 'depleted'
    assert status['distinct_architectures'] == 5
    assert all(records[gene]['failure_reason'] == 'invalid proposal' for gene in genes[5:])


def test_yolo26_failed_required_target_stops_before_refill(monkeypatch):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', 'yolo26_p2')
    monkeypatch.setattr(evolution, 'RUN_ROOT', None)
    fitness = (0.8, 0.4, 2_000_000)
    genes = ['seed', 'p2', 'p3', 'p4', 'p5']
    records = {gene: {'fitness': fitness, 'coverage_target': None}
               for gene in genes}
    for gene in genes[1:]:
        records[gene]['coverage_target'] = f'{gene}_down'
    records['p2']['fitness'] = evolution.INVALID_FITNESS_MAX
    records['p2']['failure_reason'] = 'training failed'
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    with pytest.raises(RuntimeError, match='required p2_down candidate'):
        evolution.refill_detection_initial_population(
            [_individual(gene) for gene in genes], 8, 'mixtral')


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_custom_detector_success_keeps_parent_record(monkeypatch, family):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'load_candidate_policy', lambda _: None)
    records = {'parent': {'fitness': (0.8, 0.4, 2_000_000)},
               'child': {'fitness': None}}
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    result = evolution.update_individual(_individual('child'), 'child',
                                         old_gene_id='parent', process_success=True)
    assert result[0] == 'child'
    assert 'parent' in records


def test_yolo26_failed_mutation_remains_failed_child(monkeypatch):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', 'yolo26_p2')
    monkeypatch.setattr(evolution, 'DELAYED_CHECK', True)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'parent': {'fitness': (0.8, 0.4, 2_000_000)},
        'child': {'status': 'DELAYED_CHECK', 'sub_flag': True, 'job_id': '123',
                  'fitness': None},
    })
    monkeypatch.setattr(evolution, 'LINKED_GENES', {'child': 'parent'})
    monkeypatch.setattr(evolution, 'check4job_completion', lambda *args, **kwargs: False)
    monkeypatch.setattr(evolution, 'operator_failure_reason', lambda *_: 'retry_exhausted: mutation')
    def reject(gene, _fitness, reason, status):
        evolution.GLOBAL_DATA[gene].update(fitness=evolution.INVALID_FITNESS_MAX,
                                          failure_reason=reason, status=status)
    monkeypatch.setattr(evolution, 'reject_detection_fitness', reject)
    child = _individual('child')
    assert evolution.delayed_mutate_check([child])[0][0] == 'child'
    assert evolution.GLOBAL_DATA['child']['status'] == 'FAILED: OPERATOR'
    assert 'parent' in evolution.GLOBAL_DATA


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_failed_crossover_child_is_not_mutated(monkeypatch, family):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'DELAYED_CHECK', True)
    monkeypatch.setattr(evolution, 'load_candidate_policy', lambda _: None)
    monkeypatch.setattr(evolution, 'mutation_probability', 1.0)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'parent': {'fitness': (0.8, 0.4, 2_000_000)},
        'failed_child': {'status': 'DELAYED_CHECK', 'sub_flag': True,
                         'job_id': '123', 'fitness': None},
        'healthy': {'status': 'completed', 'fitness': (0.8, 0.4, 2_000_000)},
    })
    monkeypatch.setattr(evolution, 'LINKED_GENES', {'failed_child': 'parent'})
    monkeypatch.setattr(evolution, 'check4job_completion', lambda *args, **kwargs: False)
    monkeypatch.setattr(evolution, 'record_dfine_proposal_failure', lambda *args: None)
    monkeypatch.setattr(evolution, 'operator_failure_reason',
                        lambda *_: 'retry_exhausted: crossover')

    def reject(gene, _fitness, reason, status):
        evolution.GLOBAL_DATA[gene].update(fitness=evolution.INVALID_FITNESS_MAX,
                                          failure_reason=reason, status=status)

    monkeypatch.setattr(evolution, 'reject_detection_fitness', reject)
    monkeypatch.setattr(evolution.random, 'random', lambda: 0.0)
    mutated = []
    monkeypatch.setattr(evolution.toolbox, 'mutate',
                        lambda individual, llm_model: mutated.append(individual[0]))
    offspring = [_individual('failed_child'), _individual('healthy')]

    evolution.mutate_selected_offspring(evolution.delayed_mate_check(offspring), 'mixtral')

    assert mutated == ['healthy']
    assert offspring[0][0] == 'failed_child'
    assert offspring[0].fitness.valid
    assert not offspring[1].fitness.valid
    assert evolution.GLOBAL_DATA['failed_child']['failure_reason'] == 'retry_exhausted: crossover'
    assert 'parent' in evolution.GLOBAL_DATA


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_custom_detector_generation_refill_keeps_population_size(monkeypatch, family):
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_ROOT', None)
    monkeypatch.setattr(evolution, 'FRED_MAX_REFILL_ATTEMPTS', 3)
    monkeypatch.setattr(evolution, 'log_failure_counts', lambda *_: None)
    monkeypatch.setattr(evolution, 'delayed_mutate_check', lambda fresh: fresh)
    fitness = (0.8, 0.4, 2_000_000)
    parents = [f'parent{i}' for i in range(5)]
    records = {gene: {'fitness': fitness, 'architecture_sha256': gene}
               for gene in parents}
    records['duplicate'] = {'fitness': fitness, 'architecture_sha256': parents[0]}
    records['invalid'] = {'fitness': evolution.INVALID_FITNESS_MAX,
                          'failure_reason': 'invalid proposal'}
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', records)
    monkeypatch.setattr(evolution.toolbox, 'clone', lambda individual: _individual(individual[0]))
    new_genes = iter(('child1', 'child2', 'child3'))
    def mutate(individual, llm_model):
        assert individual[0] in parents and llm_model == 'mixtral'
        gene = next(new_genes)
        records[gene] = {'fitness': None}
        return _individual(gene)
    monkeypatch.setattr(evolution.toolbox, 'mutate', mutate)
    def evaluate(fresh):
        for individual in fresh:
            records[individual[0]].update(fitness=fitness,
                                          architecture_sha256=individual[0])
            individual.fitness.values = fitness
    monkeypatch.setattr(evolution, 'check_and_update_fitness', evaluate)
    result = evolution.refill_detection_generation_population(
        [_individual(gene) for gene in [*parents, 'duplicate', 'invalid']],
        8, 'mixtral', 1)
    assert [individual[0] for individual in result] == [
        *parents, 'child1', 'child2', 'child3']
    assert records['duplicate']['population_exclusion_reason'].startswith(
        'duplicate_architecture: matches parent0')
    assert records['invalid']['failure_reason'] == 'invalid proposal'


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_custom_detector_pending_generation_zero_resumes_before_generation_one(
    monkeypatch, tmp_path, family,
):
    monkeypatch.delenv('LLMGE_BOUNDED_SESSION', raising=False)
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_TAG', f'{family}_20261022_eot0_test')
    monkeypatch.setattr(evolution, 'RUN_CHECKPOINT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'run_fingerprint', lambda: {'sha256': 'fixed'})
    monkeypatch.setattr(evolution, 'INITIAL_POPULATION_PENDING', True)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'seed': {'sub_flag': True, 'status': 'running eval', 'fitness': None,
                 'results_job': '122', 'local_output': None, 'start_time': 0},
        'new': {'sub_flag': True, 'status': 'running eval', 'fitness': None,
                'results_job': '123', 'local_output': None, 'start_time': 0},
    })
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    monkeypatch.setattr(evolution, 'population', [_individual('seed')], raising=False)
    monkeypatch.setattr(evolution, 'hof', evolution.tools.ParetoFront(), raising=False)
    evolution.persist_initial_population_checkpoint([_individual('new')])
    pending, start, state = evolution.load_checkpoint(
        folder_name=str(tmp_path), global_path=str(tmp_path))
    assert start == 0
    assert pending['initial_population_pending_evaluation'] is True
    assert [individual[0] for individual in pending['population']] == ['seed', 'new']
    assert state['GLOBAL_DATA']['new']['results_job'] == '123'
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', state['GLOBAL_DATA'])
    monkeypatch.setattr(evolution, 'population', pending['population'])
    monkeypatch.setattr(evolution, 'check4model2run',
                        lambda gene: pytest.fail(f'resubmitted {gene}'))
    def finish(gene):
        evolution.GLOBAL_DATA[gene].update(status='completed',
                                           fitness=(0.8, 0.4, 2_000_000))
    monkeypatch.setattr(evolution, 'check4results', finish)
    for individual in pending['population']:
        individual.fitness.values = evolution.PLACEHOLDER_FITNESS
    evolution.check_and_update_fitness(pending['population'], loop_delay=0)
    assert all(ind.fitness.values == (0.8, 0.4, 2_000_000)
               for ind in pending['population'])
    evolution.save_checkpoint(0, folder_name=str(tmp_path), global_path=str(tmp_path),
                              checkpoint_data={'population': pending['population'],
                                               'hof': evolution.hof,
                                               'initial_population_pending_evaluation': False})
    completed, start, _ = evolution.load_checkpoint(
        folder_name=str(tmp_path), global_path=str(tmp_path))
    assert start == 1
    assert completed['initial_population_pending_evaluation'] is False


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
@pytest.mark.parametrize('prior_snapshot', [False, True])
def test_generation_zero_recovers_after_one_checkpoint_file_write(
    monkeypatch, tmp_path, family, prior_snapshot,
):
    from src import fred_session

    monkeypatch.delenv('LLMGE_BOUNDED_SESSION', raising=False)
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_TAG', f'{family}_20261022_eot0_test')
    monkeypatch.setattr(evolution, 'RUN_CHECKPOINT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'run_fingerprint', lambda: {'sha256': 'fixed'})
    monkeypatch.setattr(evolution, 'INITIAL_POPULATION_PENDING', True)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'seed': {'status': 'running eval', 'fitness': None, 'results_job': '122'},
    })
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    monkeypatch.setattr(evolution, 'population', [_individual('seed')], raising=False)
    monkeypatch.setattr(evolution, 'hof', evolution.tools.ParetoFront(), raising=False)
    if prior_snapshot:
        evolution.persist_initial_population_checkpoint()

    evolution.GLOBAL_DATA['new'] = {
        'status': 'running eval', 'fitness': None, 'results_job': '123'}
    original_write = fred_session.atomic_write
    writes = []
    def interrupt_second_write(path, payload):
        writes.append(path)
        if len(writes) == 2:
            raise OSError('simulated controller interruption')
        original_write(path, payload)
    monkeypatch.setattr(fred_session, 'atomic_write', interrupt_second_write)
    with pytest.raises(OSError, match='simulated controller interruption'):
        evolution.persist_initial_population_checkpoint([_individual('new')])
    monkeypatch.setattr(fred_session, 'atomic_write', original_write)

    pending, start, state = evolution.load_checkpoint(
        folder_name=str(tmp_path), global_path=str(tmp_path))
    assert start == 0
    assert [individual[0] for individual in pending['population']] == ['seed', 'new']
    assert state['GLOBAL_DATA']['new']['results_job'] == '123'


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_generation_zero_creation_resume_fills_remaining_slots(
    monkeypatch, tmp_path, family,
):
    monkeypatch.delenv('LLMGE_BOUNDED_SESSION', raising=False)
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_TAG', f'{family}_20261022_eot0_test')
    monkeypatch.setattr(evolution, 'RUN_CHECKPOINT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'run_fingerprint', lambda: {'sha256': 'fixed'})
    monkeypatch.setattr(evolution, 'start_population_size', 8)
    monkeypatch.setattr(evolution, 'INITIAL_CREATION_PENDING', True)
    monkeypatch.setattr(evolution, 'INITIAL_CREATION_IDS', ['first'])
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {
        'first': {'status': 'DELAYED_CHECK', 'sub_flag': True,
                  'job_id': '42', 'fitness': None},
    })
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    monkeypatch.setattr(evolution, 'hof', evolution.tools.ParetoFront(), raising=False)
    evolution.persist_initial_creation_checkpoint()
    saved, start, state = evolution.load_checkpoint(
        folder_name=str(tmp_path), global_path=str(tmp_path))
    assert start == 0
    assert saved['initial_population_creation_pending'] is True
    assert state['GLOBAL_DATA']['first']['job_id'] == '42'
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', state['GLOBAL_DATA'])
    monkeypatch.setattr(evolution, 'INITIAL_CREATION_IDS', ['first'])
    requested = []
    def make_individual(**kwargs):
        requested.append(kwargs.get('coverage_target'))
        gene = f'new{len(requested)}'
        evolution.GLOBAL_DATA[gene] = {'status': 'DELAYED_CHECK', 'sub_flag': True,
                                       'job_id': str(42 + len(requested))}
        evolution.INITIAL_CREATION_IDS.append(gene)
        evolution.persist_initial_creation_checkpoint()
        return _individual(gene)
    monkeypatch.setattr(evolution.toolbox, 'individual', make_individual)
    checked = []
    monkeypatch.setattr(evolution, 'delayed_creation_check',
                        lambda individuals: checked.extend(ind[0] for ind in individuals))
    monkeypatch.setattr(evolution, 'verify_yolo26_initial_population', lambda _: None)
    monkeypatch.setattr(evolution, 'add_stock_seed_to_initial_population',
                        lambda individuals, allow_existing: [_individual('seed'), *individuals]
                        if allow_existing else pytest.fail('resume did not allow the copied seed'))

    result = evolution.finish_initial_creation(saved['population'], 'mixtral', resume=True)

    assert [ind[0] for ind in result] == ['seed', 'first',
                                         *(f'new{i}' for i in range(1, 7))]
    assert checked == [ind[0] for ind in result[1:]]
    assert len(requested) == 6
    if family == 'yolo26_p2':
        assert requested == ['p3_down', 'p4_down', 'p5_down', None, None, None]
    assert evolution.INITIAL_CREATION_PENDING is False


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_initial_creation_submission_checkpoints_delayed_job(monkeypatch, tmp_path, family):
    monkeypatch.delenv('LLMGE_BOUNDED_SESSION', raising=False)
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    monkeypatch.setattr(evolution, 'RUN_TAG', f'{family}_20261022_eot0_test')
    monkeypatch.setattr(evolution, 'RUN_CHECKPOINT_DIR', str(tmp_path))
    monkeypatch.setattr(evolution, 'run_fingerprint', lambda: {'sha256': 'fixed'})
    monkeypatch.setattr(evolution, 'INITIAL_CREATION_PENDING', True)
    monkeypatch.setattr(evolution, 'INITIAL_CREATION_IDS', [])
    monkeypatch.setattr(evolution, 'DELAYED_CHECK', True)
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    monkeypatch.setattr(evolution, 'hof', evolution.tools.ParetoFront(), raising=False)
    monkeypatch.setattr(evolution, 'load_yaml', lambda: {'LLM_GPU': None})
    monkeypatch.setattr(evolution, 'generate_random_string', lambda length: 'new')
    monkeypatch.setattr(evolution, 'submit_bash',
                        lambda *args, **kwargs: (True, '42', None))

    individual = evolution.create_individual(evolution.creator.Individual, 'mixtral')
    saved, start, state = evolution.load_checkpoint(
        folder_name=str(tmp_path), global_path=str(tmp_path))

    assert individual[0] == 'new' and start == 0
    assert [member[0] for member in saved['population']] == ['new']
    assert state['GLOBAL_DATA']['new']['status'] == 'DELAYED_CHECK'
    assert state['GLOBAL_DATA']['new']['job_id'] == '42'


def test_creation_resume_reuses_identical_stock_seed(monkeypatch, tmp_path):
    seed = tmp_path / 'seed.yaml'
    seed.write_text('head: []\n')
    variant = tmp_path / 'variants'
    variant.mkdir()
    (variant / 'network_seed.py').write_text(seed.read_text())
    monkeypatch.setattr(evolution, 'SEED_NETWORK', str(seed))
    monkeypatch.setattr(evolution, 'VARIANT_DIR', str(variant))
    monkeypatch.setattr(evolution, 'MODEL', 'network')
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    result = evolution.add_stock_seed_to_initial_population([], allow_existing=True)
    assert [individual[0] for individual in result] == ['seed']
    assert evolution.GLOBAL_DATA['seed']['origin'] == 'stock_seed'


@pytest.mark.parametrize('family', ['yolo26_p2', 'dfine'])
def test_custom_detector_resume_has_protocol_identity(monkeypatch, family):
    monkeypatch.setattr(protocol_fingerprint, 'compute_fingerprint',
                        lambda *_: {'sha256': 'custom-detector-protocol'})
    monkeypatch.setattr(evolution, '_ACTIVE_FINGERPRINT', None)
    monkeypatch.setattr(evolution, 'SEED_FAMILY', family)
    identity = evolution.run_fingerprint()
    assert identity['sha256'] == 'custom-detector-protocol'
    assert identity['evolution_settings']['start_population_size'] == 8
