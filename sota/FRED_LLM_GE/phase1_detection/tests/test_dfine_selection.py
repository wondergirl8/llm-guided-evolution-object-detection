"""D-FINE motifs v1 selection uses validated resource evidence and AP50:95."""

import json
import time
from types import SimpleNamespace

import pytest

import run_improved as controller
from src.fred_dfine_selection import (VERSION, parent_mechanism, prompt_feedback, resource_limits,
                                      resource_violation, validated_metrics)
from src.fred_dfine_genome import VERSION as GENOME_VERSION, seed_genome
from sota.FRED_LLM_GE.phase1_detection import candidate_runner


@pytest.fixture
def selection_state(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(controller, 'DFINE_SEARCH_SPACE', 'motifs_v1')
    monkeypatch.setattr(controller, 'SOTA_ROOT', str(tmp_path))
    monkeypatch.setattr(controller, 'GLOBAL_DATA', {})
    monkeypatch.setenv('FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME', '5')
    monkeypatch.setenv('FRED_DFINE_MAX_PEAK_MEMORY_BYTES', '1000000')
    (tmp_path / 'results').mkdir()
    return tmp_path / 'results'


def _report(folder, gene, *, ap=.4, latency=3.0, peak=900000):
    path = folder / f'{gene}_evaluation.json'
    path.write_text(json.dumps({
        'schema_version': 'fred-detection-report-v1', 'gene_id': gene,
        'mode': 'validation',
        'metrics': {'mAP50': .6, 'mAP50_95': ap, 'params': 1000},
        'metadata': {'project_split': 'validation',
                     'official_split': 'challenging_train',
                     'resource_measurement_version': VERSION,
                     'selection_resource_limits': {
                         'max_inference_ms_per_frame': 5.0,
                         'max_peak_memory_bytes': 1000000},
                     'preloaded_inference_ms_per_frame': latency,
                     'training_peak_memory_bytes': peak,
                     'inference_device': 'cuda',
                     'search_resource_probe': {
                         'full_resolution_probe_batch': 8,
                         'full_resolution_probe_shape': [720, 1280],
                         'full_resolution_probe_peak_bytes': peak - 100000}},
        'diagnostics': {'overall': {'mAP75': .3, 'mAP_small': .2,
                                    'recall_small': .25,
                                    'false_negatives_iou50': 12,
                                    'poorly_localized_iou50_to75': 4}},
    }))
    (folder / f'{gene}_results.csv').write_text(f'0.600000,{ap:.6f},1000\n')
    return path


def _record(gene):
    controller.GLOBAL_DATA[gene] = {
        'status': 'running eval', 'sub_flag': True, 'fitness': None,
        'local_output': 'job done', 'results_job': None, 'start_time': time.time(),
    }


def test_limits_are_required_and_finite():
    with pytest.raises(ValueError, match='FRED_DFINE_MAX_INFERENCE'):
        resource_limits({})
    with pytest.raises(ValueError, match='finite positive'):
        resource_limits({'FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME': 'NaN',
                         'FRED_DFINE_MAX_PEAK_MEMORY_BYTES': '100'})
    assert resource_limits({'FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME': '5',
                            'FRED_DFINE_MAX_PEAK_MEMORY_BYTES': '1000'}) == {
        'max_inference_ms_per_frame': 5.0, 'max_peak_memory_bytes': 1000}


def test_report_evidence_and_limits_gate_search_result(selection_state):
    _report(selection_state, 'eligible')
    _report(selection_state, 'slow', latency=6.0)
    _report(selection_state, 'large', peak=1000001)
    for gene in ('eligible', 'slow', 'large'):
        _record(gene)
        controller.check4results(gene)
    assert controller.GLOBAL_DATA['eligible']['status'] == 'completed'
    assert controller.GLOBAL_DATA['eligible']['dfine_resources']['peak_memory_bytes'] == 900000
    for gene in ('slow', 'large'):
        record = controller.GLOBAL_DATA[gene]
        assert record['status'] == 'FAILED: RESOURCE LIMIT'
        assert record['raw_fitness'] == (.6, .4, 1000.0)
        assert record['dfine_resources']['report_path'].endswith(f'{gene}_evaluation.json')


def test_missing_or_mismatched_resource_report_is_not_eligible(selection_state):
    path = _report(selection_state, 'stale')
    report = json.loads(path.read_text())
    report['gene_id'] = 'another_gene'
    path.write_text(json.dumps(report))
    _record('stale')
    controller.check4results('stale')
    assert controller.GLOBAL_DATA['stale']['status'] == 'FAILED: RESULT FILE'
    assert 'identity mismatch' in controller.GLOBAL_DATA['stale']['failure_reason']


def test_missing_validation_diagnostics_are_not_treated_as_success(selection_state):
    path = _report(selection_state, 'undocumented')
    report = json.loads(path.read_text())
    report.pop('diagnostics')
    path.write_text(json.dumps(report))
    _record('undocumented')
    controller.check4results('undocumented')
    assert controller.GLOBAL_DATA['undocumented']['status'] == 'FAILED: RESULT FILE'
    assert 'missing validation diagnostics' in controller.GLOBAL_DATA['undocumented']['failure_reason']


def test_accuracy_selection_retains_best_and_pareto_archive_is_separate(selection_state):
    values = [('best', .55, .40, 1200), ('small', .50, .39, 700),
              ('worse', .57, .38, 1500)]
    population = []
    for gene, map50, map5095, params in values:
        ind = controller.creator.Individual([gene])
        ind.fitness.values = (map50, map5095, params)
        population.append(ind)
    assert [ind[0] for ind in controller.select_elites(population, 1)] == ['best']
    assert [ind[0] for ind in controller.select_parents(population, 2)] == ['best', 'small']
    archive = controller.tools.ParetoFront()
    archive.update(population)
    assert {'best', 'small'} <= {ind[0] for ind in archive}


def test_feedback_mentions_primary_goal_parent_and_measured_caps():
    limits = {'max_inference_ms_per_frame': 5.0, 'max_peak_memory_bytes': 1000000}
    record = {'fitness': (.6, .4, 1000),
              'dfine_resources': {'inference_ms_per_frame': 3.0,
                                  'peak_memory_bytes': 900000,
                                  'diagnostics': {'mAP75': .3, 'mAP_small': .2,
                                                  'recall_small': .25,
                                                  'false_negatives_iou50': 12,
                                                  'poorly_localized_iou50_to75': 4}}}
    prompt = prompt_feedback('parent', record, limits,
                             'Last attempted mechanism: encoder=bidir_td050_bu050.')
    assert 'mAP50:95' in prompt and 'Parent parent' in prompt
    assert '3.000 ms/frame' in prompt and '900000 bytes' in prompt
    assert 'AP_small=0.200000' in prompt and 'missed at IoU .50=12' in prompt
    assert 'encoder=bidir_td050_bu050' in prompt
    assert resource_violation(record['dfine_resources'], limits) is None
    with pytest.raises(ValueError, match='invalid D-FINE resource'):
        resource_violation({'inference_ms_per_frame': float('nan'),
                            'peak_memory_bytes': 900000}, limits)


def test_checkpoint_requires_new_selection_version(selection_state):
    with pytest.raises(ValueError, match='selection version'):
        controller.require_dfine_selection_version({}, 'old checkpoint')
    controller.require_dfine_selection_version({'selection_version': VERSION}, 'new checkpoint')


def test_bounded_checkpoint_records_selection_version_in_both_halves(
    selection_state, tmp_path, monkeypatch,
):
    from src.fred_session import latest_pair

    monkeypatch.setenv('LLMGE_BOUNDED_SESSION', '1')
    monkeypatch.setattr(controller, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(controller, 'GLOBAL_DATA_ANCESTRY', {})
    controller.save_checkpoint(0, tmp_path, tmp_path,
                               {'population': [], 'hof': controller.tools.ParetoFront()})
    population, _, global_state = latest_pair(tmp_path, tmp_path, 'dfine')
    assert population['selection_version'] == VERSION
    assert global_state['selection_version'] == VERSION


def test_parent_mechanism_reads_genome_and_audit(tmp_path):
    seed = candidate_runner.FRED_DIR / 'seeds' / 'seed_dfine.py'
    assert 'stock seed' in parent_mechanism(seed, seed)
    seed_copy = tmp_path / 'network_seed.py'
    seed_copy.write_bytes(seed.read_bytes())
    assert 'stock seed' in parent_mechanism(seed_copy, seed)
    parent = tmp_path / 'network_parent.py'
    parent.write_bytes(seed.read_bytes())
    (tmp_path / 'network_parent.py.genome.json').write_text(json.dumps(seed_genome()))
    (tmp_path / 'network_parent.py.proposal_attempts.json').write_text(json.dumps({
        'search_space_version': GENOME_VERSION, 'target_gene': 'encoder',
        'attempts': [{'accepted': True, 'genome': seed_genome()}]}))
    assert 'Last attempted mechanism: encoder=none' in parent_mechanism(parent, seed)


def test_cached_candidate_without_resource_evidence_is_excluded(selection_state):
    ind = controller.creator.Individual(['cached'])
    ind.fitness.values = controller.PLACEHOLDER_FITNESS
    controller.GLOBAL_DATA['cached'] = {
        'status': 'completed', 'sub_flag': True, 'fitness': (.6, .4, 1000)}
    controller.check_and_update_fitness([ind], loop_delay=0)
    assert controller.GLOBAL_DATA['cached']['status'] == 'FAILED: RESOURCE EVIDENCE'
    assert controller.valid_detection_population([ind]) == []


def test_controller_writes_measured_parent_feedback_into_audited_prompt(
    selection_state, tmp_path, monkeypatch,
):
    seed = candidate_runner.FRED_DIR / 'seeds' / 'seed_dfine.py'
    parent = tmp_path / 'network_parent.py'
    parent.write_bytes(seed.read_bytes())
    (tmp_path / 'network_parent.py.genome.json').write_text(json.dumps(seed_genome()))
    (tmp_path / 'network_parent.py.proposal_attempts.json').write_text(json.dumps({
        'search_space_version': GENOME_VERSION, 'target_gene': 'decoder',
        'attempts': [{'accepted': True, 'genome': seed_genome()}]}))
    controller.GLOBAL_DATA['parent'] = {
        'fitness': (.6, .4, 1000),
        'dfine_resources': {'inference_ms_per_frame': 3.0,
                            'peak_memory_bytes': 900000,
                            'diagnostics': {'mAP75': .3, 'mAP_small': .2,
                                            'recall_small': .25,
                                            'false_negatives_iou50': 12,
                                            'poorly_localized_iou50_to75': 4}}}
    monkeypatch.setattr(controller, 'SEED_NETWORK', str(seed))
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(controller, 'GENERATION', 0)
    monkeypatch.setattr(controller, 'GEN_COUNT', -1)
    monkeypatch.setattr(controller, 'MODEL', 'network')
    monkeypatch.setattr(controller, 'generate_template', lambda *args: ('Choose a value: {}', 'normal'))
    monkeypatch.setattr(controller, 'load_yaml', lambda: {
        'llm_bash_script': '{}', 'gpu_selection': ''})
    controller.write_bash_script('mixtral', 'test', input_filename_x=str(parent),
                                 output_filename=str(tmp_path / 'network_child.py'))
    prompt = (tmp_path / '0' / 'child_model.txt').read_text()
    assert 'mAP50:95=0.400000' in prompt
    assert 'AP_small=0.200000' in prompt
    assert 'Last attempted mechanism: decoder=none' in prompt


def test_evaluation_script_exports_frozen_resource_ceilings(
    selection_state, tmp_path, monkeypatch,
):
    monkeypatch.setattr(controller, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(controller, 'GENERATION', 0)
    monkeypatch.setattr(controller, 'RUN_COMMAND', 'sbatch')
    monkeypatch.setattr(controller, 'LOCAL', False)
    monkeypatch.setattr(controller, 'CHECKPOINT_FOLDER', None)
    monkeypatch.setattr(controller, 'EVAL_RUNLINE', 'python {} {}')
    monkeypatch.setattr(controller, 'load_yaml', lambda: {'python_bash_script': '{}'})
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='Submitted batch job 42', stderr=''))
    controller.GLOBAL_DATA['candidate'] = {'sub_flag': True, 'status': 'subbed file'}
    (tmp_path / '0').mkdir()

    controller.submit_run('candidate')

    script = (tmp_path / '0' / 'candidate_model.sh').read_text()
    assert 'export FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME=5.0' in script
    assert 'export FRED_DFINE_MAX_PEAK_MEMORY_BYTES=1000000' in script
