"""Versioned D-FINE search selection inputs, separate from baseline evaluation."""

import json
import math
import os
from pathlib import Path


VERSION = 'map5095_resource_gated_v1'


def resource_limits(environment=None):
    """Require calibrated absolute ceilings before a resource-gated search."""
    environment = os.environ if environment is None else environment
    latency_name = 'FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME'
    memory_name = 'FRED_DFINE_MAX_PEAK_MEMORY_BYTES'
    latency_raw, memory_raw = environment.get(latency_name), environment.get(memory_name)
    for name, raw in ((latency_name, latency_raw), (memory_name, memory_raw)):
        if raw is None or not raw.strip():
            raise ValueError(f'{name} is required for D-FINE motifs_v1 selection')
    try:
        latency = float(latency_raw)
    except ValueError as exc:
        raise ValueError(f'{latency_name} must be a finite positive number') from exc
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError(f'{latency_name} must be a finite positive number')
    if not memory_raw.isdecimal() or int(memory_raw) <= 0:
        raise ValueError(f'{memory_name} must be a positive integer number of bytes')
    return {'max_inference_ms_per_frame': latency,
            'max_peak_memory_bytes': int(memory_raw)}


def validated_metrics(report_path, gene_id, fitness, limits):
    """Match the search report to its CSV before accepting resource evidence."""
    report = json.loads(Path(report_path).read_text())
    if not isinstance(report, dict):
        raise ValueError('D-FINE report must be an object')
    if (report.get('schema_version') != 'fred-detection-report-v1'
            or report.get('gene_id') != gene_id or report.get('mode') != 'validation'):
        raise ValueError(f'D-FINE report identity mismatch: {report_path}')
    metrics = report.get('metrics', {})
    if not isinstance(metrics, dict):
        raise ValueError('D-FINE report metrics must be an object')
    for name, expected in zip(('mAP50', 'mAP50_95', 'params'), fitness):
        value = metrics.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or abs(value - expected) > 1e-6):
            raise ValueError(f'D-FINE report {name} disagrees with result CSV')
    metadata = report.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('D-FINE report metadata must be an object')
    if (metadata.get('project_split') != 'validation'
            or metadata.get('official_split') != 'challenging_train'
            or metadata.get('resource_measurement_version') != VERSION):
        raise ValueError('D-FINE report lacks matching search resource measurements')
    if metadata.get('selection_resource_limits') != limits:
        raise ValueError('D-FINE report resource ceilings disagree with this search')
    latency = metadata.get('preloaded_inference_ms_per_frame')
    probe = metadata.get('search_resource_probe')
    probe_peak = (probe.get('full_resolution_probe_peak_bytes')
                  if isinstance(probe, dict) else None)
    peak = metadata.get('training_peak_memory_bytes')
    probe_batch = probe.get('full_resolution_probe_batch') if isinstance(probe, dict) else None
    probe_shape = probe.get('full_resolution_probe_shape') if isinstance(probe, dict) else None
    if (isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency <= 0
            or isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0
            or isinstance(probe_peak, bool) or not isinstance(probe_peak, int)
            or probe_peak <= 0 or isinstance(probe_batch, bool)
            or not isinstance(probe_batch, int) or probe_batch <= 0
            or not isinstance(probe_shape, list) or len(probe_shape) != 2
            or any(isinstance(side, bool) or not isinstance(side, int) or side < 32
                   for side in probe_shape)
            or not str(metadata.get('inference_device', '')).startswith('cuda')):
        raise ValueError('D-FINE report has missing or invalid latency/peak memory')
    details = report.get('diagnostics')
    overall = details.get('overall') if isinstance(details, dict) else None
    if not isinstance(overall, dict):
        raise ValueError('D-FINE search report is missing validation diagnostics')
    diagnostics = {}
    for name in ('mAP75', 'mAP_small', 'recall_small'):
        if name not in overall:
            raise ValueError(f'D-FINE diagnostic {name} is missing')
        value = overall.get(name)
        if (value is not None and
                (isinstance(value, bool) or not isinstance(value, (int, float))
                 or not math.isfinite(value) or not 0 <= value <= 1)):
            raise ValueError(f'D-FINE diagnostic {name} is invalid')
        diagnostics[name] = value
    for name in ('false_negatives_iou50', 'poorly_localized_iou50_to75'):
        value = overall.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'D-FINE diagnostic {name} is invalid')
        diagnostics[name] = value
    return {'inference_ms_per_frame': float(latency), 'peak_memory_bytes': peak,
            'probe_peak_memory_bytes': probe_peak,
            'report_path': str(report_path), 'inference_device': metadata.get('inference_device'),
            'probe_batch': probe_batch,
            'probe_shape': probe_shape,
            'diagnostics': diagnostics}


def resource_violation(metrics, limits):
    latency, peak = metrics['inference_ms_per_frame'], metrics['peak_memory_bytes']
    if (isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency <= 0
            or isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0):
        raise ValueError('invalid D-FINE resource measurements')
    if latency > limits['max_inference_ms_per_frame']:
        return (f"preloaded inference {metrics['inference_ms_per_frame']:.3f} ms/frame "
                f"exceeds {limits['max_inference_ms_per_frame']:.3f} ms/frame")
    if peak > limits['max_peak_memory_bytes']:
        return (f"training peak {metrics['peak_memory_bytes']} bytes "
                f"exceeds {limits['max_peak_memory_bytes']} bytes")
    return None


def parent_mechanism(parent_path, seed_path):
    """Read the already validated genome and its proposal audit for feedback."""
    from src.fred_dfine_genome import GENES, VERSION as GENOME_VERSION, load_genome

    genome = load_genome(parent_path, seed_path)
    genes = genome['genes']
    values = ', '.join(f'{name}={genes[name]}' for name in GENES)
    audit_path = Path(str(parent_path) + '.proposal_attempts.json')
    if not audit_path.exists():
        if Path(parent_path).read_bytes() != Path(seed_path).read_bytes():
            raise ValueError(f'Missing D-FINE parent proposal audit: {audit_path}')
        return f'Current mechanisms: {values}. Last attempted mechanism: stock seed.'
    audit = json.loads(audit_path.read_text())
    if not isinstance(audit, dict):
        raise ValueError(f'Invalid D-FINE parent proposal audit: {audit_path}')
    target = audit.get('target_gene')
    if audit.get('search_space_version') != GENOME_VERSION or target not in GENES:
        raise ValueError(f'Invalid D-FINE parent proposal audit: {audit_path}')
    if audit.get('operator') == 'semantic_crossover':
        if audit.get('genome') != genome:
            raise ValueError(f'D-FINE crossover audit disagrees with genome: {audit_path}')
    else:
        attempts = audit.get('attempts')
        accepted = ([entry for entry in attempts if isinstance(entry, dict)
                     and entry.get('accepted')] if isinstance(attempts, list) else [])
        if len(accepted) != 1 or accepted[0].get('genome') != genome:
            raise ValueError(f'D-FINE mutation audit disagrees with genome: {audit_path}')
    return (f'Current mechanisms: {values}. Last attempted mechanism: '
            f'{target}={genes[target]} ({audit.get("operator", "mutation")}).')


def prompt_feedback(parent_id, parent_record, limits, mechanism=None):
    """Give the LLM the search goal and measured parent cost, not held-out data."""
    lines = [
        'Search goal: improve project-validation mAP50:95.',
        'Keep measured preloaded detector inference and training peak memory '
        'within the fixed eligibility limits.',
        f"Limits: inference <= {limits['max_inference_ms_per_frame']:.3f} ms/frame; "
        f"peak memory <= {limits['max_peak_memory_bytes']} bytes.",
        'Parameter count is a tie-breaker and archive metric, not the primary goal.',
    ]
    fitness = parent_record.get('fitness') if parent_record else None
    resources = parent_record.get('dfine_resources') if parent_record else None
    if (fitness and resources and len(fitness) == 3 and
            all(math.isfinite(float(value)) for value in fitness)):
        lines.append(
            f"Parent {parent_id}: validation mAP50:95={fitness[1]:.6f}, "
            f"mAP50={fitness[0]:.6f}, params={fitness[2]:.0f}, "
            f"inference={resources['inference_ms_per_frame']:.3f} ms/frame, "
            f"peak memory={resources['peak_memory_bytes']} bytes.")
        diagnostics = resources.get('diagnostics')
        if diagnostics:
            def shown(value):
                return 'undefined' if value is None else f'{value:.6f}'
            lines.append(
                f"Validation diagnostics: AP75={shown(diagnostics['mAP75'])}, "
                f"AP_small={shown(diagnostics['mAP_small'])}, "
                f"recall_small={shown(diagnostics['recall_small'])}, "
                f"missed at IoU .50={diagnostics['false_negatives_iou50']}, "
                f"localized at .50 but missed at .75="
                f"{diagnostics['poorly_localized_iou50_to75']}.")
    if mechanism:
        lines.append(mechanism)
    return '\n'.join(lines)
