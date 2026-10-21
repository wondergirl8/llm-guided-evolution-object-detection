"""Structural checks that do not require an Ultralytics installation."""

from copy import deepcopy

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection.yolo11_search_space import (
    SEED_YAML, changed_layer_count, evaluation_cache_key, load_and_validate,
    repair_channel_values, retry_prompt,
)


def _validate(tmp_path, change):
    candidate = deepcopy(yaml.safe_load(SEED_YAML.read_text()))
    change(candidate)
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(candidate, sort_keys=False))
    return load_and_validate(path)


def test_seed_is_valid():
    assert load_and_validate(SEED_YAML)["head"][-1][2] == "Detect"


@pytest.mark.parametrize("change", [
    lambda c: c["backbone"].pop(),
    lambda c: c["head"][1][0].append(3),
    lambda c: c["head"][1].__setitem__(2, "Conv"),
    lambda c: c["backbone"][0].__setitem__(1, 5),
    lambda c: c["backbone"][0][3].__setitem__(0, 65),
    lambda c: c["backbone"][0][3].__setitem__(0, 192),
    lambda c: c["backbone"][0][3].__setitem__(2, 1),
    lambda c: c["head"][-1].__setitem__(2, "C3k2"),
    lambda c: c.__setitem__("nc", 80),
])
def test_rejects_changes_outside_search_space(tmp_path, change):
    with pytest.raises(ValueError):
        _validate(tmp_path, change)


def test_accepts_allowed_width_change(tmp_path):
    candidate = _validate(tmp_path, lambda c: c["backbone"][0][3].__setitem__(0, 128))
    assert candidate["backbone"][0][3][0] == 128


def test_retry_prompt_contains_previous_validation_error():
    assert retry_prompt('original', None, 0) == 'original'
    retry = retry_prompt('original', 'layer 3 channels must be a multiple of 64', 1)
    assert 'layer 3 channels must be a multiple of 64' in retry
    assert 'exactly one mutable layer' in retry


def test_channel_repair_is_local_and_still_validated(tmp_path):
    seed = yaml.safe_load(SEED_YAML.read_text())
    seed['backbone'][0][3][0] = 70
    chunk = yaml.safe_dump({'backbone': seed['backbone']}, sort_keys=False)
    repaired = repair_channel_values(chunk)
    assert yaml.safe_load(repaired)['backbone'][0][3][0] == 64
    candidate = yaml.safe_load(SEED_YAML.read_text())
    candidate['backbone'] = yaml.safe_load(repaired)['backbone']
    path = tmp_path / 'candidate.yaml'
    path.write_text(yaml.safe_dump(candidate, sort_keys=False))
    assert load_and_validate(path)['backbone'][0][3][0] == 64


def test_mutation_layer_count_detects_multiple_edits(tmp_path):
    candidate = yaml.safe_load(SEED_YAML.read_text())
    candidate['backbone'][0][3][0] = 128
    candidate['backbone'][1][3][0] = 64
    path = tmp_path / 'candidate.yaml'
    path.write_text(yaml.safe_dump(candidate, sort_keys=False))
    assert changed_layer_count(path, SEED_YAML) == 2


def test_cache_key_separates_event_variants():
    released = {'sha256': 'same-protocol', 'evolution_settings': {'fred_events': ''}}
    raw = {'sha256': 'same-protocol',
           'evolution_settings': {'fred_events': 'raw_w10000_polarity'}}
    assert evaluation_cache_key('same-built-graph', released) != evaluation_cache_key(
        'same-built-graph', raw)
