"""Structural checks that do not require an Ultralytics installation."""

from copy import deepcopy

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection.yolo11_search_space import SEED_YAML, load_and_validate


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
