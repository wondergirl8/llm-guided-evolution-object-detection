"""Family selection must keep generated models, results, and evolution state apart."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sota.FRED_LLM_GE.phase1_detection import candidate_runner


FAMILIES = ("yolo11", "yolo26_p2", "dfine", "tiny_yolo11")
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("family", FAMILIES)
def test_standalone_runner_uses_family_directories(family):
    with patch.object(sys, "argv", ["candidate_runner.py", "--family", family]), \
            patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    root = candidate_runner.FAMILY_SOTA_ROOTS[family]
    assert run.call_args.args[2] == str(root / "models")
    assert run.call_args.args[5] == str(root)


def test_variant_directory_follows_explicit_output_root(tmp_path):
    with patch.object(sys, "argv", ["candidate_runner.py", "--family", "dfine",
                                    "--sota_root", str(tmp_path)]), \
            patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    assert run.call_args.args[2] == str(tmp_path / "models")
    assert run.call_args.args[5] == str(tmp_path)


def test_evolution_paths_are_distinct_for_every_family():
    code = """import json
from src.cfg import constants as c
print(json.dumps({
    "seed": c.SEED_NETWORK, "root": c.SOTA_ROOT, "variant": c.VARIANT_DIR,
    "output": c.OUTPUT_DIR, "islands": c.ISLAND_CONTROLLER_RUN_NAME,
    "island_script": c.ISLAND_TEMP_SCRIPT, "rules": c.CONSTANT_RULES_PATH,
    "prompts": c.PROMPTS, "eval": c.EVAL_RUNLINE,
    "global_data": c.GLOBAL_DATA_PATH,
}))"""
    selected = {}
    for family in FAMILIES:
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "FRED_SEED_FAMILY": family},
                                capture_output=True, text=True, check=True)
        selected[family] = json.loads(result.stdout)

    for key in ("seed", "root", "variant", "output", "islands", "island_script",
                "rules", "prompts"):
        assert len({entry[key] for entry in selected.values()}) == len(FAMILIES), key
    for family, entry in selected.items():
        expected_root = candidate_runner.FAMILY_SOTA_ROOTS[family]
        if family == "yolo26_p2":
            # Fresh coverage-policy searches cannot mix models or reports with
            # the standalone seed and older YOLO26 runs.
            expected_root = expected_root.with_name("yolo26_p2_coverage_v1")
            assert f"--sota_root {expected_root}" in entry["eval"]
        assert Path(entry["root"]) == expected_root
        assert Path(entry["variant"]) == Path(entry["root"]) / "models"
        assert Path(entry["seed"]).is_file()
        assert (ROOT / entry["rules"]).is_file()
        assert family in entry["eval"]
        assert entry["global_data"] == "global_data"
