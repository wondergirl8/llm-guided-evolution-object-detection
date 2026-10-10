"""Configuration routing regression checks, without a GPU or LLM server."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
PROBE = """
import importlib
import os
import sys
import types
from pathlib import Path

# Only the existing MuJoCo device check needs torch. Model execution is not tested.
sys.modules['torch'] = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False))
sys.path.insert(0, str(Path.cwd() / 'src'))
target = ('constants_fred_yolo11' if os.environ.get('LLMGE_CONFIG', 'fred-yolo11') == 'fred-yolo11'
          else 'constants_Mujoco')
for namespace in ('src.cfg', 'cfg'):
    selected = importlib.import_module(f'{namespace}.constants')
    direct = importlib.import_module(f'{namespace}.{target}')
    public = {name for name in vars(direct) if not name.startswith('_')}
    assert set(selected.__all__) == public
    assert all(getattr(selected, name) is getattr(direct, name) for name in public)
    exported = {}
    exec(f'from {namespace}.constants import *', exported)
    assert public == set(exported) - {'__builtins__'}
    assert selected.DEFAULT_PROMPT_GROUP == (
        'FRED/Event/Normal' if target == 'constants_fred_yolo11' else 'Mujoco/Normal')
    assert len(selected.FITNESS_WEIGHTS) == (
        3 if target == 'constants_fred_yolo11' else 2)
if target == 'constants_fred_yolo11':
    # FRED must not import the other domain's dependencies.
    assert not any(name.endswith('constants_Mujoco') for name in sys.modules)
    import subprocess
    subprocess.run([sys.executable, '-c',
        'from src.cfg import constants; '
        'assert constants.DEFAULT_PROMPT_GROUP == "FRED/Event/Normal"'], check=True)
"""


@pytest.mark.parametrize("profile", [None, "mujoco", "fred-yolo11"])
def test_config_exports_and_both_import_namespaces(profile):
    environment = os.environ.copy()
    environment.pop("LLMGE_CONFIG", None)
    if profile is not None:
        environment["LLMGE_CONFIG"] = profile
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=ROOT, env=environment,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("profile", ["", "fred-yolo1", "os"])
def test_unknown_config_fails_before_loading_a_profile(profile):
    result = subprocess.run(
        [sys.executable, "-c", "from src.cfg import constants"], cwd=ROOT,
        env={**os.environ, "LLMGE_CONFIG": profile}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Unknown LLMGE_CONFIG" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
