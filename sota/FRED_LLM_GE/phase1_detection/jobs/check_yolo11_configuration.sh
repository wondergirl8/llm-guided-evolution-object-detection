#!/bin/bash
# Read-only configuration check on ICE; no GPU or training job is needed.
set -euo pipefail
test "$(git branch --show-current)" = fred-yolo11-infrastructure
git diff --quiet
git diff --cached --quiet
test ! -e "$(git rev-parse --git-path MERGE_HEAD)"
PY=data/.venv-yolo11/bin/python
test -x "$PY" || { echo "Missing existing YOLO11 environment: $PY" >&2; exit 1; }

for PROFILE in mujoco fred-yolo11; do
  LLMGE_CONFIG="$PROFILE" "$PY" - <<'PY'
import importlib
import os
import sys
from pathlib import Path

profile = os.environ['LLMGE_CONFIG']
target = 'constants_Mujoco' if profile == 'mujoco' else 'constants_fred_yolo11'
sys.path.insert(0, str(Path.cwd() / 'src'))
for namespace in ('src.cfg', 'cfg'):
    selected = importlib.import_module(f'{namespace}.constants')
    direct = importlib.import_module(f'{namespace}.{target}')
    public = {name for name in vars(direct) if not name.startswith('_')}
    assert set(selected.__all__) == public, 'Configuration exports changed'
    assert all(getattr(selected, name) is getattr(direct, name) for name in public)
    assert len(selected.FITNESS_WEIGHTS) == (2 if profile == 'mujoco' else 3)
    assert selected.DEFAULT_PROMPT_GROUP == (
        'Mujoco/Normal' if profile == 'mujoco' else 'FRED/Event/Normal')
print('PROFILE OK:', profile)
PY
done

env -u LLMGE_CONFIG "$PY" - <<'PY'
from src.cfg import constants
assert constants.DEFAULT_PROMPT_GROUP == 'FRED/Event/Normal', 'Default changed'
assert constants.FITNESS_WEIGHTS == (1.0, 1.0, -1.0)
print('DEFAULT OK: fred-yolo11')
PY
printf 'CONFIGURATION CHECK COMPLETE; no training submitted\n'
