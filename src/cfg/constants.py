"""Shared configuration: FRED defaults, with explicit MuJoCo selection.

Set LLMGE_CONFIG=mujoco before starting Python to select the MuJoCo profile.
Child processes inherit that selection; no checkout-wide symlink edit is needed.
"""

import importlib as _importlib
import os as _os

_profiles = {
    "mujoco": "constants_Mujoco",
    "fred-yolo11": "constants_fred_yolo11",
}
_profile = _os.environ.get("LLMGE_CONFIG", "fred-yolo11")
if _profile not in _profiles:
    raise ValueError(
        f"Unknown LLMGE_CONFIG {_profile!r}; choose mujoco or fred-yolo11"
    )
_config = _importlib.import_module(f".{_profiles[_profile]}", __package__)
__all__ = [name for name in vars(_config) if not name.startswith("_")]
globals().update({name: getattr(_config, name) for name in __all__})
