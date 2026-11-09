"""Verify and activate the pinned official Gen1 fork in a dedicated process."""
import hashlib
import json
import os
import tempfile
import shutil
import atexit
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "third_party/spikeyolo"


def verify_source():
    record = json.loads((SOURCE / "UPSTREAM.json").read_text())
    local_files = {"UPSTREAM.json", "INTEGRATION.md", "compat.patch"}
    actual = {str(path.relative_to(SOURCE)) for path in SOURCE.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts
              and str(path.relative_to(SOURCE)) not in local_files}
    if actual != set(record["files"]):
        raise ValueError("SpikeYOLO source snapshot has missing or additional files")
    for name, expected in record["files"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"SpikeYOLO source snapshot changed: {name}")
    return {key: record[key] for key in ("repository", "commit", "subdirectory", "model_config", "scale", "license")}


def activate():
    """Never replace an already imported PyPI Ultralytics in the current process."""
    verify_source()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if "ultralytics" not in sys.modules:
        # The old fork has a different settings schema. Never rewrite the
        # user's PyPI Ultralytics settings; use a private disposable config.
        config = Path(tempfile.mkdtemp(prefix="fred_spikeyolo_config_"))
        os.environ["YOLO_CONFIG_DIR"] = str(config)
        atexit.register(shutil.rmtree, config, ignore_errors=True)
        # Dataset checks require a font even with plots=false. Reuse the
        # installed Matplotlib font and keep CPU smoke checks offline.
        import matplotlib
        font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        shutil.copyfile(font, config / "Arial.ttf")
    for name, root in (("ultralytics", SOURCE),
                       ("spikingjelly", SOURCE / "spikingjelly-0.0.0.0.12")):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"{name} already imported from another source; use a fresh SpikeYOLO process")
        sys.path.insert(0, str(root))


def build_official(config, device="cpu"):
    activate()
    from ultralytics.nn.tasks import DetectionModel
    from spikingjelly.clock_driven.functional import reset_net

    model = DetectionModel(config, ch=3, nc=1, verbose=False).to(device)
    # The official constructor probes strides with a dummy forward.
    reset_net(model)
    return model
