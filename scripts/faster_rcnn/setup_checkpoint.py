"""Explicit, atomic download of the selected official COCO checkpoint."""
import hashlib
import os
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.network import CHECKPOINT_URL
path = Path(__file__).resolve().parents[2] / "sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/checkpoints/coco.pth"
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
        if not hashlib.sha256(temporary.read_bytes()).hexdigest().startswith("047c8118"):
            raise ValueError("Official checkpoint checksum mismatch")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
if not hashlib.sha256(path.read_bytes()).hexdigest().startswith("047c8118"):
    raise ValueError("Cached checkpoint checksum mismatch")
print(path)
