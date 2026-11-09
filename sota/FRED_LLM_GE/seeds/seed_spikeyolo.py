"""Compatibility builder for the stock v2 YAML; this file is not an LLM genome."""
from pathlib import Path

from sota.FRED_LLM_GE.phase1_detection.spikeyolo_search_space import build


def build_model(num_classes=1):
    if num_classes != 1:
        raise ValueError("FRED SpikeYOLO requires the single drone class")
    return build(Path(__file__).with_suffix(".yaml"))
