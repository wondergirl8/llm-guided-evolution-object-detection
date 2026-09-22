from pathlib import Path

import pytest
from PIL import Image

from sota.FRED_LLM_GE.phase0_data.schema import Annotation
from sota.FRED_LLM_GE.phase0_data.visualization import (
    HeldOutVisualizationError,
    render_annotation_overlay,
)


def test_held_out_overlay_is_prohibited(tmp_path: Path):
    with pytest.raises(HeldOutVisualizationError):
        render_annotation_overlay(
            Image.new("RGB", (20, 20)),
            (Annotation("0.1", (1, 1, 5, 5), 1, "drone"),),
            project_split="held_out_test",
            destination=tmp_path / "forbidden.png",
        )


def test_development_overlay_is_written_atomically(tmp_path: Path):
    destination = tmp_path / "overlay.png"
    render_annotation_overlay(
        Image.new("RGB", (20, 20)),
        (Annotation("0.1", (1, 1, 5, 5), 1, "drone"),),
        project_split="validation",
        destination=destination,
    )
    assert destination.is_file()
