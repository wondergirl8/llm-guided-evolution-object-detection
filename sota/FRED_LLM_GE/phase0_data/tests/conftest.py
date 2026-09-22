from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sequence_root(tmp_path: Path) -> Path:
    root = tmp_path / "0"
    (root / "RGB").mkdir(parents=True)
    (root / "Event" / "Frames").mkdir(parents=True)
    for index in range(2):
        rgb_name = f"Video_0_00_00_00.{index * 33333:06d}.jpg"
        event_name = f"Video_0_frame_{(index + 1) * 33333}.png"
        Image.new("RGB", (32, 24), "black").save(root / "RGB" / rgb_name)
        Image.new("RGB", (32, 24), "white").save(
            root / "Event" / "Frames" / event_name
        )
    (root / "coordinates.txt").write_text(
        "0.033333: 1, 2, 10, 12, 7, test drone\n"
        "0.066666: 4, 5, 15, 16, 7, test drone\n",
        encoding="utf-8",
    )
    return root
