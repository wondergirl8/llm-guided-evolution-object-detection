import sqlite3

from sota.FRED_LLM_GE.phase1_detection.adapters.yolo11 import selected_indices


def test_bringup_selects_only_in_bounds_annotated_frames(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    with sqlite3.connect(manifest) as connection:
        connection.executescript(
            "CREATE TABLE samples (sample_id TEXT, sequence_id TEXT, "
            "frame_index INTEGER, project_split TEXT, event_width INTEGER, event_height INTEGER);"
            "CREATE TABLE annotations (sample_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL);"
        )
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?, 1280, 720)",
            [("0:0", "0", 0, "train"), ("0:1", "0", 1, "train"),
             ("0:2", "0", 2, "train"), ("0:3", "0", 3, "train"),
             ("1:0", "1", 0, "train")],
        )
        connection.executemany(
            "INSERT INTO annotations VALUES (?, ?, ?, ?, ?)",
            [("0:1", 1, 1, 20, 20), ("0:2", 1, 700, 20, 725),
             ("0:3", 3, 3, 21, 21), ("1:0", 4, 4, 22, 22)],
        )

    assert selected_indices(manifest, "train", 2) == {"0": [1, 3]}
