import sqlite3

from sota.FRED_LLM_GE.phase1_detection.adapters.yolo11 import selected_indices


def test_bringup_selects_annotated_frames_from_first_sequence(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    with sqlite3.connect(manifest) as connection:
        connection.executescript(
            "CREATE TABLE samples (sample_id TEXT, sequence_id TEXT, "
            "frame_index INTEGER, project_split TEXT);"
            "CREATE TABLE annotations (sample_id TEXT);"
        )
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?)",
            [("0:0", "0", 0, "train"), ("0:1", "0", 1, "train"),
             ("0:2", "0", 2, "train"), ("1:0", "1", 0, "train")],
        )
        connection.executemany(
            "INSERT INTO annotations VALUES (?)", [("0:1",), ("0:2",), ("1:0",)],
        )

    assert selected_indices(manifest, "train", 2) == {"0": [1, 2]}
