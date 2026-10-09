"""Local interface checks for the first FRED YOLO11 bring-up path."""

import contextlib
import ast
import csv
import importlib.util
import io
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from contextlib import closing
from unittest.mock import patch

from sota.FRED_LLM_GE.data.yolo_export import (
    export_subset, selected_indices, yolo_labels,
)


SEED_DIR = Path(__file__).resolve().parents[1] / "seeds" / "yolo11"
sys.path.insert(0, str(SEED_DIR))
from validator import load_model_module, validate_source  # noqa: E402
import network  # noqa: E402


def fake_yolo11_config():
    backbone = [[-1, 1, "Conv", [64]] for _ in range(11)]
    head = [[-1, 1, "Conv", [64]] for _ in range(13)]
    backbone[4] = [-1, 2, "C3k2", [512, False]]
    backbone[10] = [-1, 2, "C2PSA", [1024]]
    head[5] = [-1, 2, "C3k2", [256, False]]
    head[-1] = [[16, 19, 22], 1, "Detect", [1]]
    return {"backbone": backbone, "head": head, "nc": 80, "scale": "m"}


class Yolo11InterfaceTest(unittest.TestCase):
    def test_llm_ge_configuration_and_command_construction(self):
        from src.cfg import constants

        self.assertEqual(constants.FITNESS_WEIGHTS, (1.0, 1.0, -1.0))
        self.assertEqual(constants.num_generations, 2)
        self.assertEqual(constants.crossover_probability, 0.0)
        command = constants.EVAL_RUNLINE.format(
            constants.TRAIN_FILE,
            constants.RUNLINE_TMP.format(constants.MODEL, "ABC"),
            VARIANT_DIR=constants.VARIANT_DIR,
        )
        self.assertIn("--model network_ABC", command)
        self.assertIn(f"--variant_dir {constants.VARIANT_DIR}", command)
        prompt = Path(constants.ROOT_DIR, "templates/FRED/Event/Normal/1.txt").read_text()
        self.assertEqual(prompt.count("{}"), 1)
        self.assertTrue(Path(constants.ROOT_DIR, constants.CONSTANT_RULES_PATH).is_file())

    def test_generated_variant_is_imported_after_literal_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            variant_dir = Path(temp)
            path = variant_dir / "network_ABC.py"
            source = (SEED_DIR / "network.py").read_text(encoding="utf-8")
            path.write_text(source.replace('"neck_p3": 2,', '"neck_p3": 3,'), encoding="utf-8")
            self.assertEqual(validate_source(path)["neck_p3"], 3)
            module = load_model_module("network_ABC", variant_dir)
            base = fake_yolo11_config()
            candidate = module.build_config(base)
            self.assertEqual(candidate["head"][5][1], 3)
            self.assertEqual(base["head"][5][1], 2)
            self.assertEqual(candidate["nc"], 1)
            path.write_text(path.read_text(encoding="utf-8") + "\nprint('unsafe')\n")
            with self.assertRaises(ValueError):
                load_model_module("network_ABC", variant_dir)

    def test_genome_rejects_protected_and_out_of_range_changes(self):
        base = fake_yolo11_config()
        with self.assertRaises(ValueError):
            network.build_config(base, {"backbone_p3": 100, "attention": 2, "neck_p3": 2})
        base["head"][-1][2] = "Segment"
        with self.assertRaises(ValueError):
            network.build_config(base)

    def test_phase0_box_conversion_and_invalid_box(self):
        sample = types.SimpleNamespace(
            event=types.SimpleNamespace(size=(100, 50)),
            annotations=[types.SimpleNamespace(box_xyxy=(10, 5, 30, 15))],
        )
        self.assertEqual(yolo_labels(sample), "0 0.20000000 0.20000000 0.20000000 0.20000000\n")
        sample.annotations[0].box_xyxy = (-1, 5, 30, 15)
        with self.assertRaises(ValueError):
            yolo_labels(sample)

    def test_export_selection_is_bounded_and_split_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.sqlite"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE samples (sample_id TEXT, sequence_id TEXT, frame_index INTEGER, project_split TEXT, event_width INTEGER, event_height INTEGER)")
                db.execute("CREATE TABLE annotations (sample_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL)")
                db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, 100, 50)", [
                    ("2:2", "2", 2, "train"), ("1:1", "1", 1, "train"),
                    ("1:2", "1", 2, "train"), ("3:1", "3", 1, "validation"),
                ])
                db.executemany("INSERT INTO annotations VALUES (?, 10, 5, 30, 15)",
                               [("2:2",), ("1:1",), ("1:2",), ("3:1",)])
                db.commit()
            self.assertEqual(dict(selected_indices(path, "train", 2)), {"1": [0, 1]})
            self.assertEqual(dict(selected_indices(path, "train", 3)), {"1": [0, 1]})
            self.assertEqual(dict(selected_indices(path, "validation", 1)), {"3": [0]})

    def test_phase0_export_uses_prepared_sequences_and_writes_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            manifest = temp / "manifest.sqlite"
            inventory_path = temp / "inventory.json"
            inventory_path.write_text("{}", encoding="utf-8")
            with closing(sqlite3.connect(manifest)) as db:
                db.execute("CREATE TABLE samples (sample_id TEXT, sequence_id TEXT, frame_index INTEGER, project_split TEXT, event_width INTEGER, event_height INTEGER)")
                db.execute("CREATE TABLE annotations (sample_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL)")
                db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, 100, 50)",
                               [("1:1", "1", 1, "train"), ("2:1", "2", 1, "validation")])
                db.executemany("INSERT INTO annotations VALUES (?, 10, 5, 30, 15)",
                               [("1:1",), ("2:1",)])
                db.commit()
            prepared = []
            active = []
            case = self

            class FakeImage:
                size = (100, 50)
                def convert(self, mode):
                    case.assertEqual(mode, "RGB")
                    return self
                def save(self, path):
                    path.write_bytes(b"synthetic image")

            image = FakeImage()
            sample = lambda sequence: types.SimpleNamespace(
                sequence_id=sequence, frame_index=1, event=image,
                annotations=[types.SimpleNamespace(box_xyxy=(10, 5, 30, 15))],
            )

            class FakeDataset:
                def __init__(self, **kwargs):
                    self.split = str(kwargs["project_split"])
                def __getitem__(self, index):
                    case.assertEqual(index, 0)
                    return sample("1" if self.split == "train" else "2")
                def close(self):
                    pass

            class FakeSource:
                def __init__(self, config):
                    pass
                def prepare_sequence(self, sequence, record):
                    prepared.append(sequence)
                @contextlib.contextmanager
                def activate_prepared_window(self, records):
                    active.append(records[0][0])
                    yield

            class FakeInventory:
                dataset_revision = "rev"
                inventory_hash = "hash"
                def by_sequence(self):
                    return {"1": object(), "2": object()}

            def stub(name, **attributes):
                module = types.ModuleType(name)
                module.__dict__.update(attributes)
                return module

            prefix = "sota.FRED_LLM_GE.archive.phase0_data_old."
            modules = {
                prefix + "config": stub(prefix + "config", load_config=lambda *a, **k: object()),
                prefix + "dataset": stub(prefix + "dataset", FREDDataset=FakeDataset),
                prefix + "fred_api": stub(prefix + "fred_api", HFFredSource=FakeSource),
                prefix + "inventory": stub(prefix + "inventory", load_inventory=lambda p: FakeInventory()),
                prefix + "manifest": stub(prefix + "manifest", read_manifest_metadata=lambda p: {
                    "project_split_approval_reference": "approved", "dataset_revision": "rev",
                    "annotation_policy": "coordinates.txt_unmodified_v1"}),
                prefix + "schema": stub(prefix + "schema", AccessMode=types.SimpleNamespace(
                    TRAIN="train", TRUSTED_EVALUATOR="trusted_evaluator"),
                    Modality=types.SimpleNamespace(EVENT="event"), ProjectSplit=str),
            }
            with patch.dict(sys.modules, modules):
                result = export_subset(manifest, inventory_path, temp / "export", 1, 1)
            self.assertEqual(prepared, ["1", "2"])
            self.assertEqual(active, ["1", "2"])
            self.assertTrue(result.is_file())
            self.assertTrue((temp / "export" / "images" / "val" / "2_00000001.png").is_file())
            self.assertEqual((temp / "export" / "labels" / "train" / "1_00000001.txt").read_text(),
                             "0 0.20000000 0.20000000 0.20000000 0.20000000\n")



if __name__ == "__main__":
    unittest.main()
