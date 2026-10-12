"""Local plumbing checks. Synthetic frames/provider responses are not FRED results."""

import argparse
import csv
import json
from pathlib import Path
import sys

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11"))
sys.path.insert(0, str(ROOT / "sota/FRED_LLM_GE/seeds"))

import train_ultralytics as runner
from seed_ultralytics_yolo11 import DroneDetector


@pytest.fixture(autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("scale", ["n", "m"])
def test_seed_matches_official_yolo11_exactly(scale):
    torch.manual_seed(0)
    official = DetectionModel(yaml_model_load(f"yolo11{scale}.yaml"), nc=1, verbose=False).eval()
    torch.manual_seed(0)
    seed = DroneDetector(scale=scale).eval()
    assert sum(p.numel() for p in official.parameters()) == sum(p.numel() for p in seed.parameters())
    with torch.no_grad():
        images = torch.rand(2, 3, 64, 96)
        expected = official(images)
        actual = seed(images)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for got, want in zip(actual[1], expected[1]):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_preflight_preserves_initial_weights_and_batch_statistics():
    model = DroneDetector(scale="n")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    assert runner.preflight(model, "cpu")["status"] == "passed"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def changed_neck():
    text = runner.SEED_FILE.read_text().split(runner.MARKER)[2]
    text = text.replace("self.layers = nn.ModuleList(layers)",
                        "self.layers = nn.ModuleList(layers)\n"
                        "        width = layers[5].cv2.conv.out_channels\n"
                        "        self.refine = Conv(width, width, 3)")
    return text.replace("medium = m[8]", "small = small + 0.1 * self.refine(small)\n        medium = m[8]")


def synthetic_data(sequences):
    boxes = [torch.tensor([[24., 20., 42., 34.]])] * 2
    return {"images": torch.randint(0, 256, (2, 3, 60, 96), dtype=torch.uint8),
            "boxes": boxes, "orig_boxes": boxes,
            "sample_ids": [f"synthetic/{sequences[0]}:{i}" for i in range(2)],
            "orig_size": (96, 60)}


def test_generated_code_training_csv_selection_and_resume(tmp_path, monkeypatch):
    # Exercise the real mutation/crossover operators with deterministic provider
    # responses. No real LLM service or remote FRED data is used in this test.
    from src import llm_mutation, llm_crossover
    import run_improved as evolution

    neck = changed_neck()
    monkeypatch.setattr(llm_mutation.np.random, "randint", lambda *a, **k: 2)
    # Exercise the real response parser, including an empty-block retry.
    import importlib
    llm_impl = importlib.import_module(llm_mutation.generate_augmented_code.__module__)
    responses = iter(["```python\n```", f"```python\n{neck.strip()}\n```", neck.strip()])
    monkeypatch.setattr(llm_impl, "get_llm_code_generator",
                        lambda *a: (lambda *a, **k: next(responses), None))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text((ROOT / "templates/FRED/Ultralytics/1.txt").read_text())
    child = tmp_path / "network_child.py"
    llm_mutation.augment_network(str(runner.SEED_FILE), str(child), str(prompt), llm_model="mixtral")
    crossed = tmp_path / "network_crossed.py"
    llm_crossover.augment_network(str(runner.SEED_FILE), str(child), str(crossed), llm_model="mixtral")
    for name in ("network_child", "network_crossed"):
        module, _ = runner.load_variant(name, tmp_path)
        model = module.DroneDetector(scale="n")
        assert runner.preflight(model, "cpu")["params"] > model.detector.reference_params

    monkeypatch.setattr(runner, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(runner.shared, "load_split", lambda split, sequences, *a: synthetic_data(sequences))
    population = []
    monkeypatch.setattr(evolution, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(evolution, "GLOBAL_DATA", {})
    monkeypatch.setattr(evolution, "GLOBAL_DATA_HIST", {})
    monkeypatch.setattr(evolution, "GLOBAL_DATA_ANCESTRY", {})
    for name, gene in (("network", "seed"), ("network_crossed", "crossed")):
        args = argparse.Namespace(model=name, variant_dir=tmp_path, scale="n", device="cpu",
                                  train_sequences=["5"], val_sequences=["1"], frame_step=200,
                                  img_width=96, epochs=1, batch_size=2, lr=1e-3, preflight_only=False)
        runner.main(args)
        report = json.loads((tmp_path / "results" / f"{gene}_evaluation.json").read_text())
        assert report["num_frames"] == 2
        assert report["metadata"]["training_config"]["initialization"] == "random-no-pretrained-weights"
        assert report["metadata"]["training_config"]["loss"] == "v8DetectionLoss"
        row = next(csv.reader((tmp_path / "results" / f"{gene}_results.csv").open()))
        assert len(row) == 3
        evolution.GLOBAL_DATA[gene] = {"results_job": None, "local_output": "job done"}
        evolution.check4results(gene)
        individual = evolution.creator.Individual([gene])
        individual.fitness.values = evolution.GLOBAL_DATA[gene]["fitness"]
        population.append(individual)
    survivors = evolution.select_elites(population, 2)
    assert {p[0] for p in survivors} == {"seed", "crossed"}
    checkpoints, globals_dir = tmp_path / "checkpoints", tmp_path / "global"
    evolution.save_checkpoint(1, str(checkpoints), str(globals_dir), {"population": survivors, "hof": survivors})
    loaded, next_generation, _ = evolution.load_checkpoint(str(checkpoints), global_path=str(globals_dir))
    assert next_generation == 2
    assert [p[0] for p in loaded["population"]] == [p[0] for p in survivors]


@pytest.mark.parametrize("change", ["protected", "unchanged", "bad-shape"])
def test_invalid_candidates_fail_before_training(tmp_path, change):
    source = runner.SEED_FILE.read_text()
    if change == "protected":
        source = source.replace("iou_thres=0.5", "iou_thres=0.9")
    elif change == "bad-shape":
        source = source.replace("return [small, medium, large]", "return [small[:, :1], medium, large]")
    (tmp_path / "network_invalid.py").write_text(source)
    with pytest.raises(ValueError):
        module, _ = runner.load_variant("network_invalid", tmp_path)
        runner.preflight(module.DroneDetector(scale="n"), "cpu")


def test_failed_rerun_removes_stale_fitness_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SCRIPT_DIR", tmp_path)
    (tmp_path / "results").mkdir()
    stale = tmp_path / "results/missing_results.csv"
    stale.write_text("1,1,1\n")
    args = argparse.Namespace(model="network_missing", variant_dir=tmp_path,
                              scale="n", device="cpu", preflight_only=False)
    with pytest.raises(FileNotFoundError):
        runner.main(args)
    assert not stale.exists()
    failure = json.loads((tmp_path / "results/missing_failure.json").read_text())
    assert "Traceback" in failure["error"]


def test_padding_and_empty_targets_are_valid():
    data = synthetic_data(["5"])
    data["boxes"] = [torch.empty(0, 4), torch.empty(0, 4)]
    batch = runner.training_batch(data, torch.tensor([0, 1]), "cpu")
    assert batch["img"].shape == (2, 3, 64, 96)
    assert batch["bboxes"].shape == (0, 4)
    model = DroneDetector(scale="n")
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss).all()


def test_legacy_small_seed_still_trains_and_predicts():
    from seed_yolo11 import DroneDetector as LegacyDetector
    model = LegacyDetector()
    data = synthetic_data(["5"])
    stride = runner.shared.model_stride(model, data["images"].float() / 255)
    model.output_stride = stride
    runner.shared.train(model, data, 1, 2, 1e-3, "cpu", stride)
    frames = runner.shared.predict_frames(model, data, 2, "cpu")
    assert len(frames) == 2
    assert [f["sample_id"] for f in frames] == data["sample_ids"]


def test_cli_rejects_sequences_outside_existing_training_cohort(monkeypatch):
    # Sequence 36 was identified by the team as challenging-test; no search access.
    monkeypatch.setattr(sys, "argv", ["train_ultralytics.py", "--train_sequences", "36"])
    with pytest.raises(SystemExit):
        runner.parse_args()
