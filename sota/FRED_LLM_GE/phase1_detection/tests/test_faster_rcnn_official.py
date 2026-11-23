import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from sota.FRED_LLM_GE.phase1_detection import candidate_runner, faster_rcnn_search_space as policy
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import resized_data, preserving_preflight
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch import finalize
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import (
    pareto_front, valid_fitness, freeze_manifest)
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.validator import SEED, validate_source


def test_family_budget_does_not_change_other_families():
    original = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    unchanged = copy.deepcopy(original)
    original["proxy"]["seed"] = 2
    resolved = candidate_runner.family_protocol(original, "faster_rcnn_scratch")
    assert resolved["proxy"]["img_width"] == 640
    assert resolved["proxy"]["batch_size"] == 2
    assert resolved["proxy"]["seed"] == 2
    assert resolved["challenging"]["search"]["epochs"] == 3
    assert original["proxy"]["img_width"] == unchanged["proxy"]["img_width"]
    assert original["challenging"]["search"] == unchanged["challenging"]["search"]
    assert candidate_runner.family_protocol(original, "yolo11") is original
    assert candidate_runner.family_protocol(original, "dfine") is original


def test_pareto_dominance_ties_and_failures():
    records = {"a": {"fitness": (.9, .5, 20)}, "tie": {"fitness": (.9, .5, 20)},
               "small": {"fitness": (.8, .4, 10)}, "bad": {"fitness": (.8, .4, 30)},
               "failure": {"fitness": (0, 0, 999999999)}, "nan": {"fitness": (float("nan"), .4, 2)}}
    assert {g for g, _ in pareto_front(records)} == {"a", "tie", "small"}
    assert not valid_fitness((.4, .9, 5))
    with pytest.raises(ValueError):
        pareto_front({"a": {"fitness": None}})


@pytest.mark.parametrize("replacement", [
    'class FeatureGenes:\n    values = {"fpn_channels": 512}',
    'class FeatureGenes:\n    values = {"fpn_channels": True}',
    'class FeatureGenes:\n    values = dict(fpn_channels=256)',
    'class FeatureGenes:\n    values = {"fpn_channels": 128}\n    def run(self): pass',
])
def test_rejects_invalid_scratch_proposals(tmp_path, replacement):
    parts = SEED.read_text().split("# --OPTION--")
    parts[1] = "\n" + replacement + "\n"
    path = tmp_path / "candidate.py"
    path.write_text("# --OPTION--".join(parts))
    with pytest.raises(ValueError):
        validate_source(path)


def test_policy_rejects_protected_changes_and_noop(tmp_path, monkeypatch):
    def built(path):
        genes = validate_source(path)
        return hashlib.sha256(json.dumps(genes, sort_keys=True).encode()).hexdigest(), 10
    monkeypatch.setattr(policy, "built_architecture", built)
    path = tmp_path / "candidate.py"
    path.write_text(SEED.read_text())
    assert not policy.differs_from_parents(path, SEED)
    path.write_text(SEED.read_text().replace('"fpn_channels": 256', '"fpn_channels": 128'))
    assert policy.changed_layer_count(path, SEED) == 1
    assert policy.differs_from_parents(path, SEED)
    path.write_text("import os\n" + SEED.read_text())
    with pytest.raises(ValueError):
        policy.differs_from_parents(path, SEED)
    reasons = [json.loads(line)["reason"] for line in Path(str(path) + ".proposals.jsonl").read_text().splitlines()]
    assert reasons[:2] == ["unchanged_parent", "accepted"]
    assert reasons[2].startswith("invalid_architecture")


def test_cache_key_includes_protocol_and_seed():
    first = policy.evaluation_cache_key("graph", {"seed": 0, "protocol": "a"})
    assert first != policy.evaluation_cache_key("graph", {"seed": 1, "protocol": "a"})
    assert first != policy.evaluation_cache_key("graph", {"seed": 0, "protocol": "b"})
    assert first != policy.evaluation_cache_key("other", {"seed": 0, "protocol": "a"})


def test_family_prompts_and_response_parsing():
    root = Path("templates/FRED_FasterRCNNScratch")
    chunk = SEED.read_text().split("# --OPTION--")[1].strip()
    response = f"```python\n{chunk}\n```"
    assert policy.candidate_chunks(response) == [chunk + "\n"]
    assert not policy.candidate_chunks("no code")
    for name in ("crossover.txt", "crossover_s.txt"):
        rendered = (root / "CrossOver" / name).read_text().format(chunk, chunk)
        assert "Faster R-CNN" in rendered and "YOLO" not in rendered
    assert "unchanged_parent" in policy.retry_prompt("prompt", "unchanged_parent", 1)


def test_resizing_is_lazy_and_scales_only_training_boxes():
    class Images:
        shape = (3, 3, 72, 128)
        calls = []
        def __len__(self): return 3
        def __getitem__(self, indices):
            self.calls.append(indices.tolist())
            return torch.zeros((len(indices), 3, 72, 128), dtype=torch.uint8)
    images = Images()
    boxes = [torch.tensor([[12., 8., 24., 16.]])] * 3
    data = {"images": images, "boxes": boxes, "orig_boxes": boxes, "orig_size": (128, 72)}
    resized = resized_data(data, 64)
    assert not images.calls
    assert resized["images"][1].shape == (3, 36, 64)
    assert images.calls == [[1]]
    assert resized["boxes"][0].tolist() == [[6., 4., 12., 8.]]
    assert resized["orig_boxes"][0].tolist() == [[12., 8., 24., 16.]]


def test_preflight_restores_buffers_parameters_and_mode(monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.adapters import faster_rcnn_scratch as adapter
    model = torch.nn.BatchNorm2d(3).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def probe(model):
        model.train()
        model(torch.randn(2, 3, 4, 4)).sum().backward()
        with torch.no_grad(): model.weight.add_(3)
    monkeypatch.setattr(adapter, "preflight", probe)
    preserving_preflight(model)
    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_freeze_requires_matching_generation_identity(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete", "last_completed_generation": 20}))
    monkeypatch.setattr(finalize, "identity", lambda _: {"sha256": "new"})
    monkeypatch.setattr(finalize, "summarize", lambda *a, **k: {})
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "checkpoint_gen_20.pkl").write_bytes(b"exists")
    with (checkpoints / "global_gen_20.pkl").open("wb") as handle:
        pickle.dump({"protocol_fingerprint": {"sha256": "old"}}, handle)
    with pytest.raises(ValueError, match="provenance"):
        finalize.freeze_finalists(tmp_path)


def test_finalist_freeze_adds_control_and_is_immutable(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete", "last_completed_generation": 20}))
    monkeypatch.setattr(finalize, "identity", lambda _: {"sha256": "frozen"})
    baseline = {"runs": [{"mAP50": .7, "mAP50_95": .3, "params": 20}]}
    monkeypatch.setattr(finalize, "summarize", lambda *a, **k: baseline)
    monkeypatch.setattr(finalize, "built_architecture", lambda p: ("stock" if p == SEED else "child", 20))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "checkpoint_gen_20.pkl").write_bytes(b"exists")
    path = tmp_path / "candidates/models/network_child.py"
    path.parent.mkdir(parents=True)
    path.write_text(SEED.read_text())
    archive = {"protocol_fingerprint": {"sha256": "frozen"},
               "GLOBAL_DATA": {"child": {"fitness": (.9, .5, 20)}}}
    with (checkpoints / "global_gen_20.pkl").open("wb") as handle: pickle.dump(archive, handle)
    selected = finalize.freeze_finalists(tmp_path)
    assert [f["gene_id"] for f in selected["finalists"]] == ["child", "seed"]
    assert selected["finalists"][1]["stock_control"]
    assert not selected["finalists"][1]["pareto_front"]
    assert finalize.freeze_finalists(tmp_path) == selected
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen source"):
        finalize.freeze_finalists(tmp_path)


def test_final_evaluation_resumes_weights_and_test_sequences(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import state_sha256
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["data_root"] = str(tmp_path / "data")
    train_root = tmp_path / "data/final_train_step5"
    (train_root / "done").mkdir(parents=True)
    (train_root / "data.yaml").write_text("verified export")
    training = [(str(i), "train") for i in range(172)]
    test = [(str(i), "test") for i in range(172, 231)]
    for s, _ in training: (train_root / "done" / f"{s}.json").write_text("{}")
    monkeypatch.setattr(finalize, "load_protocol", lambda _: copy.deepcopy(protocol))
    monkeypatch.setattr(finalize, "dataset_parts", lambda *a: [(train_root, training, 5)])
    monkeypatch.setattr(finalize, "challenging_split", lambda _: (training, test))
    monkeypatch.setattr(finalize, "load_exported_split", lambda *a, **k: {"lazy": True})
    calls = {"train": 0, "predict": [], "report": 0}
    class Adapter:
        def __init__(self, *a): pass
        def build(self, source, classes, device):
            model = torch.nn.Linear(2, 2)
            self.initialization = {"initial_state_sha256": state_sha256(model)}
            return model
        def train(self, model, data, settings, device, **kwargs):
            assert data == {"lazy": True} and settings["epochs"] == 20
            calls["train"] += 1
        def postprocessing(self, settings): return "fixed"
    monkeypatch.setattr(finalize, "FasterRCNNScratchAdapter", Adapter)
    source = tmp_path / "final/sources/network_child.py"
    source.parent.mkdir(parents=True)
    source.write_text(SEED.read_text())
    finalist = {"gene_id": "child", "source": str(source.relative_to(tmp_path)),
                "source_sha256": finalize.file_sha256(source), "architecture_sha256": "graph", "params": 6}
    selection = {"training": protocol["families"]["faster_rcnn_scratch"]["final"],
                 "identity": {"sha256": "frozen", "git_commit": "commit"}}
    freeze_manifest(tmp_path / "final/finalists.json", selection)
    interrupted = [False]
    def predict(adapter, model, sequences, settings, device, output):
        sequence, folder = sequences[0]
        calls["predict"].append(sequence)
        if sequence == "174" and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("interrupted")
        sample_id = f"{folder}/{sequence}:0"
        frame = {"sample_id": sample_id, "target": {"boxes": [], "labels": []},
                 "prediction": {"boxes": [], "labels": [], "scores": []}, "width": 80, "height": 40}
        return [frame], [sample_id]
    monkeypatch.setattr(finalize, "predict_test_split", predict)
    def report(payload, *, sota_root, gene_id, mode):
        assert mode == "held-out" and len(payload["expected_sample_ids"]) == 59
        calls["report"] += 1
        value = {"metadata": payload["metadata"], "num_frames": 59,
                 "metrics": {"mAP50": .8, "mAP50_95": .4, "params": 6}}
        path = Path(sota_root) / "held_out_reports" / f"{gene_id}_evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return None, path
    monkeypatch.setattr(finalize.map_reporter, "report", report)
    with pytest.raises(RuntimeError, match="interrupted"):
        finalize.evaluate_finalist(tmp_path, selection, finalist, "cpu")
    result = finalize.evaluate_finalist(tmp_path, selection, finalist, "cpu")
    assert result["num_frames"] == 59 and calls["train"] == calls["report"] == 1
    assert calls["predict"].count("172") == calls["predict"].count("173") == 1
    before = copy.deepcopy(calls)
    assert finalize.evaluate_finalist(tmp_path, selection, finalist, "cpu") == result
    assert calls == before
    chunk = tmp_path / "final/child/predictions/172.json"
    altered = json.loads(chunk.read_text())
    altered["identity"]["checkpoint_sha256"] = "changed"
    chunk.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="incompatible provenance"):
        finalize.evaluate_finalist(tmp_path, selection, finalist, "cpu")


def test_epoch_resume_preserves_optimizer_and_random_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn import train_eval
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.5]))
        def forward(self, indices, mode):
            return ((self.weight - torch.rand(1)) ** 2).sum()
        def parse_losses(self, loss): return loss, {}
    monkeypatch.setattr(train_eval, "prepare_batch", lambda m, d, i, t: {"indices": i})
    args = SimpleNamespace(epochs=2, batch_size=2, lr=.01)
    data = {"images": torch.zeros(5, 3, 4, 4)}
    torch.manual_seed(42)
    full = Model()
    train_eval.train(full, data, args)
    torch.manual_seed(42)
    partial = Model()
    path = tmp_path / "epoch.pt"
    def save_then_interrupt(epoch, optimizer):
        torch.save({"state_dict": partial.state_dict(), "optimizer": optimizer.state_dict(),
                    "completed_epochs": epoch, "cpu_rng": torch.get_rng_state(), "cuda_rng": []}, path)
        raise RuntimeError("time limit")
    with pytest.raises(RuntimeError, match="time limit"):
        train_eval.train(partial, data, args, on_epoch=save_then_interrupt)
    saved = torch.load(path, weights_only=False)
    torch.manual_seed(999)
    resumed = Model()
    resumed.load_state_dict(saved["state_dict"])
    train_eval.train(resumed, data, args, resume_state=saved)
    assert torch.equal(full.weight, resumed.weight)


def test_scratch_fingerprint_separates_dependencies_sources_and_exports(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint as fingerprints
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import REVISION
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["data_root"] = str(tmp_path)
    for relative in ("splits/challenging_train_split.txt", "splits/challenging_test_split.txt",
                     f"splits/hf_folders_{REVISION}.json", "search_train_step50/train.txt",
                     "search_val_step10/train.txt", "search_train_step50/done/1.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen")
    monkeypatch.setattr(fingerprints.yaml, "safe_load", lambda _: copy.deepcopy(protocol))
    monkeypatch.setattr(fingerprints.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="commit\n"))
    versions = {"torch": "2.1.2", "mmdet": "3.3.0"}
    monkeypatch.setattr(fingerprints.subprocess, "check_output", lambda *a, **k: json.dumps(versions))
    monkeypatch.setattr(fingerprints.importlib.metadata, "version", lambda _: "controller-torch")
    code = {"adapter": "frozen"}
    monkeypatch.setattr(fingerprints, "scratch_code_identity", lambda _: dict(code))
    root = Path(__file__).resolve().parents[4]
    def compute():
        return fingerprints.compute_fingerprint(root, "faster_rcnn_scratch",
                    "faster_rcnn_scratch_20990101_eot0_unit", SEED)
    first = compute()
    assert first["torch_version"] == "2.1.2"
    versions["mmdet"] = "changed"
    assert compute()["sha256"] != first["sha256"]
    versions["mmdet"] = "3.3.0"
    code["adapter"] = "changed"
    assert compute()["sha256"] != first["sha256"]
    code["adapter"] = "frozen"
    (tmp_path / "search_train_step50/done/1.json").write_text("changed annotations")
    assert compute()["sha256"] != first["sha256"]


def test_baseline_gate_rejects_missing_stale_or_changed_seeds(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection import faster_rcnn_baseline as baseline
    identity = {"sha256": "frozen"}
    monkeypatch.setattr(baseline, "identity", lambda _: identity)
    with pytest.raises(FileNotFoundError):
        baseline.summarize(tmp_path)
    results = {}
    for seed in (0, 1, 2):
        result = {"seed": seed, "mAP50": .8 + .01 * seed, "mAP50_95": .4,
                  "params": 10, "report_sha256": "report", "args_sha256": "args",
                  "checkpoint_sha256": "weights"}
        results[seed] = result
        path = tmp_path / "baseline" / f"seed_{seed}" / "run.json"
        freeze_manifest(path, dict(result, identity=identity, wall_time_seconds=1))
    monkeypatch.setattr(baseline, "validate_seed", lambda _, seed: dict(results[seed]))
    summary = baseline.summarize(tmp_path, expected_fingerprint=dict(identity, evolution_settings={}))
    assert summary["aggregate"]["mAP50"]["mean"] == pytest.approx(.81)
    path = tmp_path / "baseline/summary.json"
    rounded = copy.deepcopy(summary)
    rounded["aggregate"]["wall_time_seconds"]["sample_sd"] = 1e-20
    path.write_text(json.dumps(rounded))
    original_bytes = path.read_bytes()
    assert baseline.summarize(tmp_path) == rounded
    assert path.read_bytes() == original_bytes
    results[2]["checkpoint_sha256"] = "altered"
    with pytest.raises(ValueError, match="stale or incomplete"):
        baseline.summarize(tmp_path)


def test_summary_tolerates_cross_python_statistics_rounding():
    from sota.FRED_LLM_GE.phase1_detection.faster_rcnn_baseline import verify_summary
    original = {"identity": {"sha256": "exact"}, "runs": [{"checkpoint_sha256": "exact"}],
                "aggregate": {"wall_time_seconds": {"mean": 1373.3871126350325,
                               "sample_sd": 0.000247989995314225}}}
    computed = copy.deepcopy(original)
    computed["aggregate"]["wall_time_seconds"]["sample_sd"] = 0.00024798999531422503
    verify_summary(original, computed)


@pytest.mark.parametrize("change", ["statistic", "nan", "identity", "checkpoint", "fields"])
def test_summary_rejects_real_changes(change):
    from sota.FRED_LLM_GE.phase1_detection.faster_rcnn_baseline import verify_summary
    original = {"identity": {"sha256": "exact"}, "runs": [{"checkpoint_sha256": "exact"}],
                "aggregate": {"mAP50": {"mean": .8, "sample_sd": .01}}}
    altered = copy.deepcopy(original)
    if change == "statistic": altered["aggregate"]["mAP50"]["mean"] = .81
    if change == "nan": altered["aggregate"]["mAP50"]["mean"] = float("nan")
    if change == "identity": altered["identity"]["sha256"] = "changed"
    if change == "checkpoint": altered["runs"][0]["checkpoint_sha256"] = "changed"
    if change == "fields": del altered["aggregate"]["mAP50"]["sample_sd"]
    with pytest.raises(ValueError):
        verify_summary(original, altered)
