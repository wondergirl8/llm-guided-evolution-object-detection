"""CPU contracts for the experimental EMS-YOLO family; no data downloads."""
import copy
import csv

import pytest
import torch
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection import candidate_runner as runner
from sota.FRED_LLM_GE.phase1_detection import ems_yolo_search_space as policy
from sota.FRED_LLM_GE.phase1_detection.adapters.ems_yolo import EMSYoloAdapter
from sota.FRED_LLM_GE.phase1_detection.seeds.ems_yolo.detection import EMSYoloLoss, decode_predictions
from sota.FRED_LLM_GE.phase1_detection.seeds.ems_yolo.network import EMSYolo
from sota.FRED_LLM_GE.phase1_detection.ems_yolo_overfit import load_labeled_subset


def variant(tmp_path, old, new):
    path = tmp_path / "network_test.py"
    path.write_text(policy.SEED.read_text().replace(old, new))
    return path


def settings():
    protocol = runner.load_protocol(runner.DEFAULT_PROTOCOL)
    value = copy.deepcopy(protocol["families"]["ems_yolo"])
    value["training"].update(input_size=64, time_steps=4, epochs=1,
                             microbatch_size=1, nominal_batch_size=2,
                             warmup_epochs=1.0, amp=False)
    return value


@pytest.mark.parametrize("old,new", [
    ("BASE_CHANNELS = 64", "BASE_CHANNELS = 32"),
    ("BASE_CHANNELS = 64", "BASE_CHANNELS = True"),
    ("STAGE_DEPTHS = (3, 4, 6, 3)", "STAGE_DEPTHS = (3, 4, 6)"),
    ("STAGE_DEPTHS = (3, 4, 6, 3)", "STAGE_DEPTHS = (0, 4, 6, 3)"),
    ("NECK_CHANNELS = 256", "NECK_CHANNELS = __import__('os').getpid()"),
    ("return EMSYolo", "return object"),
])
def test_policy_rejects_invalid_or_protected_code(tmp_path, old, new):
    with pytest.raises((ValueError, SyntaxError)):
        policy.load_and_validate(variant(tmp_path, old, new))


def test_seed_and_mutation_have_distinct_valid_built_graphs(tmp_path):
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    seed_hash, seed_params = policy.built_architecture(policy.SEED)
    assert 20_000_000 < seed_params < 35_000_000
    assert torch.equal(before, torch.get_rng_state())
    changed = variant(tmp_path, "BASE_CHANNELS = 64", "BASE_CHANNELS = 48")
    changed_hash, changed_params = policy.built_architecture(changed)
    assert changed_hash != seed_hash and changed_params != seed_params
    assert policy.differs_from_parents(changed, policy.SEED)
    clone = tmp_path / "clone.py"
    clone.write_text(policy.SEED.read_text() + "\n# comment does not alter the graph\n")
    assert not policy.differs_from_parents(clone, policy.SEED)


def test_forward_loss_backward_and_decode():
    model = EMSYolo(base_channels=32, stage_depths=(1, 2, 2, 1), neck_channels=128)
    inputs = torch.rand(1, 4, 3, 64, 64)
    predictions = model(inputs)
    assert [tuple(value.shape) for value in predictions] == [
        (1, 3, 4, 4, 6), (1, 3, 2, 2, 6)]
    target = torch.tensor([[0., 0., 0.5, 0.5, 0.2, 0.2]])
    loss = EMSYoloLoss(model, {"box": 0.05, "obj": 1.0,
                               "obj_pw": 1.0, "anchor_t": 4.0})(predictions, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.stem.conv.weight.grad.abs().sum() > 0
    model.eval()
    boxes, scores = decode_predictions(model, model(inputs))
    assert boxes.shape == (1, 60, 4) and scores.shape == (1, 60)


def test_loss_accepts_float32_targets_with_half_predictions():
    """Match CUDA autocast: half predictions with float32 target assignment."""
    model = EMSYolo(base_channels=32, stage_depths=(1, 2, 2, 1), neck_channels=128)
    predictions = [torch.randn(1, 3, 4, 4, 6, dtype=torch.float16, requires_grad=True),
                   torch.randn(1, 3, 2, 2, 6, dtype=torch.float16, requires_grad=True)]
    target = torch.tensor([[0., 0., 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    loss_function = EMSYoloLoss(model, {"box": 0.05, "obj": 1.0,
                                        "obj_pw": 1.0, "anchor_t": 4.0})
    loss = loss_function(predictions, target)
    assert torch.isfinite(loss)
    assert loss_function.last_components["positive_assignments"] > 0
    assert 0 <= loss_function.last_components["positive_objectness_sum"]
    assert loss_function.last_components["positive_ciou_sum"] <= \
        loss_function.last_components["positive_target_iou_sum"]
    assert 0 <= loss_function.last_components["positive_target_iou_sum"]
    assert 0 <= loss_function.last_components["max_confidence"] <= 1
    loss.backward()
    assert all(prediction.grad is not None for prediction in predictions)


def test_letterbox_targets_and_one_step_training():
    adapter = EMSYoloAdapter(settings(), runner.FRED_DIR)
    images = torch.randint(0, 256, (2, 3, 32, 48), dtype=torch.uint8)
    boxes = [torch.tensor([[12., 8., 36., 24.]]), torch.empty(0, 4)]
    resized, targets, geometry = adapter._letterbox(images, boxes, torch.device("cpu"))
    assert resized.shape == (2, 3, 64, 64)
    assert torch.allclose(targets[0, 2:], torch.tensor([0.5, 31.333333 / 64, 0.5, 1 / 3]),
                          atol=1e-6)
    assert geometry == (pytest.approx(4 / 3), 0, 10, 48, 32)
    assert adapter._temporal(resized).shape == (2, 4, 3, 64, 64)
    model = EMSYolo(base_channels=32, stage_depths=(1, 2, 2, 1), neck_channels=128)
    before = model.stem.conv.weight.detach().clone()
    data = dict(images=images, boxes=boxes, orig_boxes=boxes,
                orig_size=(48, 32), sample_ids=["a", "b"])
    proxy = dict(epochs=1, batch_size=2, seed=0, eval_conf=0.05, nms_iou=0.5)
    adapter.train(model, data, proxy, "cpu")
    assert not torch.equal(before, model.stem.conv.weight)
    predictions = adapter.predict(model, images, proxy, "cpu")
    assert len(predictions) == 2
    assert all((box[:, 0::2] >= 0).all() and (box[:, 0::2] <= 48).all()
               and (box[:, 1::2] >= 0).all() and (box[:, 1::2] <= 32).all()
               for box, _ in predictions)


def test_family_registration_and_scratch_provenance():
    protocol = runner.load_protocol(runner.DEFAULT_PROTOCOL)
    family = protocol["families"]["ems_yolo"]
    assert runner.ADAPTERS["ems_yolo"] is EMSYoloAdapter
    assert family["initialization"] == "scratch"
    assert "weights" not in family and "checkpoint" not in family
    assert family["upstream"]["commit"] == "fbfc2caeac5ff4d60cbc2ad23864911df63d0f85"
    assert family["data"] == "challenging_search"


def test_baseline_validation_surfaces_candidate_failure(tmp_path):
    from sota.FRED_LLM_GE.phase1_detection import ems_yolo_baseline as baseline

    results = tmp_path / "results"
    results.mkdir()
    with (results / "seed_results.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(
            ["-inf", "-inf", "inf", "train_or_eval_failed: AMP dtype mismatch"])

    with pytest.raises(ValueError, match="AMP dtype mismatch"):
        baseline.validate(tmp_path, 0)


def test_overfit_subset_uses_only_labeled_export_frames(tmp_path):
    root = tmp_path / "search_train_step50"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    paths = []
    for index, label in enumerate(("", "0 0.5 0.5 0.25 0.5\n",
                                   "0 0.25 0.25 0.1 0.2\n")):
        path = root / "images" / f"frame_{index}.png"
        Image.new("RGB", (40, 20)).save(path)
        (root / "labels" / f"frame_{index}.txt").write_text(label)
        paths.append(str(path))
    (root / "train.txt").write_text("\n".join(paths) + "\n")

    data = load_labeled_subset(root, 2, seed=0)
    assert len(data["images"]) == 2
    assert all(len(boxes) == 1 for boxes in data["boxes"])
    assert data["orig_size"] == (40, 20)
    assert data["images"][torch.tensor([0])].shape == (1, 3, 20, 40)
