"""GPU integration check without downloading data or starting an LLM."""
import argparse
import json
import importlib.metadata
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import (
    FasterRCNNScratchAdapter, preserving_preflight)
from sota.FRED_LLM_GE.phase1_detection.candidate_runner import (
    DEFAULT_PROTOCOL, FRED_DIR, load_protocol, predict_frames)
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import prepare_batch, synthetic_data
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import state_sha256
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import freeze_manifest
from sota.FRED_LLM_GE.phase1_detection.protocol_fingerprint import scratch_code_identity


def smoke_identity():
    return {"sources": scratch_code_identity(ROOT), "versions": {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision", "mmcv", "mmengine", "mmdet")}}


def run(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires a CUDA allocation")
    torch.manual_seed(0)
    identity = smoke_identity()
    protocol = load_protocol(DEFAULT_PROTOCOL)
    family = protocol["families"]["faster_rcnn_scratch"]
    settings = dict(protocol["proxy"], **family["search"])
    adapter = FasterRCNNScratchAdapter(family, FRED_DIR)
    model = adapter.build(adapter.variant_path("network", None), 1, device)
    before = state_sha256(model)
    assert before == adapter.initialization["initial_state_sha256"]
    preserving_preflight(model)
    assert state_sha256(model) == before
    data = synthetic_data()
    if device == "cuda":
        data["images"] = torch.randint(0, 256, (2, 3, 360, 640), dtype=torch.uint8)
        data["orig_size"] = (1280, 720)
    model.train()
    loss, _ = model.parse_losses(model(**prepare_batch(model, data, [0, 1], True), mode="loss"))
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4"):
        parameters = getattr(model.backbone, name).parameters()
        assert any(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in parameters), name
    optimizer = torch.optim.SGD(model.parameters(), lr=family["lr"],
                                momentum=family["momentum"], weight_decay=family["weight_decay"])
    optimizer.step()
    model.zero_grad(set_to_none=True)
    after = state_sha256(model)
    assert after != before
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "smoke.pt"
        adapter.save(model, path)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        assert state_sha256(model) == after
    settings["img_width"] = data["images"].shape[-1]
    frames = predict_frames(adapter, model, data, settings, device)
    assert len(frames) == 2 and not frames[1]["target"]["boxes"]
    for frame in frames:
        for box in frame["prediction"]["boxes"]:
            assert 0 <= box[0] < box[2] <= frame["width"]
            assert 0 <= box[1] < box[3] <= frame["height"]
    if identity != smoke_identity():
        raise RuntimeError("Code or dependencies changed during the smoke test")
    return {"status": "passed", "device": device, "initial_state_sha256": before,
            "updated_state_sha256": after, "empty_target_checked": True,
            "all_backbone_stages_have_gradients": True, "prediction_frames": len(frames),
            "identity": identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous.get("status") != "passed" or previous.get("device") != args.device
                or previous.get("identity") != smoke_identity()):
            raise ValueError("Existing smoke test differs from this code or environment")
        print("Reusing verified smoke test", flush=True)
        raise SystemExit(0)
    result = run(args.device)
    if args.output:
        freeze_manifest(args.output, result)
    print(json.dumps(result, indent=2))
