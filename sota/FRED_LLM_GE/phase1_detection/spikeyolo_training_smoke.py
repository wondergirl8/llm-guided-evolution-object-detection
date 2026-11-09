"""Run two tiny CPU batches through the official dataset and trainer; no FRED download."""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .adapters.spikeyolo import SpikeYoloAdapter


def main():
    torch.set_num_threads(1)
    fred = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((fred / "configs/phase1/protocol.yaml").read_text())["families"]["spikeyolo"]
    settings["workers"] = 0
    settings["training"]["amp"] = False
    with tempfile.TemporaryDirectory(prefix="spikeyolo_cpu_smoke_") as temporary:
        root = Path(temporary)
        (root / "images").mkdir()
        (root / "labels").mkdir()
        random = np.random.default_rng(0)
        for index in range(4):
            Image.fromarray(random.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / f"images/{index}.png")
            (root / f"labels/{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(dict(path=str(root), train="images", val="images", names={0: "drone"})))
        adapter = SpikeYoloAdapter(settings, fred)
        model = adapter.build(fred / settings["seed_file"], 1, "cpu")
        proxy = dict(seed=0, epochs=1, img_width=64, batch_size=2, eval_conf=.05, nms_iou=.5)
        adapter.train(model, dict(data_yaml=data_yaml, frames=4), proxy, "cpu", root / "artifacts/args.yaml")
        if (root / "labels.cache").exists():
            raise ValueError("SpikeYOLO rewrote a shared export label cache")
        assert adapter.trainer.fred_batch_shape == (64, 64)
        predictions = adapter.predict(model, torch.zeros(2, 3, 40, 64, dtype=torch.uint8), proxy, "cpu")
        assert len(predictions) == 2
        adapter.save(model, root / "model.pt")

        class KnownPrediction(torch.nn.Module):
            def forward(self, images):
                # Box [20,22,40,32] in a 64-square letterbox. The 40x64
                # source gets 12 pixels top padding, so y becomes [10,20].
                return torch.tensor([[[30.], [27.], [20.], [10.], [.9]]])

        known = adapter.predict(KnownPrediction(), torch.zeros(1, 3, 40, 64, dtype=torch.uint8), proxy, "cpu")
        torch.testing.assert_close(known[0][0], torch.tensor([[20., 10., 40., 20.]]))
        torch.testing.assert_close(known[0][1], torch.tensor([.9]))
        assert all(torch.isfinite(boxes).all() and torch.isfinite(scores).all() for boxes, scores in predictions)
        print(json.dumps(dict(status="passed", batches=2, epochs=1, device="cpu",
                              params=adapter.param_count(model), predictions=len(predictions))))


if __name__ == "__main__":
    main()
