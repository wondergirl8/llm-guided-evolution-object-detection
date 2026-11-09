"""Validate official SpikeYOLO graphs on CPU, including shape and gradient contracts."""
import argparse
import copy
import hashlib
import json

import torch

from .spikeyolo_search_space import GENOME_VERSION, SEED, model_config, validate_config
from .spikeyolo_source import build_official


def graph_identity(model):
    """Capture operation choices as well as tensors: pooling kernels have no weights."""
    graph = dict(
        modules=repr(model.model),
        routing=[layer.f for layer in model.model],
        strides=model.stride.tolist(),
        state_shapes=[(name, list(value.shape)) for name, value in model.state_dict().items()])
    return hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()


def check(config, backward=False):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    # Validate even callers supplying --config-json directly; no unchecked
    # constructor strings are passed to the upstream parser.
    config = copy.deepcopy(config)
    validate_config(dict(config, genome_version=GENOME_VERSION))
    model = build_official(config)
    from spikingjelly.clock_driven.functional import reset_net
    from ultralytics.utils import DEFAULT_CFG

    head = model.model[-1]
    if (head.__class__.__name__ != "SpikeDetect" or head.nc != 1
            or head.nl != 3 or model.stride.tolist() != [8, 16, 32]):
        raise ValueError("official SpikeDetect/class/stride contract violated")
    model.args = DEFAULT_CFG
    if backward:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        for _ in range(2):
            reset_net(model)
            model.train()
            batch = dict(img=torch.rand(2, 3, 64, 64), batch_idx=torch.tensor([0., 1.]),
                         cls=torch.zeros(2, 1), bboxes=torch.tensor([[.5, .5, .25, .25]] * 2))
            loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite official SpikeYOLO loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    model.eval()
    checked_shapes = []
    # Exercise batch>1 and non-square frames to catch temporal-axis/Concat/
    # downsampling problems hidden by the upstream square stride probe.
    for batch_size, height, width in ((1, 64, 64), (2, 64, 96)):
        reset_net(model)
        images = torch.zeros(batch_size, 3, height, width)
        with torch.no_grad():
            first, levels = model(images)
            reset_net(model)
            second, _ = model(images)
        anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
        expected_levels = [(batch_size, 65, height // stride, width // stride)
                           for stride in (8, 16, 32)]
        if (tuple(first.shape) != (batch_size, 5, anchors)
                or [tuple(level.shape) for level in levels] != expected_levels
                or not torch.isfinite(first).all()
                or not all(torch.isfinite(level).all() for level in levels)
                or not torch.equal(first, second)):
            raise ValueError("SpikeYOLO output/feature-shape/sample-state reset contract violated")
        checked_shapes.append(list(first.shape))
    reset_net(model)
    return dict(architecture_hash=graph_identity(model),
                params=sum(p.numel() for p in model.parameters()), backward=backward,
                checked_shapes=checked_shapes, genome_version=GENOME_VERSION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.config_json) if args.config_json else model_config(SEED), args.backward)))


if __name__ == "__main__":
    main()
