"""Check a generated model on CPU without loading FRED or writing fitness."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--scale', choices=list('nsmlx'), default='n')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'sota/FRED_LLM_GE/phase1_detection/seeds/yolo11'))
    import torch
    from train_ultralytics import load_variant, preflight

    torch.set_num_threads(2)
    torch.manual_seed(0)
    module, _ = load_variant('network_check', args.directory)
    model = module.DroneDetector(num_classes=1, scale=args.scale).to('cpu')
    check = preflight(model, 'cpu')
    # A non-square input also catches size assumptions hidden by square inputs.
    model.eval()
    with torch.no_grad():
        features = model.detector.model[:2](torch.zeros(2, 3, 64, 96))
        channels = [block[0].conv.in_channels for block in model.detector.model[-1].cv2]
        for feature, width, stride in zip(features, channels, (8, 16, 32)):
            if tuple(feature.shape) != (2, width, 64 // stride, 96 // stride):
                raise ValueError('Candidate broke the rectangular P3/P4/P5 interface')
            if not torch.isfinite(feature).all():
                raise ValueError('Candidate produced non-finite rectangular features')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
