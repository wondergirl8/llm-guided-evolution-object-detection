"""Side-by-side FRED event frames for different event settings.

For each frame, one column per setting: the full frame with the drone box drawn
on it, and below it a zoomed crop around the drone. Each panel's title gives the
share of pixels that hold an event, which is how sparse the frame is. Frames are
read from the exported search_val_step10 datasets, which share file names across
settings.

Usage:
    uv run python analysis/scripts/fred_event_frames.py
    uv run python analysis/scripts/fred_event_frames.py --frames 14_000931 209_000691 \
        --settings released raw_w10000_palette raw_w50000_palette
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11",
                        "data_cache", "challenging")
BACKGROUND = (30, 37, 52)


def setting_label(tag):
    if tag == "released":
        return "33.3 ms (FRED released)"
    window_us, encoding = tag.removeprefix("raw_w").split("_")
    return f"{int(window_us) / 1000:g} ms ({encoding})"


def load(dataset, frame):
    return np.asarray(Image.open(os.path.join(dataset, "images", f"{frame}.png")).convert("RGB"))


def drone_box(dataset, frame, width, height):
    """First label box in pixels as (x0, y0, x1, y1)."""
    with open(os.path.join(dataset, "labels", f"{frame}.txt")) as file:
        _, x, y, w, h = map(float, file.readline().split())
    return ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)


def crop_window(box, width, height, size):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(size, 4 * max(box[2] - box[0], box[3] - box[1])) / 2
    x0 = int(np.clip(cx - half, 0, width - 2 * half))
    y0 = int(np.clip(cy - half, 0, height - 2 * half))
    return x0, y0, int(2 * half)


def event_share(image):
    return float((image != BACKGROUND).any(axis=2).mean())


def main():
    parser = argparse.ArgumentParser(description="Side-by-side FRED event frames")
    parser.add_argument("--frames", nargs="+", default=["14_000931", "209_000691"],
                        help="Frame names (<sequence>_<frame>) in the val split")
    parser.add_argument("--settings", nargs="+", default=["released", "raw_w10000_palette"],
                        help="Event tags to compare")
    parser.add_argument("--split", default="search_val_step10", help="Exported dataset base name")
    parser.add_argument("--crop", type=int, default=160, help="Minimum zoom crop size in pixels")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis",
                                                         "fred_event_frames.png"))
    args = parser.parse_args()

    n_cols = len(args.settings)
    fig, axes = plt.subplots(2 * len(args.frames), n_cols, figsize=(5.2 * n_cols, 6.4 * len(args.frames)),
                             gridspec_kw={"height_ratios": [1, 1.35] * len(args.frames)}, squeeze=False)
    for row, frame in enumerate(args.frames):
        for col, tag in enumerate(args.settings):
            dataset = os.path.join(DATA_DIR, args.split if tag == "released" else f"{args.split}_{tag}")
            image = load(dataset, frame)
            height, width = image.shape[:2]
            box = drone_box(dataset, frame, width, height)
            x0, y0, size = crop_window(box, width, height, args.crop)

            full, zoom = axes[2 * row, col], axes[2 * row + 1, col]
            full.imshow(image)
            full.add_patch(patches.Rectangle((x0, y0), size, size, fill=False, ec="#eda100", lw=1.2))
            full.set_title(f"{setting_label(tag)}\nframe {frame}: {event_share(image):.2%} of pixels have events",
                           fontsize=9.5)
            zoom.imshow(image[y0:y0 + size, x0:x0 + size], interpolation="nearest")
            zoom.add_patch(patches.Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                                             fill=False, ec="#eda100", lw=1, ls="--"))
            zoom.set_title("zoom on the drone (dashed = label box)", fontsize=8.5)
            for ax in (full, zoom):
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("FRED event frames: same moment, different accumulation windows")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
