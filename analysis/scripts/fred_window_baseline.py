"""Plot the YOLO11n baseline against the event-frame window.

Reads the plain palette rows of event_sweep/summary.csv and
renderer_retrain/summary.csv (YOLO11n, seeds 0-2, challenging_search split)
and draws mAP50 and mAP50:95 vs window with one-sd bars, marking FRED's
released 33.3 ms frames.

Usage:
    uv run python analysis/scripts/fred_window_baseline.py
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11",
                    "data_cache", "challenging")
COLOR = "#2a78d6"
METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50:95")]
LABELLED = [1, 2, 3.3, 5, 10, 20, 33.3, 50]


def load():
    sweep = pd.read_csv(os.path.join(DATA, "event_sweep", "summary.csv"))
    sweep = sweep[sweep.encoding == "palette"].rename(columns={"window_ms": "window"})
    retrain = pd.read_csv(os.path.join(DATA, "renderer_retrain", "summary.csv"))
    retrain = retrain[retrain.renderer.str.startswith("palette")]
    retrain = retrain.assign(window=retrain.window_us / 1000)
    rows = pd.concat([sweep, retrain])[["window", "runs", "mAP50_mean", "mAP50_sd",
                                        "mAP50_95_mean", "mAP50_95_sd"]]
    return rows.drop_duplicates("window").sort_values("window")


def main():
    parser = argparse.ArgumentParser(description="Plot YOLO11n score vs event-frame window")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis",
                                                         "fred_window_baseline.png"))
    args = parser.parse_args()

    rows = load()
    released = rows[(rows.window - 33.333).abs() < 0.01].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (key, title) in zip(axes, METRICS):
        ax.errorbar(rows.window, rows[f"{key}_mean"] * 100, yerr=rows[f"{key}_sd"] * 100,
                    color=COLOR, linewidth=2, marker="o", markersize=7, capsize=4,
                    markeredgecolor="white", markeredgewidth=1.5)
        # Label a few windows only; 15-20 ms sit too close together on a log axis.
        for _, r in rows[rows.window.round(1).isin(LABELLED)].iterrows():
            ax.annotate(f"{r[f'{key}_mean'] * 100:.1f}", (r.window, r[f"{key}_mean"] * 100),
                        textcoords="offset points", xytext=(9, -14), ha="left", fontsize=8,
                        color="#333333")
        ax.scatter([released.window], [released[f"{key}_mean"] * 100], s=180, facecolors="none",
                   edgecolors="#333333", linewidths=1.5, zorder=5)
        ax.annotate("FRED released\n(33.3 ms)", (released.window, released[f"{key}_mean"] * 100),
                    textcoords="offset points", xytext=(-12, 14), ha="right", fontsize=9,
                    color="#333333")
        ax.set_xscale("log")
        ax.set_xticks(LABELLED)
        ax.set_xticklabels([f"{t:g}" for t in LABELLED])
        ax.minorticks_off()
        ax.set_xlabel("Event-frame window (ms, log scale)")
        ax.set_ylabel(f"{title} %")
        ax.set_title(f"{title} (higher is better)")
        ax.grid(linestyle=":", color="#cccccc")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        low = (rows[f"{key}_mean"] - rows[f"{key}_sd"]).min() * 100
        ax.set_ylim(low - 4, (rows[f"{key}_mean"] + rows[f"{key}_sd"]).max() * 100 + 4)
    fig.suptitle("YOLO11n baseline vs event-frame window (palette frames)", fontsize=13)
    fig.text(0.5, 0.02, "challenging_search validation split, 20 epochs from scratch; "
             "mean of seeds 0-2, bars = 1 sd.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
