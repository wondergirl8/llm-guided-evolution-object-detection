"""Bar chart of the held-out challenging_test scores, 33.3 ms vs 10 ms frames.

Reads the final_eval.py reports in <data_root>/held_out_reports and draws
mAP50 and mAP50:95 for YOLO11n and the evolved model chosen at each window,
one colour per window (the same colours as fred_pareto_compare.py).

Usage:
    uv run python analysis/scripts/fred_final_eval.py
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11",
                       "data_cache", "challenging", "held_out_reports")
COLORS = ["#2a78d6", "#eb6834"]
WINDOWS = [("33.3 ms (released)", "released"), ("10 ms", "raw_w10000_palette")]
# Model label -> report name prefix per window (evolved models were picked on the search split).
MODELS = [("YOLO11n", {"released": "yolo11n", "raw_w10000_palette": "yolo11n"}),
          ("Evolved", {"released": "xXx50vzbE97eCX2d6roTlkIp2M5",
                       "raw_w10000_palette": "xXxiNjKaQJ0ncqTsHWTkI5uE6m5"})]
METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50:95")]


def load(name):
    with open(os.path.join(REPORTS, f"{name}_evaluation.json")) as f:
        return json.load(f)["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Plot challenging_test scores by frame window")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis",
                                                         "fred_final_eval.png"))
    args = parser.parse_args()

    scores = {(m, tag): load(f"{names[tag]}_{tag}") for m, names in MODELS for _, tag in WINDOWS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    width = 0.36
    for ax, (key, title) in zip(axes, METRICS):
        for w, ((window, tag), color) in enumerate(zip(WINDOWS, COLORS)):
            xs = [i + (w - 0.5) * (width + 0.02) for i in range(len(MODELS))]
            values = [scores[(m, tag)][key] for m, _ in MODELS]
            bars = ax.bar(xs, values, width, color=color, label=window, edgecolor="white", linewidth=2)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=10, color="#333333")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([f"{m}\n(" + ", ".join(dict.fromkeys(
            f"{scores[(m, tag)]['params'] / 1e6:.2f}M" for _, tag in WINDOWS)) + " params)"
            for m, _ in MODELS])
        ax.set_ylim(0, max(s[key] for s in scores.values()) * 1.18)
        ax.set_title(f"{title} (higher is better)")
        ax.set_ylabel(title)
        ax.grid(axis="y", linestyle=":", color="#cccccc")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("FRED challenging_test (held out): 10 ms vs 33.3 ms event frames", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), title="Frame window", frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)
    fig.text(0.5, 0.005, "Evolved: best LLM-GE model per window, chosen on the search split before testing. "
             "One training seed each; 203,158 test frames.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
