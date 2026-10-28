"""Overlay the 2D Pareto fronts of several FRED LLM-GE runs on shared axes.

Reads the candidates.csv that fred_pareto_graphs.py writes for each run (the
row with gene_id "seed" is that run's seed model) and draws mAP50 vs params and
mAP50:95 vs params, one colour per run: faint dots for every candidate, a
staircase and labelled dots for the front, and an x for the seed.

Usage:
    uv run python analysis/scripts/fred_pareto_compare.py \
        --run "33.3 ms (released)" analysis/results/analysis/fred_pareto_released/candidates.csv \
        --run "10 ms" analysis/results/analysis/fred_pareto_raw_w10000_palette/candidates.csv
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fred_pareto_graphs import METRICS, ROOT_DIR, front_2d

# Categorical slots 1 and 2 (blue, orange), in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def label_fronts(ax, fronts, min_dx=70, min_dy=16):
    """Label front points with their score and size, skipping any whose label
    would sit on top of one already placed. Each front's two ends go first."""
    order = []
    for front in fronts:
        rows = list(front.itertuples())
        order += [rows[0], rows[-1]] + rows[1:-1]
    placed = []
    for row in order:
        px, py = ax.transData.transform((row.x, row.y))
        if any(abs(px - qx) < min_dx and abs(py - qy) < min_dy for qx, qy in placed):
            continue
        placed.append((px, py))
        ax.annotate(f"{row.y:.1f}%, {row.x:.2f}M", (row.x, row.y), fontsize=7, color="#333333",
                    xytext=(6, 5), textcoords="offset points")


def main():
    parser = argparse.ArgumentParser(description="Compare Pareto fronts of several FRED LLM-GE runs")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CANDIDATES_CSV"), required=True,
                        help="Run label and its candidates.csv; repeat for each run")
    parser.add_argument("--title", default="FRED YOLO11n LLM-GE after 20 generations: input frame window")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis",
                                                         "fred_pareto_compare.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fronts = {ax: [] for ax in axes}
    for (label, csv), color in zip(args.run, COLORS):
        df = pd.read_csv(csv)
        seed = df[df["gene_id"] == "seed"]
        candidates = df[df["gene_id"] != "seed"]
        for ax, (metric, _) in zip(axes, METRICS):
            front = front_2d(candidates, metric)
            others = candidates[~candidates["gene_id"].isin(front["gene_id"])]
            ax.scatter(others["params"] / 1e6, others[metric] * 100, color=color, s=18, alpha=0.22,
                       linewidths=0)
            ax.step(front["params"] / 1e6, front[metric] * 100, where="post", color=color, lw=2)
            ax.scatter(front["params"] / 1e6, front[metric] * 100, color=color, s=60, zorder=3,
                       edgecolor="white", linewidths=1.5, label=f"{label} front")
            fronts[ax].append(front.assign(x=front["params"] / 1e6, y=front[metric] * 100))
            if len(seed):
                s = seed.iloc[0]
                ax.scatter(s["params"] / 1e6, s[metric] * 100, color=color, marker="x", s=80,
                           linewidths=2.2, zorder=4, label=f"YOLO11n on {label}")
    for ax, (_, label) in zip(axes, METRICS):
        ax.set_xlim(left=0)
        ax.set_xlabel("Parameters (millions)")
        ax.set_ylabel(f"{label} %")
        ax.set_title(f"{label} vs parameters (up and left is better)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
        label_fronts(ax, fronts[ax])
    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
