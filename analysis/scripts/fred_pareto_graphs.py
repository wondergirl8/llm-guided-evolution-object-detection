"""Pareto front plots for the FRED YOLO11 LLM-GE run.

Reads the global_gen_*.pkl files that run_improved.py saves after every
generation, where each candidate's fitness is (mAP50, mAP50:95, params).
Both mAPs are maximized and parameter count minimized; the Pareto front is
the set of candidates no other candidate beats on all three at once.
Outputs (in --output):

    pareto2d_final.png     mAP50 vs params and mAP50:95 vs params after the last
                           generation, each with its own 2D front (best score at
                           a given size)
    pareto2d_gen_<N>.png   the same using every candidate up to generation N
    pareto2d_fronts.gif    the per-generation plots as an animation
    best_by_generation.png best mAP50 / mAP50:95 after each generation
    candidates.csv         every valid candidate, flagged if on the final front
    pareto_models.csv      final 3-objective front with model file paths

Usage:
    uv run python analysis/scripts/fred_pareto_graphs.py
    uv run python analysis/scripts/fred_pareto_graphs.py --checkpoints fred_yolo11_checkpoints --output analysis/results/analysis/fred_pareto
"""

import argparse
import glob
import math
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO11_DIR = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11")
METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50:95")]
# train_eval.py writes (0, 0, 999999999) for candidates that crashed.
FAILED_PARAMS = 999_999_999


def is_valid_fitness(fitness):
    if fitness is None or len(fitness) != 3:
        return False
    if any(math.isinf(v) or math.isnan(v) for v in fitness):
        return False
    map50, map50_95, params = fitness
    return 0 <= map50 <= 1 and 0 <= map50_95 <= 1 and 0 < params < FAILED_PARAMS


def discover_generations(checkpoint_dir):
    generations = []
    for path in glob.glob(os.path.join(checkpoint_dir, "global_gen_*.pkl")):
        match = re.search(r"global_gen_(\d+)\.pkl$", path)
        if match:
            generations.append((int(match.group(1)), path))
    return sorted(generations)


def load_candidates(path):
    """Every valid candidate evaluated up to and including this generation."""
    with open(path, "rb") as file:
        data = pickle.load(file)
    genes = {**data.get("GLOBAL_DATA_HIST", {}), **data.get("GLOBAL_DATA", {})}
    rows = [
        {"gene_id": gene_id, "mAP50": f[0], "mAP50_95": f[1], "params": int(f[2])}
        for gene_id, attrs in genes.items()
        if is_valid_fitness(f := attrs.get("fitness"))
    ]
    return pd.DataFrame(rows, columns=["gene_id", "mAP50", "mAP50_95", "params"])


def pareto_mask(df):
    """True where no other candidate is at least as good on mAP50, mAP50:95 and
    params and strictly better on one of them."""
    mask = []
    for _, row in df.iterrows():
        no_worse = ((df["mAP50"] >= row["mAP50"]) & (df["mAP50_95"] >= row["mAP50_95"])
                    & (df["params"] <= row["params"]))
        better = ((df["mAP50"] > row["mAP50"]) | (df["mAP50_95"] > row["mAP50_95"])
                  | (df["params"] < row["params"]))
        mask.append(not (no_worse & better).any())
    return pd.Series(mask, index=df.index, dtype=bool)


def load_seed(path):
    if not os.path.isfile(path):
        return None
    map50, map50_95, params = pd.read_csv(path, header=None).iloc[0, :3]
    return {"mAP50": map50, "mAP50_95": map50_95, "params": params}


def front_2d(df, metric):
    """Candidates no other beats on this metric with at most as many params, by params."""
    rows = df.sort_values(["params", metric], ascending=[True, False])
    front, best = [], -1.0
    for _, row in rows.iterrows():
        if row[metric] > best:
            front.append(row)
            best = row[metric]
    return pd.DataFrame(front)


def plot_fronts_2d(df, title, seed, limits, path, dpi=150):
    """mAP50 vs params and mAP50:95 vs params, each with its own Pareto front."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    candidates = df[df["gene_id"] != "seed"]
    for ax, (metric, label) in zip(axes, METRICS):
        front = front_2d(candidates, metric)
        others = candidates[~candidates["gene_id"].isin(front["gene_id"])]
        ax.scatter(others["params"] / 1e6, others[metric] * 100, color="lightgray",
                   edgecolor="gray", s=25, alpha=0.7, label="Candidates")
        # Staircase: at any size, the best score reachable with at most that many params.
        ax.step(front["params"] / 1e6, front[metric] * 100, where="post", color="crimson",
                lw=1.2, alpha=0.7)
        ax.scatter(front["params"] / 1e6, front[metric] * 100, color="crimson",
                   edgecolor="black", s=55, zorder=3, label="Pareto front")
        for i, (_, row) in enumerate(front.iterrows()):
            ax.annotate(f"{row[metric] * 100:.1f}%, {row['params'] / 1e6:.2f}M",
                        (row["params"] / 1e6, row[metric] * 100), fontsize=7.5,
                        xytext=(6, -11 if i % 2 else 5), textcoords="offset points")
        if seed:
            ax.scatter(seed["params"] / 1e6, seed[metric] * 100, color="black", marker="x",
                       s=90, linewidths=2, zorder=4, label="Seed")
            ax.annotate(f"seed {seed[metric] * 100:.1f}%", (seed["params"] / 1e6, seed[metric] * 100),
                        fontsize=7.5, xytext=(-8, 7), textcoords="offset points", ha="right")
        ax.set_xlim(limits["params"][0] / 1e6, limits["params"][1] / 1e6)
        ax.set_ylim(*limits[metric])
        ax.set_xlabel("Parameters (millions)")
        ax.set_ylabel(f"{label} %")
        ax.set_title(f"{label} vs parameters")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_best_by_generation(history, seed, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (metric, label), color in zip(METRICS, ["crimson", "darkorange"]):
        ax.plot(history["generation"], history[metric] * 100, marker="o", color=color,
                label=f"Best {label}")
        if seed:
            ax.axhline(seed[metric] * 100, color=color, linestyle="--", alpha=0.6,
                       label=f"Seed {label}")
    ax.set_xticks(history["generation"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score %")
    ax.set_title("Best candidate after each generation")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "best_by_generation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_gif(frames, output_path, duration_ms=700):
    from PIL import Image

    images = [Image.open(frame) for frame in frames]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0)


def padded(values, pad=0.08, lower=None):
    low, high = min(values), max(values)
    margin = (high - low) * pad or abs(high) * pad or 1
    return (low - margin if lower is None else max(lower, low - margin), high + margin)


def n_candidates(df):
    return int((df["gene_id"] != "seed").sum())


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto fronts for the FRED LLM-GE run")
    parser.add_argument("--checkpoints", default=os.path.join(ROOT_DIR, "fred_yolo11_checkpoints"),
                        help="Folder with global_gen_*.pkl")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis", "fred_pareto"),
                        help="Folder to write plots and CSVs to")
    parser.add_argument("--seed_results", default=os.path.join(YOLO11_DIR, "results", "seed_results.csv"),
                        help="Seed score CSV (mAP50,mAP50:95,params); skipped if missing")
    parser.add_argument("--models_dir", default=os.path.join(YOLO11_DIR, "models"),
                        help="Where the run's network_<id> files are (for pareto_models.csv)")
    parser.add_argument("--no_gif", action="store_true", help="Skip the animated GIF")
    parser.add_argument("--family", default="YOLO11", help="Seed family name used in plot titles")
    parser.add_argument("--seed_file", default=os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "seeds", "seed_yolo11.py"),
                        help="Seed model file listed for the seed in pareto_models.csv")
    args = parser.parse_args()

    generations = discover_generations(args.checkpoints)
    if not generations:
        raise SystemExit(f"No global_gen_*.pkl files in {args.checkpoints}")
    os.makedirs(args.output, exist_ok=True)
    seed = load_seed(args.seed_results)

    by_gen = [(gen, load_candidates(path)) for gen, path in generations]
    by_gen = [(gen, df) for gen, df in by_gen if len(df)]
    if seed:
        # The seed is a valid model too, so it can sit on (and shape) the front.
        seed_row = pd.DataFrame([{"gene_id": "seed", **seed, "params": int(seed["params"])}])
        by_gen = [(gen, pd.concat([df, seed_row], ignore_index=True)) for gen, df in by_gen]
    last_gen, final = by_gen[-1]

    # Same axes for every generation so the GIF frames line up.
    extra = [seed] if seed else []
    limits = {"params": padded(list(final["params"]) + [s["params"] for s in extra], lower=0)}
    for metric, _ in METRICS:
        limits[metric] = padded([v * 100 for v in list(final[metric]) + [s[metric] for s in extra]],
                                lower=0)

    frames, history = [], []
    for gen, df in by_gen:
        title = f"FRED {args.family} LLM-GE: generation {gen} ({n_candidates(df)} valid candidates)"
        frames.append(plot_fronts_2d(df, title, seed, limits,
                                     os.path.join(args.output, f"pareto2d_gen_{gen}.png")))
        history.append({"generation": gen, "mAP50": df["mAP50"].max(),
                        "mAP50_95": df["mAP50_95"].max()})
    print(f"Saved {len(frames)} per-generation plots to {args.output}")

    final_title = (f"FRED {args.family} LLM-GE: final Pareto front after generation {last_gen} "
                   f"({n_candidates(final)} valid candidates)")
    print("Saved " + plot_fronts_2d(final, final_title, seed, limits,
                                    os.path.join(args.output, "pareto2d_final.png"), dpi=300))

    if not args.no_gif and len(frames) > 1:
        make_gif(frames, os.path.join(args.output, "pareto2d_fronts.gif"))
        print("Saved pareto2d_fronts.gif")
    print(f"Saved {plot_best_by_generation(pd.DataFrame(history), seed, args.output)}")

    final = final.copy()
    final["pareto_front"] = pareto_mask(final)
    final = final.sort_values("mAP50", ascending=False)
    final.to_csv(os.path.join(args.output, "candidates.csv"), index=False)

    models = final[final["pareto_front"]].sort_values("params").copy()
    seed_file = args.seed_file
    models["model_file"] = [seed_file if g == "seed"
                            else os.path.join(args.models_dir, f"network_{g}.py")
                            for g in models["gene_id"]]
    models.to_csv(os.path.join(args.output, "pareto_models.csv"), index=False)

    print(f"\nPareto front (mAP50, mAP50:95, params) after generation {last_gen}:")
    if seed:
        print(f"  seed: mAP50 {seed['mAP50']:.1%}, mAP50:95 {seed['mAP50_95']:.1%}, "
              f"params {int(seed['params']):,}")
    for _, row in models.iterrows():
        print(f"  {row['gene_id']}: mAP50 {row['mAP50']:.1%}, mAP50:95 {row['mAP50_95']:.1%}, "
              f"params {row['params']:,}")


if __name__ == "__main__":
    main()
