"""Additional 2D Pareto views for FRED LLM-GE runs.

Run this separately from fred_pareto_graphs.py to keep its existing 3D outputs.

Reads the global_gen_*.pkl files that run_improved.py saves after every
generation, where each candidate's fitness is (mAP50, mAP50:95, params).
Both mAPs are maximized and parameter count minimized; the Pareto front is
the set of candidates no other candidate beats on all three at once.
Outputs (in --output):

    pareto2d_<metric>_final.png    mAP50 / mAP50_95 versus parameters
    pareto2d_<metric>_gen_<N>.png  the same views after each generation
    pareto2d_<metric>_fronts.gif   one animation per metric
    best_by_generation.png best mAP50 / mAP50:95 after each generation
    candidates.csv         valid models with 3-objective and per-view front flags
    pareto_models.csv      final front with model file paths

Gray points show valid models; red points retain the full three-objective front.
The step line is computed separately for each 2D view: maximum accuracy under
each parameter budget. The seed participates in both fronts and has a star marker.
These are post-run projections; they do not change evolutionary selection.

Usage (only load trusted run checkpoints):
    uv run python analysis/scripts/fred_pareto_graphs_2d.py
    uv run python analysis/scripts/fred_pareto_graphs_2d.py --checkpoints fred_yolo11_checkpoints --output analysis/results/analysis/fred_pareto_2d
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
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO11_DIR = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11")
METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50:95")]
# train_eval.py writes (0, 0, 999999999) for candidates that crashed.
FAILED_PARAMS = 999_999_999


def is_valid_fitness(fitness):
    try:
        if len(fitness) != 3:
            return False
        map50, map50_95, params = (float(value) for value in fitness)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (map50, map50_95, params)):
        return False
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
        {"gene_id": gene_id, "mAP50": float(f[0]), "mAP50_95": float(f[1]), "params": int(float(f[2]))}
        for gene_id, attrs in genes.items()
        if isinstance(attrs, dict)
        and str(attrs.get("status") or "").lower() in ("", "completed", "verified")
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
    table = pd.read_csv(path, header=None)
    if table.shape != (1, 3) or not is_valid_fitness(table.iloc[0]):
        raise ValueError(f"Invalid baseline fitness in {path}")
    map50, map50_95, params = (float(value) for value in table.iloc[0])
    return {"mAP50": map50, "mAP50_95": map50_95, "params": params}


def view_front(df, metric):
    """Best accuracy attainable for each parameter budget (min cost, max score).

    Keep one coordinate for ties. A model can be dominated in this view while
    remaining on the three-objective front because of its other mAP score.
    """
    ordered = df.sort_values(["params", metric], ascending=[True, False])
    ordered = ordered.drop_duplicates("params")
    keep, best = [], -math.inf
    for index, row in ordered.iterrows():
        if row[metric] > best:
            keep.append(index)
            best = row[metric]
    return df.loc[keep].sort_values("params")


def view_pareto_mask(df, metric):
    # All models at a non-dominated coordinate count, including exact ties.
    coordinates = set(view_front(df, metric)[["params", metric]].itertuples(index=False, name=None))
    return pd.Series([(row.params, getattr(row, metric)) in coordinates
                      for row in df.itertuples()], index=df.index, dtype=bool)


def plot_front_2d(df, title, metric, label, limits, path, dpi=150):
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    color = "#ad3d29"
    front = df[pareto_mask(df)]
    ax.scatter(df["params"], df[metric] * 100, color="#9aa4af", s=38, alpha=0.8,
               label=f"All valid models (n={len(df)})", zorder=2)
    view = view_front(df, metric)
    if not view.empty and n_candidates(df):
        # No frontier is drawn below the smallest feasible parameter budget.
        xs = list(view["params"]) + [limits["params"][1]]
        ys = list(view[metric] * 100) + [view[metric].iloc[-1] * 100]
        ax.step(xs, ys, where="post", color=color, linewidth=2,
                label="Front in this view (best under parameter budget)", zorder=3)
    if n_candidates(df):
        ax.scatter(front["params"], front[metric] * 100, color=color, s=75,
                   label=f"3-objective Pareto members (n={len(front)})", zorder=4)
    seed = df[df["gene_id"] == "seed"]
    if not seed.empty:
        ax.scatter(seed["params"], seed[metric] * 100, color="#e9b44c",
                   marker="*", s=230, edgecolor="#292929", linewidth=1.1,
                   label="Baseline seed (included in fronts)", zorder=5)
    ax.set_xlim(*limits["params"])
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
    ax.set_xlabel("Parameters (millions; lower is better)")
    ax.set_ylabel(f"{label} (%) (higher is better)")
    ax.set_title(f"{label} vs. parameters\n{title}", fontsize=12, pad=12)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
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
    ax.set_title("Best recorded score after each generation (including seed)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "best_by_generation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_gif(frames, output_path, duration_ms=700):
    from PIL import Image

    images = []
    try:
        for frame in frames:
            with Image.open(frame) as image:
                images.append(image.copy())
        images[0].save(output_path, save_all=True, append_images=images[1:],
                       duration=duration_ms, loop=0)
    finally:
        for image in images:
            image.close()


def padded(values, pad=0.08, lower=None):
    low, high = min(values), max(values)
    margin = (high - low) * pad or abs(high) * pad or 1
    return (low - margin if lower is None else max(lower, low - margin), high + margin)


def n_candidates(df):
    return int((df["gene_id"] != "seed").sum())


def main():
    parser = argparse.ArgumentParser(description="Plot two 2D Pareto views for the FRED LLM-GE run")
    parser.add_argument("--checkpoints", default=os.path.join(ROOT_DIR, "fred_yolo11_checkpoints"),
                        help="Folder with global_gen_*.pkl")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis", "fred_pareto_2d"),
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
    if seed:
        # The seed is a valid model too, so it can sit on (and shape) the front.
        seed_row = pd.DataFrame([{"gene_id": "seed", **seed, "params": int(seed["params"])}])
        by_gen = [(gen, pd.concat([df[df["gene_id"] != "seed"], seed_row], ignore_index=True))
                  for gen, df in by_gen]
    by_gen = [(gen, df) for gen, df in by_gen if len(df)]
    if not by_gen:
        raise SystemExit("No valid completed candidate fitness or baseline was found; no Pareto plots generated.")
    last_gen, final = by_gen[-1]

    # Include every displayed generation when fixing animation limits.
    limits = {"params": (0, padded([value for _, df in by_gen for value in df["params"]], lower=0)[1])}

    frames, history = {metric: [] for metric, _ in METRICS}, []
    for gen, df in by_gen:
        title = f"FRED {args.family} LLM-GE: generation {gen} ({n_candidates(df)} valid candidates)"
        if not n_candidates(df):
            title += "\nBaseline only: no evaluated candidates"
        for metric, label in METRICS:
            frames[metric].append(plot_front_2d(df, title, metric, label, limits,
                os.path.join(args.output, f"pareto2d_{metric}_gen_{gen}.png")))
        history.append({"generation": gen, "mAP50": df["mAP50"].max(),
                        "mAP50_95": df["mAP50_95"].max()})
    print(f"Saved two 2D views for each of {len(by_gen)} generations to {args.output}")

    final_title = (f"FRED {args.family} LLM-GE: final Pareto front after generation {last_gen} "
                   f"({n_candidates(final)} valid candidates)")
    if not n_candidates(final):
        final_title = f"FRED {args.family}: baseline only (no evaluated candidates)"
    for metric, label in METRICS:
        print("Saved " + plot_front_2d(final, final_title, metric, label, limits,
            os.path.join(args.output, f"pareto2d_{metric}_final.png"), dpi=300))
        if not args.no_gif and len(frames[metric]) > 1:
            path = os.path.join(args.output, f"pareto2d_{metric}_fronts.gif")
            make_gif(frames[metric], path)
            print("Saved " + path)
    print(f"Saved {plot_best_by_generation(pd.DataFrame(history), seed, args.output)}")

    final = final.copy()
    final["pareto_front"] = pareto_mask(final)
    for metric, _ in METRICS:
        final[f"pareto_{metric}_params"] = view_pareto_mask(final, metric)
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
