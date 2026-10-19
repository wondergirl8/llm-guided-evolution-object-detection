"""3D Pareto front plots for the FRED YOLO11 LLM-GE run.

Reads the global_gen_*.pkl files that run_improved.py saves after every
generation, where each candidate's fitness is (mAP50, mAP50:95, params).
Both mAPs are maximized and parameter count minimized; the Pareto front is
the set of candidates no other candidate beats on all three at once.
Outputs (in --output):

    pareto3d_final.png     3D front after the last generation
    pareto3d_gen_<N>.png   3D front using every candidate up to generation N
    pareto3d_fronts.gif    the per-generation plots as an animation
    best_by_generation.png best mAP50 / mAP50:95 after each generation
    candidates.csv         every valid candidate, flagged if on the final front
    pareto_models.csv      final front with model file paths

Usage:
    uv run python analysis/scripts/fred_pareto_graphs.py
    uv run python analysis/scripts/fred_pareto_graphs.py --checkpoints fred_yolo11_checkpoints --output analysis/results/analysis/fred_pareto
"""

import argparse
import ast
import json
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


def pareto_2d_mask(df):
    """Pareto mask that uses only mAP50 and mAP50:95, both maximized."""
    mask = []
    for _, row in df.iterrows():
        no_worse = ((df["mAP50"] >= row["mAP50"])
                    & (df["mAP50_95"] >= row["mAP50_95"]))
        better = ((df["mAP50"] > row["mAP50"])
                  | (df["mAP50_95"] > row["mAP50_95"]))
        mask.append(not (no_worse & better).any())
    return pd.Series(mask, index=df.index, dtype=bool)


def load_seed(path):
    if not os.path.isfile(path):
        return None
    map50, map50_95, params = pd.read_csv(path, header=None).iloc[0]
    return {"mAP50": map50, "mAP50_95": map50_95, "params": params}


def plot_front_3d(df, title, seed, limits, path, dpi=150):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    front_mask = pareto_mask(df) & (df["gene_id"] != "seed")
    others = df[(df["gene_id"] != "seed") & ~front_mask]
    front = df[front_mask]
    for subset, style in ((others, dict(color="lightgray", edgecolor="gray", s=25,
                                        label="Candidates", alpha=0.7)),
                          (front, dict(color="crimson", edgecolor="black", s=60,
                                       label="Pareto front"))):
        ax.scatter(subset["params"], subset["mAP50"] * 100, subset["mAP50_95"] * 100,
                   depthshade=False, **style)
    # Drop lines to the floor make each front point's height readable, and each
    # front point is labeled with its metrics.
    for _, row in front.iterrows():
        x, y, z = row["params"], row["mAP50"] * 100, row["mAP50_95"] * 100
        ax.plot([x, x], [y, y], [limits["mAP50_95"][0], z], color="crimson", lw=0.8, alpha=0.5)
        dx = 0.03 * (limits["params"][1] - limits["params"][0])
        dz = 0.03 * (limits["mAP50_95"][1] - limits["mAP50_95"][0])
        ax.text(x + dx, y, z + dz, f"mAP50 {y:.1f}%\nmAP50:95 {z:.1f}%\n{int(x):,} params",
                fontsize=8, color="darkred", zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="crimson",
                          alpha=0.85))
    if seed:
        ax.scatter(seed["params"], seed["mAP50"] * 100, seed["mAP50_95"] * 100,
                   color="black", marker="x", s=110, linewidths=2, label="Seed",
                   depthshade=False)
    ax.set_xlim(*limits["params"])
    ax.set_ylim(*limits["mAP50"])
    ax.set_zlim(*limits["mAP50_95"])
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("mAP50 %")
    ax.set_zlabel("mAP50:95 %")
    # Viewed so the best corner (few params, high mAP50 and mAP50:95) faces the viewer.
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def short_model_id(gene_id):
    """Compact label used beside points while the CSV retains the full ID."""
    return gene_id.removeprefix("xXx")[:8]


def plot_generation_2d(new_models, cumulative, generation, limits, path):
    """Plot models first seen in one generation on mAP50 versus mAP50:95."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cumulative_front = cumulative.loc[pareto_2d_mask(cumulative)].sort_values("mAP50")
    global_front = set(cumulative_front["gene_id"])
    on_front = new_models["gene_id"].isin(global_front)
    ax.scatter(new_models.loc[~on_front, "mAP50"] * 100,
               new_models.loc[~on_front, "mAP50_95"] * 100,
               s=55, color="steelblue", edgecolor="black", label="Generation models")
    ax.scatter(new_models.loc[on_front, "mAP50"] * 100,
               new_models.loc[on_front, "mAP50_95"] * 100,
               s=90, color="crimson", edgecolor="black", label="Cumulative Pareto front")
    front_x = cumulative_front["mAP50"] * 100
    front_y = cumulative_front["mAP50_95"] * 100
    ax.step(front_x, front_y, where="post", color="crimson", linewidth=2,
            label="Pareto frontier")
    for _, row in new_models.iterrows():
        ax.annotate(short_model_id(row["gene_id"]),
                    (row["mAP50"] * 100, row["mAP50_95"] * 100),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(*limits["mAP50"]); ax.set_ylim(*limits["mAP50_95"])
    ax.set_xlabel("mAP50 (%)"); ax.set_ylabel("mAP50:95 (%)")
    ax.set_title(f"Models first evaluated in generation {generation}")
    ax.grid(True, linestyle=":", alpha=.6); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path


def plot_all_generations_2d(points, seed, limits, path, title):
    """Plot every model, colored by its first generation, with model labels."""
    fig, ax = plt.subplots(figsize=(14, 10))
    generations = sorted(points["generation"].unique())
    colors = plt.cm.turbo([i / max(1, len(generations) - 1) for i in range(len(generations))])
    for generation, color in zip(generations, colors):
        subset = points[points["generation"] == generation]
        ax.scatter(subset["mAP50"] * 100, subset["mAP50_95"] * 100,
                   s=45, color=color, edgecolor="black", alpha=.8,
                   label=f"Generation {generation}")
        for _, row in subset.iterrows():
            ax.annotate(short_model_id(row["gene_id"]),
                        (row["mAP50"] * 100, row["mAP50_95"] * 100),
                        xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=.85)
    front = points[pareto_2d_mask(points)].sort_values("mAP50")
    ax.scatter(front["mAP50"] * 100, front["mAP50_95"] * 100,
               s=130, facecolors="none", edgecolors="crimson", linewidths=2,
               label="Final Pareto front")
    front_x = front["mAP50"] * 100
    front_y = front["mAP50_95"] * 100
    ax.step(front_x, front_y, where="post", color="crimson", linewidth=2.5,
            label="Pareto frontier")
    if seed:
        ax.scatter(seed["mAP50"] * 100, seed["mAP50_95"] * 100,
                   marker="x", color="black", s=120, linewidths=2.5, label="Seed")
    ax.set_xlim(*limits["mAP50"]); ax.set_ylim(*limits["mAP50_95"])
    ax.set_xlabel("mAP50 (%)"); ax.set_ylabel("mAP50:95 (%)")
    ax.set_title(title); ax.grid(True, linestyle=":", alpha=.6)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=250); plt.close(fig)
    return path


def plot_best_by_generation(history, seed, output_dir, title="Best candidate after each generation"):
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
    ax.set_title(title)
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

def literal_genes(path):
    if not os.path.isfile(path): return None
    genes={}
    for chunk in open(path).read().split("# --OPTION--")[1:]:
        tree=ast.parse(chunk)
        if tree.body and isinstance(tree.body[0],ast.ClassDef):
            for statement in tree.body[0].body:
                if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=="values": genes.update(ast.literal_eval(statement.value))
    return genes or None

def plot_winner_lineage(manifest_path,generations,final,models_dir,seed_file,output_dir):
    manifest=json.load(open(manifest_path)); ancestry=manifest.get("ancestry",{}); ids=ancestry.get("GENES",[manifest["gene_id"]]); operations=ancestry.get("MUTATE_TYPE",[])
    first={}
    for generation,path in generations:
        for gene_id in load_candidates(path)["gene_id"]: first.setdefault(gene_id,generation)
    fitness=final.set_index("gene_id"); rows=[]; previous=None
    for index,gene_id in enumerate(ids):
        source=seed_file if gene_id=="seed" else os.path.join(models_dir,f"network_{gene_id}.py"); genes=literal_genes(source)
        changed=[] if previous is None or genes is None else [k for k,v in genes.items() if previous.get(k)!=v]
        row={"lineage_step":index,"generation_first_seen":first.get(gene_id),"gene_id":gene_id,"operation":operations[index] if index<len(operations) else "UNKNOWN","changed_genes":",".join(changed),"genes":json.dumps(genes,sort_keys=True) if genes else None,"model_file":source}
        if gene_id in fitness.index:
            score=fitness.loc[gene_id]; row.update(mAP50=score["mAP50"],mAP50_95=score["mAP50_95"],params=int(score["params"]))
        rows.append(row); previous=genes or previous
    lineage=pd.DataFrame(rows); lineage.to_csv(os.path.join(output_dir,"winner_lineage.csv"),index=False); scored=lineage.dropna(subset=["mAP50","mAP50_95","params"])
    if scored.empty: return
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(scored["lineage_step"],scored["mAP50"]*100,marker="o",label="mAP50"); ax.plot(scored["lineage_step"],scored["mAP50_95"]*100,marker="o",label="mAP50:95"); size=ax.twinx(); size.plot(scored["lineage_step"],scored["params"],color="gray",marker="s",linestyle="--",label="Parameters"); ax.set_title("Selected generation-20 model lineage"); ax.set_xlabel("Lineage step"); ax.set_ylabel("Validation score %"); size.set_ylabel("Parameters"); lines=ax.get_lines()+size.get_lines(); ax.legend(lines,[line.get_label() for line in lines]); fig.tight_layout(); fig.savefig(os.path.join(output_dir,"winner_lineage.png"),dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 3D Pareto fronts for the FRED LLM-GE run")
    parser.add_argument("--checkpoints", default=os.path.join(ROOT_DIR, "fred_yolo11_checkpoints"),
                        help="Folder with global_gen_*.pkl")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "analysis", "results", "analysis", "fred_pareto"),
                        help="Folder to write plots and CSVs to")
    parser.add_argument("--seed_results", default=os.path.join(YOLO11_DIR, "results", "seed_results.csv"),
                        help="Seed score CSV (mAP50,mAP50:95,params); skipped if missing")
    parser.add_argument("--title",default="FRED YOLO11 LLM-GE")
    parser.add_argument("--models_dir",default=os.path.join(YOLO11_DIR,"models"))
    parser.add_argument("--seed_file",default=os.path.join(ROOT_DIR,"sota","FRED_LLM_GE","seeds","seed_yolo11.py"))
    parser.add_argument("--winner_manifest")
    parser.add_argument("--no_gif", action="store_true", help="Skip the animated GIF")
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

    frames, history, first_seen_rows, seen = [], [], [], set()
    for gen, df in by_gen:
        title = f"{args.title}: generation {gen} ({n_candidates(df)} valid candidates)"
        frames.append(plot_front_3d(df, title, seed, limits,
                                    os.path.join(args.output, f"pareto3d_gen_{gen}.png")))
        new_models = df[(df["gene_id"] != "seed") & ~df["gene_id"].isin(seen)].copy()
        new_models["generation"] = gen
        first_seen_rows.append(new_models)
        seen.update(new_models["gene_id"])
        plot_generation_2d(new_models, df, gen, limits,
                           os.path.join(args.output, f"pareto2d_gen_{gen}.png"))
        history.append({"generation": gen, "mAP50": df["mAP50"].max(),
                        "mAP50_95": df["mAP50_95"].max()})
    print(f"Saved {len(frames)} per-generation 3D plots to {args.output}")

    final_title = (f"{args.title}: final Pareto front after generation {last_gen} "
                   f"({n_candidates(final)} valid candidates)")
    print("Saved " + plot_front_3d(final, final_title, seed, limits,
                                   os.path.join(args.output, "pareto3d_final.png"), dpi=300))
    model_points = pd.concat(first_seen_rows, ignore_index=True)
    model_points["pareto_front"] = pareto_2d_mask(model_points)
    model_points.to_csv(os.path.join(args.output, "model_points_by_generation.csv"), index=False)
    pareto2d = model_points[model_points["pareto_front"]].copy().sort_values("mAP50")
    pareto2d["mAP50_percent"] = pareto2d["mAP50"] * 100
    pareto2d["mAP50_95_percent"] = pareto2d["mAP50_95"] * 100
    pareto2d["model_file"] = [os.path.join(args.models_dir, f"network_{gene_id}.py")
                              for gene_id in pareto2d["gene_id"]]
    table_columns = ["generation", "gene_id", "mAP50", "mAP50_95", "mAP50_percent",
                     "mAP50_95_percent", "params", "model_file"]
    pareto2d[table_columns].to_csv(os.path.join(args.output, "pareto2d_models.csv"), index=False)
    markdown = ["# Two-objective Pareto-front models", "",
                "| Generation | Individual | mAP50 | mAP50:95 | Parameters |",
                "|---:|---|---:|---:|---:|"]
    for _, row in pareto2d.iterrows():
        markdown.append(f"| {int(row['generation'])} | `{row['gene_id']}` | "
                        f"{row['mAP50_percent']:.2f}% | {row['mAP50_95_percent']:.2f}% | "
                        f"{int(row['params']):,} |")
    with open(os.path.join(args.output, "pareto2d_models.md"), "w") as file:
        file.write("\n".join(markdown) + "\n")
    print("Saved " + plot_all_generations_2d(
        model_points, seed, limits, os.path.join(args.output, "pareto2d_all_generations.png"),
        args.title + ": mAP50 vs mAP50:95 by first generation"))

    if not args.no_gif and len(frames) > 1:
        make_gif(frames, os.path.join(args.output, "pareto3d_fronts.gif"))
        print("Saved pareto3d_fronts.gif")
    print(f"Saved {plot_best_by_generation(pd.DataFrame(history), seed, args.output, args.title + ': best by generation')}")

    final = final.copy()
    final["pareto_front"] = pareto_mask(final)
    final = final.sort_values("mAP50", ascending=False)
    final.to_csv(os.path.join(args.output, "candidates.csv"), index=False)

    models = final[final["pareto_front"]].sort_values("params").copy()
    models["model_file"] = [args.seed_file if g == "seed"
                            else os.path.join(args.models_dir, f"network_{g}.py")
                            for g in models["gene_id"]]
    models.to_csv(os.path.join(args.output, "pareto_models.csv"), index=False)
    if args.winner_manifest:
        plot_winner_lineage(args.winner_manifest,generations,final,args.models_dir,args.seed_file,args.output)

    print(f"\nPareto front (mAP50, mAP50:95, params) after generation {last_gen}:")
    if seed:
        print(f"  seed: mAP50 {seed['mAP50']:.1%}, mAP50:95 {seed['mAP50_95']:.1%}, "
              f"params {int(seed['params']):,}")
    for _, row in models.iterrows():
        print(f"  {row['gene_id']}: mAP50 {row['mAP50']:.1%}, mAP50:95 {row['mAP50_95']:.1%}, "
              f"params {row['params']:,}")


if __name__ == "__main__":
    main()
