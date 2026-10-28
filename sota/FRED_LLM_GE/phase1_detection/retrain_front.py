"""Retrain each LLM-GE run's Pareto-front models on more training seeds.

During the search every candidate is trained once, with proxy seed 0, so its
search score is one noisy sample. This trains the front models (from
fred_pareto_graphs.py's pareto_models.csv) again with other seeds under the same
protocol and event setting, then averages them with the search score as seed 0.
YOLO11n's scores per seed come from event_sweep.py.

    # 1. one job per (front model, seed), plus jobs.txt and a Slurm array script
    uv run python sota/FRED_LLM_GE/phase1_detection/retrain_front.py setup \\
        --front released analysis/results/analysis/fred_pareto_released/pareto_models.csv \\
        --front raw_w10000_palette analysis/results/analysis/fred_pareto_raw_w10000_palette/pareto_models.csv
    # 2. once the jobs finish
    uv run python sota/FRED_LLM_GE/phase1_detection/retrain_front.py summary

Model files are read from the run's own models folder,
seeds/ultralytics_yolo11[_<events tag>]/models, and each job writes to
<data_root>/front_retrain/<events tag>/<gene_id>/seed<k>/. Genes whose files are
identical are trained once.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import yaml

try:
    from .event_sweep import REPO_ROOT, SBATCH, mean_sd, read_result
    from .export_split import DEFAULT_PROTOCOL, FRED_DIR, events_from_tag
except ImportError:
    from event_sweep import REPO_ROOT, SBATCH, mean_sd, read_result
    from export_split import DEFAULT_PROTOCOL, FRED_DIR, events_from_tag

SEEDS_DIR = FRED_DIR / "phase1_detection" / "seeds"


def run_root(tag):
    """The LLM-GE run's SOTA_ROOT (constants_fred_yolo11.py appends the events tag)."""
    return SEEDS_DIR / ("ultralytics_yolo11" if tag == "released" else f"ultralytics_yolo11_{tag}")


def front_genes(tag, pareto_csv):
    """Front gene ids with distinct model files, smallest first."""
    with open(pareto_csv) as file:
        rows = sorted((r for r in csv.DictReader(file) if r["gene_id"] != "seed"),
                      key=lambda r: int(r["params"]))
    genes, seen = [], set()
    for row in rows:
        model = run_root(tag) / "models" / f"network_{row['gene_id']}.py"
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            genes.append(row["gene_id"])
    return genes


def setup(protocol, out_dir, fronts, seeds, family):
    commands, plan = [], {}
    for tag, pareto_csv in fronts:
        plan[tag] = front_genes(tag, pareto_csv)
        for gene in plan[tag]:
            for seed in seeds:
                job_dir = out_dir / tag / gene / f"seed{seed}"
                job_dir.mkdir(parents=True, exist_ok=True)
                job_protocol = copy.deepcopy(protocol)
                job_protocol["challenging"]["events"] = events_from_tag(tag)
                job_protocol["proxy"]["seed"] = seed
                (job_dir / "protocol.yaml").write_text(yaml.safe_dump(job_protocol, sort_keys=False))
                commands.append(
                    f"uv run python {FRED_DIR / 'phase1_detection' / 'candidate_runner.py'} "
                    f"--family {family} --model network_{gene} "
                    f"--variant_dir {run_root(tag) / 'models'} "
                    f"--protocol {job_dir / 'protocol.yaml'} --sota_root {job_dir}")
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = out_dir / "jobs.txt"
    jobs.write_text("\n".join(commands) + "\n")
    script = out_dir / "train_front.sbatch"
    log_dir = REPO_ROOT / "run_job_outputs" / "front_retrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    script.write_text(SBATCH.replace("event_sweep", "front_retrain").format(
        repo=REPO_ROOT, out=log_dir, last=len(commands) - 1, script=script, jobs=jobs))
    for tag, genes in plan.items():
        print(f"{tag}: {len(genes)} front models")
    print(f"{len(commands)} jobs (seeds {' '.join(map(str, seeds))}; the search score counts as seed 0)")
    print(f"from {REPO_ROOT}:\n  sbatch --array=0-{len(commands) - 1}%4 {script}")


def summary(out_dir, sweep_dir):
    plan = json.loads((out_dir / "plan.json").read_text())
    header = "setting,model,params,runs,mAP50_search,mAP50_mean,mAP50_sd,mAP50_95_search,mAP50_95_mean,mAP50_95_sd"
    lines = [header]
    print(f"{'setting':<20}{'model':<30}{'params':>10}{'runs':>6}{'mAP50 mean ± sd':>18}"
          f"{'mAP50:95 mean ± sd':>20}{'search mAP50':>14}")
    for tag, genes in plan.items():
        # YOLO11n first: all its seeds come from the event sweep.
        rows = [("YOLO11n", None, [read_result(p) for p in
                                   sorted((sweep_dir / tag).glob("seed*/results/seed_results.csv"))])]
        for gene in genes:
            search = read_result(run_root(tag) / "results" / f"{gene}_results.csv")
            retrained = [read_result(p) for p in sorted((out_dir / tag / gene).glob("seed*/results/*_results.csv"))]
            rows.append((gene, search, [search] + retrained))
        for name, search, scores in rows:
            scores = [s for s in scores if s]
            if not scores:
                print(f"{tag:<20}{name:<30}{'-':>10}{0:>6}")
                continue
            m50, sd50 = mean_sd([s[0] for s in scores])
            m95, sd95 = mean_sd([s[1] for s in scores])
            params = scores[0][2]
            shown_search = f"{search[0]:.4f}" if search else "-"
            print(f"{tag:<20}{name:<30}{params / 1e6:>9.2f}M{len(scores):>6}"
                  f"{f'{m50:.4f} ± {sd50:.4f}':>18}{f'{m95:.4f} ± {sd95:.4f}':>20}{shown_search:>14}")
            lines.append(",".join(str(v) for v in [
                tag, name, params, len(scores), search[0] if search else "", f"{m50:.6f}", f"{sd50:.6f}",
                search[1] if search else "", f"{m95:.6f}", f"{sd95:.6f}"]))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"-> {out_dir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Retrain LLM-GE Pareto-front models on more seeds")
    parser.add_argument("command", choices=["setup", "summary"])
    parser.add_argument("--front", nargs=2, action="append", metavar=("EVENTS_TAG", "PARETO_CSV"),
                        help="A run's events tag and its pareto_models.csv; repeat per run (setup only)")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--family", default="yolo11")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    data_root = FRED_DIR / protocol["challenging"]["data_root"]
    out_dir = data_root / "front_retrain"
    if args.command == "setup":
        if not args.front:
            parser.error("setup needs at least one --front")
        if 0 in args.seeds:
            parser.error("seed 0 is the search score; retrain other seeds")
        setup(protocol, out_dir, args.front, args.seeds, args.family)
    else:
        summary(out_dir, data_root / "event_sweep")


if __name__ == "__main__":
    main()
