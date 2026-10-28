"""Score evolved renderers on more training seeds, against plain palette frames
at the same window.

The renderer run (FRED_SEED_FAMILY=renderer) trains YOLO11n once per renderer,
with proxy seed 0, so a renderer that beats the released frames by a few
thousandths may only have drawn a lucky seed. This scores the chosen renderers
again with other seeds, and the seed renderer (released palette frames) cut to
each other window on every seed, so each renderer is compared with plain
palette frames that wait just as long.

    # 1. one job per (renderer, seed), plus jobs.txt and a Slurm array script
    uv run python sota/FRED_LLM_GE/phase1_detection/retrain_renderers.py setup \\
        --genes xXx1b8y2QjtJktieR8P4J9lgNpm xXxYBInRiVcQFKdxH2K0IRh4wOl --palette 16666 15000
    # 2. once the jobs finish
    uv run python sota/FRED_LLM_GE/phase1_detection/retrain_renderers.py summary

Evolved renderers are read from seeds/renderer/models (or --run_root) and their search score
counts as seed 0. Each setup adds to the earlier plan, so jobs.txt holds only
the new jobs while the summary covers all of them. Palette renderers are
written to <out>/models. Palette frames
at 10 ms and 33.3 ms already have seeds 0-2 from event_sweep.py, which the
summary includes. Each job writes to <data_root>/renderer_retrain/<name>/seed<k>/.
"""

import argparse
import copy
import json
import re
from pathlib import Path

import yaml

try:
    from .event_sweep import REPO_ROOT, SBATCH, mean_sd, read_result
    from .export_split import DEFAULT_PROTOCOL, FRED_DIR
    from .render_runner import SEED_RENDERER
except ImportError:
    from event_sweep import REPO_ROOT, SBATCH, mean_sd, read_result
    from export_split import DEFAULT_PROTOCOL, FRED_DIR
    from render_runner import SEED_RENDERER

RUN_ROOT = FRED_DIR / "phase1_detection" / "seeds" / "renderer"
# Palette settings event_sweep.py already scored on seeds 0-2.
SWEPT = {"palette 33333 (released)": "released", "palette 10000": "raw_w10000_palette"}


def palette_renderer(models, window_us):
    """The seed renderer with WINDOW_US = window_us, as models/network_palette_w<window_us>.py."""
    text, count = re.subn(r"^WINDOW_US = \d+$", f"WINDOW_US = {window_us}",
                          SEED_RENDERER.read_text(), flags=re.M)
    if count != 1:
        raise ValueError(f"{SEED_RENDERER}: expected one WINDOW_US line")
    path = models / f"network_palette_w{window_us}.py"
    path.write_text(text)
    return path.stem.split("network_")[1]


def setup(protocol, out_dir, genes, windows, seeds, run_root):
    models = out_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    jobs = [(gene, run_root / "models", seeds) for gene in genes]
    # Palette renderers have no search score, so they also need seed 0.
    jobs += [(palette_renderer(models, w), models, sorted({0, *seeds})) for w in windows]
    commands = []
    for name, variant_dir, job_seeds in jobs:
        if not (variant_dir / f"network_{name}.py").is_file():
            raise FileNotFoundError(variant_dir / f"network_{name}.py")
        for seed in job_seeds:
            job_dir = out_dir / name / f"seed{seed}"
            job_dir.mkdir(parents=True, exist_ok=True)
            job_protocol = copy.deepcopy(protocol)
            job_protocol["proxy"]["seed"] = seed
            (job_dir / "protocol.yaml").write_text(yaml.safe_dump(job_protocol, sort_keys=False))
            commands.append(
                f"uv run python {FRED_DIR / 'phase1_detection' / 'render_runner.py'} "
                f"--model network_{name} --variant_dir {variant_dir} "
                f"--protocol {job_dir / 'protocol.yaml'} --sota_root {job_dir}")
    # Add to an earlier plan, so the summary still covers its renderers.
    plan_file = out_dir / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.is_file() else {"genes": [], "palette": []}
    plan = {"genes": list(dict.fromkeys(plan["genes"] + genes)),
            "run_roots": dict(plan.get("run_roots", {}), **{g: str(run_root) for g in genes}),
            "palette": sorted(set(plan["palette"] + windows), reverse=True)}
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    jobs_file = out_dir / "jobs.txt"
    jobs_file.write_text("\n".join(commands) + "\n")
    script = out_dir / "train_renderers.sbatch"
    log_dir = REPO_ROOT / "run_job_outputs" / "renderer_retrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    script.write_text(SBATCH.replace("event_sweep", "renderer_retrain").format(
        repo=REPO_ROOT, out=log_dir, last=len(commands) - 1, script=script, jobs=jobs_file))
    print(f"{len(genes)} evolved renderers (seeds {' '.join(map(str, seeds))}; the search score counts as seed 0), "
          f"{len(windows)} palette windows (seeds {' '.join(map(str, sorted({0, *seeds})))}): {len(commands)} jobs")
    print(f"from {REPO_ROOT}:\n  sbatch --array=0-{len(commands) - 1}%4 {script}")


def summary(out_dir, sweep_dir):
    plan = json.loads((out_dir / "plan.json").read_text())
    rows = [(label, [read_result(p) for p in sorted((sweep_dir / tag).glob("seed*/results/seed_results.csv"))])
            for label, tag in SWEPT.items()]
    rows += [(f"palette {w}", [read_result(p) for p in sorted((out_dir / f"palette_w{w}").glob(
        "seed*/results/*_results.csv"))]) for w in plan["palette"]]
    for gene in plan["genes"]:
        run_root = Path(plan.get("run_roots", {}).get(gene, RUN_ROOT))
        search = read_result(run_root / "results" / f"{gene}_results.csv")
        rows.append((gene, [search] + [read_result(p) for p in sorted((out_dir / gene).glob(
            "seed*/results/*_results.csv"))]))
    header = "renderer,window_us,runs,mAP50_mean,mAP50_sd,mAP50_95_mean,mAP50_95_sd"
    lines = [header]
    print(f"{'renderer':<30}{'window':>8}{'runs':>6}{'mAP50 mean ± sd':>18}{'mAP50:95 mean ± sd':>20}")
    for name, scores in rows:
        scores = [s for s in scores if s]
        if not scores:
            print(f"{name:<30}{'-':>8}{0:>6}")
            continue
        m50, sd50 = mean_sd([s[0] for s in scores])
        m95, sd95 = mean_sd([s[1] for s in scores])
        # Sweep results hold the model's parameter count, not the window.
        window = name.split()[1] if name.startswith("palette") else scores[0][2]
        print(f"{name:<30}{window:>8}{len(scores):>6}{f'{m50:.4f} ± {sd50:.4f}':>18}{f'{m95:.4f} ± {sd95:.4f}':>20}")
        lines.append(f"{name},{window},{len(scores)},{m50:.6f},{sd50:.6f},{m95:.6f},{sd95:.6f}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"-> {out_dir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Score evolved renderers on more seeds against palette frames")
    parser.add_argument("command", choices=["setup", "summary"])
    parser.add_argument("--genes", nargs="+", default=[], help="Evolved renderer gene ids (setup only)")
    parser.add_argument("--palette", type=int, nargs="+", default=[],
                        help="Windows in us to score plain palette frames at (setup only)")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--run_root", default=str(RUN_ROOT),
                        help="Renderer run the --genes come from, e.g. seeds/renderer_w2000 (setup only)")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    data_root = FRED_DIR / protocol["challenging"]["data_root"]
    out_dir = data_root / "renderer_retrain"
    if args.command == "setup":
        if not args.genes and not args.palette:
            parser.error("setup needs --genes or --palette")
        if 0 in args.seeds:
            parser.error("seed 0 is the search score; retrain other seeds")
        setup(protocol, out_dir, args.genes, args.palette, args.seeds, Path(args.run_root))
    else:
        summary(out_dir, data_root / "event_sweep")


if __name__ == "__main__":
    main()
