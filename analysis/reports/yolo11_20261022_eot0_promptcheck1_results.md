# FRED YOLO11n evolution: completed generation-20 run

Run tag: `yolo11_20261022_eot0_promptcheck1`. The controller evaluated the stock YOLO11n seed in generation 0, then completed evolutionary generations 1–20. The run status is `complete` at generation 20 (`fred_runs/yolo11_20261022_eot0_promptcheck1/status.json:2-3`). The first controller saved generation 9 before its server-time parsing error (`run_job_outputs/islands/slurm-6479225.out:14599-14608`); the second loaded that checkpoint and finished generation 20 (`run_job_outputs/islands/slurm-6484898.out:13-19,12548-12553`). The final `checkpoint_gen_20.pkl` and `global_gen_20.pkl` are both present.

## Figures and source data

- [Final two-panel Pareto plot](../results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/pareto2d_final.png)
- [Best score after each generation](../results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/best_by_generation.png)
- [Animated generation-by-generation fronts](../results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/pareto2d_fronts.gif)
- [All valid candidate records](../results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/candidates.csv) and [three-objective Pareto records](../results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/pareto_models.csv)

The left panel shows mAP50 against parameter count; the right shows mAP50:95 against parameter count. Each red 2D front is computed for that panel's accuracy metric and model size. `pareto_models.csv` instead uses *both* accuracy metrics and size simultaneously, so its membership can differ from either red front (`analysis/scripts/fred_pareto_graphs.py:78-105,109-158,249-265`). The black cross is the stock seed. Overlapping points are annotated with `×N`, but the labels around 1.3–1.4 million parameters crowd each other; use the CSV for exact identities and scores. The plots contain all **valid, scored candidate records** saved through generation 20, including the seed. Rejected and unscored intermediate records are excluded by the plotter's fitness filter (`analysis/scripts/fred_pareto_graphs.py:50-74,225-229`). The generated image/CSV directory is ignored by `.gitignore`; regenerate it with:

```bash
MPLCONFIGDIR=/tmp/llmge-mpl .venv/bin/python analysis/scripts/fred_pareto_graphs.py \
  --family yolo11 \
  --checkpoints fred_runs/yolo11_20261022_eot0_promptcheck1/checkpoints \
  --models_dir fred_runs/yolo11_20261022_eot0_promptcheck1/candidates/models \
  --output analysis/results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1
```

## Main result

All accuracy values below are **search-validation** results on 6,733 event frames from FRED's official `challenging_train` split, not challenging-test results (`fred_runs/yolo11_20261022_eot0_promptcheck1/candidates/results/xXxlhk4QygNwrjKhaujkG3IYhGx_evaluation.json:15-26`; `sota/FRED_LLM_GE/configs/phase1/protocol.yaml:37-65`). The saved training arguments for the leading candidate show 20 epochs, batch 16, image size 1280, seed 0, SGD, and `pretrained: false` (`fred_runs/yolo11_20261022_eot0_promptcheck1/candidates/training_args/xXxlhk4QygNwrjKhaujkG3IYhGx/args.yaml:5-22,73-75`).

| Model | First seen | mAP50 | mAP50:95 | Parameters | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| Stock seed (`seed`) | 0 | 84.2307% | 43.1665% | 2,590,035 | The in-run comparison point. |
| `xXxlhk4QygNwrjKhaujkG3IYhGx` | 20 | **85.9272%** | 45.0720% | 1,379,279 | Highest mAP50; on the three-objective front. |
| `xXx9RRLA8jFvH8Yj7rvDFpMlFLe` | 10 | 84.9857% | **45.1919%** | 1,388,439 | Highest mAP50:95; on the three-objective front. |
| `xXx5FrwxtdXM9Fs4O34VKQDlUgN` | 9 | 82.1244% | 39.2868% | **1,303,967** | Smallest evaluated graph; lower accuracy is its tradeoff. |

Scores and sizes are recorded in `analysis/results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/candidates.csv:2,10,24,56`; the leading candidate's unrounded metrics are in its evaluation report, lines 20–23. The highest-mAP50 candidate exceeds the in-run stock seed by **1.6965 percentage points mAP50** and **1.9055 points mAP50:95**, with **46.75% fewer parameters**. It strictly dominates the stock seed on all three objectives. Across the final checkpoint, 16 candidate records representing 14 distinct built graphs strictly dominate the seed. The smallest graph uses 49.65% fewer parameters than the seed but scores lower on both accuracy metrics; it remains on a Pareto front because no scored graph is smaller (`candidates.csv:24,56`; `pareto_models.csv:2`).

An independent comparison of every row in `candidates.csv` reproduced its 14-row three-objective front, and every `model_file` named in `pareto_models.csv` exists. Those 14 records represent **11 distinct built graphs**. The two 2D panels have 10 and 9 frontier records respectively, each at eight distinct score/size coordinates. The stock seed is not on either 2D front. The front clusters between 1,303,967 and 1,388,439 parameters, close to the validator's 1,295,018-parameter lower bound (`sota/FRED_LLM_GE/phase1_detection/yolo11_search_space.py:14-16,191-193`; `pareto_models.csv:2-15`). This is an observed concentration, not proof that the wider search space has been exhausted.

## Search progress and reliability

The checkpoint history shows cumulative valid records and best scores, rather than only the individuals alive in each generation:

| Through generation | Valid records | Best mAP50 | Best mAP50:95 | Smallest model |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 85.2680% | 43.7952% | 2,564,019 |
| 5 | 30 | 85.2680% | 44.1907% | 1,407,691 |
| 10 | 55 | 85.2680% | 45.1919% | 1,303,967 |
| 15 | 71 | 85.2680% | 45.1919% | 1,303,967 |
| 20 | 82 | 85.9272% | 45.1919% | 1,303,967 |

These values were recomputed from `fred_runs/yolo11_20261022_eot0_promptcheck1/checkpoints/global_gen_{0,5,10,15,20}.pkl` using the plotter's validity rule (`analysis/scripts/fred_pareto_graphs.py:50-57,62-74`). Generation 0 contained eight distinct successful built graphs, including the stock seed (`run_job_outputs/islands/slurm-6479225.out:1696-1701`). The strongest mAP50:95 value first appeared by generation 10; the strongest mAP50 value appeared in generation 20. The number of distinct successfully trained graphs grew from 8 in generation 0 to 54 by generation 10 and 78 by generation 20. The slower second-half growth is an observation; the logs do not isolate one cause.

The final checkpoint contains **152** gene records: **78** distinct successfully trained graph records, **4** additional valid records that reused cached fitness for those same graphs, **26** explicitly rejected proposals, and **44** submitted crossover records with no fitness. I obtained these counts by merging `GLOBAL_DATA_HIST` and `GLOBAL_DATA` from `global_gen_20.pkl`, applying the plotter's finite-fitness rule, then comparing `architecture_hash`, `results_job`, and `failure_category` fields. Those 44 are unscored intermediates, not 44 additional failed model evaluations; the controller applies mutation after crossover before evaluating offspring (`run_improved.py:1600-1635`). Three groups of valid records have identical built-graph hashes and scores; the four duplicate records have no evaluation job ID, consistent with cache reuse. The plotter therefore shows 82 valid records, while only 78 distinct graphs were trained (`fred_runs/yolo11_20261022_eot0_promptcheck1/checkpoints/global_gen_20.pkl`; `analysis/results/analysis/fred_pareto/yolo11_20261022_eot0_promptcheck1/candidates.csv:1-83`).

The 26 recorded rejections break down as **16 `unchanged_parent`**, **7 `duplicate_graph`**, and **3 `invalid_architecture`**. All 16 unchanged-parent rejections were mutation outputs. The three invalid architectures were two crossover outputs and one mutation output. A representative mutation exhausted five attempts because its built graph kept matching the parent (`run_job_outputs/evolution/slurm-6480428.out:581`); a crossover changed a required downsampling pattern (`run_job_outputs/evolution/slurm-6481956.out:2543-2549`); another produced 1,249,543 parameters, below the allowed 1,295,018 (`run_job_outputs/evolution/slurm-6484966.out:772-778`). The final checkpoint has no recorded `llm_unavailable`, `submission_failed`, or `gpu_unsupported` failure categories. The generation-17 log reports four unchanged-parent rejections among eight offspring (`run_job_outputs/islands/slurm-6484898.out:9127`).

The parent pool after invalid removal ranged from **4 to 8** during generations 1–20. It had only four parents entering generations 18 and 20 (`run_job_outputs/islands/slurm-6484898.out:9205-9212,11232-11239`). This reduces the variety available for selection and mating. It did not prevent completion: generation 20 ended with six scored population members and zero recorded failures (`run_job_outputs/islands/slurm-6484898.out:12522-12553`). Maintaining eight parents after generation 0 was not enforced in this run; do not describe every generation as having population eight.

## Comparison limits and next validation

The independent stock baseline trained seeds 0, 1, and 2 from scratch. Its mean ± sample SD is **83.8617% ± 1.7431 points mAP50** and **42.3691% ± 1.9163 points mAP50:95**, at 2,590,035 parameters (`sota/FRED_LLM_GE/baseline_validation/phase1_detection/yolo11_stock/summary.json:33-47`). Its seed-0 scores exactly match the in-run stock seed's reported scores (`summary.json:55-59`; `fred_runs/yolo11_20261022_eot0_promptcheck1/candidates/results/seed_evaluation.json:20-23`). The best candidate's gains over seed 0 are about the same size as these observed baseline seed-to-seed standard deviations, so one candidate training seed cannot establish a robust accuracy improvement.

The baseline was made at Git commit `646b12d8`, while this evolution checkpoint reports `938b42be`; therefore their full provenance fingerprints differ (`summary.json:17`; the leading candidate report, line 12). A local comparison found equal hashes for the protocol, seed YAML, split lists, model adapter, candidate runner, fitness/reporting code, and equal PyTorch and Ultralytics versions. Their `training_config_identity` values also match (leading candidate report, line 14; baseline seed-0 report, line 14). The only changed file among the baseline's listed code hashes is `yolo11_search_space.py`: the intervening changes add prompt retry/parsing/repair and adjust the cache key; its model-build and validation functions are unchanged (`sota/FRED_LLM_GE/phase1_detection/yolo11_search_space.py:77-96,96-159,186-200`). These checks support a protocol-level comparison, but the different commit prevents calling the two runs bit-for-bit identical. The controller also resumed after a local, uncommitted server-time parser fix; that fix affects scheduling, not candidate training or evaluation (`run_improved.py:44-78`).

The results were selected on the same search-validation sequences used to report them. The available evidence supports the conclusion that this run found smaller architectures with promising validation scores; it does **not** yet establish a generalization gain on unseen flights. The next controlled comparison is to retrain a small, preselected set of nonduplicate front graphs—especially `xXxlhk4QygNwrjKhaujkG3IYhGx` and `xXx9RRLA8jFvH8Yj7rvDFpMlFLe`—with seeds 0, 1, and 2 through the same frozen runner and protocol. Compare each distribution with the stock baseline, then freeze model selection before the one-time challenging-test evaluation (`sota/FRED_LLM_GE/configs/phase1/protocol.yaml:37-70`).
