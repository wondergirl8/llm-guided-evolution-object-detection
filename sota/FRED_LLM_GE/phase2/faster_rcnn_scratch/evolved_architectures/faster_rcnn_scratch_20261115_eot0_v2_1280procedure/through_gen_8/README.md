# Generation-8 evolved architectures

These are the 56 successfully scored architecture sources included at the generation-8 cutoff. `manifest.json` records validation metrics, parameter counts, source hashes, and PACE task directories. These small files define architectures; they do not contain trained weights.

Selected for 50-epoch final retraining: `g05_slot4_p01.py` (28,957,013 parameters).

PACE repository root:
`/storage/ice1/3/8/mgrushina3/llm-guided-evolution-object-detection`

Search weights and provenance for each model:
`fred_runs/faster_rcnn_scratch_20261115_eot0_v2_1280procedure/tasks/<model_id>/`

- `source.py`: architecture source.
- `resolved.json`: resolved genes and identity.
- `result.json`: search-validation scores.
- `model.pt`: completed 20-epoch search checkpoint.
- `training.pt`: optimizer/RNG/batch-resume state.

The stopped evolution checkpoint is `fred_runs/faster_rcnn_scratch_20261115_eot0_v2_1280procedure/checkpoints/global_gen_8.pkl`.

Independent 50-epoch final-training seed checkpoints are under:
`fred_runs/faster_rcnn_scratch_20261121_eot0_v2_final_g05s4_50ep/tasks/final_000_sXX/`, where XX is 01 through 30. During training, `training.pt` is the resumable checkpoint; `model.pt` is written after all 50 epochs finish. Official-test reports follow completed training.

Weights, datasets, and mutable run state are excluded from Git. Search-validation metrics must not be presented as official-test results.
