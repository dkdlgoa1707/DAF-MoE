# Phase 2 HPO and Final Evaluation

`runners/run_phase2.py` is the single Phase 2 orchestration entrypoint. Search
spaces live in `configs/hpo/phase2/`, and dataset-level task/metric inputs live
in `configs/experiments/phase2/base/`.

## Global contract

- Optuna `TPESampler(seed=42)` with `NopPruner` and RDB-backed studies.
- Study names and SQLite files are `{dataset}__{model}__phase2-v2__{signature12}`. The signature covers the protocol, dataset schema, model implementation, task/metric, search space, base config, and effective target policy.
- HPO constructs train and validation only. Seed 42 cannot enter final results.
- A study stops after 50 finite `COMPLETE` trials. `FAIL`, OOM, invalid shape,
  nonfinite metrics, `RUNNING` trials, and trials with mismatched signature attributes do not count.
- `--complete-trials` values other than 50 require explicit `--smoke`.
- Final evaluation defaults to the common seed registry 43 through 57.
- Every seed refits its split-specific preprocessing and checks the stored
  protocol, split, preprocessing, search-space, and resolved-config hashes
  before resuming.
- Neural early stopping restores the best state in memory and writes the
  explicit trial/seed checkpoint. XGBoost uses 300-round early stopping,
  CatBoost uses `od_type=Iter, od_wait=300`, and RealMLP uses pytabkit's
  estimator-owned epoch selection.

## Method mapping

| Method | Search YAML | Execution | Early stopping | Main rank |
|---|---|---|---|---:|
| DAF-MoE v2 | `daf_moe_v2.yaml` | PyTorch trainer | configured patience | yes |
| XGBoost | `xgboost.yaml` | native sklearn API | 300 rounds | yes |
| CatBoost | `catboost.yaml` | native CatBoost API | Iter/300 | yes |
| MLP | `mlp.yaml` | PyTorch trainer | 400 max / 16 patience | yes |
| ResNet | `resnet.yaml` | PyTorch trainer | 400 max / 16 patience | yes |
| FT-Transformer | `ft_transformer.yaml` | PyTorch trainer | configured patience | yes |
| TabR full | `tabr.yaml` | PyTorch trainer | configured patience | yes |
| TabM-mini | `tabm.yaml` | PyTorch trainer | configured patience | yes |
| TabM†-mini | `tabm_ple.yaml` | PyTorch trainer | configured patience | no |
| ModernNCA | `modernnca.yaml` | PyTorch trainer | configured patience | yes |
| RealMLP-HPO | `realmlp.yaml` | official pytabkit | official epoch selection | yes |
| TabICLv2 | `tabicl.yaml` (fixed only) | official pretrained estimator | no HPO | yes |

Pytabkit 1.7.3's `RealMLP_HPO_*` constructor owns an internal random search and
does not accept an individual sampled architecture. To preserve the common
Optuna TPE protocol and the fixed search space, each outer trial uses the same
package's official `RealMLP_TD_*` single-configuration estimator. Pytabkit still
owns preprocessing, training, and epoch selection; the aggregate method is the
50-trial RealMLP-HPO baseline. No local RealMLP approximation is reachable.

## Commands

```bash
python runners/run_phase2.py hpo \
  --base-config configs/experiments/phase2/base/adult.yaml \
  --search-space configs/hpo/phase2/mlp.yaml

python runners/run_phase2.py final \
  --base-config configs/experiments/phase2/base/adult.yaml \
  --search-space configs/hpo/phase2/mlp.yaml \
  --best-config configs/experiments/phase2/mlp/adult_census_income_best.yaml
```

TabICLv2 final evaluation omits `--best-config`. Full HPO and 15-seed commands
must not be launched until the readiness gate from Prompt 6 passes.
