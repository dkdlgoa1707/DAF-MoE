# Phase 2 Implementation Status

This document maps the fixed `phase2-v2` protocol to the implementation that
was behavior-tested. It does not claim bitwise identity with every upstream
repository. Full readiness is decided only by `scripts/phase2_preflight.sh`.

## Model matrix

| Method | Version / behavior reference | Preprocessing | Runner | HPO YAML | Smoke evidence | Dependency / remaining blocker |
|---|---|---|---|---|---|---|
| DAF-MoE v2 | Local v2 contract: train-only PLE, FiLM DGG, lightweight expert | `DAFV2Adapter` | `runners/run_phase2.py` | `daf_moe_v2.yaml` | architecture + data leakage tests | PyTorch; full HPO not run |
| XGBoost | 2.1.4, native categorical `hist` | `XGBoostFrameAdapter` on raw frames | native branch of `runners/run_phase2.py` | `xgboost.yaml` | contract + official tiny fit/predict | `xgboost==2.1.4` |
| CatBoost | 1.2.10, Plain/SymmetricTree/Bernoulli | `CatBoostFrameAdapter` on raw frames | native branch | `catboost.yaml` | contract + official tiny fit/predict | `catboost==1.2.10` |
| RTDL MLP | Plain MLP with modified quantile inputs | `RTDLQuantileAdapter` | PyTorch branch | `mlp.yaml` | forward, mask, early stopping, one-trial/two-seed smoke | PyTorch; full HPO not run |
| RTDL ResNet | ReLU, BatchNorm, residual/hidden dropout separation | `RTDLQuantileAdapter` | PyTorch branch | `resnet.yaml` | normalization/dropout/forward tests | PyTorch; full HPO not run |
| FT-Transformer | Feature tokenizer, 8 heads, three dropout paths | `RTDLQuantileAdapter` | PyTorch branch | `ft_transformer.yaml` | tokenizer/dropout/forward tests | PyTorch; full HPO not run |
| TabR full | TALENT reference `08301d670`, full context/value path | `TabRAdapter` | PyTorch retrieval branch | `tabr.yaml` | brute-force retrieval, row-ID exclusion, CPU store | PyTorch; full HPO not run |
| TabM-mini | Plain mini, k=32, BatchEnsemble, no PLE | `TabMAdapter` | PyTorch branch | `tabm.yaml` | member loss, sharing, no-PLE tests | PyTorch; full HPO not run |
| ModernNCA | TALENT reference `08301d670`, Euclidean/SNS path | `ModernNCAAdapter` | PyTorch retrieval branch | `modernnca.yaml` | retrieval, duplicate, sample-rate, CPU store tests | PyTorch; full HPO not run |
| RealMLP-HPO | pytabkit 1.7.3 official `RealMLP_TD_*` per outer trial | train-median numerical imputation, then estimator-owned transforms | native branch | `realmlp.yaml` | leakage/fail-fast + official tiny fit/predict | `pytabkit[models]==1.7.3`, Python >=3.9 |
| TabICLv2 | tabicl 2.1.1, fixed v2 checkpoints, 8 estimators | `TabICLFrameAdapter`; train+val context | native branch, final only | `tabicl.yaml` (fixed) | classifier/regressor checkpoint fit/predict | `tabicl==2.1.1`, Python >=3.10 |
| TabM PLE (secondary) | TabM-mini plus updated PLE | `TabMPLEAdapter` | PyTorch branch | `tabm_ple.yaml` | separate module/state path and PLE tests | excluded from main rank; full HPO not run |

## Integration coverage

- The unit matrix covers regression, binary, and multiclass protocol fixtures;
  every tunable search schema is sampled, guarded as a finite COMPLETE trial,
  materialized, and checked against two final seed manifests.
- A real tiny MLP path executes sealed HPO data construction, in-memory best-state restoration without persistent HPO trial checkpoints, and final seeds 43 and 44. Model behavior tests separately
  execute all local neural forward paths and both retrieval backward paths.
- Native estimators keep model-specific frame ownership and fail fast when
  their exact dependency is absent or incompatible. The readiness gate runs
  tiny fit/predict for XGBoost, CatBoost, RealMLP, and both TabICLv2 tasks.
  There is no local approximation fallback for RealMLP or TabICL.
- Full 50-trial HPO and 15-seed evaluation have intentionally not been run.

## Readiness gate

Run:

```bash
bash scripts/phase2_preflight.sh
```

The command validates protocol/config/routing/data/dependency contracts, runs
the full unit and architecture verification suite, writes
`results/phase2/preflight_report.json`, and generates (but does not execute)
`scripts/phase2_hpo_commands.sh` (99 jobs) and
`scripts/phase2_final_commands.sh` (108 jobs). Retrieval timeout-only reports
return `READY_FOR_HPO_WITH_RETRIEVAL_WARNING`; dependency, correctness, nonfinite,
and OOM failures remain blockers. `--skip-tests` is available for regenerating the launch file,
but deliberately returns nonzero and can never declare readiness.

The full model set requires a Python 3.10+ environment with the exact pins in
`requirements.txt`. Missing or mismatched optional packages are red blockers,
not skipped baselines.
