# DAF-MoE Phase 2 Experiment Protocol

Protocol version: `phase2-v2`

This document is the source of truth for Phase 2 data preparation, model
selection, HPO, and final evaluation. Results without a matching protocol and
manifest hash are not Phase 2 results.

## Datasets

The nine datasets are California, Adult, Higgs Small, Covertype, Allstate, BNP,
NHANES, MIMIC-III, and MIMIC-IV.

Raw rows are split before any fitted preprocessing. Every method reuses the
same indices for a given seed:

- train: 80%
- validation: 10%
- test: 10%
- classification is stratified when the class counts permit it

HPO sampling, splitting, and model training use seed 42. The test partition is
sealed from objectives, early stopping, and configuration selection. Final
evaluation uses seeds 43 through 57, excludes seed 42, and uses the same raw
split indices and training seed across methods for each outer seed.

## HPO

Tunable methods use `Optuna TPESampler(seed=42)` and require 50 valid
`COMPLETE` trials per dataset. Failed, OOM, invalid, and nonfinite trials do not
count toward 50. Cross-trial pruning is disabled. Search ranges are defined by
the Phase 2 model-specific HPO configuration and may not be inferred from old
best configs.

## Methods

Main methods included in average-rank and omnibus tests:

1. DAF-MoE v2
2. XGBoost
3. CatBoost
4. RTDL MLP
5. RTDL ResNet
6. FT-Transformer
7. full TabR
8. plain TabM-mini
9. ModernNCA
10. RealMLP-HPO
11. TabICLv2

`TabM-dagger-mini` (`tabm_ple`, updated PLE) is a secondary control. It receives
the same 50-trial HPO and 15-seed evaluation but is excluded from main ranks and
the omnibus test.

## Leakage Boundary

Each model owns a registered preprocessing adapter. DAF statistics are never
forced onto baselines. Every learned statistic, vocabulary, boundary, target
mapping, and target transform is fitted on the train partition only.

The HPO API constructs only train and validation partitions. Its public return
type has no test member. Final evaluation is a separate API that exposes test
only after model and preprocessing configuration are fixed.

### Categorical contract

- Nulls become the reserved `[MISSING]` state before string conversion.
- `[MISSING]` and train-unseen `[UNK]` have different IDs.
- `[UNK]` is excluded from train vocabulary, cardinality, and frequency.
- Unknown transform frequency is zero.
- Validation and test values cannot resize vocabularies, embeddings, or output
  dimensions.

### Numerical missing contract

Custom neural adapters impute with the train mean and emit an independent
`x_numerical_missing` binary tensor. Models must not infer missingness from the
imputed value. Native-missing methods such as XGBoost, CatBoost, and TabICLv2 do
not receive forced numerical imputation.

The missing mask does not add features and never receives PLE. DAF-MoE v2 adds
a learned per-feature missing embedding to each numerical token selected by the
mask.

### DAF-MoE v2 statistics

For each numerical feature, finite observed train values alone determine mean,
population standard deviation, frozen empirical CDF, unbiased skewness
(`scipy.stats.skew(..., bias=False)`), and PLE quantiles. Imputed copies are not
counted as observations.

The preservation value is standardized with observed-train mean and scale. PLE
boundaries are quantiles of observed raw values transformed to that same
standardized scale. Duplicate boundaries remain deterministic and are handled
by the encoder denominator clamp. Constant and all-missing features use scale
1, skew 0, and repeated zero boundaries.

### Regression targets

Target transformations are declared before evaluation. Custom neural methods
standardize regression targets with train-only mean and population standard
deviation. Validation, HPO objective, and final metrics inverse-transform both
targets and predictions before scoring in the original target unit. Native
estimators retain their official native target scale. Constant train targets
use a deterministic unit-scale fallback recorded in the manifest.

## Provenance

Every run manifest records:

- git SHA and protocol version
- dataset/schema version and schema hash
- split-index hash
- feature and target preprocessing classes/versions and fitted-state hashes
- effective target policy, train target mean/std, fallback flag, and metric scale
- model implementation version and canonical study signature/components
- missing and unseen-category counts by partition
- random seed and deterministic subsample size

Resume is valid only when protocol, dataset/schema, model implementation,
task/metric, search-space, base-config, target-policy, split, preprocessing, and
manifest hashes match exactly. Only finite `COMPLETE` trials carrying matching
study and search-space attributes contribute to the HPO completion count.
