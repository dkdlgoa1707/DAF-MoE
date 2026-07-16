# Phase 2 Native/Official Models

These models bypass `DAFPreprocessor` and the repository PyTorch trainer. Raw
rows are split by `RawSplitRegistry`; target mappings and every categorical
schema are fitted on the train partition only.

| Model | Pinned dependency | Input boundary | Protocol-fixed behavior |
|---|---|---|---|
| XGBoost | `xgboost==2.1.4` | pandas frame; numeric NaN untouched; train-frozen `category` dtype | `hist`, native categorical, depthwise |
| CatBoost | `catboost==1.2.10` | raw categorical strings; categorical NaN becomes `[MISSING]`; numeric NaN untouched | Plain, SymmetricTree, Bernoulli |
| RealMLP-HPO | `pytabkit[models]==1.7.3` | train-median numerical imputation, then frame plus categorical column names | official estimator owns robust scaling, clipping, encoding, and embedding search |
| TabICLv2 | `tabicl==2.1.1` | raw pandas frame with categorical dtype | 8 estimators; v2 classifier/regressor checkpoints; no HPO or fine-tuning |

The audited pytabkit source revision is
`c126ea51187c5080b91f28d352481dbd3b2194b0` (Apache-2.0). The audited TabICL
source revision is `46b91961db4f8873dd049ec09990698a435e1e29` (BSD-3-Clause).

XGBoost maps train-unseen categories to pandas categorical code `-1`, which the
pinned native-categorical path treats as missing. Original categorical missing
values remain the explicit `[MISSING]` training category. CatBoost keeps unseen
strings native. TabICLv2 maps unseen strings to reserved `[UNK]` while keeping it
outside the fitted train vocabulary.

Pytabkit 1.7.3 documents numerical missing-value imputation as the sole manual
preprocessing required by its RealMLP sklearn interface and rejects continuous
NaNs. The adapter therefore fits medians on the outer train partition only;
all RealMLP transforms after imputation remain estimator-owned.

During final TabICLv2 evaluation, context consists only of that outer seed's
train and validation rows. Test rows are queries and their labels never enter
context. Covertype context is deterministically limited to 400,000 rows and the
selected raw row-ID hash is stored in the run manifest.

The main protocol entrypoint is `runners/run_phase2.py`, which dispatches these
models to the native execution branch. `runners/run_phase2_native.py` remains a
low-level native-only diagnostic entrypoint; its `hpo-validation` mode
constructs only train and validation partitions. Missing or mismatched official
dependencies raise a compatibility error; no local approximation is used.
