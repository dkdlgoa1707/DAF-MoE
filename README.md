# DAF-MoE: Distribution-Aware Feature-level Mixture of Experts

> **Official PyTorch Implementation**
> *This repository contains the code for the paper "DAF-MoE: Distribution-Aware Feature-level Mixture of Experts for Tabular Data".*

## 📌 Introduction

**DAF-MoE** is a novel deep learning architecture designed for heterogeneous tabular data. Unlike standard Transformers that treat all features uniformly, DAF-MoE employs a **Distribution-Guided Gating (DGG)** mechanism to route features to specialized experts based on their statistical characteristics (e.g., outliers vs. common values).

This repository provides:

* Complete implementation of **DAF-MoE** and baselines (FT-Transformer, ResNet, MLP).
* Scripts for **Hyperparameter Optimization (HPO)** using Optuna.
* Automated scripts to **reproduce the main results** (15 random seeds).
* **Comprehensive Analysis Suite:** Tools for expert visualization, noise robustness tests, statistical significance checks, and ablation summaries.

## Version History

- **v1 (KDD 2026 submission)**: Original architecture. Source: `src/models/daf_moe/`. Best configs: `configs/experiments/{dataset}_daf_moe_best.yaml`.
- **v1.5 (Phase 1 exploration)**: Configurable ablation variants for architecture design exploration. Source: `src/models/daf_moe_v15/`. Phase 1 results: `results/phase1_v15/PHASE1_REPORT.md`.
- **v2 (Phase 2, AAAI 2027 submission)**: Finalized architecture with FiLM-based Distribution-Guided Gating, per-feature PLE numerical embedding, lightweight dual-pathway expert, and entity-only categorical embedding. Source: `src/models/daf_moe_v2/`.

---

## 🛠️ 1. Installation

Since this repository is provided for anonymous review, please **download the source code as a ZIP file** and extract it.

### Prerequisites

* Python 3.10
* CUDA (Recommended for Deep Learning models)

### Setup

1. Open your terminal and navigate to the extracted project root:

```bash
cd DAF-MoE-1EF0

```

### Option A: Conda (Recommended)

Use this option to automatically set up the environment using `environment.yml`.

```bash
# 1. Create Conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate daf_moe

# 3. Install the package in editable mode (Important!)
pip install -e .

```

### Option B: Pip (Standard)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the package in editable mode (Important!)
pip install -e .

```

---

## 📂 2. Data Preparation

Due to licensing and size constraints, raw data files are **not included** in this repository.
Please download the datasets and place them in the `data/` directory.

### Directory Structure

```text
data/
├── adult.csv
├── california_housing.csv
├── covertype.csv
├── higgs_small.csv
...

```

* **Note:** The schema for each dataset is defined in `configs/datasets/*.yaml`.

---

## 🚀 3. Phase 2 Readiness and Execution

Phase 2 uses one protocol-aware entrypoint for the 11 main methods and the
secondary TabM† control. Before starting any HPO, run the readiness gate:

```bash
bash scripts/phase2_preflight.sh
```

The gate checks the 9 dataset files, model/adaptor routing, exact dependency
versions, protocol constants, and the full smoke suite. It prints `READY_FOR_HPO` when no blocker remains, or
`READY_FOR_HPO_WITH_RETRIEVAL_WARNING` with exit code 0 when the only remaining
issue is a retrieval scale timeout. It generates but does not execute
`scripts/phase2_hpo_commands.sh` (99 HPO jobs) and
`scripts/phase2_final_commands.sh` (108 final jobs). The old combined script
is deprecated and fails fast.
The machine-readable report is written to
`results/phase2/preflight_report.json`.

### A. Hyperparameter Optimization

```bash
python runners/run_phase2.py hpo \
  --base-config configs/experiments/phase2/base/adult.yaml \
  --search-space configs/hpo/phase2/mlp.yaml \
  --best-output configs/experiments/phase2/mlp/adult_best.yaml \
  --device cuda
```

HPO is fixed to Optuna TPE seed 42 and 50 finite `COMPLETE` trials. Failed,
OOM, invalid, and nonfinite trials do not count. Each invocation defaults to
`--max-attempts 200` and `--max-consecutive-failures 10`; stopping preserves the
signed SQLite DB and JSON artifacts for resume. HPO best states stay in memory
and do not create trial `.pth` files. TabICLv2 is fixed and has no HPO command.

### B. Final Evaluation

```bash
python runners/run_phase2.py final \
  --base-config configs/experiments/phase2/base/adult.yaml \
  --search-space configs/hpo/phase2/mlp.yaml \
  --best-config configs/experiments/phase2/mlp/adult_best.yaml \
  --device cuda
```

The default final seed registry is 43 through 57. Each seed creates a fresh raw
split and train-fitted preprocessor; seed 42 is rejected. Result reuse requires
an exact protocol/config/split/preprocessing manifest match.

For fixed TabICLv2, omit `--best-config`:

```bash
python runners/run_phase2.py final \
  --base-config configs/experiments/phase2/base/adult.yaml \
  --search-space configs/hpo/phase2/tabicl.yaml
```

XGBoost, CatBoost, RealMLP, and TabICLv2 are dispatched through their
official/native execution branch and never through the DAF preprocessor. See
`docs/phase2_experiment_protocol.md`, `docs/phase2_hpo_and_execution.md`, and
`docs/phase2_implementation_status.md` for the exact contracts and pins.

---

## ⚡ 4. Legacy v1/v1.5 Reproduction

The original best-config runner remains available for the v1/v1.5 result
layout. It is not the Phase 2 HPO or final-evaluation entrypoint.

```bash
# Syntax: bash scripts/reproduce_results.sh <CONFIG_PATH> [GPU_ID]

bash scripts/reproduce_results.sh configs/experiments/adult_daf_moe_best.yaml 0

```

---

## 🔬 5. Ablation Studies

To validate the contribution of each component (e.g., removal of the Preservation Path or Distribution Tokens):

```bash
python runners/run_ablation.py

```

* **Output:** Results are saved in `results/ablation/`.

---

## 📊 6. Analysis & Evaluation

We provide a comprehensive suite of scripts to generate the tables and figures presented in the paper.

### A. Main Performance Summary (Table 2 & 4)

Aggregates metrics across all datasets and models, and generates the performance ranking chart.

```bash
python analysis/summarize_results.py

```

### B. Expert Visualization (Figure 4)

Visualizes how the Distribution-Guided Gating mechanism routes features based on their statistical properties (e.g., Mode vs. Tail).

```bash
# Default dataset: California Housing
python analysis/analyze_expert.py

```

### C. Noise Robustness Analysis (Figure 3)

Evaluates model stability by injecting Gaussian noise into input features and comparing performance degradation against baselines.

```bash
python analysis/analyze_noise_robustness.py

```

### D. Statistical Significance Test (Table 2 Support)

Performs 95% Confidence Interval (CI) overlap tests to determine statistical parity between DAF-MoE and GBDTs.

```bash
python analysis/significance_test.py

```

### E. Ablation Study Summary (Table 5)

Summarizes the results from `runners/run_ablation.py` to quantify the impact of specific architectural components.

```bash
python analysis/summarize_ablation.py

```

---

## 📁 Project Structure

```text
DAF-MoE/
├── src/                    # Source code (Models, Loss, Data Loader)
├── configs/                # Configuration files (YAML)
├── scripts/                # Shell scripts for execution
├── runners/                # Python runners for Trees & Ablation
├── analysis/               # Analysis & Evaluation scripts
├── results/                # Output logs
├── train.py                # Training entry point
├── runners/run_phase2.py   # Phase 2 HPO/final entry point
├── tune.py                 # Retired legacy HPO entry point (fail-fast)
└── setup.py                # Package installation script

```
