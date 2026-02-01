📄 README.md 
Markdown
# DAF-MoE: Distribution-Aware Feature-level Mixture of Experts

> **Official PyTorch Implementation**
> *This repository contains the code for the paper "DAF-MoE: Robust Tabular Deep Learning via Distribution-Aware Routing".*

## 📌 Introduction
**DAF-MoE** is a novel deep learning architecture designed for heterogeneous tabular data. Unlike standard Transformers that treat all features uniformly, DAF-MoE employs a **Distribution-Guided Gating (DGG)** mechanism to route features to specialized experts based on their statistical characteristics (e.g., outliers vs. common values).

This repository provides:
- Complete implementation of **DAF-MoE** and baselines (FT-Transformer, ResNet, MLP).
- Scripts for **Hyperparameter Optimization (HPO)** using Optuna.
- Automated scripts to **reproduce the main results** (15 random seeds).
- Analysis tools for **statistical significance** and **robustness evaluation**.

---

## 🛠️ 1. Installation

This repository provides two ways to set up the environment: **Conda (Recommended)** or **Pip**.

### Prerequisites
- Python 3.10
- CUDA (Recommended for Deep Learning models)

### Option A: Conda (Recommended)
If you use Anaconda or Miniconda, you can create the environment automatically using `environment.yml`.

```bash
# 1. Clone the repository
git clone [https://github.com/anonymous/daf-moe.git](https://github.com/anonymous/daf-moe.git)
cd daf-moe

# 2. Create Conda environment (Python 3.10 + Dependencies)
conda env create -f environment.yml

# 3. Activate environment
conda activate daf_moe

# 4. Install the package in editable mode (Essential)
pip install -e .

📂 2. Data Preparation
Due to licensing and size constraints, raw data files are not included in this repository. Please download the datasets and place them in the data/ directory.

Directory Structure
Plaintext
data/
├── adult.csv
├── california_housing.csv
├── creditcard.csv
├── higgs_small.csv
...
Configuring Datasets: The schema for each dataset (target column, numerical/categorical features) is defined in configs/datasets/*.yaml. You do not need to modify these unless you are using custom data.

🚀 3. Reproduction (Main Results)
We provide the best hyperparameter configurations found in our experiments in configs/experiments/*_best.yaml. You can reproduce the results reported in the paper (Table 2) using the following scripts.

A. Deep Learning Models (15 Seeds)
To train DAF-MoE or Deep Baselines (FT-Transformer, ResNet) over 15 fixed seeds (43-57):

Bash
# Syntax: bash scripts/reproduce_results.sh <CONFIG_PATH> [GPU_ID]

# Example: Reproduce DAF-MoE on Adult dataset
bash scripts/reproduce_results.sh configs/experiments/adult_daf_moe_best.yaml 0

# Example: Reproduce FT-Transformer on California Housing
bash scripts/reproduce_results.sh configs/experiments/california_ft_transformer_best.yaml 0
B. Tree-Based Models (XGBoost / CatBoost)
Tree models are trained using runners/run_batch_trees.py or individual run scripts.

Bash
# Run evaluation for XGBoost on Adult dataset
python runners/run_trees.py --dataset adult --model xgboost --eval

⚡ 4. Hyperparameter Optimization (Optional)
If you wish to re-tune the hyperparameters from scratch, use the scripts/run_hpo.sh script. This utilizes Optuna with the TPE sampler.

Bash
# Syntax: bash scripts/run_hpo.sh <BASE_CONFIG> <HPO_SEARCH_SPACE> <METRIC> <TRIALS> <GPU_ID>

# Example: Tune DAF-MoE on Higgs dataset (Maximize Accuracy)
bash scripts/run_hpo.sh \
    configs/experiments/higgs_small_daf_moe.yaml \
    configs/hpo/daf_moe.yaml \
    acc 50 0
Note: The best parameters found will be saved to configs/experiments/{dataset}_{model}_best.yaml.

🔬 5. Ablation Studies
To validate the contribution of each component (e.g., removing the Raw Path or Specialization Loss), run the ablation script. This automates training for various model variants.

Bash
# Run full ablation study (Structural & Loss ablations)
python runners/run_ablation.py
Output: Results are saved in results/ablation/.

📊 6. Analysis & Evaluation
After training, you can generate summaries and perform statistical tests using the scripts in the analysis/ directory.

A. Summarize Results
Aggregates metrics across all seeds and calculates Mean ± Std.

Bash
python analysis/summarize_results.py
Output: results/analysis/final_summary.csv

B. Statistical Significance Test
Performs Welch's t-test to compare Deep Learning models against GBDT baselines.

Bash
python analysis/compare_baselines.py
Output: results/analysis/dl_vs_gbdt_final.csv

C. Robustness Evaluation (Hard Samples)
Evaluates model performance on "Hard Samples" (Outliers identified by Isolation Forest) to demonstrate robustness.

Bash
python analysis/eval_robustness.py
Output: results/analysis/model_comparison_multiseed.csv

📁 Project Structure
Plaintext
DAF-MoE/
├── src/                    # Source code (Models, Loss, Data Loader)
│   ├── models/             # DAF-MoE and Baseline implementations
│   ├── losses/             # Custom loss functions (Spec, Repel, Bal)
│   └── trainer.py          # Training loop
├── configs/                # Configuration files (YAML)
│   ├── datasets/           # Dataset schemas
│   └── experiments/        # Hyperparameters (Templates & Best)
├── scripts/                # Shell scripts for easy execution
│   ├── reproduce_results.sh
│   └── run_hpo.sh
├── runners/                # Python runners for Trees & Ablation
├── analysis/               # Analysis & Evaluation scripts
├── results/                # Output logs (Scores, JSONs)
├── train.py                # Main training entry point
├── tune.py                 # HPO entry point
└── setup.py                # Package installation script
📜 License
This project is licensed under the MIT License.
