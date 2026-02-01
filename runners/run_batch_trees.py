"""
Batch Runner for Tree Models
============================

Description:
    Executes `run_trees.py` sequentially for multiple datasets and models.
    Useful for running overnight experiments.

    Modes:
    - MODE = "tune": Run HPO for all specified datasets.
    - MODE = "eval": Run final evaluation for all specified datasets.

Usage:
    python runners/run_batch_trees.py
"""

import subprocess
import time

# ====================================================
# [Configuration] Dataset List & Task Type
# ====================================================
DATASET_INFO = {
    # Regression Datasets
    "california": "regression",
    "allstate": "regression",     
    "year_prediction": "regression",

    # Classification Datasets
    "adult": "classification",
    "higgs_small": "classification",
    "covertype": "classification",
    "creditcard": "classification",
    "bnp": "classification",
    "nhanes": "classification",
    "mimic3": "classification",
    "mimic4": "classification", 
}

MODELS = ["xgboost", "catboost"]

# 🔥 Mode Selection
# "tune": Run Hyperparameter Optimization
# "eval": Run Final Evaluation (15 Seeds)
MODE = "eval"  

TRIALS = 50  # Used only in "tune" mode

def run_command(cmd):
    """Executes a shell command and reports errors."""
    print(f"\n🚀 [Running] {cmd}")
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"🚨 [Error] Failed: {cmd}")

def main():
    start_total = time.time()
    
    print("="*60)
    print(f"🔥 Batch Runner Start (Mode: {MODE} | Datasets: {len(DATASET_INFO)})")
    print("="*60)

    for dataset, task_type in DATASET_INFO.items():
        for model in MODELS:
            print(f"\n>> 🛠️ Processing {model} on {dataset} ({task_type})...")
            
            # Construct command relative to project root
            base_cmd = f"python runners/run_trees.py --dataset {dataset} --model {model} --task_type {task_type}"
            
            if MODE == "tune":
                cmd = f"{base_cmd} --tune --trials {TRIALS}"
            else:
                cmd = f"{base_cmd} --eval"
            
            run_command(cmd)
            
            # Cooldown to prevent overheating
            time.sleep(3)

    elapsed = time.time() - start_total
    print("\n" + "="*60)
    print(f"✅ All Jobs Finished! Total Time: {elapsed/60:.2f} min")
    print("="*60)

if __name__ == "__main__":
    main()