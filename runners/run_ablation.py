"""
Ablation Study Execution Script
===============================

Description:
    This script automates the ablation study experiments to validate the contribution
    of each component in DAF-MoE. It trains model variants by selectively disabling
    specific modules (Structural Ablation) or loss terms (Loss Ablation).

    Variants include:
    1. Structural Ablation (Hardware Check):
       - 'wo_raw_path': Disables the raw feature bypass path.
       - 'wo_deep_path': Disables the deep representation path.
       - 'wo_dist_token': Removes the distribution token.
    
    2. Loss Ablation (Software Check):
       - 'wo_spec_loss': Disables the specialization loss (expert focus).
       - 'wo_repel_loss': Disables the repel loss (expert diversity).
       - 'wo_aux_loss': Disables both auxiliary losses (Standard MoE).

    It iterates through all specified datasets and random seeds (5 seeds) for statistical reliability.
    Results (JSON logs) are saved in `results/ablation/`.

Usage:
    python runners/run_ablation.py
"""

import os
import time

# ====================================================
# 🔥 [Configuration] Datasets & Seeds & Paths
# ====================================================
# List of datasets (< 100k samples recommended for ablation speed)
DATASETS = ["california", "adult", "higgs_small", "nhanes", "mimic3", "mimic4"]

GPU_ID = "0"
SEEDS = [43, 44, 45, 46, 47]  # 5 Random Seeds
RESULT_DIR = "results/ablation"  # Output directory for ablation logs

# Ensure output directory exists (relative to project root)
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================================================
# 🧪 [Configuration] Ablation Variants
# ====================================================
VARIANTS = {
    # ================================================
    # Set 1: Structural Ablation (Architecture)
    # ================================================
    # Note: "full_model" is skipped as it is already trained in main experiments.
    
    "wo_raw_path":      {"--use_raw_path": "False"},   # Key component: Remove Raw Path
    "wo_deep_path":     {"--use_deep_path": "False"},  # Sanity Check: Remove Deep Path
    "wo_dist_token":    {"--use_dist_token": "False"}, # Remove Distribution Token

    # ================================================
    # Set 2: Loss Ablation (Optimization Objective)
    # ================================================
    
    "wo_spec_loss":     {"--lambda_spec": "0.0"},      # Disable Specialization Loss
    "wo_repel_loss":    {"--lambda_repel": "0.0"},     # Disable Repel Loss
    "wo_aux_loss":      {"--lambda_spec": "0.0", "--lambda_repel": "0.0"} # Disable Both (Standard MoE)
}

def run_experiment(dataset, variant_name, flags, seed):
    print(f"\n🚀 [Start] {dataset} | {variant_name} | Seed {seed}")
    
    # Path to the best hyperparameter config (relative to project root)
    config_path = f"configs/experiments/{dataset}_daf_moe_best.yaml"
    
    if not os.path.exists(config_path):
        print(f"🚨 Config not found: {config_path}")
        return

    # Construct the command string
    # - Run `train.py` from the project root
    # - Save results specifically to the ablation directory
    cmd = (f"python train.py --config {config_path} --gpu_ids {GPU_ID} "
           f"--seed {seed} --verbose --result_dir {RESULT_DIR}")
    
    # Append variant-specific flags
    for k, v in flags.items():
        cmd += f" {k} {v}"
        
    print(f"   Command: {cmd}")
    
    # Execute command
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Error in {variant_name} (Seed {seed})")
    else:
        print(f"✅ Done {variant_name} (Seed {seed})")
    
    # Slight delay to prevent file I/O conflicts and overheating
    time.sleep(1)

def main():
    total_runs = len(DATASETS) * len(VARIANTS) * len(SEEDS)
    print(f"🔥 Starting Full Ablation Study")
    print(f"   - Datasets: {DATASETS}")
    print(f"   - Variants: {list(VARIANTS.keys())}")
    print(f"   - Seeds: {SEEDS}")
    print(f"   - Total Runs: {total_runs}")
    print(f"   - Save Directory: {RESULT_DIR}")
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"📂 Processing Dataset: {dataset}")
        print(f"{'='*60}")
        
        for name, flags in VARIANTS.items():
            for seed in SEEDS:
                run_experiment(dataset, name, flags, seed)

if __name__ == "__main__":
    main()