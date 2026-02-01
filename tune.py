"""
Hyperparameter Optimization Script
==================================

Description:
    Executes Hyperparameter Optimization (HPO) using Optuna with the TPE sampler.
    It supports resuming interrupted studies and saving the best configuration found.
    
    Key Features:
    - Resume capability (checks existing study DB).
    - Seeded sampler for reproducibility.
    - Automatic creation of trial-specific temporary configs.

Usage:
    python tune.py --base_config ... --hpo_config ... --trials 50
"""

import optuna
from optuna.samplers import TPESampler
import argparse
import yaml
import os
import sys
import logging
from train import main as train_main

class TrainArgs:
    """
    Mock class to emulate argparse.Namespace for passing arguments to train_main directly.
    """
    def __init__(self, config_path, gpu_ids="0", seed=42):
        self.config = config_path
        self.gpu_ids = gpu_ids
        self.data_root = None
        self.seed = seed
        self.verbose = True 

def suggest_params(trial, hpo_config):
    """Maps HPO config specification to Optuna trial suggestions."""
    params = {}
    for param_name, spec in hpo_config.items():
        type_name = spec.get('type')
        if type_name == 'float':
            params[param_name] = trial.suggest_float(
                param_name, float(spec['low']), float(spec['high']), 
                log=spec.get('log', False), step=spec.get('step', None)
            )
        elif type_name == 'int':
            params[param_name] = trial.suggest_int(
                param_name, int(spec['low']), int(spec['high']), 
                step=spec.get('step', 1), log=spec.get('log', False)
            )
        elif type_name == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, spec['choices'])
    return params

def objective(trial, args, base_config_dict, hpo_config_dict):
    config = base_config_dict.copy()
    config['optimize_metric'] = args.metric 
    
    suggested_params = suggest_params(trial, hpo_config_dict)
    config.update(suggested_params)
    
    # Create temporary config file for the current trial
    os.makedirs("configs/temp_hpo", exist_ok=True)
    temp_config_path = f"configs/temp_hpo/{args.study_name}_trial_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    score = 0
    try:
        # Pass seed to TrainArgs to ensure consistent data splits within the trial
        train_args = TrainArgs(temp_config_path, gpu_ids=args.gpu_ids, seed=args.seed)
        train_args.verbose = False # Suppress verbose logs during tuning
        score = train_main(train_args)
    except Exception as e:
        print(f"🚨 Trial {trial.number} Failed: {e}")
        # Return worst possible score on failure
        score = 0 if config.get('task_type', 'classification') == 'classification' else float('inf')
    
    # Cleanup temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, required=True, help="Path to base configuration file")
    parser.add_argument('--hpo_config', type=str, required=True, help="Path to HPO search space configuration")
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--trials', type=int, default=50, help="Total number of trials to run")
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--metric', type=str, default='acc', help="Target metric to optimize (acc, auroc, rmse, etc.)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for the sampler")
    args = parser.parse_args()

    # Load Configs
    with open(args.base_config, 'r') as f:
        base_config_dict = yaml.safe_load(f)
    with open(args.hpo_config, 'r') as f:
        hpo_config_dict = yaml.safe_load(f)

    # Auto-generate Study Name if not provided
    if args.study_name is None:
        ds_name = base_config_dict.get('dataset_name')
        if not ds_name or ds_name in ["default", "unknown"]:
            filename = os.path.basename(args.base_config)
            ds_name = filename.split('_')[0]
        md_name = base_config_dict.get('model_name', 'model')
        args.study_name = f"{ds_name}_{md_name}_opt"

    # Logging Setup
    # Note: 'results/tuning' should be added to .gitignore
    os.makedirs("results/tuning", exist_ok=True)
    optuna_logger = optuna.logging.get_logger("optuna")
    if optuna_logger.hasHandlers(): optuna_logger.handlers.clear()
    
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(f"results/tuning/{args.study_name}.log")
    optuna_logger.addHandler(stream_handler)
    optuna_logger.addHandler(file_handler)
    optuna_logger.setLevel(logging.INFO)

    # Determine optimization direction
    if args.metric in ['acc', 'auroc', 'f1', 'auprc']:
        direction = 'maximize'
    else:
        direction = 'minimize'
    
    # Update base config with the optimization metric
    base_config_dict['optimize_metric'] = args.metric

    print(f"🔥 Optimization Start: {args.study_name}")
    print(f"🎯 Target Metric: {args.metric} ({direction})")
    print(f"🎲 Seed: {args.seed}")
    
    storage_name = f"sqlite:///results/tuning/{args.study_name}.db"
    
    # Fixed seed sampler for reproducibility
    sampler = TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        load_if_exists=True,
        direction=direction,
        sampler=sampler
    )
    
    # Smart Resume Logic: Calculate remaining trials
    n_trials_left = args.trials - len(study.trials)

    if n_trials_left > 0:
        print(f"⏩ Resuming... ({len(study.trials)} done, {n_trials_left} to go)")
        study.optimize(
            lambda trial: objective(trial, args, base_config_dict, hpo_config_dict), 
            n_trials=n_trials_left
        )
    else:
        print(f"✅ Study already has {len(study.trials)} trials (Target: {args.trials}). Skipping...")

    print("\n" + "="*50)
    print(f"🏆 Best Params ({args.metric}={study.best_value:.4f}):")
    for k, v in study.best_params.items():
        print(f"  🔹 {k}: {v}")
    
    # Save Best Config
    final_config = yaml.safe_load(open(args.base_config))
    final_config.update(study.best_params)
    base_name = os.path.basename(args.base_config).replace('.yaml', '')
    save_path = f"configs/experiments/{base_name}_best.yaml"
    
    with open(save_path, 'w') as f:
        yaml.dump(final_config, f, default_flow_style=False)
        
    print(f"💾 Final Best Config saved to: {save_path}")