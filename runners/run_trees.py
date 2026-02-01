"""
Tree Model Runner (XGBoost / CatBoost)
======================================

Description:
    This script handles Hyperparameter Optimization (HPO) and Final Evaluation 
    for Tree-based models (XGBoost, CatBoost).
    
    Modes:
    1. --tune: Runs Optuna optimization to find best hyperparameters.
    2. --eval: Runs final evaluation with 15 random seeds using best params.

Usage:
    python runners/run_trees.py --dataset adult --model xgboost --tune --trials 50
    python runners/run_trees.py --dataset adult --model xgboost --eval
"""

import argparse
import os
import gc
import yaml
import json
import time
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm

# Optimization for CPU performance
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Ensure directories exist
os.makedirs("configs/experiments", exist_ok=True)
os.makedirs("results/scores", exist_ok=True)

# Imbalanced datasets requiring AUPRC optimization
AUPRC_DATASETS = ['creditcard', 'mimic3', 'mimic4']

def load_data_config(dataset_name):
    config_path = f"configs/datasets/{dataset_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg

def get_data(dataset_name, data_cfg, model_type):
    csv_path = data_cfg.get('csv_path', f"data/{dataset_name}.csv")
    if not os.path.exists(csv_path):
        csv_path = f"data/{os.path.basename(csv_path)}"
    
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    target_col = data_cfg.get('target_col', 'target')
    cat_cols = data_cfg.get('cat_cols', [])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocessing
    num_cols = [c for c in X.columns if c not in cat_cols]
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)

    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    if cat_cols:
        for col in cat_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                
    return X, y, cat_cols

def get_model(model_type, task_type, params, cat_cols=None, seed=42):
    params = params.copy()

    if model_type == 'xgboost':
        ModelClass = XGBRegressor if task_type == 'regression' else XGBClassifier
        model = ModelClass(
            **params,
            random_state=seed,
            tree_method='hist', 
            device='cpu',       
            enable_categorical=False,
            n_jobs=-1,
            early_stopping_rounds=50
        )
    elif model_type == 'catboost':
        if 'colsample_bytree' in params:
            params['colsample_bylevel'] = params.pop('colsample_bytree')
            
        ModelClass = CatBoostRegressor if task_type == 'regression' else CatBoostClassifier
        model = ModelClass(
            **params,
            random_seed=seed,
            task_type="CPU",      
            thread_count=-1,
            verbose=0,
            early_stopping_rounds=50,
            bootstrap_type='Bernoulli'
        )
    return model

# =========================================================
# HPO Logic
# =========================================================
def objective(trial, dataset_name, model_type, task_type, X, y, cat_cols):
    stratify_param = y if task_type == 'classification' else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=stratify_param, random_state=42
    )
    
    stratify_temp = y_temp if task_type == 'classification' else None
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=42
    )
    
    if model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
    elif model_type == 'catboost':
        params = {
            'boosting_type': 'Plain', 
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        }

    model = get_model(model_type, task_type, params, cat_cols, seed=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    if task_type == 'regression':
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    
    else: # Classification
        n_classes = len(np.unique(y))
        
        if n_classes > 2:
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)
        else:
            if dataset_name in AUPRC_DATASETS:
                preds_proba = model.predict_proba(X_val)[:, 1]
                try:
                    return average_precision_score(y_val, preds_proba)
                except:
                    return 0.0
            else:
                preds = model.predict(X_val)
                return accuracy_score(y_val, preds)

def run_hpo(args):
    print(f"🚀 Start HPO for {args.model} on {args.dataset}")
    data_cfg = load_data_config(args.dataset)
    
    if args.task_type:
        task_type = args.task_type
    else:
        task_type = data_cfg.get('task_type', 'classification')
    
    direction = 'minimize' if task_type == 'regression' else 'maximize'
    
    print("⏳ Pre-loading data into memory...")
    X, y, cat_cols = get_data(args.dataset, data_cfg, args.model)
    
    study = optuna.create_study(direction=direction, study_name=f"{args.dataset}_{args.model}")
    study.optimize(lambda trial: objective(trial, args.dataset, args.model, task_type, X, y, cat_cols), n_trials=args.trials)
    
    print(f"🏆 Best Params: {study.best_params}")
    
    save_path = f"configs/experiments/{args.dataset}_{args.model}_best.yaml"
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f)
    print(f"💾 Config saved to {save_path}")
    
    del study, X, y
    gc.collect()

# =========================================================
# Final Evaluation
# =========================================================
def run_eval(args):
    print(f"📊 Start Final Evaluation for {args.model} on {args.dataset} (15 Seeds)")
    
    config_path = f"configs/experiments/{args.dataset}_{args.model}_best.yaml"
    if not os.path.exists(config_path):
        print(f"🚨 Config not found! Run --tune first.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        best_params = yaml.safe_load(f)
        
    data_cfg = load_data_config(args.dataset)
    if args.task_type:
        task_type = args.task_type
    else:
        task_type = data_cfg.get('task_type', 'classification')

    X, y, cat_cols = get_data(args.dataset, data_cfg, args.model)
    n_classes = len(np.unique(y))
    
    seeds = range(43, 43 + 15)
    scores = []
    
    for seed in tqdm(seeds, desc="Running Seeds"):
        stratify_param = y if task_type == 'classification' else None
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=stratify_param, random_state=seed
        )
        stratify_temp = y_temp if task_type == 'classification' else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=seed
        )
        
        model = get_model(args.model, task_type, best_params, cat_cols, seed=seed)
        
        start_train = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - start_train
        
        start_inf = time.time()
        
        metric_dict = {}
        if task_type == 'regression':
            preds = model.predict(X_test)
            inf_time = time.time() - start_inf
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            metric_dict['rmse'] = rmse
            main_score = rmse
        else: # Classification
            preds = model.predict(X_test)
            inf_time = time.time() - start_inf
            acc = accuracy_score(y_test, preds)
            metric_dict['acc'] = acc
            
            if n_classes > 2:
                auprc, auroc = 0.0, 0.0
            else:
                preds_proba = model.predict_proba(X_test)[:, 1]
                auprc = average_precision_score(y_test, preds_proba)
                try:
                    auroc = roc_auc_score(y_test, preds_proba)
                except:
                    auroc = 0.5
            
            metric_dict['auprc'] = auprc
            metric_dict['auroc'] = auroc
            main_score = auprc if args.dataset in AUPRC_DATASETS else acc
            
        metric_dict['train_time'] = train_time
        metric_dict['inference_time'] = inf_time
        scores.append(main_score)
        
        result_json = {
            "dataset": args.dataset,
            "model": args.model,
            "seed": seed,
            "metrics": metric_dict,
            "config": best_params
        }
        with open(f"results/scores/{args.dataset}_{args.model}_seed{seed}.json", "w", encoding='utf-8') as f:
            json.dump(result_json, f, indent=4)

    print(f"\n✅ Final Results ({args.dataset}): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--task_type', type=str, default=None, choices=['regression', 'classification'])
    
    args = parser.parse_args()
    
    if args.tune: run_hpo(args)
    if args.eval: run_eval(args)