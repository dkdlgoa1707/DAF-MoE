"""
Robustness Evaluation Script for DAF-MoE and Baselines
======================================================

Description:
    This script evaluates the robustness of trained models (DAF-MoE, Deep Baselines) 
    and Tree-based models (XGBoost, CatBoost) on 'Hard Samples'.
    
    'Hard Samples' are defined in two ways:
    1. Feature Outliers: Samples with low probability density in the input space,
       identified using Isolation Forest (Unsupervised).
    2. Target Tail (Regression only): Samples where the target value deviates 
       significantly from the mean distribution.

    The script iterates through multiple random seeds to ensure statistical reliability 
    and reports the Mean ± Std of performance metrics (RMSE for Regression, AUPRC for Classification).

Usage:
    python analysis/eval_robustness.py

Author: [Anonymous for Double-Blind Review]
Date: 2026-02-01
"""

import torch
import numpy as np
import pandas as pd
import os
import yaml
import time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# [DAF-MoE Modules]
from src.models.factory import create_model
from src.data.loader import get_dataloaders
from src.configs.default_config import DAFConfig
from src.utils.common import seed_everything

# ==========================================
# ⚙️ Configuration
# ==========================================
DATASETS = ['california', 'adult', 'higgs_small', 'nhanes', 'mimic3', 'mimic4']
MODELS = ['daf_moe', 'ft_transformer', 'resnet', 'mlp', 'xgboost', 'catboost']
GPU_ID = "0"
SEEDS = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57] # 15 Seeds for Robustness
TOP_K_PERCENT = 0.05 # Analyze top [0.01, 0.05, 0.1]% hardest samples

# ==========================================
# 🛠️ 1. Tree Model Handler
# ==========================================
def get_tree_data(dataset_name):
    """
    Loads and preprocesses data specifically for Tree-based models (XGBoost, CatBoost).
    Unlike Deep Learning models, Tree models handle raw features but require label encoding.
    """
    config_path = f"configs/datasets/{dataset_name}.yaml"
    if not os.path.exists(config_path):
        config_path = f"configs/datasets/{dataset_name}.yaml"
    
    with open(config_path, 'r') as f: data_cfg = yaml.safe_load(f)
    
    csv_path = data_cfg.get('csv_path', f"data/{dataset_name}.csv")
    if not os.path.exists(csv_path): csv_path = f"data/{os.path.basename(csv_path)}"
    
    df = pd.read_csv(csv_path, skipinitialspace=True)
    target_col = data_cfg.get('target_col', 'target')
    cat_cols = data_cfg.get('cat_cols', [])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Simple Imputation for Trees
    num_cols = [c for c in X.columns if c not in cat_cols]
    if num_cols: X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if cat_cols: X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
        
    for col in cat_cols:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
    if 'task_type' in data_cfg:
        task_type = data_cfg['task_type']
    else:
        # Auto-detect task type
        if y.dtype in [float, np.float32, np.float64] and len(np.unique(y)) > 20:
            task_type = 'regression'
        else:
            task_type = 'classification'
            
    return X, y, task_type

def train_tree_model_on_fly(dataset, model_name, seed):
    """
    Trains a tree-based model (XGBoost/CatBoost) from scratch using the specified seed.
    Since tree training is fast, we retrain on-the-fly to ensure exact seed reproducibility.
    
    Returns:
        model: Trained model object.
        X_test, y_test: Test set data for evaluation.
        task_type: 'regression' or 'classification'.
    """
    # Load Best Hyperparameters
    config_path = f"configs/experiments/{dataset}_{model_name}_best.yaml"
    if not os.path.exists(config_path):
        params = {} # Fallback to default
    else:
        with open(config_path, 'r') as f: params = yaml.safe_load(f)

    X, y, task_type = get_tree_data(dataset)
    
    # Stratified Split (None for Regression)
    stratify = y if task_type == 'classification' else None
    
    # 80/20 Train/Temp split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=stratify, random_state=seed)
    # 10/10 Val/Test split
    stratify_temp = y_temp if task_type == 'classification' else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=seed)
    
    if model_name == 'xgboost':
        ModelClass = XGBRegressor if task_type == 'regression' else XGBClassifier
        model = ModelClass(**params, random_state=seed, device='cpu', n_jobs=4)
    elif model_name == 'catboost':
        if 'colsample_bytree' in params: params['colsample_bylevel'] = params.pop('colsample_bytree')
        ModelClass = CatBoostRegressor if task_type == 'regression' else CatBoostClassifier
        model = ModelClass(**params, random_seed=seed, verbose=0, thread_count=4)
        
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model, X_test, y_test, task_type

# ==========================================
# 🛠️ 2. Deep Model Handler
# ==========================================
def load_deep_model(dataset, model_name, seed, device):
    """
    Loads a trained Deep Learning model (DAF-MoE, FT-Transformer, etc.) from a checkpoint.
    """
    config_path = f"configs/experiments/{dataset}_{model_name}_best.yaml"
    if not os.path.exists(config_path): return None, None, None, None

    config = DAFConfig()
    with open(config_path, 'r') as f: exp_args = yaml.safe_load(f)
    for k, v in exp_args.items(): 
        if hasattr(config, k): setattr(config, k, v)
    
    with open(config.data_config_path, 'r') as f: 
        data_cfg = yaml.safe_load(f)
        config.dataset_name = data_cfg['dataset_name']
    config.seed = seed
    
    ckpt_path = f"checkpoints/{config.dataset_name}_{config.model_name}_seed{seed}_best.pth"
    if not os.path.exists(ckpt_path): 
        return None, None, None, None

    # Load Data to initialize model dimensions
    _, _, test_loader = get_dataloaders(config, data_cfg)

    model = create_model(config).to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except RuntimeError as e:
        print(f"      🚨 Model Load Failed: {e}")
        return None, None, None, None
        
    model.eval()
    return model, test_loader, config.task_type

# ==========================================
# 🚀 3. Metric Calculator
# ==========================================
def calculate_metrics(preds, targets, indices, task_type, prefix=""):
    """
    Calculates evaluation metrics for a specific subset of data defined by 'indices'.
    
    Args:
        preds (np.array): Model predictions (probabilities or values).
        targets (np.array): Ground truth labels.
        indices (np.array): Indices of samples to evaluate (e.g., hard samples).
        task_type (str): 'regression' or 'classification'.
        prefix (str): Prefix for result keys (e.g., "Hard_").
        
    Returns:
        dict: Computed metrics.
    """
    if len(indices) == 0: return {}
    
    sub_preds = preds[indices]
    sub_targets = targets[indices]
    
    results = {}
    
    if task_type == 'regression':
        rmse = np.sqrt(mean_squared_error(sub_targets, sub_preds))
        results[f"{prefix}RMSE"] = rmse
    else:
        # 1. Accuracy
        sub_preds_label = (sub_preds > 0.5).astype(int)
        acc = accuracy_score(sub_targets, sub_preds_label)
        results[f"{prefix}ACC"] = acc
        
        # 2. AUROC
        try:
            if len(np.unique(sub_targets)) > 1:
                auroc = roc_auc_score(sub_targets, sub_preds)
            else:
                auroc = 0.5
            results[f"{prefix}AUROC"] = auroc
        except:
            results[f"{prefix}AUROC"] = 0.0
            
        # 3. AUPRC (Primary Metric for Imbalanced Data)
        try:
            auprc = average_precision_score(sub_targets, sub_preds)
            results[f"{prefix}AUPRC"] = auprc
        except:
            results[f"{prefix}AUPRC"] = 0.0
            
    return results

def run_analysis():
    """
    Main Analysis Loop:
    1. Iterates over Datasets and Models.
    2. Runs inference across multiple random seeds (SEEDS).
    3. Identifies 'Hard Samples' using Isolation Forest (Unsupervised).
    4. Computes metrics on both Full Set and Hard Subset.
    5. Aggregates results (Mean ± Std) and saves to CSV.
    """
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    final_summary = []
    
    print(f"🔥 Unified Analysis (Multi-Seed): {SEEDS}")
    print(f"📌 Datasets: {DATASETS}")
    print(f"📌 Models: {MODELS}")
    print(f"📌 Hard Sample Threshold: Top {TOP_K_PERCENT*100}% Outliers")
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"📦 Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        for model_name in MODELS:
            print(f"   ▶ Analyzing {model_name}...")
            
            seed_metrics_list = []
            
            pbar = tqdm(SEEDS, desc=f"      Running Seeds", leave=False)
            for seed in pbar:
                try:
                    # --- [A] Load / Train Model ---
                    if model_name in ['xgboost', 'catboost']:
                        model, X_test, y_test, task_type = train_tree_model_on_fly(dataset, model_name, seed)
                        if task_type == 'regression':
                            preds = model.predict(X_test)
                        else:
                            preds = model.predict_proba(X_test)[:, 1]
                        
                        targets = y_test if isinstance(y_test, np.ndarray) else y_test.values
                        X_features = X_test if isinstance(X_test, np.ndarray) else X_test.values
                        
                    else: # Deep Model
                        result = load_deep_model(dataset, model_name, seed, device)
                        if result[0] is None: 
                            continue # Skip if checkpoint missing
                        model, loader, task_type = result
                        
                        preds_list, targets_list, X_feat_list = [], [], []
                        with torch.no_grad():
                            for inputs, target in loader:
                                for k in inputs: inputs[k] = inputs[k].to(device)
                                out = model(**inputs)
                                logits = out['logits']
                                
                                if task_type == 'regression':
                                    p = logits.view(-1).cpu().numpy()
                                else:
                                    if logits.shape[1] == 1:
                                        p = torch.sigmoid(logits).view(-1).cpu().numpy()
                                    else:
                                        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                                
                                preds_list.append(p)
                                targets_list.append(target.cpu().numpy())
                                if 'x_numerical' in inputs: X_feat_list.append(inputs['x_numerical'].cpu().numpy())
                                
                        preds = np.concatenate(preds_list)
                        targets = np.concatenate(targets_list)
                        X_features = np.concatenate(X_feat_list) if X_feat_list else None

                    # --- [B] Calculate Metrics per Seed ---
                    current_seed_metrics = {}
                    
                    # 1. Full Data Metrics
                    full_m = calculate_metrics(preds, targets, np.arange(len(targets)), task_type, prefix="Full_")
                    current_seed_metrics.update(full_m)
                    
                    # 2. Hard Feature Outlier Metrics (Isolation Forest)
                    if X_features is not None:
                        if X_features.ndim == 3:
                            N, F, C = X_features.shape
                            X_flat = X_features.reshape(N, -1)
                        else:
                            X_flat = X_features
                        
                        # Train Isolation Forest for each seed (Fairness)
                        iso = IsolationForest(contamination=TOP_K_PERCENT, random_state=seed, n_jobs=4)
                        iso.fit(X_flat)
                        scores = -iso.decision_function(X_flat)
                        # Higher score = More anomalous
                        feat_idx = np.argsort(scores)[::-1][:int(len(targets)*TOP_K_PERCENT)]
                        
                        feat_m = calculate_metrics(preds, targets, feat_idx, task_type, prefix="Hard_Feat_")
                        current_seed_metrics.update(feat_m)
                        
                    # 3. Hard Target Tail Metrics (Regression Only)
                    if task_type == 'regression':
                        dev = np.abs(targets - np.mean(targets))
                        tail_idx = np.argsort(dev)[::-1][:int(len(targets)*TOP_K_PERCENT)]
                        tail_m = calculate_metrics(preds, targets, tail_idx, task_type, prefix="Hard_Target_")
                        current_seed_metrics.update(tail_m)
                        
                    seed_metrics_list.append(current_seed_metrics)
                    
                except Exception as e:
                    # print(f"Error in seed {seed}: {e}")
                    continue
            
            # --- [C] Aggregate Seeds (Mean ± Std) ---
            if len(seed_metrics_list) > 0:
                agg_row = {'Dataset': dataset, 'Model': model_name, 'Task': task_type}
                all_keys = seed_metrics_list[0].keys()
                
                for key in all_keys:
                    values = [m[key] for m in seed_metrics_list if key in m]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        agg_row[f"{key}_Mean"] = mean_val
                        agg_row[f"{key}_Std"] = std_val
                        agg_row[f"{key}_Str"] = f"{mean_val:.4f} ± {std_val:.4f}"
                
                final_summary.append(agg_row)
                
                # Print Summary for current model
                main_key = 'Full_RMSE' if task_type == 'regression' else 'Full_AUPRC'
                hard_key = 'Hard_Feat_RMSE' if task_type == 'regression' else 'Hard_Feat_AUPRC'
                res_str = f"      ✅ {agg_row.get(f'{main_key}_Str', 'N/A')} | Hard: {agg_row.get(f'{hard_key}_Str', 'N/A')}"
                print(res_str)
            else:
                print("      ❌ Failed to collect results for any seed.")

    # --- [D] Save Final CSV ---
    if not final_summary:
        print("❌ No results to save.")
        return

    df = pd.DataFrame(final_summary)
    
    # Column Sorting for Readability
    str_cols = [c for c in df.columns if c.endswith('_Str')]
    raw_cols = [c for c in df.columns if c not in str_cols and c not in ['Dataset', 'Model', 'Task']]
    cols = ['Dataset', 'Model', 'Task'] + sorted(str_cols) + sorted(raw_cols)
    df = df[cols]

    print("\n" + "="*60)
    print("📊 Final Aggregated Results (Mean ± Std)")
    print("="*60)
    
    os.makedirs("results/analysis", exist_ok=True)
    df.to_csv(f"results/analysis/model_comparison_{TOP_K_PERCENT}.csv", index=False)
    print(f"\n💾 Saved to results/analysis/model_comparison_{TOP_K_PERCENT}.csv")

if __name__ == "__main__":
    run_analysis()