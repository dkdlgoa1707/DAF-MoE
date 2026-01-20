import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 프로젝트 모듈
from src.configs.default_config import DAFConfig
from src.data.preprocessor import DAFPreprocessor
from src.data.dataset import DAFDataset
from src.models.daf_moe_transformer import DAFMoETransformer
from src.losses.daf_moe_loss import DAFLoss
from src.utils.metrics import Evaluator

def load_config(yaml_path):
    """
    [Hybrid Config System]
    1. default_config.py (Schema & Defaults)
    2. Experiment YAML (Override Params)
    3. Dataset YAML (Features & CSV Path)
    """
    config = DAFConfig()
    
    # 1. Experiment Config 로드
    print(f"📄 Loading Experiment Config: {yaml_path}")
    with open(yaml_path, 'r') as f:
        exp_args = yaml.safe_load(f)
        
    for k, v in exp_args.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    # 2. Data Schema YAML 로드
    data_cfg = None
    if config.data_config_path:
        print(f"📄 Loading Data Schema: {config.data_config_path}")
        with open(config.data_config_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
    else:
        raise ValueError("Experiment YAML must contain 'data_config_path'")
            
    return config, data_cfg

def main(args):
    # [Step 0] GPU Setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 System Config: Device={device}, Visible GPUs={args.gpu_ids}")

    # [Step 1] Configuration Load
    config, data_cfg = load_config(args.config)
    
    # [Change] YAML에서만 경로를 가져옴 (CLI 로직 삭제됨)
    try:
        csv_path = data_cfg['csv_path']
        NUMERICAL_COLS = data_cfg['numerical_cols']
        CATEGORICAL_COLS = data_cfg['categorical_cols']
        TARGET_COL = data_cfg['target_col']
    except KeyError as e:
        raise KeyError(f"🚨 YAML file is missing a required key: {e}. Please check {config.data_config_path}")

    print(f"📊 Dataset: {data_cfg.get('dataset_name', 'Unknown')}")
    print(f"   Source: {csv_path}")
    print(f"   Target: {TARGET_COL}")
    print(f"   Features: {len(NUMERICAL_COLS)} Num, {len(CATEGORICAL_COLS)} Cat")

    # [Step 2] Data Load & Split
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"🚨 Data file not found at: {csv_path}")

    print(f"📂 Reading CSV...")
    df = pd.read_csv(csv_path)
    
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    # Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"✂️  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # [Step 3] Preprocessing (Fit on Train ONLY)
    print("🛠️ Fitting Preprocessor...")
    preprocessor = DAFPreprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, config=config)
    preprocessor.fit(X_train)

    def create_dataset(X_split, y_split):
        X_num, X_cat_idx, X_cat_meta = preprocessor.transform(X_split)
        return DAFDataset(X_num, X_cat_idx, X_cat_meta, y_split.values, config.task_type)

    train_ds = create_dataset(X_train, y_train)
    val_ds = create_dataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # [Step 4] Model Initialization
    config.n_numerical = len(NUMERICAL_COLS)
    config.n_categorical = len(CATEGORICAL_COLS)
    config.n_features = config.n_numerical + config.n_categorical
    
    vocab_sum = sum([len(le.classes_) for le in preprocessor.label_encoders.values()])
    config.total_cats = vocab_sum + 10 

    model = DAFMoETransformer(config)
    
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = DAFLoss(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    evaluator = Evaluator(task_type=config.task_type)

    # [Step 5] Training Loop
    print("\n🎬 Starting Training Loop...")
    best_metric = -float('inf') if config.task_type == 'classification' else float('inf')
    best_epoch = 0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config.epochs):
        # TRAIN
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for inputs, targets in pbar:
            for k in inputs: inputs[k] = inputs[k].to(device)
            targets = targets.to(device)

            logits, hist, meta = model(**inputs)
            losses = criterion(logits, targets, hist, meta)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f"{losses['total'].item():.4f}"})

        # VALIDATION
        model.eval()
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                for k in inputs: inputs[k] = inputs[k].to(device)
                targets = targets.to(device)
                logits, _, _ = model(**inputs)
                val_preds.append(logits)
                val_targets.append(targets)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        metrics = evaluator(val_targets, val_preds)
        
        # Log & Save
        val_log = f"   >>> [Val] "
        for k, v in metrics.items():
            val_log += f"{k.upper()}: {v:.4f} | "
        print(val_log)

        current_metric = metrics.get('auroc', -metrics.get('mse', 0))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            save_path = os.path.join(save_dir, f"{config.dataset_name}_best.pth")
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"   🏆 New Best Model Saved!")

    print(f"\n✅ Training Finished. Best Epoch: {best_epoch}, Best Score: {best_metric:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [Clean] --data_path 삭제됨. Config만 입력받음.
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('--gpu_ids', type=str, default='1,2', help='GPU IDs to use')
    
    args = parser.parse_args()
    main(args)