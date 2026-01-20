import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data.preprocessor import DAFPreprocessor
from src.data.dataset import DAFDataset

def get_dataloaders(config, data_cfg):
    """
    YAML 설정에 따라 데이터를 로드하고 전처리를 수행한 뒤 DataLoader를 반환합니다.
    """
    csv_path = data_cfg.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"🚨 Data file not found: {csv_path}")

    print(f"📂 Preparing Data from {csv_path}...")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    num_cols = data_cfg['numerical_cols']
    cat_cols = data_cfg['categorical_cols']
    target_col = data_cfg['target_col']
    
    X = df[num_cols + cat_cols]
    y = df[target_col]

    # 2. Stratified Split (8:1:1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # 3. Fit Preprocessor (Train Only)
    print("🛠️ Fitting Preprocessor (Train Only)...")
    preprocessor = DAFPreprocessor(num_cols, cat_cols, config=config)
    preprocessor.fit(X_train)
    
    # Config Update (자동 감지)
    vocab_sum = sum([len(le.classes_) for le in preprocessor.label_encoders.values()])
    config.n_numerical = len(num_cols)
    config.n_categorical = len(cat_cols)
    config.n_features = config.n_numerical + config.n_categorical
    config.total_cats = vocab_sum + 10 # 여유분

    # 4. Create Datasets
    def _create_ds(X_split, y_split):
        X_num, X_cat_idx, X_cat_meta = preprocessor.transform(X_split)
        return DAFDataset(X_num, X_cat_idx, X_cat_meta, y_split.values, config.task_type)

    train_ds = _create_ds(X_train, y_train)
    val_ds = _create_ds(X_val, y_val)
    
    # 5. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✅ Data Ready: Train({len(train_ds)}), Val({len(val_ds)})")
    
    return train_loader, val_loader, preprocessor