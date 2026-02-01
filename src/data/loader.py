import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .preprocessor import DAFPreprocessor
from .dataset import DAFDataset

def get_dataloaders(config, data_cfg):
    """
    Loads raw CSV, applies DAF-preprocessing, and returns Train/Val/Test DataLoaders.
    """
    print(f"📂 Loading Data from: {data_cfg['csv_path']}")
    df = pd.read_csv(data_cfg['csv_path'], skipinitialspace=True)

    num_cols = data_cfg.get('num_cols', [])
    cat_cols = data_cfg.get('cat_cols', [])
    target_col = data_cfg['target_col']
    
    # Target Encoding for Classification
    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
        print(f"🔄 Encoding Target Column '{target_col}'...")
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    else:
        print(f"✅ Target '{target_col}' is already numerical.")

    X = df[num_cols + cat_cols]
    y = df[target_col]

    # Stratified Split
    stratify_param = y if config.task_type == 'classification' else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=stratify_param, random_state=config.seed
    )
    
    stratify_temp = y_temp if config.task_type == 'classification' else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=config.seed
    )
    
    # Preprocessing (Fit on Train, Transform All)
    print("🛠️ Fitting Preprocessor...")
    preprocessor = DAFPreprocessor(num_cols, cat_cols, config)
    preprocessor.fit(X_train)
    
    # Update Config with Dataset Info
    X_num_tr, X_cat_tr, _ = preprocessor.transform(X_train)
    config.total_cats = int(X_cat_tr.max() + 1) if X_cat_tr.size > 0 else 0
    config.n_numerical = X_num_tr.shape[1]
    config.n_categorical = X_cat_tr.shape[1]
    config.n_features = config.n_numerical + config.n_categorical
    
    print(f"📊 Dataset Stats: Total Cats({config.total_cats}), Features({config.n_features})")

    # Helper to wrap transformed arrays into DAFDataset
    def wrap_dataset(X_arrays, y_series):
        return DAFDataset(*X_arrays, y=y_series.values)
    
    train_loader = DataLoader(
        wrap_dataset(preprocessor.transform(X_train), y_train), 
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        wrap_dataset(preprocessor.transform(X_val), y_val), 
        batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        wrap_dataset(preprocessor.transform(X_test), y_test), 
        batch_size=config.batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader