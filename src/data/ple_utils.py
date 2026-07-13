import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocessor import DAFPreprocessor


def compute_ple_boundaries(config, data_cfg):
    """
    Computes PLE quantile boundaries from the same train split used by loader.py.
    Boundaries are calculated on the normalized numerical channel consumed by v1.5.
    """
    df = pd.read_csv(data_cfg['csv_path'], skipinitialspace=True)
    num_cols = data_cfg.get('num_cols', [])
    cat_cols = data_cfg.get('cat_cols', [])
    target_col = data_cfg['target_col']

    if len(num_cols) == 0:
        return []

    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    X = df[num_cols + cat_cols]
    y = df[target_col]

    stratify_param = y if config.task_type == 'classification' else None
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=stratify_param, random_state=config.seed
    )

    preprocessor = DAFPreprocessor(num_cols, cat_cols, config)
    preprocessor.fit(X_train)
    X_num_train, _, _ = preprocessor.transform(X_train)
    normalized_values = X_num_train[:, :, 0]

    quantiles = np.linspace(0.0, 1.0, config.ple_n_bins + 1)
    boundaries = []
    for feature_idx in range(normalized_values.shape[1]):
        feature_bounds = np.quantile(normalized_values[:, feature_idx], quantiles)
        feature_bounds = np.maximum.accumulate(feature_bounds)
        for idx in range(1, len(feature_bounds)):
            if feature_bounds[idx] <= feature_bounds[idx - 1]:
                feature_bounds[idx] = feature_bounds[idx - 1] + 1e-6
        boundaries.append(feature_bounds.astype(float).tolist())

    return boundaries


def inject_ple_boundaries(config, data_cfg):
    config.ple_boundaries = compute_ple_boundaries(config, data_cfg)
    return config
