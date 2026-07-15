"""Run pretrained TabICL inference on Phase 2 datasets.

TabICL is evaluated through its official sklearn-style API. It does not use the
repository's PyTorch training loop and does not perform gradient-based fitting.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.data.loader import get_dataloaders
from src.utils.common import seed_everything


DATASETS = [
    "california",
    "adult",
    "higgs_small",
    "covertype",
    "allstate",
    "bnp",
    "nhanes",
    "mimic3",
    "mimic4",
]
SEEDS = list(range(42, 57))
COVERTYPE_SUBSAMPLE = 400_000


def load_config_and_data(dataset_name, seed):
    config_path = Path("configs/experiments") / f"{dataset_name}_daf_moe_best.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        experiment = yaml.safe_load(file)
    with Path(experiment["data_config_path"]).open(encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    config = DAFConfig()
    for key, value in experiment.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed

    loaders = get_dataloaders(config, data_config)
    return config, loaders


def to_flat_arrays(loader):
    """Convert DAFDataset batches while preserving categorical integer dtypes."""
    numerical_parts = []
    categorical_parts = []
    targets = []
    for inputs, target in loader:
        numerical_parts.append(inputs["x_numerical"][:, :, 0].numpy())
        categorical_parts.append(inputs["x_categorical_idx"].numpy())
        targets.append(target.numpy())

    numerical = np.concatenate(numerical_parts)
    categorical = np.concatenate(categorical_parts).astype(np.int64, copy=False)
    columns = {
        **{f"num_{i}": numerical[:, i] for i in range(numerical.shape[1])},
        **{f"cat_{i}": categorical[:, i] for i in range(categorical.shape[1])},
    }
    frame = pd.DataFrame(columns)
    categorical_features = list(
        range(numerical.shape[1], numerical.shape[1] + categorical.shape[1])
    )
    return frame, np.concatenate(targets), categorical_features


def fit_with_categorical_features(
    model, x_train, y_train, x_test, categorical_features
):
    """Fit across TabICL API versions without losing categorical semantics."""
    if categorical_features and "categorical_features" in inspect.signature(model.fit).parameters:
        model.fit(
            x_train,
            y_train,
            categorical_features=categorical_features,
        )
        return x_train, x_test

    # TabICLv2 infers categorical columns from pandas dtypes instead of a fit
    # argument. Category values remain integer-coded in both frames.
    if categorical_features:
        x_train = x_train.copy()
        x_test = x_test.copy()
        for index in categorical_features:
            column = x_train.columns[index]
            x_train[column] = x_train[column].astype("category")
            x_test[column] = x_test[column].astype("category")
    model.fit(x_train, y_train)
    return x_train, x_test


def evaluate_predictions(model, config, y_test, predictions, x_test):
    metric_name = config.optimize_metric.lower()
    if config.task_type == "regression":
        value = mean_squared_error(y_test, predictions) ** 0.5
        return "rmse", float(value)

    if metric_name in {"auprc", "auroc", "roc_auc"}:
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] != 2:
            raise ValueError(
                f"{metric_name.upper()} evaluation currently requires binary classification."
            )
        positive_probabilities = probabilities[:, 1]
        if metric_name == "auprc":
            return "auprc", float(
                average_precision_score(y_test, positive_probabilities)
            )
        return "auroc", float(roc_auc_score(y_test, positive_probabilities))
    return "acc", float(accuracy_score(y_test, predictions))


def run_dataset(dataset_name, seed, device, subsample=None):
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:
        raise ImportError(
            "TabICL is not installed. Install the Phase 2 dependencies from requirements.txt."
        ) from exc

    seed_everything(seed)
    config, (train_loader, _, test_loader) = load_config_and_data(dataset_name, seed)
    x_train, y_train, categorical_features = to_flat_arrays(train_loader)
    x_test, y_test, test_categorical_features = to_flat_arrays(test_loader)
    if categorical_features != test_categorical_features:
        raise ValueError("Train/test categorical feature positions do not match.")

    if subsample and len(x_train) > subsample:
        indices = np.random.RandomState(seed).permutation(len(x_train))[:subsample]
        x_train = x_train.iloc[indices].reset_index(drop=True)
        y_train = y_train[indices]
        print(f"  Subsampled context to {len(x_train):,} rows")

    estimator_class = (
        TabICLRegressor if config.task_type == "regression" else TabICLClassifier
    )
    model = estimator_class(device=device, random_state=seed)

    start = time.time()
    x_train, x_test = fit_with_categorical_features(
        model,
        x_train,
        y_train,
        x_test,
        categorical_features,
    )
    predictions = model.predict(x_test)
    elapsed = time.time() - start
    metric_name, metric_value = evaluate_predictions(
        model, config, y_test, predictions, x_test
    )

    result = {
        "model": "tabicl",
        "dataset": dataset_name,
        "seed": seed,
        "metric": metric_name,
        "value": metric_value,
        "elapsed_seconds": elapsed,
        "n_train_used": len(x_train),
    }
    output_dir = Path("results/phase2/tabicl") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"seed{seed}.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(
        f"[TabICL] {dataset_name} seed={seed}: "
        f"{metric_name}={metric_value:.4f}, elapsed={elapsed:.1f}s"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained TabICL on Phase 2 datasets."
    )
    parser.add_argument("--gpu-id", default="0", help="CUDA device index")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda:0"

    failures = []
    for dataset in args.datasets:
        subsample = COVERTYPE_SUBSAMPLE if dataset == "covertype" else None
        for seed in args.seeds:
            try:
                run_dataset(dataset, seed, device=device, subsample=subsample)
            except Exception as exc:
                failures.append((dataset, seed, str(exc)))
                print(f"[Fail] {dataset} seed={seed}: {exc}")

    if failures:
        raise SystemExit(f"TabICL failed for {len(failures)} run(s).")


if __name__ == "__main__":
    main()
