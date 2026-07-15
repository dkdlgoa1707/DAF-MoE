"""Run pretrained TabICL inference on Phase 2 datasets.

TabICL is evaluated through its official sklearn-style API. It does not use the
repository's PyTorch training loop and does not perform gradient-based fitting.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_error

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
    """Convert DAFDataset batches to the dense matrix expected by TabICL."""
    features = []
    targets = []
    for inputs, target in loader:
        numerical = inputs["x_numerical"][:, :, 0]
        categorical = inputs["x_categorical_idx"].float()
        features.append(
            np.concatenate(
                [numerical.numpy(), categorical.numpy()], axis=1
            )
        )
        targets.append(target.numpy())
    return np.concatenate(features), np.concatenate(targets)


def evaluate_predictions(model, config, y_test, predictions, x_test):
    metric_name = config.optimize_metric.lower()
    if config.task_type == "regression":
        value = mean_squared_error(y_test, predictions) ** 0.5
        return "rmse", float(value)

    if metric_name == "auprc":
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] != 2:
            raise ValueError("AUPRC evaluation currently requires binary classification.")
        return "auprc", float(average_precision_score(y_test, probabilities[:, 1]))
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
    x_train, y_train = to_flat_arrays(train_loader)
    x_test, y_test = to_flat_arrays(test_loader)

    if subsample and len(x_train) > subsample:
        indices = np.random.RandomState(seed).permutation(len(x_train))[:subsample]
        x_train, y_train = x_train[indices], y_train[indices]
        print(f"  Subsampled context to {len(x_train):,} rows")

    estimator_class = (
        TabICLRegressor if config.task_type == "regression" else TabICLClassifier
    )
    model = estimator_class(device=device, random_state=seed)

    start = time.time()
    model.fit(x_train, y_train)
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
