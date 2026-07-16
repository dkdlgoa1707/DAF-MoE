"""Shared metric registry for native Phase 2 runners."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
)


METRIC_DIRECTIONS = {
    "rmse": "minimize",
    "acc": "maximize",
    "accuracy": "maximize",
    "auprc": "maximize",
    "auroc": "maximize",
    "roc_auc": "maximize",
}


def metric_direction(metric_name):
    try:
        return METRIC_DIRECTIONS[metric_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2 metric: {metric_name}") from exc


def evaluate_metric(metric_name, targets, predictions=None, probabilities=None):
    name = metric_name.lower()
    targets = np.asarray(targets).reshape(-1)
    if name == "rmse":
        if predictions is None:
            raise ValueError("RMSE requires predictions.")
        return float(mean_squared_error(targets, np.asarray(predictions).reshape(-1)) ** 0.5)
    if name in {"acc", "accuracy"}:
        if predictions is None:
            raise ValueError("Accuracy requires predictions.")
        return float(accuracy_score(targets, np.asarray(predictions).reshape(-1)))
    if name in {"auprc", "auroc", "roc_auc"}:
        if probabilities is None:
            raise ValueError(f"{name} requires positive-class probabilities.")
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 2:
            if probabilities.shape[1] != 2:
                raise ValueError(f"{name} requires binary probabilities, got {probabilities.shape}.")
            probabilities = probabilities[:, 1]
        probabilities = probabilities.reshape(-1)
        if name == "auprc":
            return float(average_precision_score(targets, probabilities))
        return float(roc_auc_score(targets, probabilities))
    raise ValueError(f"Unsupported Phase 2 metric: {metric_name}")
