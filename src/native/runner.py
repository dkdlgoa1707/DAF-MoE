"""Execution primitives shared by native Phase 2 entrypoints."""

from dataclasses import dataclass
import time

import numpy as np

from src.data.provenance import stable_hash

from .data import build_tabicl_context
from .estimators import (
    create_catboost,
    create_realmlp,
    create_tabicl,
    create_xgboost,
)
from .metrics import evaluate_metric


@dataclass(frozen=True)
class NativeRunResult:
    model_name: str
    metric_name: str
    metric_value: float
    elapsed_seconds: float
    manifest: dict


def _predictions_and_probabilities(estimator, frame, task_type):
    predictions = estimator.predict(frame)
    probabilities = None
    if task_type == "classification" and hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(frame)
    return predictions, probabilities


def fit_hpo_estimator(model_name, task_type, data, seed, params=None):
    """Fit on train and use validation only; a test partition cannot be supplied."""
    normalized = model_name.lower()
    if normalized == "xgboost":
        estimator, dependency, resolved = create_xgboost(task_type, seed, params)
        estimator.fit(
            data.train.frame,
            data.train.targets,
            eval_set=[(data.validation.frame, data.validation.targets)],
            verbose=False,
        )
    elif normalized == "catboost":
        estimator, dependency, resolved = create_catboost(task_type, seed, params)
        estimator.fit(
            data.train.frame,
            data.train.targets,
            eval_set=(data.validation.frame, data.validation.targets),
            cat_features=data.frame_adapter.cat_features,
            verbose=False,
        )
    elif normalized == "realmlp":
        estimator, dependency, resolved = create_realmlp(task_type, seed, params)
        estimator.fit(
            data.train.frame,
            data.train.targets,
            X_val=data.validation.frame,
            y_val=data.validation.targets,
            cat_col_names=data.frame_adapter.cat_col_names,
        )
    else:
        raise ValueError(f"{model_name} has no native HPO fit path.")
    return estimator, dependency, resolved


def evaluate_hpo_estimator(estimator, data, task_type, metric_name):
    predictions, probabilities = _predictions_and_probabilities(
        estimator, data.validation.frame, task_type
    )
    return evaluate_metric(
        metric_name,
        data.validation.targets,
        predictions=predictions,
        probabilities=probabilities,
    )


def run_tabicl_final(data, dataset_name, task_type, metric_name, seed, device=None):
    context = build_tabicl_context(data, dataset_name, seed)
    estimator, dependency, resolved = create_tabicl(task_type, seed, device=device)
    started = time.perf_counter()
    estimator.fit(context.context_frame, context.context_targets)
    predictions, probabilities = _predictions_and_probabilities(
        estimator, context.query_frame, task_type
    )
    elapsed = time.perf_counter() - started
    value = evaluate_metric(
        metric_name,
        context.query_targets,
        predictions=predictions,
        probabilities=probabilities,
    )
    manifest = dict(data.manifest)
    manifest.update(
        {
            "dependency": dependency,
            "resolved_config": resolved,
            "context_row_count": len(context.context_row_ids),
            "context_index_hash": context.context_index_hash,
            "context_subsample_size": context.subsample_size,
            "test_query_only": True,
        }
    )
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    return NativeRunResult(
        model_name="tabicl",
        metric_name=metric_name,
        metric_value=value,
        elapsed_seconds=elapsed,
        manifest=manifest,
    )


def run_native_final(
    model_name,
    data,
    dataset_name,
    task_type,
    metric_name,
    seed,
    params=None,
    device=None,
):
    """Run one final outer seed without hiding estimator failures."""
    normalized = model_name.lower()
    if normalized == "tabicl":
        if params:
            raise ValueError("TabICLv2 has no HPO or user-tunable parameters.")
        return run_tabicl_final(
            data, dataset_name, task_type, metric_name, seed, device=device
        )

    started = time.perf_counter()
    estimator, dependency, resolved = fit_hpo_estimator(
        normalized, task_type, data, seed, params=params
    )
    predictions, probabilities = _predictions_and_probabilities(
        estimator, data.test.frame, task_type
    )
    elapsed = time.perf_counter() - started
    value = evaluate_metric(
        metric_name,
        data.test.targets,
        predictions=predictions,
        probabilities=probabilities,
    )
    manifest = dict(data.manifest)
    manifest.update(
        {
            "dependency": dependency,
            "resolved_config": resolved,
            "test_query_only": True,
        }
    )
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    return NativeRunResult(
        model_name=normalized,
        metric_name=metric_name,
        metric_value=value,
        elapsed_seconds=elapsed,
        manifest=manifest,
    )
