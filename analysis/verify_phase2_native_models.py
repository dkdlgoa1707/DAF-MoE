#!/usr/bin/env python
"""Fit/predict smoke checks for pinned official Phase 2 estimators."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import RawDataset  # noqa: E402
from src.native.data import prepare_native_final, prepare_native_hpo  # noqa: E402
from src.native.dependencies import dependency_report  # noqa: E402
from src.native.runner import (  # noqa: E402
    evaluate_hpo_estimator,
    fit_hpo_estimator,
    run_native_final,
)


def _raw(task_type):
    n_rows = 60
    index = np.arange(n_rows)
    frame = pd.DataFrame(
        {
            "num": np.sin(index / 4.0),
            "num_nan": np.where(index % 13 == 0, np.nan, index / 10.0),
            "cat": np.resize(np.array(["a", "b", None], dtype=object), n_rows),
        }
    )
    if task_type == "regression":
        target = pd.Series(np.cos(index / 6.0), name="target")
    else:
        target = pd.Series(index % 2, name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num", "num_nan"),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name="native-smoke",
        schema_version="native-smoke-v1",
    )


def _assert_finite(name, value):
    if not np.isfinite(float(value)):
        raise AssertionError(f"{name} produced nonfinite metric: {value}")


def verify_xgboost():
    data = prepare_native_hpo(_raw("classification"), "xgboost", "classification", 42)
    params = {
        "n_estimators": 4,
        "early_stopping_rounds": 2,
        "max_depth": 2,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    }
    estimator, report, _ = fit_hpo_estimator(
        "xgboost", "classification", data, 42, params=params
    )
    value = evaluate_hpo_estimator(estimator, data, "classification", "acc")
    _assert_finite("xgboost", value)
    return report["installed_version"], value


def verify_catboost():
    data = prepare_native_hpo(_raw("classification"), "catboost", "classification", 42)
    params = {
        "iterations": 4,
        "od_type": "Iter",
        "od_wait": 2,
        "depth": 2,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bylevel": 1.0,
        "l2_leaf_reg": 1.0,
        "leaf_estimation_iterations": 1,
        "one_hot_max_size": 8,
        "max_ctr_complexity": 2,
    }
    estimator, report, _ = fit_hpo_estimator(
        "catboost", "classification", data, 42, params=params
    )
    value = evaluate_hpo_estimator(estimator, data, "classification", "acc")
    _assert_finite("catboost", value)
    return report["installed_version"], value


def verify_realmlp():
    report = dependency_report("realmlp")
    if not report["compatible"]:
        raise RuntimeError(report["install_command"])
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

    data = prepare_native_hpo(_raw("classification"), "realmlp", "classification", 42)
    estimator = RealMLP_TD_Classifier(
        device="cpu",
        random_state=42,
        n_epochs=1,
        n_cv=1,
        n_refit=0,
        hidden_sizes=[16],
        batch_size=16,
        verbosity=0,
    )
    estimator.fit(
        data.train.frame,
        data.train.targets,
        X_val=data.validation.frame,
        y_val=data.validation.targets,
        cat_col_names=data.frame_adapter.cat_col_names,
    )
    probabilities = estimator.predict_proba(data.validation.frame)
    if probabilities.shape != (len(data.validation.targets), 2):
        raise AssertionError(f"Unexpected RealMLP probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise AssertionError("RealMLP produced nonfinite probabilities.")
    return report["installed_version"], float(probabilities[:, 1].mean())


def verify_tabicl(task_type):
    report = dependency_report("tabicl")
    if not report["compatible"]:
        raise RuntimeError(report["install_command"])
    metric = "rmse" if task_type == "regression" else "acc"
    data = prepare_native_final(_raw(task_type), "tabicl", task_type, 43)
    result = run_native_final(
        "tabicl",
        data,
        data.train.frame.attrs.get("dataset_name", "native-smoke"),
        task_type,
        metric,
        43,
        device="cpu",
    )
    _assert_finite(f"tabicl-{task_type}", result.metric_value)
    expected_checkpoint = f"tabicl-{task_type.replace('classification', 'classifier').replace('regression', 'regressor')}-v2-20260212.ckpt"
    actual_checkpoint = result.manifest["resolved_config"]["checkpoint_version"]
    if actual_checkpoint != expected_checkpoint:
        raise AssertionError(
            f"TabICL checkpoint mismatch: {actual_checkpoint} != {expected_checkpoint}"
        )
    return report["installed_version"], result.metric_value


def main():
    checks = (
        ("xgboost", verify_xgboost),
        ("catboost", verify_catboost),
        ("realmlp", verify_realmlp),
        ("tabicl-classification", lambda: verify_tabicl("classification")),
        ("tabicl-regression", lambda: verify_tabicl("regression")),
    )
    for name, check in checks:
        version, value = check()
        print(f"{name}: version={version} value={float(value):.6g} PASS")
    print("PHASE2_NATIVE_MODELS_SMOKE_PASSED")


if __name__ == "__main__":
    main()
