"""Factories for pinned official/native estimators."""

from .dependencies import require_dependency


TABICL_CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"


def _merge_fixed(params, fixed):
    resolved = dict(params or {})
    collisions = {
        key: (resolved[key], value)
        for key, value in fixed.items()
        if key in resolved and resolved[key] != value
    }
    if collisions:
        raise ValueError(f"Parameters conflict with fixed protocol fields: {collisions}")
    resolved.update(fixed)
    return resolved


def create_xgboost(task_type, seed, params=None):
    module, report = require_dependency("xgboost")
    resolved = _merge_fixed(
        params,
        {
            "tree_method": "hist",
            "enable_categorical": True,
            "grow_policy": "depthwise",
            "random_state": int(seed),
        },
    )
    estimator_class = (
        module.XGBRegressor if task_type == "regression" else module.XGBClassifier
    )
    return estimator_class(**resolved), report, resolved


def create_catboost(task_type, seed, params=None):
    module, report = require_dependency("catboost")
    resolved = _merge_fixed(
        params,
        {
            "boosting_type": "Plain",
            "grow_policy": "SymmetricTree",
            "bootstrap_type": "Bernoulli",
            "random_seed": int(seed),
            "verbose": False,
        },
    )
    estimator_class = (
        module.CatBoostRegressor
        if task_type == "regression"
        else module.CatBoostClassifier
    )
    return estimator_class(**resolved), report, resolved


def _realmlp_trial_params(task_type, params):
    """Map the fixed outer-Optuna space to official RealMLP-TD constructor keys."""
    params = dict(params)
    embedding = params["numerical_embedding"]
    label_smoothing = float(params["label_smoothing"])
    if task_type == "regression":
        label_smoothing = 0.0
    return {
        "num_emb_type": "none" if embedding is None else str(embedding).lower(),
        "add_front_scale": bool(params["scaling_layer"]),
        "lr": float(params["learning_rate"]),
        "p_drop": float(params["dropout"]),
        "act": str(params["activation"]).lower(),
        "hidden_sizes": list(params["hidden_shape"]),
        "wd": float(params["weight_decay"]),
        "plr_sigma": float(params["first_embedding_init_std"]),
        "use_ls": task_type == "classification" and label_smoothing > 0.0,
        "ls_eps": label_smoothing,
    }


def create_realmlp(task_type, seed, params=None):
    """Create official pytabkit RealMLP.

    With no sampled parameters this exposes pytabkit's RealMLP-HPO pipeline.
    With outer Optuna parameters it uses the official RealMLP-TD single-config
    estimator, because RealMLP_HPO_* does not accept individual architecture
    parameters in pytabkit 1.7.3.
    """
    _, report = require_dependency("realmlp")
    from pytabkit.models.sklearn.sklearn_interfaces import (
        RealMLP_HPO_Classifier,
        RealMLP_HPO_Regressor,
        RealMLP_TD_Classifier,
        RealMLP_TD_Regressor,
    )

    if params:
        resolved = _realmlp_trial_params(task_type, params)
        resolved["random_state"] = int(seed)
        estimator_class = (
            RealMLP_TD_Regressor
            if task_type == "regression"
            else RealMLP_TD_Classifier
        )
    else:
        resolved = {"random_state": int(seed)}
        estimator_class = (
            RealMLP_HPO_Regressor
            if task_type == "regression"
            else RealMLP_HPO_Classifier
        )
    return estimator_class(**resolved), report, resolved


def create_tabicl(task_type, seed, device=None):
    module, report = require_dependency("tabicl")
    if task_type == "regression":
        estimator_class = module.TabICLRegressor
        checkpoint = TABICL_REGRESSOR_CHECKPOINT
    else:
        estimator_class = module.TabICLClassifier
        checkpoint = TABICL_CLASSIFIER_CHECKPOINT
    resolved = {
        "n_estimators": 8,
        "checkpoint_version": checkpoint,
        "random_state": int(seed),
        "device": device,
    }
    return estimator_class(**resolved), report, resolved
