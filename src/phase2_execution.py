"""Shared HPO/final execution for neural and native Phase 2 methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from src.configs.default_config import DAFConfig
from src.data.phase2_loader import (
    get_phase2_dataloaders,
    get_phase2_hpo_dataloaders,
)
from src.losses.factory import create_criterion
from src.models.factory import create_model
from src.native.data import prepare_native_final, prepare_native_hpo
from src.native.runner import (
    evaluate_hpo_estimator,
    fit_hpo_estimator,
    run_native_final,
)
from src.phase2_results import build_execution_manifest
from src.trainer import Trainer
from src.utils.common import seed_everything
from src.utils.metrics import Evaluator


NATIVE_MODELS = frozenset({"xgboost", "catboost", "realmlp", "tabicl"})
CONFIG_METADATA_FIELDS = frozenset(
    {
        "optimizer",
        "scheduler",
        "preprocessing",
        "activation",
        "normalization",
        "architecture",
        "plr_lite",
        "distance",
        "sampling",
        "estimator_family",
        "external_hpo",
        "regression_label_smoothing",
    }
)


class SilentLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


@dataclass(frozen=True)
class ExecutionOutcome:
    metric_value: float
    metrics: dict
    split_hash: str
    preprocessing_hash: str
    manifest: dict
    best_epoch_or_iteration: Optional[int] = None
    epochs_completed: Optional[int] = None


def materialize_neural_config(
    base_config,
    search_space,
    resolved_config,
    task_type,
    metric_name,
    seed,
    checkpoint_path=None,
):
    search_space.validate_resolved(resolved_config, task_type=task_type)
    base_config = dict(base_config)
    if base_config.get("model_name", search_space.model_name) != search_space.model_name:
        raise ValueError("Base config model_name does not match search-space model_name.")
    forbidden = sorted(set(base_config).intersection(search_space.forbidden))
    if forbidden:
        raise ValueError(f"Base config contains forbidden fields: {forbidden}")
    for key, fixed_value in search_space.fixed.items():
        if key in base_config and base_config[key] != fixed_value:
            raise ValueError(
                f"Base config collides with fixed field {key}: "
                f"{base_config[key]} != {fixed_value}"
            )

    config = DAFConfig()
    combined = dict(base_config)
    combined.update(resolved_config)
    combined["model_name"] = search_space.model_name
    combined["task_type"] = task_type
    combined["optimize_metric"] = metric_name
    combined["seed"] = int(seed)
    unknown = sorted(
        key
        for key in combined
        if not hasattr(config, key) and key not in CONFIG_METADATA_FIELDS
    )
    if unknown:
        raise ValueError(f"Unknown Phase 2 config fields: {unknown}")
    applied = {}
    for key, value in combined.items():
        if hasattr(config, key):
            setattr(config, key, value)
            applied[key] = value
    config.explicit_fields = frozenset(applied)
    config.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
    return config


def _manifest_fields(manifest):
    return manifest["split_index_hash"], manifest["fitted_state_hash"]


def _evaluate_torch_model(model, loader, config, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions.append(model(**inputs)["logits"].detach().cpu())
            targets.append(batch_targets.detach().cpu())
    if not predictions:
        raise ValueError("Cannot evaluate an empty data loader.")
    return Evaluator(task_type=config.task_type)(
        torch.cat(targets), torch.cat(predictions)
    )


def _fit_neural(raw_dataset, config, include_test, device):
    seed_everything(config.seed)
    if include_test:
        train_loader, validation_loader, test_loader = get_phase2_dataloaders(
            config, raw_dataset
        )
    else:
        train_loader, validation_loader = get_phase2_hpo_dataloaders(
            config, raw_dataset
        )
        test_loader = None
    model = create_model(config).to(device)
    criterion = create_criterion(config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        config,
        device,
        SilentLogger(),
        verbose=False,
    )
    metric_value = float(trainer.fit(train_loader, validation_loader))
    if include_test:
        metrics = _evaluate_torch_model(model, test_loader, config, device)
        metric_value = float(metrics[config.optimize_metric])
    else:
        metrics = {config.optimize_metric: metric_value}
    return trainer, metrics, metric_value


def execute_hpo_trial(
    raw_dataset,
    base_config,
    search_space,
    resolved_config,
    task_type,
    metric_name,
    checkpoint_path=None,
    device=None,
):
    seed = 42
    model_name = search_space.model_name
    if model_name == "tabicl":
        raise ValueError("TabICLv2 has no HPO objective.")
    if model_name in NATIVE_MODELS:
        data = prepare_native_hpo(raw_dataset, model_name, task_type, seed=seed)
        estimator, dependency, estimator_config = fit_hpo_estimator(
            model_name, task_type, data, seed, params=resolved_config
        )
        value = float(evaluate_hpo_estimator(estimator, data, task_type, metric_name))
        best_iteration = getattr(estimator, "best_iteration", None)
        if best_iteration is None and hasattr(estimator, "get_best_iteration"):
            best_iteration = estimator.get_best_iteration()
        split_hash, preprocessing_hash = _manifest_fields(data.manifest)
        manifest = build_execution_manifest(
            data.manifest,
            model_name,
            resolved_config,
            search_space.schema_hash,
            seed,
        )
        manifest["dependency"] = dependency
        manifest["estimator_config"] = estimator_config
        return ExecutionOutcome(
            metric_value=value,
            metrics={metric_name: value},
            split_hash=split_hash,
            preprocessing_hash=preprocessing_hash,
            manifest=manifest,
            best_epoch_or_iteration=best_iteration,
        )

    config = materialize_neural_config(
        base_config,
        search_space,
        resolved_config,
        task_type,
        metric_name,
        seed,
        checkpoint_path=checkpoint_path,
    )
    trainer, metrics, value = _fit_neural(
        raw_dataset, config, include_test=False, device=device or torch.device("cpu")
    )
    split_hash, preprocessing_hash = _manifest_fields(config.phase2_manifest)
    manifest = build_execution_manifest(
        config.phase2_manifest,
        model_name,
        resolved_config,
        search_space.schema_hash,
        seed,
    )
    return ExecutionOutcome(
        metric_value=value,
        metrics=metrics,
        split_hash=split_hash,
        preprocessing_hash=preprocessing_hash,
        manifest=manifest,
        best_epoch_or_iteration=trainer.best_epoch,
        epochs_completed=trainer.epochs_completed,
    )


def execute_final_seed(
    raw_dataset,
    base_config,
    search_space,
    resolved_config,
    task_type,
    metric_name,
    seed,
    checkpoint_path=None,
    device=None,
):
    if int(seed) == 42:
        raise ValueError("HPO seed 42 cannot be used for final evaluation.")
    model_name = search_space.model_name
    if model_name in NATIVE_MODELS:
        data = prepare_native_final(raw_dataset, model_name, task_type, seed=int(seed))
        result = run_native_final(
            model_name,
            data,
            raw_dataset.dataset_name,
            task_type,
            metric_name,
            int(seed),
            params=None if model_name == "tabicl" else resolved_config,
            device=str(device) if device is not None else None,
        )
        manifest = build_execution_manifest(
            result.manifest,
            model_name,
            resolved_config,
            search_space.schema_hash,
            seed,
        )
        return ExecutionOutcome(
            metric_value=float(result.metric_value),
            metrics={metric_name: float(result.metric_value)},
            split_hash=manifest["split_index_hash"],
            preprocessing_hash=manifest["fitted_state_hash"],
            manifest=manifest,
        )

    config = materialize_neural_config(
        base_config,
        search_space,
        resolved_config,
        task_type,
        metric_name,
        int(seed),
        checkpoint_path=checkpoint_path,
    )
    trainer, metrics, value = _fit_neural(
        raw_dataset, config, include_test=True, device=device or torch.device("cpu")
    )
    manifest = build_execution_manifest(
        config.phase2_manifest,
        model_name,
        resolved_config,
        search_space.schema_hash,
        seed,
    )
    return ExecutionOutcome(
        metric_value=value,
        metrics=metrics,
        split_hash=manifest["split_index_hash"],
        preprocessing_hash=manifest["fitted_state_hash"],
        manifest=manifest,
        best_epoch_or_iteration=trainer.best_epoch,
        epochs_completed=trainer.epochs_completed,
    )
