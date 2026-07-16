"""Phase 2 data preparation entrypoints with sealed HPO test access."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .adapters import AdapterOutput, get_adapter
from .dataset import DAFDataset
from .phase2_dataset import Phase2TensorDataset
from .provenance import build_run_manifest
from .splits import RawDataset, RawSplitRegistry, TrainOnlyTargetEncoder


@dataclass(frozen=True)
class PreparedPartition:
    inputs: dict
    targets: object
    row_ids: object


@dataclass(frozen=True)
class PreparedHPOData:
    train: PreparedPartition
    validation: PreparedPartition
    adapter: object
    target_encoder: TrainOnlyTargetEncoder
    manifest: dict


@dataclass(frozen=True)
class PreparedFinalData:
    train: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition
    adapter: object
    target_encoder: TrainOnlyTargetEncoder
    manifest: dict


def _adapter_kwargs(config):
    return {
        "n_bins": getattr(config, "ple_n_bins", 48),
        "seed": config.seed,
    }


def _prepare_partition(raw_partition, adapter, target_encoder):
    transformed: AdapterOutput = adapter.transform(raw_partition.features)
    return PreparedPartition(
        inputs=dict(transformed.inputs),
        targets=target_encoder.transform(raw_partition.target),
        row_ids=raw_partition.row_ids.copy(),
    ), transformed


def _fit_components(raw_dataset, train_partition, config):
    adapter = get_adapter(
        config.model_name,
        raw_dataset.numerical_columns,
        raw_dataset.categorical_columns,
        **_adapter_kwargs(config),
    ).fit(train_partition.features)
    if hasattr(adapter, "apply_to_config"):
        adapter.apply_to_config(config)

    regression_policy = getattr(config, "regression_target_policy", "identity")
    target_encoder = TrainOnlyTargetEncoder(
        config.task_type, regression_policy=regression_policy
    ).fit(train_partition.target)
    if config.task_type == "classification":
        n_classes = len(target_encoder.class_mapping)
        if config.model_name.lower() in {"tabr", "modernnca"}:
            # Canonical retrieval models embed class labels and predict class
            # distributions, including two logits for binary classification.
            config.out_dim = n_classes
        else:
            config.out_dim = 1 if n_classes == 2 else n_classes
    return adapter, target_encoder


def prepare_phase2_hpo(raw_dataset: RawDataset, config) -> PreparedHPOData:
    """Prepare train/validation without constructing or exposing a test partition."""
    partitions = RawSplitRegistry(raw_dataset, config.task_type, config.seed).for_hpo()
    adapter, target_encoder = _fit_components(raw_dataset, partitions.train, config)
    train, train_report = _prepare_partition(partitions.train, adapter, target_encoder)
    validation, validation_report = _prepare_partition(
        partitions.validation, adapter, target_encoder
    )
    manifest = build_run_manifest(
        dataset_name=raw_dataset.dataset_name,
        schema_version=raw_dataset.schema_version,
        schema_hash=raw_dataset.schema_hash,
        split_hash=partitions.split_hash,
        adapter=adapter,
        seed=config.seed,
        subsample_size=getattr(config, "subsample", None),
        missing_counts={
            "train": train_report.missing_counts,
            "validation": validation_report.missing_counts,
        },
        unseen_category_counts={
            "train": train_report.unseen_category_counts,
            "validation": validation_report.unseen_category_counts,
        },
    )
    return PreparedHPOData(
        train=train,
        validation=validation,
        adapter=adapter,
        target_encoder=target_encoder,
        manifest=manifest,
    )


def prepare_phase2_final(raw_dataset: RawDataset, config) -> PreparedFinalData:
    partitions = RawSplitRegistry(raw_dataset, config.task_type, config.seed).for_final()
    adapter, target_encoder = _fit_components(raw_dataset, partitions.train, config)
    train, train_report = _prepare_partition(partitions.train, adapter, target_encoder)
    validation, validation_report = _prepare_partition(
        partitions.validation, adapter, target_encoder
    )
    test, test_report = _prepare_partition(partitions.test, adapter, target_encoder)
    manifest = build_run_manifest(
        dataset_name=raw_dataset.dataset_name,
        schema_version=raw_dataset.schema_version,
        schema_hash=raw_dataset.schema_hash,
        split_hash=partitions.split_hash,
        adapter=adapter,
        seed=config.seed,
        subsample_size=getattr(config, "subsample", None),
        missing_counts={
            "train": train_report.missing_counts,
            "validation": validation_report.missing_counts,
            "test": test_report.missing_counts,
        },
        unseen_category_counts={
            "train": train_report.unseen_category_counts,
            "validation": validation_report.unseen_category_counts,
            "test": test_report.unseen_category_counts,
        },
    )
    return PreparedFinalData(
        train=train,
        validation=validation,
        test=test,
        adapter=adapter,
        target_encoder=target_encoder,
        manifest=manifest,
    )


def _daf_dataset(partition):
    return DAFDataset(
        partition.inputs["x_numerical"],
        partition.inputs["x_categorical_idx"],
        partition.inputs["x_categorical_meta"],
        y=partition.targets,
        x_numerical_missing=partition.inputs["x_numerical_missing"],
        row_ids=partition.row_ids,
    )


def _neural_dataset(partition, model_name):
    if model_name.lower().startswith("daf_moe_v2"):
        return _daf_dataset(partition)
    return Phase2TensorDataset(partition.inputs, partition.targets, partition.row_ids)


def _make_loaders(config, prepared, include_test):
    model_name = config.model_name.lower()
    if "frame" in prepared.train.inputs:
        raise ValueError(f"{model_name} uses a native runner, not PyTorch DataLoaders.")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        _neural_dataset(prepared.train, model_name),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        _neural_dataset(prepared.validation, model_name),
        batch_size=config.batch_size,
        shuffle=False,
    )
    if not include_test:
        return train_loader, validation_loader
    test_loader = DataLoader(
        _neural_dataset(prepared.test, model_name),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def get_phase2_hpo_dataloaders(config, raw_dataset):
    prepared = prepare_phase2_hpo(raw_dataset, config)
    config.phase2_manifest = prepared.manifest
    config.target_encoder = prepared.target_encoder
    return _make_loaders(config, prepared, include_test=False)


def get_phase2_dataloaders(config, raw_dataset):
    prepared = prepare_phase2_final(raw_dataset, config)
    config.phase2_manifest = prepared.manifest
    config.target_encoder = prepared.target_encoder
    return _make_loaders(config, prepared, include_test=True)
