"""Leakage-safe raw dataset loading and split-index registry for Phase 2."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


HPO_SEED = 42
FINAL_SEEDS = tuple(range(43, 58))
SPLIT_VERSION = "phase2-80-10-10-v1"
TRAIN_SUBSAMPLE_THRESHOLD = 100_000
TRAIN_SUBSAMPLE_POLICY_VERSION = "phase2-train-only-100k-v1"
TRAIN_SUBSAMPLE_SEED_POLICY = "execution_seed"


def _stable_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RawDataset:
    features: pd.DataFrame
    target: pd.Series
    numerical_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]
    target_column: str
    dataset_name: str
    schema_version: str

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "dataset_name": self.dataset_name,
                "schema_version": self.schema_version,
                "target": self.target_column,
                "numerical": list(self.numerical_columns),
                "categorical": list(self.categorical_columns),
                "dtypes": {name: str(dtype) for name, dtype in self.features.dtypes.items()},
            }
        )


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int
    version: str = SPLIT_VERSION

    @property
    def split_hash(self) -> str:
        return _stable_hash(
            {
                "version": self.version,
                "seed": self.seed,
                "train": self.train.tolist(),
                "validation": self.validation.tolist(),
                "test": self.test.tolist(),
            }
        )


@dataclass(frozen=True)
class RawPartition:
    features: pd.DataFrame
    target: pd.Series
    row_ids: np.ndarray


@dataclass(frozen=True)
class HPOPartitions:
    """Train/validation-only view. A test member intentionally does not exist."""

    train: RawPartition
    validation: RawPartition
    split_hash: str
    dataset_schema_hash: str
    dataset_schema_version: str
    train_subsample: Optional[Mapping]


@dataclass(frozen=True)
class FinalPartitions:
    train: RawPartition
    validation: RawPartition
    test: RawPartition
    split_hash: str
    dataset_schema_hash: str
    dataset_schema_version: str
    train_subsample: Optional[Mapping]


class TrainOnlyTargetEncoder:
    """Fit class mappings or regression transforms using the train target only."""

    version = "phase2-target-v2"

    def __init__(self, task_type: str, regression_policy: str = "identity"):
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if regression_policy not in {"identity", "standardize"}:
            raise ValueError(f"Unsupported regression policy: {regression_policy}")
        self.task_type = task_type
        self.regression_policy = regression_policy
        self.class_mapping = None
        self.mean = 0.0
        self.std = 1.0
        self.std_fallback = False

    def fit(self, target: pd.Series):
        if target.isna().any():
            raise ValueError("Target contains missing values.")
        if self.task_type == "classification":
            classes = sorted(target.unique().tolist(), key=lambda value: str(value))
            self.class_mapping = {value: index for index, value in enumerate(classes)}
        elif self.regression_policy == "standardize":
            values = target.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Regression target contains nonfinite values.")
            self.mean = float(values.mean())
            std = float(values.std(ddof=0))
            self.std_fallback = not (np.isfinite(std) and std > 0.0)
            self.std = 1.0 if self.std_fallback else std
        return self

    def state_dict(self):
        mapping = None
        if self.class_mapping is not None:
            mapping = sorted(
                ((str(key), int(value)) for key, value in self.class_mapping.items()),
                key=lambda item: item[1],
            )
        return {
            "version": self.version,
            "task_type": self.task_type,
            "regression_policy": self.regression_policy,
            "class_mapping": mapping,
            "mean": self.mean if self.task_type == "regression" else None,
            "std": self.std if self.task_type == "regression" else None,
            "std_fallback": self.std_fallback,
        }

    @property
    def state_hash(self):
        return _stable_hash(self.state_dict())

    def transform(self, target: pd.Series) -> np.ndarray:
        if self.task_type == "classification":
            mapped = target.map(self.class_mapping)
            if mapped.isna().any():
                unseen = target[mapped.isna()].unique().tolist()
                raise ValueError(f"Unseen target classes outside train split: {unseen}")
            return mapped.to_numpy(dtype=np.float32)
        values = target.to_numpy(dtype=np.float64)
        if self.regression_policy == "standardize":
            values = (values - self.mean) / self.std
        return values.astype(np.float32)

    def inverse_transform(self, values) -> np.ndarray:
        values = np.asarray(values)
        if self.task_type == "classification":
            inverse = {index: value for value, index in self.class_mapping.items()}
            return np.asarray([inverse[int(value)] for value in values.reshape(-1)])
        if self.regression_policy == "standardize":
            return values * self.std + self.mean
        return values


def load_raw_dataset(data_config: Mapping, csv_path: Optional[str] = None) -> RawDataset:
    path = Path(csv_path or data_config["csv_path"])
    frame = pd.read_csv(path, skipinitialspace=True)
    numerical = tuple(data_config.get("num_cols", []))
    categorical = tuple(data_config.get("cat_cols", []))
    target_column = data_config["target_col"]
    required = set(numerical + categorical + (target_column,))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing configured columns: {missing}")
    if len(set(numerical).intersection(categorical)):
        raise ValueError("Numerical and categorical column lists overlap.")

    return RawDataset(
        features=frame.loc[:, numerical + categorical].copy(),
        target=frame[target_column].copy(),
        numerical_columns=numerical,
        categorical_columns=categorical,
        target_column=target_column,
        dataset_name=data_config.get("dataset_name", path.stem),
        schema_version=str(data_config.get("schema_version", "1")),
    )


def _split_with_optional_stratification(indices, target, test_size, seed, stratify):
    stratify_values = target.iloc[indices] if stratify else None
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_values,
        )
    except ValueError:
        if not stratify:
            raise
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def create_split_indices(target: pd.Series, task_type: str, seed: int) -> SplitIndices:
    indices = np.arange(len(target), dtype=np.int64)
    stratify = task_type == "classification"
    train, temporary = _split_with_optional_stratification(
        indices, target, test_size=0.2, seed=seed, stratify=stratify
    )
    validation, test = _split_with_optional_stratification(
        np.asarray(temporary),
        target,
        test_size=0.5,
        seed=seed,
        stratify=stratify,
    )
    return SplitIndices(
        train=np.sort(np.asarray(train, dtype=np.int64)),
        validation=np.sort(np.asarray(validation, dtype=np.int64)),
        test=np.sort(np.asarray(test, dtype=np.int64)),
        seed=seed,
    )


class RawSplitRegistry:
    def __init__(self, dataset: RawDataset, task_type: str, seed: int):
        self.dataset = dataset
        self.task_type = task_type
        self.seed = int(seed)
        full_indices = create_split_indices(dataset.target, task_type, self.seed)
        self.full_train_size = len(full_indices.train)
        self.train_subsample_applied = (
            self.full_train_size > TRAIN_SUBSAMPLE_THRESHOLD
        )
        if self.train_subsample_applied:
            selected_train = np.random.default_rng(self.seed).choice(
                full_indices.train,
                size=TRAIN_SUBSAMPLE_THRESHOLD,
                replace=False,
            )
            selected_train = np.sort(np.asarray(selected_train, dtype=np.int64))
            self.indices = SplitIndices(
                train=selected_train,
                validation=full_indices.validation,
                test=full_indices.test,
                seed=self.seed,
                version=f"{SPLIT_VERSION}+{TRAIN_SUBSAMPLE_POLICY_VERSION}",
            )
            policy_identity = {
                "policy_version": TRAIN_SUBSAMPLE_POLICY_VERSION,
                "max_train_samples": TRAIN_SUBSAMPLE_THRESHOLD,
                "seed_policy": TRAIN_SUBSAMPLE_SEED_POLICY,
                "threshold_rule": "apply_when_full_train_exceeds_max",
            }
            self.dataset_schema_hash = _stable_hash(
                {
                    "base_dataset_schema_hash": dataset.schema_hash,
                    "train_subsample": policy_identity,
                }
            )
            self.dataset_schema_version = (
                f"{dataset.schema_version}+{TRAIN_SUBSAMPLE_POLICY_VERSION}:"
                f"max={TRAIN_SUBSAMPLE_THRESHOLD}:"
                f"seed={TRAIN_SUBSAMPLE_SEED_POLICY}"
            )
        else:
            self.indices = full_indices
            self.dataset_schema_hash = dataset.schema_hash
            self.dataset_schema_version = str(dataset.schema_version)

    def _train_subsample_manifest(self, include_test: bool):
        if not self.train_subsample_applied:
            return None
        manifest = {
            "applied": True,
            "policy_version": TRAIN_SUBSAMPLE_POLICY_VERSION,
            "max_train_samples": TRAIN_SUBSAMPLE_THRESHOLD,
            "threshold_rule": "apply_when_full_train_exceeds_max",
            "seed_policy": TRAIN_SUBSAMPLE_SEED_POLICY,
            "seed": self.seed,
            "full_train_size": self.full_train_size,
            "selected_train_size": len(self.indices.train),
            "selected_row_ids_hash": _stable_hash(self.indices.train.tolist()),
            "validation_size": len(self.indices.validation),
        }
        if include_test:
            manifest["test_size"] = len(self.indices.test)
        return manifest

    def _partition(self, indices) -> RawPartition:
        indices = np.asarray(indices, dtype=np.int64)
        return RawPartition(
            features=self.dataset.features.iloc[indices].copy(),
            target=self.dataset.target.iloc[indices].copy(),
            row_ids=indices.copy(),
        )

    def for_hpo(self) -> HPOPartitions:
        return HPOPartitions(
            train=self._partition(self.indices.train),
            validation=self._partition(self.indices.validation),
            split_hash=self.indices.split_hash,
            dataset_schema_hash=self.dataset_schema_hash,
            dataset_schema_version=self.dataset_schema_version,
            train_subsample=self._train_subsample_manifest(include_test=False),
        )

    def for_final(self) -> FinalPartitions:
        return FinalPartitions(
            train=self._partition(self.indices.train),
            validation=self._partition(self.indices.validation),
            test=self._partition(self.indices.test),
            split_hash=self.indices.split_hash,
            dataset_schema_hash=self.dataset_schema_hash,
            dataset_schema_version=self.dataset_schema_version,
            train_subsample=self._train_subsample_manifest(include_test=True),
        )
