"""Leakage-safe raw-frame contracts for native Phase 2 estimators."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from src.data.adapters import MISSING_TOKEN, UNKNOWN_TOKEN
from src.data.phase2_loader import prepare_phase2_final, prepare_phase2_hpo
from src.data.provenance import stable_hash


NATIVE_MODELS = frozenset({"xgboost", "catboost", "realmlp", "tabicl"})
TABICL_COVERTYPE_CONTEXT_SIZE = 400_000


@dataclass(frozen=True)
class NativePartition:
    frame: pd.DataFrame
    targets: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class NativeHPOData:
    train: NativePartition
    validation: NativePartition
    frame_adapter: object
    target_encoder: object
    manifest: dict


@dataclass(frozen=True)
class NativeFinalData:
    train: NativePartition
    validation: NativePartition
    test: NativePartition
    frame_adapter: object
    target_encoder: object
    manifest: dict


@dataclass(frozen=True)
class TabICLContext:
    context_frame: pd.DataFrame
    context_targets: np.ndarray
    context_row_ids: np.ndarray
    query_frame: pd.DataFrame
    query_targets: np.ndarray
    query_row_ids: np.ndarray
    context_index_hash: str
    subsample_size: Optional[int]


class NativeFrameAdapter:
    version = "native-frame-v1"

    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = tuple(numerical_columns)
        self.categorical_columns = tuple(categorical_columns)

    def fit(self, frame):
        self.columns = tuple(frame.columns)
        self.train_categories = {}
        for column in self.categorical_columns:
            values = _categorical_strings(frame[column])
            self.train_categories[column] = tuple(sorted(values.unique().tolist()))
        return self

    @property
    def state_hash(self):
        return stable_hash(self.state_dict())

    def state_dict(self):
        return {
            "version": self.version,
            "columns": self.columns,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "train_categories": self.train_categories,
        }

    def transform(self, frame):
        return frame.loc[:, self.columns].copy()


class XGBoostFrameAdapter(NativeFrameAdapter):
    """Frozen pandas categories; unseen values become native missing code -1."""

    version = "xgboost-pandas-category-native-missing-v1"

    def transform(self, frame):
        output = super().transform(frame)
        for column in self.categorical_columns:
            values = _categorical_strings(output[column])
            dtype = pd.CategoricalDtype(
                categories=list(self.train_categories[column]), ordered=False
            )
            output[column] = values.astype(dtype)
        return output

    def state_dict(self):
        return {
            **super().state_dict(),
            "unseen_rule": "pandas category code -1; XGBoost native missing",
            "missing_rule": MISSING_TOKEN,
        }


class CatBoostFrameAdapter(NativeFrameAdapter):
    """Raw categorical strings and native numerical NaN for CatBoost."""

    version = "catboost-raw-string-native-nan-v1"

    @property
    def cat_features(self):
        return list(self.categorical_columns)

    def transform(self, frame):
        output = super().transform(frame)
        for column in self.categorical_columns:
            output[column] = _categorical_strings(output[column])
        return output


class RealMLPFrameAdapter(NativeFrameAdapter):
    """Train-median numeric imputation required by pytabkit's sklearn API.

    Pytabkit owns every subsequent RealMLP transformation. Its official
    interface explicitly rejects continuous NaNs and documents numerical
    imputation as the sole manual preprocessing requirement.
    """

    version = "pytabkit-realmlp-train-median-v2"

    def fit(self, frame):
        super().fit(frame)
        self.numeric_medians = {}
        for column in self.numerical_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = values.median(skipna=True)
            self.numeric_medians[column] = (
                float(median) if np.isfinite(median) else 0.0
            )
        return self

    @property
    def cat_col_names(self):
        return list(self.categorical_columns)

    def transform(self, frame):
        output = super().transform(frame)
        for column in self.numerical_columns:
            values = pd.to_numeric(output[column], errors="coerce")
            output[column] = values.fillna(self.numeric_medians[column])
        return output

    def state_dict(self):
        return {
            **super().state_dict(),
            "numeric_imputation": "train_median",
            "numeric_medians": self.numeric_medians,
            "downstream_preprocessing_owner": "pytabkit",
        }


class TabICLFrameAdapter(NativeFrameAdapter):
    """Raw pandas input with train-frozen category dtype and explicit unknown."""

    version = "tabiclv2-pandas-category-v1"

    def transform(self, frame):
        output = super().transform(frame)
        for column in self.categorical_columns:
            values = _categorical_strings(output[column])
            known = set(self.train_categories[column])
            values = values.where(values.isin(known), UNKNOWN_TOKEN)
            categories = list(self.train_categories[column])
            if UNKNOWN_TOKEN not in categories:
                categories.append(UNKNOWN_TOKEN)
            dtype = pd.CategoricalDtype(categories=categories, ordered=False)
            output[column] = values.astype(dtype)
        return output

    def state_dict(self):
        return {
            **super().state_dict(),
            "unseen_rule": UNKNOWN_TOKEN,
            "missing_rule": MISSING_TOKEN,
            "unknown_in_train_vocabulary": False,
        }


FRAME_ADAPTERS = {
    "xgboost": XGBoostFrameAdapter,
    "catboost": CatBoostFrameAdapter,
    "realmlp": RealMLPFrameAdapter,
    "tabicl": TabICLFrameAdapter,
}


def _categorical_strings(series):
    return series.where(series.notna(), MISSING_TOKEN).astype(str)


def _prepare_config(model_name, task_type, seed, regression_target_policy):
    normalized = model_name.lower()
    if normalized not in NATIVE_MODELS:
        raise ValueError(f"{model_name} is not a Phase 2 native model.")
    return SimpleNamespace(
        model_name=normalized,
        task_type=task_type,
        seed=int(seed),
        regression_target_policy=regression_target_policy,
        ple_n_bins=48,
        subsample=None,
        out_dim=None,
    )


def _partition(prepared_partition, adapter):
    return NativePartition(
        frame=adapter.transform(prepared_partition.inputs["frame"]),
        targets=np.asarray(prepared_partition.targets),
        row_ids=np.asarray(prepared_partition.row_ids, dtype=np.int64),
    )


def _manifest(base_manifest, model_name, adapter):
    manifest = dict(base_manifest)
    manifest.update(
        {
            "model_name": model_name,
            "native_frame_adapter_class": adapter.__class__.__name__,
            "native_frame_adapter_version": adapter.version,
            "native_frame_state_hash": adapter.state_hash,
        }
    )
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def prepare_native_hpo(
    raw_dataset,
    model_name,
    task_type,
    seed=42,
    regression_target_policy="identity",
):
    """Construct train/validation only. The result has no test attribute."""
    config = _prepare_config(model_name, task_type, seed, regression_target_policy)
    prepared = prepare_phase2_hpo(raw_dataset, config)
    adapter = FRAME_ADAPTERS[config.model_name](
        raw_dataset.numerical_columns, raw_dataset.categorical_columns
    ).fit(prepared.train.inputs["frame"])
    return NativeHPOData(
        train=_partition(prepared.train, adapter),
        validation=_partition(prepared.validation, adapter),
        frame_adapter=adapter,
        target_encoder=prepared.target_encoder,
        manifest=_manifest(prepared.manifest, config.model_name, adapter),
    )


def prepare_native_final(
    raw_dataset,
    model_name,
    task_type,
    seed,
    regression_target_policy="identity",
):
    config = _prepare_config(model_name, task_type, seed, regression_target_policy)
    prepared = prepare_phase2_final(raw_dataset, config)
    adapter = FRAME_ADAPTERS[config.model_name](
        raw_dataset.numerical_columns, raw_dataset.categorical_columns
    ).fit(prepared.train.inputs["frame"])
    return NativeFinalData(
        train=_partition(prepared.train, adapter),
        validation=_partition(prepared.validation, adapter),
        test=_partition(prepared.test, adapter),
        frame_adapter=adapter,
        target_encoder=prepared.target_encoder,
        manifest=_manifest(prepared.manifest, config.model_name, adapter),
    )


def build_tabicl_context(data, dataset_name, seed):
    """Build final train+validation context and keep test rows query-only."""
    context_frame = pd.concat(
        [data.train.frame, data.validation.frame], axis=0, ignore_index=True
    )
    context_targets = np.concatenate([data.train.targets, data.validation.targets])
    context_row_ids = np.concatenate([data.train.row_ids, data.validation.row_ids])
    subsample_size = None
    normalized_name = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized_name == "covertype" and len(context_frame) > TABICL_COVERTYPE_CONTEXT_SIZE:
        selected = np.random.default_rng(int(seed)).choice(
            len(context_frame), size=TABICL_COVERTYPE_CONTEXT_SIZE, replace=False
        )
        selected.sort()
        context_frame = context_frame.iloc[selected].reset_index(drop=True)
        context_targets = context_targets[selected]
        context_row_ids = context_row_ids[selected]
        subsample_size = TABICL_COVERTYPE_CONTEXT_SIZE
    return TabICLContext(
        context_frame=context_frame,
        context_targets=context_targets,
        context_row_ids=context_row_ids,
        query_frame=data.test.frame,
        query_targets=data.test.targets,
        query_row_ids=data.test.row_ids,
        context_index_hash=stable_hash(context_row_ids),
        subsample_size=subsample_size,
    )
