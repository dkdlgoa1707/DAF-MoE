"""Model-specific preprocessing adapter registry for Phase 2."""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Type

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import QuantileTransformer

from .provenance import stable_hash


MISSING_TOKEN = "[MISSING]"
UNKNOWN_TOKEN = "[UNK]"


@dataclass(frozen=True)
class AdapterOutput:
    inputs: Mapping[str, object]
    missing_counts: Mapping[str, int]
    unseen_category_counts: Mapping[str, int]


class Phase2Adapter:
    version = "1"

    def fit(self, frame: pd.DataFrame):
        raise NotImplementedError

    def transform(self, frame: pd.DataFrame) -> AdapterOutput:
        raise NotImplementedError

    @property
    def state_hash(self):
        return stable_hash(self.state_dict())

    def state_dict(self):
        raise NotImplementedError


class _CategoricalVocabularyMixin:
    def _fit_categorical(self, frame, categorical_columns):
        self.categorical_columns = tuple(categorical_columns)
        self.category_states = {}
        offset = 0
        n_rows = max(len(frame), 1)
        for column in self.categorical_columns:
            values = frame[column].where(frame[column].notna(), MISSING_TOKEN).astype(str)
            known = sorted(
                value for value in values.unique().tolist() if value != MISSING_TOKEN
            )
            local_mapping = {MISSING_TOKEN: 0}
            local_mapping.update({value: index + 1 for index, value in enumerate(known)})
            unknown_local_id = len(local_mapping)
            frequencies = (values.value_counts(dropna=False) / n_rows).to_dict()
            cardinality = len(local_mapping)
            self.category_states[column] = {
                "offset": offset,
                "mapping": local_mapping,
                "unknown_local_id": unknown_local_id,
                "frequencies": frequencies,
                "cardinality": cardinality,
                "known_cardinality": len(known),
            }
            offset += cardinality + 1
        self.total_cats = offset

    @property
    def categorical_cardinalities(self):
        return [
            self.category_states[column]["cardinality"] + 1
            for column in self.categorical_columns
        ]

    @property
    def train_categorical_cardinalities(self):
        return [
            self.category_states[column]["cardinality"]
            for column in self.categorical_columns
        ]

    @property
    def known_categorical_cardinalities(self):
        return [
            self.category_states[column]["known_cardinality"]
            for column in self.categorical_columns
        ]

    def _transform_categorical_local(self, frame):
        n_rows = len(frame)
        ids = []
        unseen_counts = {}
        for column in self.categorical_columns:
            state = self.category_states[column]
            values = frame[column].where(frame[column].notna(), MISSING_TOKEN).astype(str)
            unseen = ~values.isin(state["mapping"])
            unseen_counts[column] = int(unseen.sum())
            local_ids = values.map(state["mapping"])
            ids.append(
                local_ids.fillna(state["unknown_local_id"]).to_numpy(np.int64)
            )
        if not ids:
            return np.zeros((n_rows, 0), dtype=np.int64), unseen_counts
        return np.stack(ids, axis=1), unseen_counts

    def _transform_categorical(self, frame):
        local_ids, unseen_counts = self._transform_categorical_local(frame)
        n_rows = len(frame)
        ids = []
        frequencies = []
        cardinalities = []
        denominator = max(getattr(self, "n_train_rows", 0), 2)
        for index, column in enumerate(self.categorical_columns):
            state = self.category_states[column]
            values = frame[column].where(frame[column].notna(), MISSING_TOKEN).astype(str)
            ids.append(local_ids[:, index] + state["offset"])
            frequencies.append(
                values.map(state["frequencies"]).fillna(0.0).to_numpy(np.float32)
            )
            log_cardinality = np.log(max(state["cardinality"], 1)) / np.log(denominator)
            cardinalities.append(np.full(n_rows, log_cardinality, dtype=np.float32))

        if not self.categorical_columns:
            return (
                np.zeros((n_rows, 0), dtype=np.int64),
                np.zeros((n_rows, 0, 2), dtype=np.float32),
                unseen_counts,
            )
        categorical_ids = np.stack(ids, axis=1)
        metadata = np.stack(
            [np.stack(frequencies, axis=1), np.stack(cardinalities, axis=1)], axis=-1
        )
        return categorical_ids, metadata.astype(np.float32), unseen_counts

    def _apply_common_config(self, config):
        config.n_numerical = len(self.numerical_columns)
        config.n_categorical = len(self.categorical_columns)
        config.n_features = config.n_numerical + config.n_categorical
        config.total_cats = self.total_cats
        config.cat_cardinalities = self.categorical_cardinalities
        config.cat_train_cardinalities = self.train_categorical_cardinalities
        config.cat_known_cardinalities = self.known_categorical_cardinalities


class DAFV2Adapter(Phase2Adapter, _CategoricalVocabularyMixin):
    """Train-only DAF v2 statistics in the standardized preservation scale."""

    version = "daf-v2-observed-stats-v1"

    def __init__(self, numerical_columns, categorical_columns, n_bins=48, **kwargs):
        self.numerical_columns = tuple(numerical_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.n_bins = int(n_bins)
        if self.n_bins < 1:
            raise ValueError("ple_n_bins must be at least one.")
        self.numeric_states = {}
        self.ple_boundaries = None
        self.n_train_rows = 0

    def fit(self, frame):
        self.n_train_rows = len(frame)
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        boundaries = []
        for column in self.numerical_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                mean, scale, feature_skew = 0.0, 1.0, 0.0
                observed_sorted = np.empty(0, dtype=np.float64)
                feature_boundaries = np.zeros(self.n_bins + 1, dtype=np.float64)
            else:
                mean = float(observed.mean())
                observed_std = float(observed.std(ddof=0))
                scale = observed_std if np.isfinite(observed_std) and observed_std > 0 else 1.0
                if observed.size >= 3 and observed_std > 0:
                    feature_skew = float(skew(observed, bias=False))
                    if not np.isfinite(feature_skew):
                        feature_skew = 0.0
                else:
                    feature_skew = 0.0
                observed_sorted = np.sort(observed)
                raw_boundaries = np.quantile(observed, quantiles)
                feature_boundaries = (raw_boundaries - mean) / scale
            self.numeric_states[column] = {
                "mean": mean,
                "scale": scale,
                "skew": feature_skew,
                "observed_sorted": observed_sorted,
                "observed_count": int(observed.size),
            }
            boundaries.append(feature_boundaries.astype(np.float32))

        self.ple_boundaries = (
            np.stack(boundaries, axis=0)
            if boundaries
            else np.zeros((0, self.n_bins + 1), dtype=np.float32)
        )
        self._fit_categorical(frame, self.categorical_columns)
        return self

    def apply_to_config(self, config):
        self._apply_common_config(config)
        config.ple_boundaries = self.ple_boundaries.tolist()
        return config

    def _transform_numerical(self, frame):
        n_rows = len(frame)
        channels = []
        masks = []
        missing_counts = {}
        for column in self.numerical_columns:
            state = self.numeric_states[column]
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            missing = ~np.isfinite(values)
            missing_counts[column] = int(missing.sum())
            filled = values.copy()
            filled[missing] = state["mean"]
            standardized = (filled - state["mean"]) / state["scale"]
            observed_sorted = state["observed_sorted"]
            if observed_sorted.size:
                empirical_cdf = np.searchsorted(
                    observed_sorted, filled, side="right"
                ).astype(np.float64) / observed_sorted.size
            else:
                empirical_cdf = np.full(n_rows, 0.5, dtype=np.float64)
            feature_skew = np.full(n_rows, state["skew"], dtype=np.float64)
            channels.append(np.stack([standardized, empirical_cdf, feature_skew], axis=-1))
            masks.append(missing.astype(np.float32))

        if not self.numerical_columns:
            return (
                np.zeros((n_rows, 0, 3), dtype=np.float32),
                np.zeros((n_rows, 0), dtype=np.float32),
                missing_counts,
            )
        return (
            np.stack(channels, axis=1).astype(np.float32),
            np.stack(masks, axis=1).astype(np.float32),
            missing_counts,
        )

    def transform(self, frame):
        numerical, numerical_missing, missing_counts = self._transform_numerical(frame)
        categorical, categorical_meta, unseen_counts = self._transform_categorical(frame)
        return AdapterOutput(
            inputs={
                "x_numerical": numerical,
                "x_numerical_missing": numerical_missing,
                "x_categorical_idx": categorical,
                "x_categorical_meta": categorical_meta,
            },
            missing_counts=missing_counts,
            unseen_category_counts=unseen_counts,
        )

    def state_dict(self):
        numeric = {
            column: {
                "mean": state["mean"],
                "scale": state["scale"],
                "skew": state["skew"],
                "observed_sorted": state["observed_sorted"],
                "observed_count": state["observed_count"],
            }
            for column, state in self.numeric_states.items()
        }
        return {
            "version": self.version,
            "n_bins": self.n_bins,
            "numeric": numeric,
            "ple_boundaries": self.ple_boundaries,
            "categorical": self.category_states,
            "total_cats": self.total_cats,
        }


class CustomNeuralAdapter(Phase2Adapter, _CategoricalVocabularyMixin):
    """Train-mean scalar inputs used by retrieval models in Prompt 3."""

    version = "custom-neural-mean-mask-v1"

    def __init__(self, numerical_columns, categorical_columns, **kwargs):
        self.numerical_columns = tuple(numerical_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.n_train_rows = 0
        self.numeric_means = {}

    def fit(self, frame):
        self.n_train_rows = len(frame)
        for column in self.numerical_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            mean = values.mean(skipna=True)
            self.numeric_means[column] = float(mean) if np.isfinite(mean) else 0.0
        self._fit_categorical(frame, self.categorical_columns)
        return self

    def apply_to_config(self, config):
        self._apply_common_config(config)
        return config

    def _mean_imputed(self, frame):
        values = []
        masks = []
        missing_counts = {}
        for column in self.numerical_columns:
            column_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                np.float64, copy=True
            )
            missing = ~np.isfinite(column_values)
            missing_counts[column] = int(missing.sum())
            column_values[missing] = self.numeric_means[column]
            values.append(column_values.astype(np.float32))
            masks.append(missing.astype(np.float32))
        n_rows = len(frame)
        numerical = np.stack(values, axis=1) if values else np.zeros((n_rows, 0), np.float32)
        numerical_missing = (
            np.stack(masks, axis=1) if masks else np.zeros((n_rows, 0), np.float32)
        )
        return numerical, numerical_missing, missing_counts

    def transform(self, frame):
        numerical, numerical_missing, missing_counts = self._mean_imputed(frame)
        categorical, unseen_counts = self._transform_categorical_local(frame)
        return AdapterOutput(
            inputs={
                "x_numerical_values": numerical,
                "x_numerical_missing": numerical_missing,
                "x_categorical_idx": categorical,
            },
            missing_counts=missing_counts,
            unseen_category_counts=unseen_counts,
        )

    def state_dict(self):
        return {
            "version": self.version,
            "numeric_means": self.numeric_means,
            "categorical": self.category_states,
            "total_cats": self.total_cats,
        }


class RTDLQuantileAdapter(CustomNeuralAdapter):
    """RTDL noisy quantile-normal preprocessing fitted on train only."""

    version = "rtdl-noisy-quantile-mask-v1"

    def __init__(self, numerical_columns, categorical_columns, seed=42, **kwargs):
        super().__init__(numerical_columns, categorical_columns)
        self.seed = int(seed)
        self.quantile_transformer = None

    def fit(self, frame):
        super().fit(frame)
        if self.numerical_columns:
            values, _, _ = self._mean_imputed(frame)
            noisy_values = values.astype(np.float64, copy=True)
            stds = np.std(noisy_values, axis=0, keepdims=True)
            noise = 1e-3
            noise_std = noise / np.maximum(stds, noise)
            noisy_values += noise_std * np.random.default_rng(self.seed).standard_normal(
                noisy_values.shape
            )
            n_quantiles = max(min(len(frame) // 30, 1000), 10)
            self.quantile_transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=1_000_000_000,
                random_state=self.seed,
            ).fit(noisy_values)
        return self

    def transform(self, frame):
        numerical, numerical_missing, missing_counts = self._mean_imputed(frame)
        if self.quantile_transformer is not None:
            numerical = self.quantile_transformer.transform(numerical).astype(np.float32)
        categorical, unseen_counts = self._transform_categorical_local(frame)
        return AdapterOutput(
            inputs={
                "x_numerical_values": numerical,
                "x_numerical_missing": numerical_missing,
                "x_categorical_idx": categorical,
            },
            missing_counts=missing_counts,
            unseen_category_counts=unseen_counts,
        )

    def state_dict(self):
        transformer_state = None
        if self.quantile_transformer is not None:
            transformer_state = {
                "quantiles": self.quantile_transformer.quantiles_,
                "references": self.quantile_transformer.references_,
                "n_quantiles": self.quantile_transformer.n_quantiles_,
            }
        return {
            **super().state_dict(),
            "version": self.version,
            "seed": self.seed,
            "quantile_transformer": transformer_state,
        }


class StandardizedNeuralAdapter(CustomNeuralAdapter):
    """Observed-train z-standardization for ModernNCA."""

    version = "train-zscore-mask-onehot-v1"

    def __init__(self, numerical_columns, categorical_columns, **kwargs):
        super().__init__(numerical_columns, categorical_columns)
        self.numeric_scales = {}

    def fit(self, frame):
        super().fit(frame)
        for column in self.numerical_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                np.float64, copy=True
            )
            observed = values[np.isfinite(values)]
            scale = float(observed.std(ddof=0)) if observed.size else 1.0
            self.numeric_scales[column] = (
                scale if np.isfinite(scale) and scale > 0.0 else 1.0
            )
        return self

    def transform(self, frame):
        numerical, numerical_missing, missing_counts = self._mean_imputed(frame)
        for index, column in enumerate(self.numerical_columns):
            numerical[:, index] = (
                numerical[:, index] - self.numeric_means[column]
            ) / self.numeric_scales[column]
        categorical, unseen_counts = self._transform_categorical_local(frame)
        return AdapterOutput(
            inputs={
                "x_numerical_values": numerical,
                "x_numerical_missing": numerical_missing,
                "x_categorical_idx": categorical,
            },
            missing_counts=missing_counts,
            unseen_category_counts=unseen_counts,
        )

    def state_dict(self):
        return {
            **super().state_dict(),
            "version": self.version,
            "numeric_scales": self.numeric_scales,
        }


class TabRAdapter(RTDLQuantileAdapter):
    """TabR noisy-quantile scalars; PLR and one-hot remain model intrinsic."""

    version = "tabr-noisy-quantile-plr-input-v1"


class ModernNCAAdapter(StandardizedNeuralAdapter):
    """ModernNCA z-scored scalars; PLR and one-hot remain model intrinsic."""

    version = "modernnca-zscore-plr-input-v1"


class TabMAdapter(RTDLQuantileAdapter):
    """TabM scalar/one-hot contract, with optional updated PLE boundaries."""

    version = "tabm-noisy-quantile-onehot-v1"

    def __init__(
        self,
        numerical_columns,
        categorical_columns,
        seed=42,
        n_bins=48,
        use_ple=False,
        **kwargs,
    ):
        super().__init__(numerical_columns, categorical_columns, seed=seed)
        self.n_bins = int(n_bins)
        self.use_ple = bool(use_ple)
        self.ple_boundaries = None

    def fit(self, frame):
        super().fit(frame)
        if self.use_ple:
            transformed = super().transform(frame).inputs["x_numerical_values"]
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            self.ple_boundaries = (
                np.quantile(transformed, quantiles, axis=0).T.astype(np.float32)
                if transformed.shape[1]
                else np.zeros((0, self.n_bins + 1), dtype=np.float32)
            )
        return self

    def apply_to_config(self, config):
        super().apply_to_config(config)
        if self.use_ple:
            config.ple_boundaries = self.ple_boundaries.tolist()
        else:
            config.ple_boundaries = None
        return config

    def state_dict(self):
        return {
            **super().state_dict(),
            "version": self.version,
            "use_ple": self.use_ple,
            "n_bins": self.n_bins,
            "ple_boundaries": self.ple_boundaries,
        }


class TabMPLEAdapter(TabMAdapter):
    version = "tabm-updated-ple-onehot-v1"

    def __init__(self, numerical_columns, categorical_columns, **kwargs):
        super().__init__(
            numerical_columns,
            categorical_columns,
            use_ple=True,
            **kwargs,
        )


class NativeRawAdapter(Phase2Adapter):
    """No numeric imputation/scaling; native estimators own preprocessing."""

    version = "native-raw-v1"

    def __init__(self, numerical_columns, categorical_columns, **kwargs):
        self.numerical_columns = tuple(numerical_columns)
        self.categorical_columns = tuple(categorical_columns)

    def fit(self, frame):
        self.columns = tuple(frame.columns)
        return self

    def transform(self, frame):
        output = frame.loc[:, self.columns].copy()
        return AdapterOutput(
            inputs={"frame": output},
            missing_counts={column: int(output[column].isna().sum()) for column in output},
            unseen_category_counts={},
        )

    def state_dict(self):
        return {"version": self.version, "columns": self.columns}


ADAPTER_REGISTRY: Dict[str, Type[Phase2Adapter]] = {}


def register_adapter(names: Sequence[str], adapter_class: Type[Phase2Adapter]):
    for name in names:
        ADAPTER_REGISTRY[name] = adapter_class


register_adapter(("daf_moe_v2",), DAFV2Adapter)
register_adapter(("mlp", "resnet", "ft_transformer"), RTDLQuantileAdapter)
register_adapter(("tabm",), TabMAdapter)
register_adapter(("tabm_ple",), TabMPLEAdapter)
register_adapter(("tabr",), TabRAdapter)
register_adapter(("modernnca",), ModernNCAAdapter)
register_adapter(("xgboost", "catboost", "realmlp", "tabicl"), NativeRawAdapter)


def get_adapter(model_name, numerical_columns, categorical_columns, **kwargs):
    normalized = model_name.lower()
    if normalized.startswith("daf_moe_v2"):
        normalized = "daf_moe_v2"
    try:
        adapter_class = ADAPTER_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"No Phase 2 preprocessing adapter registered for {model_name}") from exc
    return adapter_class(numerical_columns, categorical_columns, **kwargs)
