import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.configs.default_config import DAFConfig
from src.data.adapters import MISSING_TOKEN, UNKNOWN_TOKEN
from src.data.splits import RawDataset
from src.models.factory import create_model
from src.native.data import (
    CatBoostFrameAdapter,
    RealMLPFrameAdapter,
    TabICLFrameAdapter,
    XGBoostFrameAdapter,
    build_tabicl_context,
    prepare_native_final,
    prepare_native_hpo,
)
from src.native.dependencies import (
    DEPENDENCIES,
    DependencyCompatibilityError,
    dependency_report,
    require_dependency,
)
from src.native.estimators import create_tabicl


def _raw_dataset(n_rows=60, dataset_name="mixed"):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "num": np.linspace(-2.0, 2.0, n_rows),
            "num_nan": rng.normal(size=n_rows),
            "cat": np.resize(np.array(["a", "b", None], dtype=object), n_rows),
        }
    )
    frame.loc[[1, 9, 23], "num_nan"] = np.nan
    target = pd.Series(np.arange(n_rows) % 2, name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num", "num_nan"),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name=dataset_name,
        schema_version="test-v1",
    )


class NativeDataContractTests(unittest.TestCase):
    def test_hpo_has_no_test_and_preserves_numeric_nan(self):
        raw = _raw_dataset()
        data = prepare_native_hpo(raw, "xgboost", "classification", seed=42)
        self.assertFalse(hasattr(data, "test"))
        train_ids = data.train.row_ids
        expected = raw.features.iloc[train_ids]["num_nan"].isna().to_numpy()
        actual = data.train.frame["num_nan"].isna().to_numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertIsInstance(data.train.frame["cat"].dtype, pd.CategoricalDtype)

    def test_xgboost_missing_and_unseen_are_distinct(self):
        train = pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", None]})
        query = pd.DataFrame({"num": [np.nan, 2.0], "cat": [None, "new"]})
        adapter = XGBoostFrameAdapter(("num",), ("cat",)).fit(train)
        transformed = adapter.transform(query)
        self.assertEqual(transformed["cat"].iloc[0], MISSING_TOKEN)
        self.assertTrue(pd.isna(transformed["cat"].iloc[1]))
        self.assertTrue(np.isnan(transformed["num"].iloc[0]))
        self.assertNotIn(UNKNOWN_TOKEN, adapter.train_categories["cat"])

    def test_catboost_uses_strings_explicit_features_and_native_numeric_nan(self):
        train = pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", None]})
        query = pd.DataFrame({"num": [np.nan], "cat": ["new"]})
        adapter = CatBoostFrameAdapter(("num",), ("cat",)).fit(train)
        transformed = adapter.transform(query)
        self.assertEqual(adapter.cat_features, ["cat"])
        self.assertEqual(transformed["cat"].iloc[0], "new")
        self.assertTrue(np.isnan(transformed["num"].iloc[0]))
        self.assertEqual(adapter.transform(train)["cat"].iloc[1], MISSING_TOKEN)

    def test_realmlp_only_imputes_numeric_from_train_median(self):
        train = pd.DataFrame({"num": [1.0, np.nan, 5.0], "cat": [None, "a", "b"]})
        adapter = RealMLPFrameAdapter(("num",), ("cat",)).fit(train)
        query = pd.DataFrame({"num": [np.nan, 100.0], "cat": ["new", None]})
        transformed = adapter.transform(query)
        np.testing.assert_allclose(transformed["num"], [3.0, 100.0])
        self.assertEqual(transformed["cat"].iloc[0], "new")
        self.assertIsNone(transformed["cat"].iloc[1])
        self.assertEqual(adapter.cat_col_names, ["cat"])
        self.assertEqual(adapter.state_dict()["numeric_imputation"], "train_median")

    def test_tabicl_context_is_train_plus_validation_and_test_is_query_only(self):
        raw = _raw_dataset()
        data = prepare_native_final(raw, "tabicl", "classification", seed=43)
        context = build_tabicl_context(data, raw.dataset_name, seed=43)
        expected_ids = np.concatenate([data.train.row_ids, data.validation.row_ids])
        np.testing.assert_array_equal(context.context_row_ids, expected_ids)
        self.assertTrue(set(context.context_row_ids).isdisjoint(context.query_row_ids))
        self.assertEqual(
            len(context.context_frame), len(data.train.frame) + len(data.validation.frame)
        )
        self.assertIsInstance(context.context_frame["cat"].dtype, pd.CategoricalDtype)

        adapter = TabICLFrameAdapter(("num",), ("cat",)).fit(
            pd.DataFrame({"num": [1.0], "cat": ["a"]})
        )
        query = adapter.transform(pd.DataFrame({"num": [2.0], "cat": ["new"]}))
        self.assertEqual(query["cat"].iloc[0], UNKNOWN_TOKEN)
        self.assertNotIn(UNKNOWN_TOKEN, adapter.train_categories["cat"])

    def test_covertype_context_subsample_is_deterministic_and_hashed(self):
        raw = _raw_dataset(n_rows=80, dataset_name="Covertype")
        data = prepare_native_final(raw, "tabicl", "classification", seed=43)
        with patch("src.native.data.TABICL_COVERTYPE_CONTEXT_SIZE", 17):
            first = build_tabicl_context(data, raw.dataset_name, seed=43)
            second = build_tabicl_context(data, raw.dataset_name, seed=43)
            other = build_tabicl_context(data, raw.dataset_name, seed=44)
        self.assertEqual(len(first.context_frame), 17)
        self.assertEqual(first.subsample_size, 17)
        np.testing.assert_array_equal(first.context_row_ids, second.context_row_ids)
        self.assertEqual(first.context_index_hash, second.context_index_hash)
        self.assertFalse(np.array_equal(first.context_row_ids, other.context_row_ids))
        self.assertNotEqual(first.context_index_hash, other.context_index_hash)
        self.assertTrue(set(first.context_row_ids).isdisjoint(first.query_row_ids))

    def test_train_schema_is_unchanged_by_validation_perturbation(self):
        raw = _raw_dataset()
        first = prepare_native_hpo(raw, "xgboost", "classification", seed=42)
        changed_features = raw.features.copy()
        changed_features.loc[first.validation.row_ids, "cat"] = "validation-only"
        changed = RawDataset(
            features=changed_features,
            target=raw.target,
            numerical_columns=raw.numerical_columns,
            categorical_columns=raw.categorical_columns,
            target_column=raw.target_column,
            dataset_name=raw.dataset_name,
            schema_version=raw.schema_version,
        )
        second = prepare_native_hpo(changed, "xgboost", "classification", seed=42)
        self.assertEqual(first.frame_adapter.state_hash, second.frame_adapter.state_hash)
        self.assertEqual(
            list(first.train.frame["cat"].cat.categories),
            list(second.train.frame["cat"].cat.categories),
        )


class NativeDependencyTests(unittest.TestCase):
    def test_dependency_contract_is_exact_and_never_falls_back(self):
        self.assertEqual(DEPENDENCIES["realmlp"].version, "1.7.3")
        self.assertEqual(DEPENDENCIES["tabicl"].version, "2.1.1")
        for model_name in DEPENDENCIES:
            report = dependency_report(model_name)
            if not report["compatible"]:
                with self.assertRaises(DependencyCompatibilityError):
                    require_dependency(model_name)

    def test_official_models_cannot_fall_back_to_pytorch_factory(self):
        for model_name in ("xgboost", "catboost", "realmlp", "tabicl"):
            config = DAFConfig()
            config.model_name = model_name
            with self.assertRaisesRegex(ValueError, "run_phase2_native.py"):
                create_model(config)

    @unittest.skipUnless(importlib.util.find_spec("tabicl"), "tabicl==2.1.1 not installed")
    def test_tabicl_constructor_contract_when_installed(self):
        report = dependency_report("tabicl")
        if not report["compatible"]:
            self.skipTest(report["install_command"])
        estimator, _, resolved = create_tabicl("classification", 43, device="cpu")
        self.assertEqual(resolved["n_estimators"], 8)
        self.assertEqual(
            resolved["checkpoint_version"], "tabicl-classifier-v2-20260212.ckpt"
        )
        self.assertEqual(estimator.n_estimators, 8)


if __name__ == "__main__":
    unittest.main()
