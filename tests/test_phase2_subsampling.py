import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.configs.default_config import DAFConfig
from src.data.phase2_loader import prepare_phase2_final, prepare_phase2_hpo
from src.data.splits import RawDataset, RawSplitRegistry, create_split_indices
from src.hpo.engine import create_phase2_study, valid_complete_count
from src.hpo.identity import StudyIdentity, build_study_identity
from src.hpo.schema import load_search_space
from src.native.data import prepare_native_hpo


def _raw_dataset(n_rows, task_type="classification", dataset_name="fixture"):
    values = np.arange(n_rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "num": values,
            "cat": np.asarray([f"c{index % 7}" for index in range(n_rows)]),
        }
    )
    if task_type == "classification":
        target = pd.Series(np.arange(n_rows) % 2, name="target")
    else:
        target = pd.Series(values * 3.0 + 11.0, name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num",),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name=dataset_name,
        schema_version="fixture-v1",
    )


def _config(seed=42, task_type="classification"):
    return DAFConfig(
        model_name="daf_moe_v2",
        seed=seed,
        task_type=task_type,
        n_layers=1,
        d_emb=8,
        n_heads=2,
        n_experts=2,
        top_k=1,
        ple_n_bins=4,
        batch_size=16,
        out_dim=1,
    )


def _base(task_type="classification"):
    return {
        "data_config_path": "unused.yaml",
        "task_type": task_type,
        "optimize_metric": "acc" if task_type == "classification" else "rmse",
        "model_name": "mlp",
        "batch_size": 16,
    }


class TrainSubsamplePolicyTests(unittest.TestCase):
    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_train_only_size_and_held_out_partitions_remain_full(self):
        raw = _raw_dataset(250)
        full = create_split_indices(raw.target, "classification", 42)
        partitions = RawSplitRegistry(raw, "classification", 42).for_final()

        self.assertEqual(len(full.train), 200)
        self.assertEqual(len(partitions.train.row_ids), 100)
        self.assertEqual(len(partitions.validation.row_ids), len(full.validation))
        self.assertEqual(len(partitions.test.row_ids), len(full.test))
        np.testing.assert_array_equal(partitions.validation.row_ids, full.validation)
        np.testing.assert_array_equal(partitions.test.row_ids, full.test)
        self.assertTrue(set(partitions.train.row_ids).issubset(set(full.train)))
        self.assertTrue(set(partitions.train.row_ids).isdisjoint(full.validation))
        self.assertTrue(set(partitions.train.row_ids).isdisjoint(full.test))
        self.assertEqual(partitions.train_subsample["selected_train_size"], 100)
        self.assertEqual(partitions.train_subsample["full_train_size"], 200)
        self.assertEqual(partitions.train_subsample["seed"], 42)

    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_non_large_dataset_is_unchanged(self):
        raw = _raw_dataset(120)
        full = create_split_indices(raw.target, "classification", 42)
        registry = RawSplitRegistry(raw, "classification", 42)
        partitions = registry.for_final()

        np.testing.assert_array_equal(partitions.train.row_ids, full.train)
        np.testing.assert_array_equal(partitions.validation.row_ids, full.validation)
        np.testing.assert_array_equal(partitions.test.row_ids, full.test)
        self.assertIsNone(partitions.train_subsample)
        self.assertEqual(partitions.dataset_schema_hash, raw.schema_hash)
        self.assertEqual(partitions.dataset_schema_version, raw.schema_version)

    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_subsample_is_deterministic_and_varies_by_execution_seed(self):
        raw = _raw_dataset(250)
        first = RawSplitRegistry(raw, "classification", 42).for_final()
        second = RawSplitRegistry(raw, "classification", 42).for_final()
        other = RawSplitRegistry(raw, "classification", 43).for_final()

        np.testing.assert_array_equal(first.train.row_ids, second.train.row_ids)
        self.assertEqual(
            first.train_subsample["selected_row_ids_hash"],
            second.train_subsample["selected_row_ids_hash"],
        )
        self.assertFalse(np.array_equal(first.train.row_ids, other.train.row_ids))
        self.assertNotEqual(
            first.train_subsample["selected_row_ids_hash"],
            other.train_subsample["selected_row_ids_hash"],
        )

    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_preprocessing_and_regression_target_fit_after_subsampling(self):
        raw = _raw_dataset(250, task_type="regression")
        registry = RawSplitRegistry(raw, "regression", 42)
        selected = registry.indices.train
        excluded = np.setdiff1d(np.arange(len(raw.target)), selected)

        first = prepare_phase2_hpo(raw, _config(task_type="regression"))
        selected_features = raw.features.iloc[selected]
        selected_target = raw.target.iloc[selected]
        self.assertAlmostEqual(
            first.adapter.numeric_states["num"]["mean"],
            float(selected_features["num"].mean()),
        )
        self.assertAlmostEqual(first.target_encoder.mean, float(selected_target.mean()))

        changed_features = raw.features.copy()
        changed_target = raw.target.copy()
        changed_features.iloc[excluded, changed_features.columns.get_loc("num")] = 1e12
        changed_features.iloc[excluded, changed_features.columns.get_loc("cat")] = (
            "outside-selected-train"
        )
        changed_target.iloc[excluded] = -1e12
        changed = RawDataset(
            features=changed_features,
            target=changed_target,
            numerical_columns=raw.numerical_columns,
            categorical_columns=raw.categorical_columns,
            target_column=raw.target_column,
            dataset_name=raw.dataset_name,
            schema_version=raw.schema_version,
        )
        second = prepare_phase2_hpo(changed, _config(task_type="regression"))

        self.assertEqual(first.adapter.state_hash, second.adapter.state_hash)
        self.assertEqual(first.target_encoder.state_hash, second.target_encoder.state_hash)
        self.assertEqual(first.manifest["subsample_size"], 100)
        self.assertEqual(
            first.manifest["train_subsample"]["selected_row_ids_hash"],
            second.manifest["train_subsample"]["selected_row_ids_hash"],
        )

    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_neural_and_native_paths_receive_identical_train_rows(self):
        raw = _raw_dataset(250)
        neural = prepare_phase2_hpo(raw, _config())
        native = prepare_native_hpo(raw, "xgboost", "classification", seed=42)

        np.testing.assert_array_equal(neural.train.row_ids, native.train.row_ids)
        np.testing.assert_array_equal(
            neural.validation.row_ids, native.validation.row_ids
        )
        self.assertEqual(len(neural.train.row_ids), 100)
        self.assertEqual(native.manifest["subsample_size"], 100)


class TrainSubsampleStudyIdentityTests(unittest.TestCase):
    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_large_policy_is_signature_isolated_and_starts_empty(self):
        raw = _raw_dataset(250, dataset_name="large-fixture")
        space = load_search_space("configs/hpo/phase2/mlp.yaml")
        subsampled = build_study_identity(raw, _base(), space)
        legacy_components = dict(subsampled.components)
        legacy_components["dataset_schema_hash"] = raw.schema_hash
        legacy_components["dataset_schema_version"] = raw.schema_version
        legacy = StudyIdentity.from_components(legacy_components)

        self.assertNotEqual(legacy.signature, subsampled.signature)
        self.assertNotEqual(legacy.study_name, subsampled.study_name)
        self.assertIn("max=100", subsampled.components["dataset_schema_version"])
        self.assertIn(
            "seed=execution_seed",
            subsampled.components["dataset_schema_version"],
        )

        with tempfile.TemporaryDirectory() as directory:
            old_study = create_phase2_study(
                legacy, "maximize", legacy.default_storage_url(directory)
            )
            trial = old_study.ask()
            trial.set_user_attr("study_signature", legacy.signature)
            trial.set_user_attr("search_space_hash", space.schema_hash)
            old_study.tell(trial, 0.5)
            self.assertEqual(valid_complete_count(old_study, legacy), 1)

            new_study = create_phase2_study(
                subsampled,
                "maximize",
                subsampled.default_storage_url(directory),
            )
            self.assertEqual(valid_complete_count(new_study, subsampled), 0)

    @patch("src.data.splits.TRAIN_SUBSAMPLE_THRESHOLD", 100)
    def test_non_large_identity_components_remain_legacy_exact(self):
        raw = _raw_dataset(120, dataset_name="small-fixture")
        space = load_search_space("configs/hpo/phase2/mlp.yaml")
        identity = build_study_identity(raw, _base(), space)
        self.assertEqual(identity.components["dataset_schema_hash"], raw.schema_hash)
        self.assertEqual(
            identity.components["dataset_schema_version"], raw.schema_version
        )


if __name__ == "__main__":
    unittest.main()
