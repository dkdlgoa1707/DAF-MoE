import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.configs.default_config import DAFConfig
from src.data.adapters import DAFV2Adapter, MISSING_TOKEN, UNKNOWN_TOKEN
from src.data.loader import get_dataloaders
from src.data.phase2_loader import prepare_phase2_hpo
from src.data.provenance import build_run_manifest
from src.data.splits import (
    RawDataset,
    RawSplitRegistry,
    TrainOnlyTargetEncoder,
    create_split_indices,
)
from src.models.factory import create_model


def make_raw_dataset(size=100):
    frame = pd.DataFrame(
        {
            "num": np.linspace(-3.0, 3.0, size),
            "constant": np.full(size, 7.0),
            "all_missing": np.full(size, np.nan),
            "cat": np.where(np.arange(size) % 3 == 0, "a", "b"),
        }
    )
    frame.loc[::11, "num"] = np.nan
    frame.loc[::13, "cat"] = None
    target = pd.Series(np.arange(size) % 2, name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num", "constant", "all_missing"),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name="synthetic",
        schema_version="test-v1",
    )


def make_v2_config(seed=42):
    return DAFConfig(
        model_name="daf_moe_v2",
        seed=seed,
        task_type="classification",
        n_layers=1,
        d_emb=16,
        n_heads=4,
        n_experts=4,
        top_k=2,
        ple_n_bins=4,
        batch_size=16,
        out_dim=1,
    )


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_seeded(self):
        target = pd.Series(np.arange(200) % 4)
        first = create_split_indices(target, "classification", 42)
        second = create_split_indices(target, "classification", 42)
        other = create_split_indices(target, "classification", 43)
        np.testing.assert_array_equal(first.train, second.train)
        self.assertEqual(first.split_hash, second.split_hash)
        self.assertNotEqual(first.split_hash, other.split_hash)
        self.assertEqual(len(first.train), 160)
        self.assertEqual(len(first.validation), 20)
        self.assertEqual(len(first.test), 20)
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.validation) & set(first.test))

    def test_hpo_partition_has_no_test_surface(self):
        raw = make_raw_dataset()
        partitions = RawSplitRegistry(raw, "classification", 42).for_hpo()
        self.assertFalse(hasattr(partitions, "test"))
        prepared = prepare_phase2_hpo(raw, make_v2_config())
        self.assertFalse(hasattr(prepared, "test"))


class DAFV2AdapterTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "num": [1.0, 2.0, np.nan, 4.0],
                "constant": [5.0, 5.0, 5.0, np.nan],
                "all_missing": [np.nan, np.nan, np.nan, np.nan],
                "cat": ["a", None, "a", "b"],
            }
        )
        self.adapter = DAFV2Adapter(
            ("num", "constant", "all_missing"), ("cat",), n_bins=4
        ).fit(self.train)

    def test_observed_only_statistics_and_ple_scale(self):
        state = self.adapter.numeric_states
        self.assertAlmostEqual(state["num"]["mean"], 7.0 / 3.0)
        self.assertEqual(state["num"]["observed_count"], 3)
        self.assertEqual(state["constant"]["scale"], 1.0)
        self.assertEqual(state["constant"]["skew"], 0.0)
        self.assertEqual(state["all_missing"]["observed_count"], 0)
        self.assertEqual(state["all_missing"]["skew"], 0.0)
        self.assertEqual(self.adapter.ple_boundaries.shape, (3, 5))
        np.testing.assert_array_equal(self.adapter.ple_boundaries[1], np.zeros(5))
        np.testing.assert_array_equal(self.adapter.ple_boundaries[2], np.zeros(5))

        observed = np.asarray([1.0, 2.0, 4.0])
        mean = observed.mean()
        scale = observed.std(ddof=0)
        expected = (np.quantile(observed, np.linspace(0, 1, 5)) - mean) / scale
        np.testing.assert_allclose(self.adapter.ple_boundaries[0], expected, rtol=1e-6)

    def test_missing_and_unknown_are_distinct(self):
        transformed = self.adapter.transform(
            pd.DataFrame(
                {
                    "num": [np.nan, 2.0],
                    "constant": [5.0, 5.0],
                    "all_missing": [np.nan, np.nan],
                    "cat": [None, "never-seen"],
                }
            )
        )
        ids = transformed.inputs["x_categorical_idx"][:, 0]
        self.assertNotEqual(int(ids[0]), int(ids[1]))
        self.assertEqual(transformed.inputs["x_categorical_meta"][1, 0, 0], 0.0)
        self.assertEqual(transformed.unseen_category_counts["cat"], 1)
        mapping = self.adapter.category_states["cat"]["mapping"]
        self.assertIn(MISSING_TOKEN, mapping)
        self.assertNotIn(UNKNOWN_TOKEN, mapping)
        self.assertEqual(transformed.inputs["x_numerical_missing"][0, 0], 1.0)
        self.assertEqual(transformed.inputs["x_numerical"][0, 0, 0], 0.0)

    def test_transform_cannot_mutate_fitted_state(self):
        before = self.adapter.state_hash
        self.adapter.transform(
            pd.DataFrame(
                {
                    "num": [1e12],
                    "constant": [-1e12],
                    "all_missing": [9.0],
                    "cat": ["new-category"],
                }
            )
        )
        self.assertEqual(before, self.adapter.state_hash)
        self.assertEqual(self.adapter.total_cats, 4)


class LeakageAndManifestTests(unittest.TestCase):
    def test_validation_and_test_perturbations_do_not_change_fit(self):
        raw = make_raw_dataset()
        indices = create_split_indices(raw.target, "classification", 42)
        perturbed_features = raw.features.copy()
        held_out = np.concatenate([indices.validation, indices.test])
        perturbed_features.iloc[held_out, 0] = 1e9
        perturbed_features.iloc[held_out, 3] = "held-out-only"
        perturbed = RawDataset(
            features=perturbed_features,
            target=raw.target.copy(),
            numerical_columns=raw.numerical_columns,
            categorical_columns=raw.categorical_columns,
            target_column=raw.target_column,
            dataset_name=raw.dataset_name,
            schema_version=raw.schema_version,
        )

        first_config = make_v2_config()
        second_config = make_v2_config()
        first = prepare_phase2_hpo(raw, first_config)
        second = prepare_phase2_hpo(perturbed, second_config)
        self.assertEqual(first.adapter.state_hash, second.adapter.state_hash)
        self.assertEqual(first_config.total_cats, second_config.total_cats)
        self.assertEqual(first_config.out_dim, second_config.out_dim)
        self.assertNotEqual(
            first.validation.inputs["x_numerical"].tolist(),
            second.validation.inputs["x_numerical"].tolist(),
        )

    def test_manifest_hash_and_target_inverse_are_deterministic(self):
        raw = make_raw_dataset()
        config = make_v2_config()
        prepared = prepare_phase2_hpo(raw, config)
        kwargs = dict(
            dataset_name=raw.dataset_name,
            schema_version=raw.schema_version,
            schema_hash=raw.schema_hash,
            split_hash=prepared.manifest["split_index_hash"],
            adapter=prepared.adapter,
            target_encoder=prepared.target_encoder,
            target_policy=config.effective_target_policy,
            seed=42,
            git_sha="abc123",
        )
        first = build_run_manifest(**kwargs)
        second = build_run_manifest(**kwargs)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])

        encoder = TrainOnlyTargetEncoder("regression", "standardize").fit(
            pd.Series([10.0, 20.0, 30.0])
        )
        encoded = encoder.transform(pd.Series([15.0, 25.0]))
        np.testing.assert_allclose(encoder.inverse_transform(encoded), [15.0, 25.0])


class LoaderAndCompatibilityTests(unittest.TestCase):
    def test_v2_loader_injects_ple_and_missing_mask(self):
        raw = make_raw_dataset(size=60)
        frame = raw.features.copy()
        frame["target"] = raw.target
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            frame.to_csv(path, index=False)
            data_config = {
                "dataset_name": "synthetic",
                "csv_path": str(path),
                "target_col": "target",
                "num_cols": list(raw.numerical_columns),
                "cat_cols": list(raw.categorical_columns),
            }
            config = make_v2_config()
            train_loader, _, _ = get_dataloaders(config, data_config)
            self.assertEqual(np.asarray(config.ple_boundaries).shape, (3, 5))
            inputs, _ = next(iter(train_loader))
            self.assertIn("x_numerical_missing", inputs)
            model = create_model(config).eval()
            with torch.no_grad():
                output = model(**inputs)
            self.assertEqual(output["logits"].shape[0], len(inputs["x_numerical"]))

    def test_v1_and_v15_forward_contracts_remain_available(self):
        for model_name in ("daf_moe", "daf_moe_v15"):
            config = DAFConfig(
                model_name=model_name,
                n_numerical=2,
                n_categorical=1,
                n_features=3,
                total_cats=4,
                d_emb=16,
                n_heads=4,
                n_layers=1,
                n_experts=4,
                top_k=2,
                out_dim=1,
            )
            model = create_model(config).eval()
            with torch.no_grad():
                output = model(
                    torch.randn(3, 2, 3),
                    torch.randint(0, 4, (3, 1)),
                    torch.rand(3, 1, 2),
                )
            self.assertEqual(output["logits"].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
