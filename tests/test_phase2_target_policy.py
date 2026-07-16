from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from src.data.phase2_loader import prepare_phase2_final, prepare_phase2_hpo
from src.data.splits import RawDataset, RawSplitRegistry, TrainOnlyTargetEncoder
from src.native.data import prepare_native_hpo
from src.phase2_execution import _evaluate_torch_model
from src.utils.metrics import Evaluator


def _raw(target=None):
    n_rows = 80
    index = np.arange(n_rows)
    return RawDataset(
        features=pd.DataFrame(
            {
                "num": index.astype(np.float64),
                "cat": np.resize(np.array(["a", "b"], dtype=object), n_rows),
            }
        ),
        target=pd.Series(
            index.astype(np.float64) * 10.0 if target is None else target,
            name="target",
        ),
        numerical_columns=("num",),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name="target-fixture",
        schema_version="fixture-v1",
    )


def _config(model_name="mlp"):
    return SimpleNamespace(
        model_name=model_name,
        task_type="regression",
        seed=42,
        ple_n_bins=8,
        subsample=None,
        out_dim=1,
    )


class _OffsetModel(torch.nn.Module):
    def forward(self, x):
        return {"logits": x + 1.0}


class TargetPolicyTests(unittest.TestCase):
    def test_population_standardization_inverse_and_original_unit_rmse(self):
        encoder = TrainOnlyTargetEncoder("regression", "standardize").fit(
            pd.Series([10.0, 20.0, 30.0])
        )
        self.assertAlmostEqual(encoder.mean, 20.0)
        self.assertAlmostEqual(encoder.std, np.std([10.0, 20.0, 30.0], ddof=0))
        encoded = encoder.transform(pd.Series([10.0, 30.0]))
        np.testing.assert_allclose(
            encoder.inverse_transform(encoded), [10.0, 30.0], rtol=1e-6
        )

        metrics = Evaluator("regression", target_transform=encoder)(
            torch.tensor(encoded), torch.tensor(encoded + 1.0)
        )
        self.assertAlmostEqual(metrics["rmse"], encoder.std, places=5)

        loader = [
            (
                {"x": torch.tensor(encoded).reshape(-1, 1)},
                torch.tensor(encoded).reshape(-1, 1),
            )
        ]
        config = SimpleNamespace(task_type="regression", target_encoder=encoder)
        objective_metrics = _evaluate_torch_model(
            _OffsetModel(), loader, config, torch.device("cpu")
        )
        self.assertAlmostEqual(objective_metrics["rmse"], encoder.std, places=5)

    def test_heldout_target_changes_do_not_change_train_fitted_state(self):
        raw = _raw()
        train_ids = set(
            RawSplitRegistry(raw, "regression", 42).indices.train.tolist()
        )
        perturbed_target = raw.target.copy()
        heldout = [index for index in range(len(raw.target)) if index not in train_ids]
        perturbed_target.iloc[heldout] += 1_000_000.0
        perturbed = _raw(perturbed_target)

        first = prepare_phase2_final(raw, _config())
        second = prepare_phase2_final(perturbed, _config())
        self.assertEqual(first.target_encoder.state_hash, second.target_encoder.state_hash)
        self.assertEqual(
            first.manifest["target_fitted_state_hash"],
            second.manifest["target_fitted_state_hash"],
        )
        self.assertEqual(first.manifest["target_policy"], "standardize")
        self.assertEqual(first.manifest["metric_scale"], "original_target_unit")
        self.assertFalse(first.manifest["target_std_fallback"])

    def test_native_regression_keeps_original_targets(self):
        raw = _raw()
        prepared = prepare_native_hpo(raw, "xgboost", "regression", seed=42)
        expected = raw.target.iloc[prepared.train.row_ids].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(prepared.train.targets, expected)
        self.assertEqual(prepared.manifest["target_policy"], "native")
        self.assertEqual(prepared.target_encoder.regression_policy, "identity")


if __name__ == "__main__":
    unittest.main()
