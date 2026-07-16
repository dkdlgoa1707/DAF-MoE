import logging
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.configs.default_config import DAFConfig
from src.data.adapters import RTDLQuantileAdapter, get_adapter
from src.data.phase2_dataset import Phase2TensorDataset
from src.models.baselines.config_validation import validate_model_config
from src.models.baselines.ft_transformer import FTTransformerWrapper
from src.models.baselines.tabm import (
    TabMOneHotEncoding,
    UpdatedPiecewiseLinearEmbedding,
    aggregate_member_predictions,
)
from src.models.factory import create_model
from src.trainer import Trainer, compute_baseline_loss
from train import load_config


def make_frame(size=90):
    index = np.arange(size)
    frame = pd.DataFrame(
        {
            "num_a": np.sin(index / 7.0) + index / 50.0,
            "num_b": np.cos(index / 5.0),
            "binary": np.where(index % 2, "yes", "no"),
            "multi": np.asarray(["a", "b", "c"])[index % 3],
        }
    )
    frame.loc[::13, "num_a"] = np.nan
    frame.loc[::17, "binary"] = None
    return frame


def make_prepared(model_name, task_type="regression"):
    frame = make_frame()
    adapter = get_adapter(
        model_name,
        ("num_a", "num_b"),
        ("binary", "multi"),
        seed=42,
        n_bins=4,
    ).fit(frame)
    config = DAFConfig(
        model_name=model_name,
        task_type=task_type,
        out_dim=1,
        n_layers=2,
        d_token=16,
        n_heads=8,
        d_ff_factor=1.5,
        d_hidden_factor=2.0,
        dropout=0.13,
        hidden_dropout=0.21,
        attention_dropout=0.17,
        ffn_dropout=0.23,
        residual_dropout=0.07,
        k=32,
        ple_n_bins=4,
        cat_embedding_dim=5,
    )
    adapter.apply_to_config(config)
    transformed = adapter.transform(frame.iloc[:8])
    inputs = {
        key: torch.as_tensor(
            value,
            dtype=torch.long if np.asarray(value).dtype.kind in "iu" else torch.float32,
        )
        for key, value in transformed.inputs.items()
    }
    return adapter, config, inputs


class AdapterBehaviorTests(unittest.TestCase):
    def test_rtdl_adapter_is_quantile_scalar_and_not_daf_metadata(self):
        adapter, config, inputs = make_prepared("mlp")
        self.assertIsInstance(adapter, RTDLQuantileAdapter)
        self.assertEqual(
            set(inputs),
            {"x_numerical_values", "x_numerical_missing", "x_categorical_idx"},
        )
        self.assertTrue(torch.isfinite(inputs["x_numerical_values"]).all())
        self.assertEqual(inputs["x_numerical_values"].ndim, 2)
        self.assertEqual(config.cat_cardinalities, [4, 5])
        self.assertEqual(inputs["x_numerical_missing"][0, 0].item(), 1.0)

    def test_all_nonretrieval_models_consume_missing_mask(self):
        for model_name in ("mlp", "resnet", "ft_transformer", "tabm", "tabm_ple"):
            with self.subTest(model=model_name):
                _, config, inputs = make_prepared(model_name)
                model = create_model(config).eval()
                observed = {key: value.clone() for key, value in inputs.items()}
                missing = {key: value.clone() for key, value in inputs.items()}
                observed["x_numerical_missing"].zero_()
                missing["x_numerical_missing"].fill_(1.0)
                with torch.no_grad():
                    first = model(**observed)["logits"]
                    second = model(**missing)["logits"]
                self.assertFalse(torch.allclose(first, second))


class CanonicalModelBehaviorTests(unittest.TestCase):
    def test_classification_and_regression_forward(self):
        for model_name in ("mlp", "resnet", "ft_transformer", "tabm", "tabm_ple"):
            for task_type in ("classification", "regression"):
                with self.subTest(model=model_name, task=task_type):
                    _, config, inputs = make_prepared(model_name, task_type)
                    model = create_model(config).eval()
                    with torch.no_grad():
                        outputs = model(**inputs)
                    self.assertEqual(outputs["logits"].shape, (8, 1))
                    self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_resnet_batchnorm_and_independent_dropouts(self):
        _, config, _ = make_prepared("resnet")
        model = create_model(config)
        block = model.blocks[0].layers
        self.assertIsInstance(block.normalization, nn.BatchNorm1d)
        self.assertAlmostEqual(block.hidden_dropout.p, 0.21)
        self.assertAlmostEqual(block.residual_dropout.p, 0.07)

    def test_ft_tokenizer_and_three_dropout_wirings(self):
        _, config, _ = make_prepared("ft_transformer")
        model = create_model(config)
        self.assertIsInstance(model, FTTransformerWrapper)
        self.assertIsNotNone(model.numerical_tokenizer.missing_weight)
        block = model.blocks[0]
        self.assertAlmostEqual(block.attention.dropout.p, 0.17)
        self.assertAlmostEqual(block.ffn[2].p, 0.23)
        self.assertAlmostEqual(block.attention_residual_dropout.p, 0.07)
        self.assertAlmostEqual(block.ffn_residual_dropout.p, 0.07)
        self.assertIsNone(block.attention_normalization)
        self.assertIsInstance(model.blocks[1].attention_normalization, nn.LayerNorm)

    def test_tabm_plain_and_ple_are_separate_and_backbone_is_shared(self):
        _, plain_config, _ = make_prepared("tabm")
        _, ple_config, _ = make_prepared("tabm_ple")
        plain = create_model(plain_config)
        ple = create_model(ple_config)
        self.assertIsNone(plain.num_embedding)
        self.assertFalse(
            any(isinstance(module, UpdatedPiecewiseLinearEmbedding) for module in plain.modules())
        )
        self.assertIsInstance(ple.num_embedding, UpdatedPiecewiseLinearEmbedding)
        self.assertTrue(plain.rank_included)
        self.assertFalse(ple.rank_included)
        self.assertEqual(plain.blocks[0][0].weight.ndim, 2)
        self.assertEqual(plain.output.weight.ndim, 3)
        self.assertEqual(plain.affine.weight.shape[0], 32)
        self.assertNotEqual(set(plain.state_dict()), set(ple.state_dict()))

    def test_tabm_feature_chunk_initialization(self):
        _, plain_config, _ = make_prepared("tabm")
        _, ple_config, _ = make_prepared("tabm_ple")
        plain = create_model(plain_config)
        ple = create_model(ple_config)
        self.assertEqual(plain.affine.feature_chunks, (2, 2, 3, 5))
        self.assertEqual(ple.affine.feature_chunks, (17, 17, 3, 5))
        for model in (plain, ple):
            start = 0
            chunk_scalars = []
            for width in model.affine.feature_chunks:
                chunk = model.affine.weight[:, start : start + width]
                torch.testing.assert_close(chunk, chunk[:, :1].expand_as(chunk))
                chunk_scalars.append(chunk[:, 0])
                start += width
            self.assertTrue(
                any(
                    not torch.equal(chunk_scalars[0], other)
                    for other in chunk_scalars[1:]
                )
            )

    def test_tabm_binary_scalar_and_reserved_states(self):
        encoder = TabMOneHotEncoding([4], [2])
        values = torch.tensor([[1], [2], [0], [3]])
        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        torch.testing.assert_close(encoder(values), expected)

    def test_tabm_memberwise_loss_and_probability_aggregation(self):
        logits_k = torch.tensor([[[-4.0], [4.0]]])
        targets = torch.ones(1)
        aggregate = aggregate_member_predictions(logits_k, "classification")
        outputs = {"logits": aggregate, "logits_k": logits_k}
        criterion = nn.BCEWithLogitsLoss()
        memberwise = compute_baseline_loss(outputs, targets, criterion, out_dim=1)
        loss_of_mean = criterion(aggregate, targets.view_as(aggregate))
        expected = criterion(logits_k, torch.ones_like(logits_k))
        torch.testing.assert_close(memberwise, expected)
        self.assertFalse(torch.isclose(memberwise, loss_of_mean))
        torch.testing.assert_close(torch.sigmoid(aggregate), torch.tensor([[0.5]]))


class ConfigAndTrainingContractTests(unittest.TestCase):
    def test_bad_daf_fields_fail_fast_for_baseline(self):
        config, _ = load_config("configs/experiments/higgs_small_ft_transformer_best.yaml")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_model_config(config)

        plain = DAFConfig(model_name="tabm", ple_boundaries=[[0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Plain TabM"):
            validate_model_config(plain)

    def test_tiny_training_exercises_early_stopping(self):
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(1))

            def forward(self, x_numerical_values, **kwargs):
                logits = x_numerical_values[:, :1] * 0.0 + self.bias
                return {"logits": logits, "aux_loss": None}

        inputs = {
            "x_numerical_values": np.ones((12, 1), dtype=np.float32),
            "x_numerical_missing": np.zeros((12, 1), dtype=np.float32),
            "x_categorical_idx": np.zeros((12, 0), dtype=np.int64),
        }
        dataset = Phase2TensorDataset(inputs, np.ones(12), np.arange(12))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        config = DAFConfig(
            model_name="constant",
            task_type="regression",
            out_dim=1,
            epochs=10,
            patience=2,
            optimize_metric="rmse",
        )
        model = ConstantModel()
        trainer = Trainer(
            model,
            nn.MSELoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            config,
            torch.device("cpu"),
            logging.getLogger("phase2-early-stopping-test"),
            verbose=False,
        )
        trainer.fit(loader, loader)
        self.assertTrue(trainer.stopped_early)
        self.assertEqual(trainer.best_epoch, 0)
        self.assertEqual(trainer.epochs_completed, 3)


if __name__ == "__main__":
    unittest.main()
