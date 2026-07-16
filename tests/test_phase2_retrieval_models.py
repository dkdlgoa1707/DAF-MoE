import logging
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.configs.default_config import DAFConfig
from src.data.adapters import ModernNCAAdapter, TabRAdapter, get_adapter
from src.data.phase2_dataset import Phase2TensorDataset
from src.models.baselines.modernnca import ModernNCAWrapper
from src.models.baselines.retrieval import (
    CandidateStore,
    OneHotWithUnknownIgnored,
    streaming_topk,
)
from src.models.baselines.tabr import TabRWrapper
from src.models.factory import create_model
from src.trainer import Trainer
from src.utils.metrics import Evaluator


def retrieval_config(model_name, task_type="regression"):
    return DAFConfig(
        model_name=model_name,
        task_type=task_type,
        out_dim=1 if task_type == "regression" else 2,
        n_numerical=1,
        n_categorical=0,
        n_features=1,
        cat_cardinalities=[],
        cat_train_cardinalities=[],
        cat_known_cardinalities=[],
        tabr_n_candidates=96,
        tabr_d_main=8,
        tabr_d_multiplier=2.0,
        tabr_encoder_n_blocks=0,
        tabr_predictor_n_blocks=1,
        tabr_context_dropout=0.0,
        tabr_dropout0=0.0,
        tabr_dropout1=0.0,
        nca_dim=8,
        nca_d_block=16,
        nca_n_blocks=0,
        nca_sample_rate=0.5,
        nca_temperature=1.0,
        plr_n_frequencies=3,
        plr_embedding_dim=4,
        plr_frequency_scale=0.1,
        retrieval_candidate_chunk_size=2,
        dropout=0.0,
    )


def model_inputs(values, row_ids):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return {
        "x_numerical_values": values,
        "x_numerical_missing": torch.zeros_like(values),
        "x_categorical_idx": torch.empty(len(values), 0, dtype=torch.long),
        "row_ids": torch.as_tensor(row_ids, dtype=torch.long),
    }


class RetrievalAdapterTests(unittest.TestCase):
    def test_tabr_quantile_and_modernnca_zscore_are_distinct(self):
        frame = pd.DataFrame(
            {
                "num": [1.0, 2.0, np.nan, 4.0, 8.0] * 8,
                "cat": ["a", "b", None, "a", "b"] * 8,
            }
        )
        tabr = get_adapter("tabr", ("num",), ("cat",), seed=42).fit(frame)
        nca = get_adapter("modernnca", ("num",), ("cat",), seed=42).fit(frame)
        self.assertIsInstance(tabr, TabRAdapter)
        self.assertIsInstance(nca, ModernNCAAdapter)
        self.assertIsNotNone(tabr.quantile_transformer)
        self.assertAlmostEqual(nca.numeric_scales["num"], np.std([1, 2, 4, 8]))

        transformed = nca.transform(
            pd.DataFrame({"num": [2.0, np.nan], "cat": ["new", None]})
        )
        unknown_id = transformed.inputs["x_categorical_idx"][0, 0]
        missing_id = transformed.inputs["x_categorical_idx"][1, 0]
        self.assertNotEqual(unknown_id, missing_id)
        encoder = OneHotWithUnknownIgnored(nca.train_categorical_cardinalities)
        encoded = encoder(torch.as_tensor(transformed.inputs["x_categorical_idx"]))
        self.assertEqual(encoded[0].sum().item(), 0.0)
        self.assertEqual(encoded[1].sum().item(), 1.0)


class CandidateStoreTests(unittest.TestCase):
    def test_streaming_topk_matches_bruteforce_and_uses_row_ids(self):
        inputs = model_inputs([0.0, 0.0, 1.0, 2.0], [10, 11, 12, 13])
        store = CandidateStore(inputs, torch.arange(4.0))
        query = torch.tensor([[0.0]])
        distances, indices = streaming_topk(
            query,
            torch.tensor([10]),
            store,
            lambda chunk: chunk["x_numerical_values"],
            k=2,
            chunk_size=2,
            squared=True,
        )
        brute = torch.cdist(query, inputs["x_numerical_values"]).square()
        brute[0, 0] = torch.inf
        expected_distances, expected_indices = brute.topk(2, largest=False)
        torch.testing.assert_close(distances, expected_distances)
        torch.testing.assert_close(indices, expected_indices)
        self.assertEqual(indices[0, 0].item(), 1)

        _, heldout_indices = streaming_topk(
            query,
            torch.tensor([999]),
            store,
            lambda chunk: chunk["x_numerical_values"],
            k=2,
            chunk_size=2,
            squared=True,
        )
        self.assertEqual(set(heldout_indices[0].tolist()), {0, 1})

    def test_large_store_remains_cpu_backed(self):
        size = 50_000
        inputs = model_inputs(torch.linspace(-1, 1, size), torch.arange(size))
        store = CandidateStore(inputs, torch.zeros(size))
        self.assertTrue(all(value.device.type == "cpu" for value in store.inputs.values()))
        self.assertEqual(store.targets.device.type, "cpu")
        self.assertEqual(store.row_ids.device.type, "cpu")
        self.assertEqual(store.provenance["candidate_count"], size)


class TabRBehaviorTests(unittest.TestCase):
    def test_indices_weights_self_exclusion_and_duplicate_preservation(self):
        torch.manual_seed(7)
        model = TabRWrapper(retrieval_config("tabr")).eval()
        candidates = model_inputs([0.0, 0.0, 1.0, 2.0], [10, 11, 12, 13])
        model.set_candidates(candidates, torch.tensor([0.0, 1.0, 2.0, 3.0]))
        query = model_inputs([0.0], [10])
        with torch.no_grad():
            query_key = model._encode(query)[1]
            all_keys = model._encode(
                {key: value for key, value in candidates.items() if key != "row_ids"}
            )[1]
            brute = torch.cdist(query_key, all_keys).square()
            brute[0, 0] = torch.inf
            expected_indices = brute.topk(3, largest=False).indices
            output = model(**query)
        history = output["history"]
        torch.testing.assert_close(history["retrieval_indices"], expected_indices)
        self.assertNotIn(0, history["retrieval_indices"][0].tolist())
        self.assertIn(1, history["retrieval_indices"][0].tolist())
        self.assertAlmostEqual(history["retrieval_weights"].sum().item(), 1.0)

        heldout = model_inputs([0.0], [999])
        with torch.no_grad():
            heldout_history = model(**heldout)["history"]
        self.assertIn(0, heldout_history["retrieval_indices"][0].tolist())
        self.assertIn(1, heldout_history["retrieval_indices"][0].tolist())

    def test_regression_and_classification_label_encoders(self):
        regression = TabRWrapper(retrieval_config("tabr", "regression"))
        classification = TabRWrapper(retrieval_config("tabr", "classification"))
        self.assertIsInstance(regression.label_encoder, nn.Linear)
        self.assertIsInstance(classification.label_encoder, nn.Embedding)
        labels = torch.tensor([0, 1])
        self.assertEqual(classification._label_embeddings(labels).shape, (2, 8))


class ModernNCABehaviorTests(unittest.TestCase):
    def test_topk_self_exclusion_and_class_probabilities(self):
        torch.manual_seed(11)
        config = retrieval_config("modernnca", "classification")
        config.nca_n_neighbors = 3
        model = ModernNCAWrapper(config).eval()
        candidates = model_inputs([0.0, 0.0, 1.0, 2.0], [10, 11, 12, 13])
        model.set_train_context(candidates, torch.tensor([0, 1, 0, 1]))
        with torch.no_grad():
            output = model(**model_inputs([0.0], [10]))
        indices = output["history"]["retrieval_indices"][0].tolist()
        self.assertNotIn(0, indices)
        self.assertIn(1, indices)
        probabilities = output["logits"].exp()
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))

    def test_uniform_sns_changes_actual_candidate_count(self):
        torch.manual_seed(13)
        config = retrieval_config("modernnca", "regression")
        config.nca_n_neighbors = -1
        config.nca_sample_rate = 0.5
        model = ModernNCAWrapper(config).train()
        candidates = model_inputs(torch.arange(10.0), torch.arange(10))
        model.set_train_context(candidates, torch.arange(10.0))
        output = model(**model_inputs([0.5], [999]))
        self.assertEqual(model.last_sampled_candidate_count, 5)
        self.assertEqual(output["history"]["sampled_candidate_count"], 5)
        self.assertEqual(output["logits"].shape, (1, 1))

    def test_tabr_and_modernnca_backward_paths(self):
        candidates = model_inputs([0.0, 0.0, 1.0, 2.0], [10, 11, 12, 13])
        query = model_inputs([0.0, 1.0], [10, 12])
        labels = torch.tensor([0.0, 1.0, 2.0, 3.0])
        for model_name in ("tabr", "modernnca"):
            with self.subTest(model=model_name):
                config = retrieval_config(model_name, "regression")
                config.nca_sample_rate = 1.0
                model = create_model(config).train()
                if model_name == "tabr":
                    model.set_candidates(candidates, labels)
                else:
                    model.set_train_context(candidates, labels)
                output = model(**query)
                output["logits"].square().mean().backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_two_logit_binary_metrics_use_positive_class_probability(self):
        targets = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor(
            [[3.0, -3.0], [2.0, -2.0], [-2.0, 2.0], [-3.0, 3.0]]
        )
        metrics = Evaluator("classification")(targets, logits)
        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["auprc"], 1.0)

class TrainerCandidateBoundaryTests(unittest.TestCase):
    def test_trainer_wires_cpu_train_only_store_and_manifest(self):
        class StoreSpy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.devices = None
                self.rows = None

            def set_candidates(self, inputs, targets):
                self.devices = {value.device.type for value in inputs.values()}
                self.rows = inputs["row_ids"].clone()

            def candidate_provenance(self):
                return {"candidate_count": len(self.rows), "storage_device": "cpu"}

            def forward(self, x_numerical_values, **kwargs):
                return {"logits": x_numerical_values[:, :1] * 0 + self.weight}

        inputs = {
            "x_numerical_values": np.arange(6, dtype=np.float32).reshape(-1, 1),
            "x_numerical_missing": np.zeros((6, 1), dtype=np.float32),
            "x_categorical_idx": np.zeros((6, 0), dtype=np.int64),
        }
        dataset = Phase2TensorDataset(inputs, np.arange(6), np.arange(100, 106))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        config = DAFConfig(model_name="store_spy", task_type="regression")
        config.phase2_manifest = {"protocol_version": "test", "manifest_hash": "old"}
        model = StoreSpy()
        trainer = Trainer(
            model,
            nn.MSELoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            config,
            torch.device("cpu"),
            logging.getLogger("candidate-boundary-test"),
            verbose=False,
        )
        trainer._wire_retrieval_context(loader)
        self.assertEqual(model.devices, {"cpu"})
        torch.testing.assert_close(model.rows, torch.arange(100, 106))
        self.assertEqual(
            config.phase2_manifest["retrieval_candidates"]["candidate_count"], 6
        )
        self.assertNotEqual(config.phase2_manifest["manifest_hash"], "old")


if __name__ == "__main__":
    unittest.main()
