"""Full TabR with CPU-backed exact retrieval.

Architecture reference: yandex-research/tabular-dl-tabr, mirrored by TALENT
commit 08301d670a7c854bcf3a73298763484ba58eecdb.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieval import (
    CandidateStore,
    OneHotWithUnknownIgnored,
    PLREmbeddingsLite,
    streaming_topk,
)


def _make_block(d_main, d_block, dropout0, dropout1, prenorm):
    modules = []
    if prenorm:
        modules.append(nn.LayerNorm(d_main))
    modules.extend(
        [
            nn.Linear(d_main, d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main),
            nn.Dropout(dropout1),
        ]
    )
    return nn.Sequential(*modules)


class TabRWrapper(nn.Module):
    """ICLR 2024 full TabR, including encoder, retrieval mixer, and predictor."""

    upstream_reference = "TALENT@08301d670a7c854bcf3a73298763484ba58eecdb"

    def __init__(self, config):
        super().__init__()
        self.n_numerical = int(config.n_numerical)
        self.out_dim = int(config.out_dim)
        self.is_regression = config.task_type == "regression"
        self.context_size = int(config.tabr_n_candidates)
        if self.context_size != 96:
            raise ValueError("Phase 2 full TabR fixes context size at 96.")
        self.candidate_chunk_size = int(config.retrieval_candidate_chunk_size)
        if self.candidate_chunk_size <= 0:
            raise ValueError("retrieval_candidate_chunk_size must be positive.")

        d_main = int(config.tabr_d_main)
        d_block = int(d_main * float(config.tabr_d_multiplier))
        encoder_n_blocks = int(config.tabr_encoder_n_blocks)
        predictor_n_blocks = int(config.tabr_predictor_n_blocks)
        dropout0 = float(config.tabr_dropout0)
        dropout1 = float(config.tabr_dropout1)
        if encoder_n_blocks not in {0, 1}:
            raise ValueError("TabR encoder_n_blocks must be 0 or 1.")
        if predictor_n_blocks not in {1, 2}:
            raise ValueError("TabR predictor_n_blocks must be 1 or 2.")

        self.num_embeddings = (
            PLREmbeddingsLite(
                self.n_numerical,
                int(config.plr_n_frequencies),
                float(config.plr_frequency_scale),
                int(config.plr_embedding_dim),
            )
            if self.n_numerical
            else None
        )
        train_cardinalities = list(
            getattr(config, "cat_train_cardinalities", None) or []
        )
        if len(train_cardinalities) != config.n_categorical:
            raise ValueError("TabR requires train-fitted categorical cardinalities.")
        self.one_hot_encoder = OneHotWithUnknownIgnored(train_cardinalities)
        d_in = (
            self.n_numerical * int(config.plr_embedding_dim)
            + self.n_numerical
            + self.one_hot_encoder.output_dim
        )
        if d_in <= 0:
            raise ValueError("TabR requires at least one input feature.")

        self.linear = nn.Linear(d_in, d_main)
        self.encoder_blocks = nn.ModuleList(
            [
                _make_block(
                    d_main,
                    d_block,
                    dropout0,
                    dropout1,
                    prenorm=index > 0,
                )
                for index in range(encoder_n_blocks)
            ]
        )
        self.mixer_normalization = nn.LayerNorm(d_main) if encoder_n_blocks else None
        self.key_projection = nn.Linear(d_main, d_main)
        self.label_encoder = (
            nn.Linear(1, d_main)
            if self.is_regression
            else nn.Embedding(self.out_dim, d_main)
        )
        self.value_transform = nn.Sequential(
            nn.Linear(d_main, d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.context_dropout = nn.Dropout(float(config.tabr_context_dropout))
        self.predictor_blocks = nn.ModuleList(
            [
                _make_block(d_main, d_block, dropout0, dropout1, prenorm=True)
                for _ in range(predictor_n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_main),
            nn.ReLU(),
            nn.Linear(d_main, self.out_dim),
        )
        self.candidate_store = None
        self._reset_label_encoder()

    def _reset_label_encoder(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1.0 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            nn.init.uniform_(self.label_encoder.weight, -1.0, 1.0)

    def set_candidates(self, inputs, targets):
        self.candidate_store = CandidateStore(inputs, targets)

    def clear_candidates(self):
        self.candidate_store = None

    def candidate_provenance(self):
        return None if self.candidate_store is None else self.candidate_store.provenance

    def _feature_vector(self, inputs):
        values = inputs["x_numerical_values"]
        parts = []
        if self.num_embeddings is not None:
            parts.append(self.num_embeddings(values).flatten(1))
            parts.append(inputs["x_numerical_missing"].float())
        if self.one_hot_encoder.train_cardinalities:
            parts.append(self.one_hot_encoder(inputs["x_categorical_idx"].long()))
        return torch.cat(parts, dim=1).float()

    def _encode(self, inputs):
        x = self.linear(self._feature_vector(inputs))
        for block in self.encoder_blocks:
            x = x + block(x)
        key_input = x if self.mixer_normalization is None else self.mixer_normalization(x)
        return x, self.key_projection(key_input)

    def _label_embeddings(self, labels):
        if self.is_regression:
            return self.label_encoder(labels.float().unsqueeze(-1))
        return self.label_encoder(labels.long())

    def _retrieve(self, query_key, query_row_ids):
        if self.candidate_store is None:
            raise RuntimeError("TabR training candidates have not been set.")
        width = min(self.context_size, len(self.candidate_store))
        distances, indices = streaming_topk(
            query_key,
            query_row_ids,
            self.candidate_store,
            lambda chunk: self._encode(chunk)[1],
            width,
            self.candidate_chunk_size,
            squared=True,
        )
        selected_inputs, selected_labels, _ = self.candidate_store.gather(
            indices, query_key.device
        )
        flat_inputs = {
            name: value.flatten(0, 1) for name, value in selected_inputs.items()
        }
        context_key = self._encode(flat_inputs)[1].reshape(
            query_key.shape[0], indices.shape[1], -1
        )
        similarities = (
            -query_key.square().sum(-1, keepdim=True)
            + 2.0
            * (query_key[:, None, :] * context_key).sum(-1)
            - context_key.square().sum(-1)
        )
        weights = F.softmax(similarities, dim=-1)
        return indices, weights, context_key, selected_labels

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
        row_ids=None,
        **kwargs,
    ):
        inputs = {
            "x_numerical_values": x_numerical_values,
            "x_numerical_missing": x_numerical_missing,
            "x_categorical_idx": x_categorical_idx,
        }
        x, query_key = self._encode(inputs)
        history = None
        if self.candidate_store is not None:
            indices, weights, context_key, labels = self._retrieve(query_key, row_ids)
            dropped_weights = self.context_dropout(weights)
            values = self._label_embeddings(labels) + self.value_transform(
                query_key[:, None, :] - context_key
            )
            x = x + torch.sum(dropped_weights.unsqueeze(-1) * values, dim=1)
            history = {
                "retrieval_indices": indices.detach(),
                "retrieval_weights": weights.detach(),
            }

        for block in self.predictor_blocks:
            x = x + block(x)
        return {
            "logits": self.head(x),
            "aux_loss": None,
            "history": history,
            "psi_x": None,
        }
