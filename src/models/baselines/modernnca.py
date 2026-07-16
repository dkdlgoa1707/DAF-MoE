"""Canonical ModernNCA with CPU-backed stochastic neighbor sampling.

Architecture reference: TALENT commit
08301d670a7c854bcf3a73298763484ba58eecdb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieval import (
    CandidateStore,
    OneHotWithUnknownIgnored,
    PLREmbeddingsLite,
    streaming_topk,
)


class _NCABlock(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, x):
        return self.block(x)


class ModernNCAWrapper(nn.Module):
    """ICLR 2025 ModernNCA with Euclidean soft-neighbor prediction."""

    upstream_reference = "TALENT@08301d670a7c854bcf3a73298763484ba58eecdb"

    def __init__(self, config):
        super().__init__()
        self.n_numerical = int(config.n_numerical)
        self.out_dim = int(config.out_dim)
        self.is_regression = config.task_type == "regression"
        self.temperature = float(config.nca_temperature)
        if self.temperature != 1.0:
            raise ValueError("Phase 2 ModernNCA fixes temperature=1.0.")
        self.sample_rate = float(config.nca_sample_rate)
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError("ModernNCA sample_rate must be in (0, 1].")
        self.n_neighbors = int(config.nca_n_neighbors)
        self.candidate_chunk_size = int(config.retrieval_candidate_chunk_size)

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
            raise ValueError("ModernNCA requires train-fitted categorical cardinalities.")
        self.one_hot_encoder = OneHotWithUnknownIgnored(train_cardinalities)
        d_in = (
            self.n_numerical * int(config.plr_embedding_dim)
            + self.n_numerical
            + self.one_hot_encoder.output_dim
        )
        if d_in <= 0:
            raise ValueError("ModernNCA requires at least one input feature.")

        dim = int(config.nca_dim)
        n_blocks = int(config.nca_n_blocks)
        if n_blocks < 0:
            raise ValueError("ModernNCA n_blocks cannot be negative.")
        self.encoder = nn.Linear(d_in, dim)
        self.post_blocks = nn.ModuleList(
            [
                _NCABlock(dim, int(config.nca_d_block), float(config.dropout))
                for _ in range(n_blocks)
            ]
        )
        self.final_normalization = nn.BatchNorm1d(dim) if n_blocks else None
        self.candidate_store = None
        self.last_sampled_candidate_count = 0

    def set_train_context(self, inputs, targets):
        self.candidate_store = CandidateStore(inputs, targets)

    def clear_train_context(self):
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

    def _embed(self, inputs):
        x = self.encoder(self._feature_vector(inputs))
        for block in self.post_blocks:
            x = block(x)
        if self.final_normalization is not None:
            x = self.final_normalization(x)
        return x

    def _sample_candidate_indices(self):
        if self.candidate_store is None:
            raise RuntimeError("ModernNCA training context has not been set.")
        n_candidates = len(self.candidate_store)
        if not self.training:
            indices = torch.arange(n_candidates)
        else:
            sample_size = max(1, int(n_candidates * self.sample_rate))
            indices = torch.randperm(n_candidates)[:sample_size]
        self.last_sampled_candidate_count = len(indices)
        return indices

    def _target_values(self, labels, dtype):
        if self.is_regression:
            return labels.float().unsqueeze(-1)
        return F.one_hot(labels.long(), self.out_dim).to(dtype)

    def _topk_prediction(self, query, query_row_ids, candidate_indices):
        distances, indices = streaming_topk(
            query,
            query_row_ids,
            self.candidate_store,
            self._embed,
            min(self.n_neighbors, len(candidate_indices)),
            self.candidate_chunk_size,
            candidate_indices=candidate_indices,
            squared=False,
        )
        selected_inputs, labels, _ = self.candidate_store.gather(indices, query.device)
        flat_inputs = {
            name: value.flatten(0, 1) for name, value in selected_inputs.items()
        }
        keys = self._embed(flat_inputs).reshape(query.shape[0], indices.shape[1], -1)
        distances = torch.linalg.vector_norm(query[:, None, :] - keys, dim=-1)
        weights = F.softmax(-distances / self.temperature, dim=-1)
        values = self._target_values(labels, query.dtype)
        prediction = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        return prediction, {
            "retrieval_indices": indices.detach(),
            "retrieval_weights": weights.detach(),
        }

    def _full_prediction(self, query, query_row_ids, candidate_indices):
        device = query.device
        query_ids = None if query_row_ids is None else query_row_ids.to(device).long()
        running_max = torch.full((len(query),), -torch.inf, device=device)
        denominator = torch.zeros(len(query), device=device)
        numerator = torch.zeros(len(query), self.out_dim, device=device)

        for _, cpu_inputs, cpu_labels, cpu_row_ids in self.candidate_store.iter_chunks(
            self.candidate_chunk_size, candidate_indices
        ):
            inputs = {name: value.to(device) for name, value in cpu_inputs.items()}
            keys = self._embed(inputs)
            scores = -torch.cdist(query, keys, p=2) / self.temperature
            if query_ids is not None:
                scores = scores.masked_fill(
                    query_ids[:, None] == cpu_row_ids.to(device)[None, :],
                    -torch.inf,
                )
            chunk_max = scores.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - new_max),
                torch.zeros_like(running_max),
            )
            weights = torch.where(
                torch.isfinite(new_max[:, None]),
                torch.exp(scores - new_max[:, None]),
                torch.zeros_like(scores),
            )
            values = self._target_values(cpu_labels.to(device), query.dtype)
            denominator = denominator * old_scale + weights.sum(dim=1)
            numerator = numerator * old_scale.unsqueeze(-1) + weights @ values
            running_max = new_max

        if (denominator <= 0).any():
            raise ValueError("No valid ModernNCA candidate remains after row-ID exclusion.")
        return numerator / denominator.unsqueeze(-1), {
            "sampled_candidate_count": self.last_sampled_candidate_count,
        }

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
        row_ids=None,
        **kwargs,
    ):
        if self.candidate_store is None:
            raise RuntimeError("ModernNCA training context has not been set.")
        inputs = {
            "x_numerical_values": x_numerical_values,
            "x_numerical_missing": x_numerical_missing,
            "x_categorical_idx": x_categorical_idx,
        }
        query = self._embed(inputs)
        candidate_indices = self._sample_candidate_indices()
        if self.n_neighbors > 0:
            prediction, history = self._topk_prediction(
                query, row_ids, candidate_indices
            )
        else:
            prediction, history = self._full_prediction(
                query, row_ids, candidate_indices
            )

        logits = (
            prediction
            if self.is_regression
            else prediction.clamp_min(1e-7).log()
        )
        return {
            "logits": logits,
            "aux_loss": None,
            "history": history,
            "psi_x": None,
        }
