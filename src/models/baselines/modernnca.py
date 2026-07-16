"""ModernNCA with official uniform SNS and full evaluation semantics.

Reference: LAMDA-Tabular/TALENT commit
08301d670a7c854bcf3a73298763484ba58eecdb (MIT license).
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieval import CandidateStore, OneHotWithUnknownIgnored, PLREmbeddingsLite


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
        if self.n_neighbors > 0:
            raise ValueError("Official ModernNCA uses all sampled candidates, not top-k.")
        self.candidate_chunk_size = int(config.retrieval_candidate_chunk_size)
        if self.candidate_chunk_size <= 0:
            raise ValueError("retrieval_candidate_chunk_size must be positive.")

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
        self.last_effective_candidate_count = 0
        self.last_query_count = 0
        self.total_query_count = 0
        self.total_candidate_comparisons = 0
        self.total_prediction_seconds = 0.0
        self.measure_retrieval_performance = bool(
            getattr(config, "retrieval_measure_performance", False)
        )

    def set_train_context(self, inputs, targets):
        self.candidate_store = CandidateStore(
            inputs, targets, device=next(self.parameters()).device
        )

    def clear_train_context(self):
        self.candidate_store = None

    def candidate_provenance(self):
        if self.candidate_store is None:
            return None
        return {
            **self.candidate_store.provenance,
            "upstream_reference": self.upstream_reference,
            "sns_policy": "uniform_without_query_batch",
            "sample_rate": self.sample_rate,
            "training_candidate_policy": "sample_train_then_readd_query_batch",
            "evaluation_candidate_policy": "all_train_candidates",
            "evaluation_mode": "exact_streaming_softmax",
            "normalization_policy": "single_logical_candidate_batch",
            "last_sampled_candidate_count": self.last_sampled_candidate_count,
            "last_effective_candidate_count": self.last_effective_candidate_count,
            "last_query_count": self.last_query_count,
            "total_query_count": self.total_query_count,
            "total_candidate_comparisons": self.total_candidate_comparisons,
            "prediction_seconds": self.total_prediction_seconds,
            "queries_per_second": (
                self.total_query_count / self.total_prediction_seconds
                if self.total_prediction_seconds > 0
                else None
            ),
            "candidate_comparisons_per_second": (
                self.total_candidate_comparisons / self.total_prediction_seconds
                if self.total_prediction_seconds > 0
                else None
            ),
        }

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

    def _target_values(self, labels, dtype):
        if self.is_regression:
            return labels.float().unsqueeze(-1)
        return F.one_hot(labels.long(), self.out_dim).to(dtype)

    def _logical_candidates(self, query, query_row_ids):
        device = query.device
        if self.training:
            if query_row_ids is None:
                raise ValueError("ModernNCA training requires query row IDs.")
            query_positions = self.candidate_store.positions_for_row_ids(query_row_ids)
            keep = torch.ones(len(self.candidate_store), dtype=torch.bool)
            keep[query_positions] = False
            available = torch.arange(len(self.candidate_store))[keep]
            sample_size = int(len(available) * self.sample_rate)
            if len(available) and sample_size == 0:
                sample_size = 1
            sampled = available[torch.randperm(len(available))[:sample_size]]
            sampled_inputs, sampled_labels, sampled_row_ids = self.candidate_store.gather(
                sampled, device
            )
            sampled_keys = (
                self._embed(sampled_inputs)
                if len(sampled)
                else query.new_empty((0, query.shape[1]))
            )
            query_labels = self.candidate_store.targets[query_positions].to(device)
            candidate_keys = torch.cat([query, sampled_keys], dim=0)
            candidate_labels = torch.cat([query_labels, sampled_labels], dim=0)
            candidate_row_ids = torch.cat(
                [query_row_ids.to(device).long(), sampled_row_ids], dim=0
            )
            self.last_sampled_candidate_count = len(sampled)
        else:
            indices = torch.arange(len(self.candidate_store))
            candidate_inputs, candidate_labels, candidate_row_ids = (
                self.candidate_store.gather(indices, device)
            )
            candidate_keys = self._embed(candidate_inputs)
            self.last_sampled_candidate_count = len(indices)
        self.last_effective_candidate_count = len(candidate_keys)
        return candidate_keys, candidate_labels, candidate_row_ids

    def _full_prediction(
        self, query, query_row_ids, candidate_keys, candidate_labels, candidate_row_ids
    ):
        prediction_started = (
            time.perf_counter() if self.measure_retrieval_performance else None
        )
        device = query.device
        query_ids = None if query_row_ids is None else query_row_ids.to(device).long()
        running_max = torch.full((len(query),), -torch.inf, device=device)
        denominator = torch.zeros(len(query), device=device)
        numerator = torch.zeros(len(query), self.out_dim, device=device)
        target_values = self._target_values(candidate_labels, query.dtype)

        for start in range(0, len(candidate_keys), self.candidate_chunk_size):
            stop = min(start + self.candidate_chunk_size, len(candidate_keys))
            keys = candidate_keys[start:stop]
            scores = -torch.cdist(query, keys, p=2) / self.temperature
            if query_ids is not None:
                scores = scores.masked_fill(
                    query_ids[:, None] == candidate_row_ids[start:stop][None, :],
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
            denominator = denominator * old_scale + weights.sum(dim=1)
            numerator = (
                numerator * old_scale.unsqueeze(-1)
                + weights @ target_values[start:stop]
            )
            running_max = new_max

        if (denominator <= 0).any():
            raise ValueError("No valid ModernNCA candidate remains after self-exclusion.")
        self.last_query_count = len(query)
        if self.measure_retrieval_performance:
            if query.device.type == "cuda":
                torch.cuda.synchronize(query.device)
            self.total_prediction_seconds += (
                time.perf_counter() - prediction_started
            )
        self.total_query_count += len(query)
        self.total_candidate_comparisons += len(query) * len(candidate_keys)
        return numerator / denominator.unsqueeze(-1), {
            "sampled_candidate_count": self.last_sampled_candidate_count,
            "effective_candidate_count": self.last_effective_candidate_count,
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
        candidate_keys, candidate_labels, candidate_row_ids = (
            self._logical_candidates(query, row_ids)
        )
        prediction, history = self._full_prediction(
            query,
            row_ids,
            candidate_keys,
            candidate_labels,
            candidate_row_ids,
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
