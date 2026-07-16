"""Shared canonical components for Phase 2 neural baselines."""

import math

import torch
import torch.nn as nn


def resolve_width(config):
    width = config.d_token if config.d_token is not None else config.d_emb
    if width is None or int(width) <= 0:
        raise ValueError("Model width must be a positive integer.")
    return int(width)


def categorical_cardinalities(config):
    cardinalities = list(getattr(config, "cat_cardinalities", None) or [])
    if len(cardinalities) != config.n_categorical:
        if config.n_categorical:
            raise ValueError(
                "cat_cardinalities must be fitted by the model-specific adapter "
                "before model construction."
            )
        return []
    if any(int(cardinality) <= 0 for cardinality in cardinalities):
        raise ValueError("Categorical cardinalities must be positive.")
    return [int(cardinality) for cardinality in cardinalities]


class FeaturewiseCategoricalEmbeddings(nn.Module):
    """A distinct learned embedding table for every categorical feature."""

    def __init__(self, cardinalities, d_embedding):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        for embedding in self.embeddings:
            nn.init.kaiming_uniform_(embedding.weight, a=math.sqrt(5))

    @property
    def output_dim(self):
        if not self.embeddings:
            return 0
        return len(self.embeddings) * self.embeddings[0].embedding_dim

    def forward(self, values):
        if not self.embeddings:
            return values.new_empty(values.shape[0], 0, dtype=torch.float32)
        return torch.cat(
            [embedding(values[:, index]) for index, embedding in enumerate(self.embeddings)],
            dim=-1,
        )


class DenseFeatureInput(nn.Module):
    """Numerical scalars, explicit missing scalars, and categorical embeddings."""

    def __init__(self, config):
        super().__init__()
        self.n_numerical = int(config.n_numerical)
        embedding_dim = int(getattr(config, "cat_embedding_dim", 16))
        if embedding_dim <= 0:
            raise ValueError("cat_embedding_dim must be positive.")
        cardinalities = categorical_cardinalities(config)
        self.categorical = FeaturewiseCategoricalEmbeddings(
            cardinalities, embedding_dim
        )
        self.output_dim = 2 * self.n_numerical + self.categorical.output_dim

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
    ):
        parts = []
        if self.n_numerical:
            parts.extend([x_numerical_values, x_numerical_missing])
        if self.categorical.embeddings:
            parts.append(self.categorical(x_categorical_idx.long()))
        if not parts:
            raise ValueError("At least one input feature is required.")
        return torch.cat(parts, dim=-1)
