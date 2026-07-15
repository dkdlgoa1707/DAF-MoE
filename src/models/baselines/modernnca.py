"""ModernNCA baseline adapted from the official TALENT implementation.

Reference: https://github.com/LAMDA-Tabular/TALENT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return x + self.block(x)


class ModernNCAWrapper(nn.Module):
    """Learn an embedding space and predict from soft nearest neighbours."""

    def __init__(self, config):
        super().__init__()
        self.d_emb = config.d_token if config.d_token is not None else config.d_emb
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        self.out_dim = config.out_dim
        self.is_regression = config.task_type == "regression"
        self.temperature = float(getattr(config, "nca_temperature", 1.0))
        self.n_neighbors = int(getattr(config, "nca_n_neighbors", -1))

        self.num_proj = nn.ModuleList(
            nn.Linear(1, self.d_emb) for _ in range(self.n_numerical)
        )
        self.cat_embed = (
            nn.Embedding(config.total_cats + 1, self.d_emb)
            if self.n_categorical
            else None
        )
        input_dim = (self.n_numerical + self.n_categorical) * self.d_emb
        self.encoder = nn.Linear(input_dim, self.d_emb)
        self.blocks = nn.ModuleList(
            _NCABlock(self.d_emb, 2 * self.d_emb, config.dropout)
            for _ in range(max(0, config.n_layers - 1))
        )
        self.final_norm = nn.BatchNorm1d(self.d_emb) if self.blocks else nn.Identity()
        self.head = nn.Linear(self.d_emb, self.out_dim)

        self.register_buffer("train_x_numerical", torch.empty(0), persistent=False)
        self.register_buffer(
            "train_x_categorical", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer("train_labels", torch.empty(0), persistent=False)

    def set_train_context(self, x_numerical, x_categorical_idx=None, train_y=None):
        """Set the training rows and labels used by the NCA prediction rule."""
        if isinstance(x_numerical, dict):
            inputs = x_numerical
            if train_y is None:
                train_y = x_categorical_idx
            x_numerical = inputs["x_numerical"]
            x_categorical_idx = inputs["x_categorical_idx"]
        if x_categorical_idx is None or train_y is None:
            raise ValueError(
                "ModernNCA context requires numerical, categorical, and label tensors."
            )

        self.train_x_numerical = x_numerical.detach()
        self.train_x_categorical = x_categorical_idx.detach().long()
        self.train_labels = train_y.detach().reshape(-1)

    def clear_train_context(self):
        device = self.train_labels.device
        self.train_x_numerical = torch.empty(0, device=device)
        self.train_x_categorical = torch.empty(0, dtype=torch.long, device=device)
        self.train_labels = torch.empty(0, device=device)

    def _embed(self, x_numerical, x_categorical_idx):
        pieces = []
        values = x_numerical[:, :, :1]
        pieces.extend(proj(values[:, i]) for i, proj in enumerate(self.num_proj))
        if self.cat_embed is not None:
            cat_tokens = self.cat_embed(x_categorical_idx.long())
            pieces.extend(cat_tokens[:, i] for i in range(self.n_categorical))
        if not pieces:
            raise ValueError("ModernNCA requires at least one input feature.")

        x = self.encoder(torch.cat(pieces, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def _nca_prediction(self, query, candidates):
        distances = torch.cdist(query, candidates)
        if candidates.shape[0] > 1:
            nearest_distances, nearest_indices = distances.min(dim=1)
            has_self = torch.isclose(
                nearest_distances,
                torch.zeros_like(nearest_distances),
                atol=1e-8,
                rtol=0.0,
            )
            rows = torch.arange(query.shape[0], device=query.device)[has_self]
            self_mask = torch.zeros_like(distances, dtype=torch.bool)
            self_mask[rows, nearest_indices[has_self]] = True
            distances = distances.masked_fill(self_mask, torch.inf)

        if 0 < self.n_neighbors < candidates.shape[0]:
            distances, indices = distances.topk(self.n_neighbors, largest=False)
            labels = self.train_labels[indices]
        else:
            labels = self.train_labels.unsqueeze(0).expand(query.shape[0], *self.train_labels.shape)

        weights = F.softmax(-distances / max(self.temperature, 1e-6), dim=-1)
        if self.is_regression:
            return torch.sum(weights * labels.float().reshape(labels.shape[0], -1), dim=-1, keepdim=True)

        if self.out_dim == 1:
            probabilities = torch.sum(
                weights * labels.float().reshape(labels.shape[0], -1), dim=-1
            ).clamp(1e-6, 1.0 - 1e-6)
            return torch.logit(probabilities).unsqueeze(-1)

        one_hot = F.one_hot(labels.long(), num_classes=self.out_dim).to(query.dtype)
        probabilities = torch.sum(weights[..., None] * one_hot, dim=1)
        return torch.log(probabilities.clamp_min(1e-7))

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        query = self._embed(x_numerical, x_categorical_idx)
        logits = self.head(query)
        if self.train_labels.numel():
            candidates = self._embed(
                self.train_x_numerical, self.train_x_categorical
            )
            logits = self._nca_prediction(query, candidates)

        return {
            "logits": logits,
            "aux_loss": None,
            "history": None,
            "psi_x": None,
        }
