"""TabR retrieval-augmented tabular baseline.

The retrieval block follows the official TabR formulation: learned keys select
nearest training candidates, whose label embeddings and key differences are
mixed into the query representation.

Reference: https://github.com/yandex-research/tabular-dl-tabr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        hidden = 2 * width
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class TabRWrapper(nn.Module):
    """TabR wrapper with an explicit training-candidate memory."""

    def __init__(self, config):
        super().__init__()
        self.d_emb = config.d_token if config.d_token is not None else config.d_emb
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        self.out_dim = config.out_dim
        self.is_regression = config.task_type == "regression"
        self.n_candidates = int(getattr(config, "tabr_n_candidates", 96))
        self.temperature = float(getattr(config, "tabr_temperature", 1.0))

        self.num_proj = nn.ModuleList(
            nn.Linear(1, self.d_emb) for _ in range(self.n_numerical)
        )
        self.cat_embed = (
            nn.Embedding(config.total_cats + 1, self.d_emb)
            if self.n_categorical
            else None
        )

        input_dim = (self.n_numerical + self.n_categorical) * self.d_emb
        self.input_proj = nn.Linear(input_dim, self.d_emb)
        self.encoder = nn.ModuleList(
            _ResidualBlock(self.d_emb, config.dropout)
            for _ in range(max(0, config.n_layers - 1))
        )
        self.key_proj = nn.Linear(self.d_emb, self.d_emb)
        self.label_encoder = (
            nn.Linear(1, self.d_emb)
            if self.is_regression
            else nn.Embedding(max(2, self.out_dim), self.d_emb)
        )
        self.value_transform = nn.Sequential(
            nn.Linear(self.d_emb, 2 * self.d_emb),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * self.d_emb, self.d_emb, bias=False),
        )
        self.context_dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_emb),
            nn.ReLU(),
            nn.Linear(self.d_emb, self.out_dim),
        )

        self.register_buffer(
            "candidate_x_numerical", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "candidate_x_categorical", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer("candidate_y", torch.empty(0), persistent=False)

    def set_candidates(self, x_numerical, x_categorical_idx=None, candidate_y=None):
        """Set the training rows used as retrieval candidates.

        ``x_numerical`` may be either the numerical tensor or the input dict
        emitted by ``DAFDataset``. Candidate tensors follow the module device
        when ``model.to(device)`` is called because they are registered buffers.
        """
        if isinstance(x_numerical, dict):
            inputs = x_numerical
            if candidate_y is None:
                candidate_y = x_categorical_idx
            x_numerical = inputs["x_numerical"]
            x_categorical_idx = inputs["x_categorical_idx"]
        if x_categorical_idx is None or candidate_y is None:
            raise ValueError("TabR candidates require numerical, categorical, and label tensors.")

        self.candidate_x_numerical = x_numerical.detach()
        self.candidate_x_categorical = x_categorical_idx.detach().long()
        self.candidate_y = candidate_y.detach()

    def clear_candidates(self):
        device = self.candidate_y.device
        self.candidate_x_numerical = torch.empty(0, device=device)
        self.candidate_x_categorical = torch.empty(0, dtype=torch.long, device=device)
        self.candidate_y = torch.empty(0, device=device)

    def _encode(self, x_numerical, x_categorical_idx):
        pieces = []
        values = x_numerical[:, :, :1]
        pieces.extend(proj(values[:, i]) for i, proj in enumerate(self.num_proj))
        if self.cat_embed is not None:
            cat_tokens = self.cat_embed(x_categorical_idx.long())
            pieces.extend(cat_tokens[:, i] for i in range(self.n_categorical))
        if not pieces:
            raise ValueError("TabR requires at least one input feature.")

        x = self.input_proj(torch.cat(pieces, dim=-1))
        for block in self.encoder:
            x = block(x)
        return x, self.key_proj(x)

    def _label_embeddings(self, labels):
        if self.is_regression:
            return self.label_encoder(labels.float().reshape(-1, 1))
        return self.label_encoder(labels.long().reshape(-1))

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        query, query_key = self._encode(x_numerical, x_categorical_idx)

        if self.candidate_y.numel():
            _, candidate_key = self._encode(
                self.candidate_x_numerical, self.candidate_x_categorical
            )
            n_context = min(self.n_candidates, candidate_key.shape[0])
            if n_context <= 0:
                raise ValueError("tabr_n_candidates must be positive.")

            distances = torch.cdist(query_key, candidate_key).square()
            context_idx = distances.topk(n_context, largest=False).indices
            context_key = candidate_key[context_idx]
            similarities = -distances.gather(1, context_idx) / max(self.temperature, 1e-6)
            weights = self.context_dropout(F.softmax(similarities, dim=-1))

            label_values = self._label_embeddings(self.candidate_y)[context_idx]
            values = label_values + self.value_transform(
                query_key[:, None, :] - context_key
            )
            query = query + torch.sum(weights[..., None] * values, dim=1)

        return {
            "logits": self.head(query),
            "aux_loss": None,
            "history": None,
            "psi_x": None,
        }
