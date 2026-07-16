import math

import torch
import torch.nn as nn


class PLEEncoder(nn.Module):
    """Per-feature Piecewise Linear Encoder (Gorishniy et al., 2022)."""

    def __init__(self, boundaries, d_emb):
        super().__init__()
        boundaries = torch.as_tensor(boundaries, dtype=torch.float32)
        if boundaries.ndim != 2:
            raise ValueError(
                "ple_boundaries must have shape [n_numerical, ple_n_bins + 1]."
            )
        self.n_num = boundaries.shape[0]
        self.T = boundaries.shape[1] - 1
        self.weight = nn.Parameter(torch.empty(self.n_num, self.T, d_emb))
        self.bias = nn.Parameter(torch.zeros(self.n_num, d_emb))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('boundaries', boundaries)

    def forward(self, values):
        left = self.boundaries[:, :-1].unsqueeze(0)
        right = self.boundaries[:, 1:].unsqueeze(0)
        denom = (right - left).clamp_min(1e-6)
        ratios = ((values.unsqueeze(-1) - left) / denom).clamp(0.0, 1.0)
        return torch.einsum('bnt,ntd->bnd', ratios, self.weight) + self.bias.unsqueeze(0)


class DAFEmbeddingV2(nn.Module):
    """Distribution-aware feature embedding for DAF-MoE v2.

    Numerical features use per-feature PLE. Categorical features use entity
    embeddings only; their distribution metadata remains available to routing.
    """

    def __init__(self, config):
        super().__init__()
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        self.d_emb = config.d_emb

        boundaries = getattr(config, 'ple_boundaries', None)
        if boundaries is None:
            raise ValueError(
                "DAFEmbeddingV2 requires config.ple_boundaries (from preflight)."
            )
        if isinstance(boundaries, dict):
            boundaries = list(boundaries.values())
        expected_shape = (self.n_numerical, config.ple_n_bins + 1)
        boundary_shape = tuple(torch.as_tensor(boundaries).shape)
        if boundary_shape != expected_shape:
            raise ValueError(
                f"ple_boundaries shape {boundary_shape} != expected {expected_shape}"
            )
        self.ple_encoder = PLEEncoder(boundaries, self.d_emb)
        self.numerical_missing_embedding = nn.Parameter(
            torch.zeros(1, self.n_numerical, self.d_emb)
        )

        self.categorical_embed = nn.Embedding(config.total_cats, self.d_emb)

        self.feature_identity = nn.Parameter(
            torch.randn(1, config.n_features, self.d_emb)
        )
        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.gelu = nn.GELU()

    def forward(
        self,
        x_numerical,
        x_categorical_idx,
        x_categorical_meta,
        x_numerical_missing=None,
    ):
        h_num = self.gelu(self.ple_encoder(x_numerical[:, :, 0]))
        h_cat = self.gelu(self.categorical_embed(x_categorical_idx))
        if x_numerical_missing is not None:
            h_num = h_num + (
                x_numerical_missing.unsqueeze(-1) * self.numerical_missing_embedding
            )

        h_merged = torch.cat([h_num, h_cat], dim=1)
        h_0 = self.layer_norm(h_merged + self.feature_identity)

        r_j = torch.cat(
            [x_numerical[:, :, 0:1], x_categorical_idx.unsqueeze(-1).float()],
            dim=1,
        )

        device = x_numerical.device
        batch_size = x_numerical.shape[0]
        feature_mask = torch.cat(
            [
                torch.ones((batch_size, self.n_numerical), device=device),
                torch.zeros((batch_size, self.n_categorical), device=device),
            ],
            dim=1,
        )

        meta_num = x_numerical[:, :, 1:]
        cat_freq = x_categorical_meta[:, :, 0:1]
        cat_card = x_categorical_meta[:, :, 1:2]
        sign = (x_categorical_idx.unsqueeze(-1) % 2) * 2 - 1
        tilde_f_j = 0.5 + sign * 0.5 * (1.0 - cat_freq)
        psi_x = torch.cat(
            [meta_num, torch.cat([tilde_f_j, cat_card], dim=-1)], dim=1
        )

        return h_0, r_j, feature_mask, psi_x.detach()
