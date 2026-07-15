"""RealMLP-style PyTorch baseline with the RealMLP-TD core components.

The official pytabkit estimator owns preprocessing, fitting, and early stopping,
so its sklearn API cannot be embedded safely in this repository's Trainer. This
module implements the neural portion directly: PLR-style numerical embeddings,
learnable front scaling, NTK-parameterized dense layers, parametric activations,
and the three-layer TD default.

Reference: https://github.com/dholzmueller/pytabkit
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PLREmbedding(nn.Module):
    def __init__(self, n_features, hidden=16, out_dim=4, sigma=0.1):
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(n_features, hidden) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, hidden, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_features, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, values):
        periodic = torch.cos(2.0 * math.pi * values.unsqueeze(-1) * self.frequency)
        projected = torch.einsum("bnh,nhd->bnd", periodic, self.weight)
        return F.relu(projected + self.bias)


class _NTKLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.scale = in_features ** -0.5
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class _ParametricActivation(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        self.mix = nn.Parameter(torch.ones(width))

    def forward(self, x):
        activated = F.selu(x) if self.kind == "selu" else F.mish(x)
        return x + (activated - x) * self.mix


class _RealMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, activation):
        super().__init__()
        self.linear = _NTKLinear(in_features, out_features)
        self.activation = _ParametricActivation(out_features, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


class RealMLPWrapper(nn.Module):
    """Trainer-compatible neural RealMLP-TD baseline."""

    def __init__(self, config):
        super().__init__()
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        hidden_dim = int(getattr(config, "realmlp_hidden_dim", 256))
        n_layers = int(getattr(config, "realmlp_n_layers", 3))
        if n_layers < 1:
            raise ValueError("realmlp_n_layers must be at least 1.")

        self.num_embedding = (
            _PLREmbedding(self.n_numerical) if self.n_numerical else None
        )
        self.cat_embed = (
            nn.Embedding(config.total_cats + 1, 8)
            if self.n_categorical
            else None
        )
        input_dim = self.n_numerical * 4 + self.n_categorical * 8
        if input_dim == 0:
            raise ValueError("RealMLP requires at least one input feature.")

        self.front_scale = nn.Parameter(torch.ones(input_dim))
        activation = "mish" if config.task_type == "regression" else "selu"
        widths = [input_dim] + [hidden_dim] * n_layers
        self.blocks = nn.Sequential(
            *[
                _RealMLPBlock(
                    widths[i], widths[i + 1], config.dropout, activation
                )
                for i in range(n_layers)
            ]
        )
        self.head = _NTKLinear(hidden_dim, config.out_dim)

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        pieces = []
        if self.num_embedding is not None:
            pieces.append(self.num_embedding(x_numerical[:, :, 0]).flatten(1))
        if self.cat_embed is not None:
            pieces.append(self.cat_embed(x_categorical_idx.long()).flatten(1))
        x = torch.cat(pieces, dim=-1) * self.front_scale
        logits = self.head(self.blocks(x))
        return {
            "logits": logits,
            "aux_loss": None,
            "history": None,
            "psi_x": None,
        }
