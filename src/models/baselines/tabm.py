"""Plain TabM-mini and the secondary updated-PLE TabM-mini control."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import categorical_cardinalities, resolve_width


class TabMOneHotEncoding(nn.Module):
    """Train-fitted one-hot encoding while keeping true binary values scalar."""

    def __init__(self, cardinalities, known_cardinalities):
        super().__init__()
        if len(cardinalities) != len(known_cardinalities):
            raise ValueError("TabM categorical metadata lengths do not match.")
        self.cardinalities = tuple(cardinalities)
        self.known_cardinalities = tuple(known_cardinalities)
        self.feature_widths = tuple(
            3 if known == 2 else cardinality
            for cardinality, known in zip(cardinalities, known_cardinalities)
        )

    @property
    def output_dim(self):
        return sum(self.feature_widths)

    def forward(self, values):
        encoded = []
        for index, (cardinality, known) in enumerate(
            zip(self.cardinalities, self.known_cardinalities)
        ):
            feature = values[:, index]
            if known == 2:
                unknown_id = cardinality - 1
                binary_value = (feature == 2).to(torch.float32)
                missing = (feature == 0).to(torch.float32)
                unknown = (feature == unknown_id).to(torch.float32)
                encoded.append(torch.stack([binary_value, missing, unknown], dim=-1))
            else:
                encoded.append(F.one_hot(feature, cardinality).to(torch.float32))
        if not encoded:
            return values.new_empty(values.shape[0], 0, dtype=torch.float32)
        return torch.cat(encoded, dim=-1)


class UpdatedPiecewiseLinearEmbedding(nn.Module):
    """Version-B style PLE: piecewise encoding followed by per-feature linear maps."""

    def __init__(self, boundaries, d_embedding):
        super().__init__()
        boundaries = torch.as_tensor(boundaries, dtype=torch.float32)
        if boundaries.ndim != 2 or boundaries.shape[1] < 2:
            raise ValueError("TabM PLE boundaries must have shape [n_num, n_bins + 1].")
        self.register_buffer("boundaries", boundaries)
        self.weight = nn.Parameter(
            torch.empty(boundaries.shape[0], boundaries.shape[1] - 1, d_embedding)
        )
        self.bias = nn.Parameter(torch.empty(boundaries.shape[0], d_embedding))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = (boundaries.shape[1] - 1) ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def output_dim(self):
        return self.weight.shape[0] * self.weight.shape[-1]

    def forward(self, values):
        left = self.boundaries[:, :-1].unsqueeze(0)
        right = self.boundaries[:, 1:].unsqueeze(0)
        ratios = ((values.unsqueeze(-1) - left) / (right - left).clamp_min(1e-6)).clamp(
            0.0, 1.0
        )
        return torch.einsum("bnt,ntd->bnd", ratios, self.weight) + self.bias


class MiniEnsembleAffine(nn.Module):
    """The only non-shared backbone transform in official TabM-mini."""

    def __init__(self, k, d_in, initialization):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d_in))
        if initialization == "random-signs":
            with torch.no_grad():
                self.weight.bernoulli_(0.5).mul_(2).add_(-1)
        elif initialization == "normal":
            nn.init.normal_(self.weight)
        else:
            raise ValueError(f"Unknown TabM affine initialization: {initialization}")

    def forward(self, x):
        return x.unsqueeze(1) * self.weight.unsqueeze(0)


class EnsembleLinear(nn.Module):
    """Independent output heads for k members, matching TabM LinearEnsemble."""

    def __init__(self, k, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d_in, d_out))
        self.bias = nn.Parameter(torch.empty(k, d_out))
        bound = d_in ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.einsum("bkd,kdo->bko", x, self.weight) + self.bias


def aggregate_member_predictions(logits_k, task_type):
    if task_type == "regression":
        return logits_k.mean(dim=1)
    if logits_k.shape[-1] == 1:
        probability = torch.sigmoid(logits_k).mean(dim=1)
        return torch.logit(probability.clamp(1e-7, 1.0 - 1e-7))
    probability = torch.softmax(logits_k, dim=-1).mean(dim=1)
    return probability.clamp_min(1e-12).log()


class _TabMMiniBase(nn.Module):
    use_ple = False
    rank_included = True

    def __init__(self, config):
        super().__init__()
        self.k = int(config.k)
        if self.k != 32:
            raise ValueError("Phase 2 TabM-mini fixes k=32.")
        self.task_type = config.task_type
        n_numerical = int(config.n_numerical)
        cardinalities = categorical_cardinalities(config)
        known_cardinalities = list(
            getattr(config, "cat_known_cardinalities", None) or []
        )
        self.categorical = TabMOneHotEncoding(
            cardinalities,
            known_cardinalities,
        )

        self.num_embedding = None
        if self.use_ple and n_numerical:
            boundaries = getattr(config, "ple_boundaries", None)
            if boundaries is None:
                raise ValueError("tabm_ple requires train-fitted PLE boundaries.")
            d_embedding = int(getattr(config, "tabm_ple_embedding_dim", 16))
            self.num_embedding = UpdatedPiecewiseLinearEmbedding(
                boundaries,
                d_embedding,
            )
            d_num = self.num_embedding.output_dim
        else:
            d_num = n_numerical

        d_in = d_num + n_numerical + self.categorical.output_dim
        if d_in <= 0:
            raise ValueError("TabM requires at least one input feature.")
        d_block = resolve_width(config)
        n_layers = int(config.n_layers)
        if n_layers <= 0:
            raise ValueError("TabM n_layers must be positive.")
        self.affine = MiniEnsembleAffine(
            self.k,
            d_in,
            "normal" if self.use_ple else "random-signs",
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in if index == 0 else d_block, d_block),
                    nn.ReLU(),
                    nn.Dropout(float(config.dropout)),
                )
                for index in range(n_layers)
            ]
        )
        self.output = EnsembleLinear(self.k, d_block, config.out_dim)

    def _flat_input(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
    ):
        parts = []
        if x_numerical_values.shape[1]:
            numerical = (
                x_numerical_values
                if self.num_embedding is None
                else self.num_embedding(x_numerical_values).flatten(1)
            )
            parts.extend([numerical, x_numerical_missing])
        if self.categorical.cardinalities:
            parts.append(self.categorical(x_categorical_idx.long()))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
        **kwargs,
    ):
        x = self.affine(
            self._flat_input(
                x_numerical_values,
                x_numerical_missing,
                x_categorical_idx,
            )
        )
        for block in self.blocks:
            x = block(x)
        logits_k = self.output(x)
        logits = aggregate_member_predictions(logits_k, self.task_type)
        return {"logits": logits, "logits_k": logits_k, "aux_loss": None}


class TabMWrapper(_TabMMiniBase):
    """Main Phase 2 method: plain ICLR 2025 TabM-mini."""


class TabMPLEWrapper(_TabMMiniBase):
    """Secondary control: TabM-mini with updated PLE (TabM dagger)."""

    use_ple = True
    rank_included = False
