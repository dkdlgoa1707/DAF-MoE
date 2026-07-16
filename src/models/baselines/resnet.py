"""Canonical RTDL tabular ResNet baseline."""

from collections import OrderedDict

import torch.nn as nn

from .components import DenseFeatureInput, resolve_width


class ResNetBlock(nn.Module):
    def __init__(self, d_block, d_hidden, hidden_dropout, residual_dropout):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("normalization", nn.BatchNorm1d(d_block)),
                    ("linear1", nn.Linear(d_block, d_hidden)),
                    ("activation", nn.ReLU()),
                    ("hidden_dropout", nn.Dropout(hidden_dropout)),
                    ("linear2", nn.Linear(d_hidden, d_block)),
                    ("residual_dropout", nn.Dropout(residual_dropout)),
                ]
            )
        )

    def forward(self, x):
        return x + self.layers(x)


class TabularResNet(nn.Module):
    """Gorishniy/RTDL ResNet with BatchNorm and separate dropouts."""

    def __init__(self, config):
        super().__init__()
        self.feature_input = DenseFeatureInput(config)
        d_block = resolve_width(config)
        d_hidden = int(d_block * float(config.d_hidden_factor))
        if d_hidden <= 0 or int(config.n_layers) <= 0:
            raise ValueError("ResNet depth and hidden width must be positive.")
        hidden_dropout = float(
            getattr(config, "hidden_dropout", None)
            if getattr(config, "hidden_dropout", None) is not None
            else config.dropout
        )
        residual_dropout = float(config.residual_dropout)

        self.input_projection = nn.Linear(self.feature_input.output_dim, d_block)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(
                    d_block,
                    d_hidden,
                    hidden_dropout,
                    residual_dropout,
                )
                for _ in range(int(config.n_layers))
            ]
        )
        self.output = nn.Sequential(
            OrderedDict(
                [
                    ("normalization", nn.BatchNorm1d(d_block)),
                    ("activation", nn.ReLU()),
                    ("linear", nn.Linear(d_block, config.out_dim)),
                ]
            )
        )

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
        **kwargs,
    ):
        x = self.feature_input(
            x_numerical_values,
            x_numerical_missing,
            x_categorical_idx,
        )
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return {"logits": self.output(x), "aux_loss": None}
