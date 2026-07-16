"""RTDL-style MLP baseline."""

from collections import OrderedDict

import torch.nn as nn

from .components import DenseFeatureInput, resolve_width


class MLP(nn.Module):
    """Plain RTDL MLP over model-specific scalar/embedding preprocessing."""

    def __init__(self, config):
        super().__init__()
        self.feature_input = DenseFeatureInput(config)
        default_width = resolve_width(config)
        n_layers = int(config.n_layers)
        if n_layers <= 0:
            raise ValueError("MLP n_layers must be positive.")

        first = int(getattr(config, "first_width", None) or default_width)
        middle = int(getattr(config, "middle_width", None) or default_width)
        last = int(getattr(config, "last_width", None) or default_width)
        widths = [first] if n_layers == 1 else [first] + [middle] * (n_layers - 2) + [last]
        if any(width <= 0 for width in widths):
            raise ValueError("MLP widths must be positive.")

        blocks = []
        d_in = self.feature_input.output_dim
        for index, width in enumerate(widths):
            blocks.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("linear", nn.Linear(d_in, width)),
                            ("activation", nn.ReLU()),
                            ("dropout", nn.Dropout(float(config.dropout))),
                        ]
                    )
                )
            )
            d_in = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d_in, config.out_dim)

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
        for block in self.blocks:
            x = block(x)
        return {"logits": self.head(x), "aux_loss": None}
