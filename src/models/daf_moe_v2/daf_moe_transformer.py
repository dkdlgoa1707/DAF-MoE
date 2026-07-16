import torch
import torch.nn as nn

from .embedding import DAFEmbeddingV2 as DAFEmbedding
from .transformer_block import TransformerBlockV2 as TransformerBlock


class DAFMoETransformerV2(nn.Module):
    """Final DAF-MoE v2 architecture for Phase 2 experiments."""

    def __init__(self, config):
        super().__init__()

        if hasattr(config, 'd_ff_factor') and config.d_ff_factor is not None:
            config.d_ff = int(config.d_emb * config.d_ff_factor)
        elif not hasattr(config, 'd_ff'):
            config.d_ff = config.d_emb * 4

        self.embedding = DAFEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_emb)

        self.head = nn.Sequential(
            nn.Linear(config.d_emb * config.n_features, config.d_emb),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_emb, config.out_dim),
        )

    def forward(
        self,
        x_numerical,
        x_categorical_idx,
        x_categorical_meta,
        x_numerical_missing=None,
    ):
        h, r_j, feature_mask, psi_x = self.embedding(
            x_numerical, x_categorical_idx, x_categorical_meta, x_numerical_missing
        )

        routing_history = []
        for block in self.blocks:
            h, weights, mu = block(h, psi_x, r_j, feature_mask)
            routing_history.append({'weights': weights, 'mu': mu})

        h = self.ln_f(h)
        h = torch.flatten(h, start_dim=1)
        logits = self.head(h)

        return {
            'logits': logits,
            'history': routing_history,
            'psi_x': psi_x,
            'aux_loss': None,
        }
