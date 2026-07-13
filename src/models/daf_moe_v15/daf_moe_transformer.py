import torch
import torch.nn as nn
from .embedding import DAFEmbeddingV15 as DAFEmbedding
from .transformer_block import TransformerBlockV15 as TransformerBlock


class DAFMoETransformerV15(nn.Module):
    """
    DAF-MoE v1.5 Phase 1 model.
    """
    def __init__(self, config):
        super().__init__()

        if hasattr(config, 'd_ff_factor') and config.d_ff_factor is not None:
            config.d_ff = int(config.d_emb * config.d_ff_factor)
        else:
            if not hasattr(config, 'd_ff'):
                config.d_ff = config.d_emb * 4

        self.embedding = DAFEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_emb)

        self.head = nn.Sequential(
            nn.Linear(config.d_emb * config.n_features, config.d_emb),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_emb, config.out_dim)
        )

    def forward(self, x_numerical, x_categorical_idx, x_categorical_meta):
        h, r_j, feature_mask, psi_x = self.embedding(
            x_numerical, x_categorical_idx, x_categorical_meta
        )

        routing_history = []
        for block in self.blocks:
            h, weights, mu = block(h, psi_x, r_j, feature_mask)
            routing_history.append({'weights': weights, 'mu': mu})

        h = self.ln_f(h)
        h = torch.flatten(h, start_dim=1)
        logits = self.head(h)

        return {
            "logits": logits,
            "history": routing_history,
            "psi_x": psi_x,
            "aux_loss": None
        }
