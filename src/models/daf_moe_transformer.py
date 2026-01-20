import torch
import torch.nn as nn
from .embedding import DAFEmbedding
from .transformer_block import TransformerBlock

class DAFMoETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        h, raw_values, feature_mask, unified_metadata = self.embedding(
            x_numerical, x_categorical_idx, x_categorical_meta
        )
        
        routing_history = []
        for block in self.blocks:
            h, weights, mu = block(h, unified_metadata, raw_values, feature_mask)
            routing_history.append({'weights': weights, 'mu': mu})
            
        h = self.ln_f(h)
        h = torch.flatten(h, start_dim=1)
        logits = self.head(h)
        
        return logits, routing_history, unified_metadata