import torch.nn as nn
from .router import DAFRouterV15 as DAFRouter
from .expert import DAFMoELayerV15 as DAFMoELayer


class TransformerBlockV15(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_emb)
        self.attn = nn.MultiheadAttention(config.d_emb, config.n_heads, batch_first=True, dropout=config.dropout)
        self.ln2 = nn.LayerNorm(config.d_emb)
        self.router = DAFRouter(config)
        self.moe_layer = DAFMoELayer(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, psi_x, r_j, feature_mask):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ln2(x)

        gating_weights, selected_indices, mu = self.router(x, psi_x)
        moe_out = self.moe_layer(x, gating_weights, selected_indices, mu, r_j, feature_mask)

        x = residual + self.dropout(moe_out)

        return x, gating_weights, mu
