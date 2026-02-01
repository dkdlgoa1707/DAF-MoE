import torch.nn as nn
from .router import DAFRouter
from .expert import DAFMoELayer

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_emb)
        self.attn = nn.MultiheadAttention(config.d_emb, config.n_heads, batch_first=True, dropout=config.dropout)
        self.ln2 = nn.LayerNorm(config.d_emb)
        self.router = DAFRouter(config)
        self.moe_layer = DAFMoELayer(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, psi_x, r_j, feature_mask):
        """
        Args:
            x (Tensor): Latent representation h^(l-1)
            psi_x (Tensor): Distributional position metadata
            r_j (Tensor): Raw value vector for preservation
            feature_mask (Tensor): Mask
        """
        # 1. Contextual Interaction (Equation 4)
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)
        
        # 2. Distribution-Adaptive MoE Layer
        residual = x
        x = self.ln2(x)
        
        # Router: [Weights, Indices, Mu]
        gating_weights, selected_indices, mu = self.router(x, psi_x)
        
        # Expert Layer: [B, S, D]
        moe_out = self.moe_layer(x, gating_weights, selected_indices, mu, r_j, feature_mask)
        
        x = residual + self.dropout(moe_out)
        
        return x, gating_weights, mu