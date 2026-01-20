import torch
import torch.nn as nn

class DAFEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        self.d_emb = config.d_emb
        
        # 3 Channels for Numerical: [Value, P, Gamma]
        self.numerical_proj = nn.Linear(3, self.d_emb)
        self.categorical_embed = nn.Embedding(config.total_cats, self.d_emb)
        self.categorical_meta_proj = nn.Linear(2, self.d_emb)
        
        self.feature_identity = nn.Parameter(torch.randn(1, config.n_features, self.d_emb))
        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.gelu = nn.GELU()

    def forward(self, x_numerical, x_categorical_idx, x_categorical_meta):
        # 1. Embedding
        h_num = self.gelu(self.numerical_proj(x_numerical))
        h_cat = self.gelu(self.categorical_embed(x_categorical_idx) + 
                          self.categorical_meta_proj(x_categorical_meta))
        
        h_merged = torch.cat([h_num, h_cat], dim=1)
        h_0 = self.layer_norm(h_merged + self.feature_identity)

        # 2. Raw Values & Mask
        raw_values = torch.cat([x_numerical[:, :, 0:1], x_categorical_idx.unsqueeze(-1).float()], dim=1)
        
        dev = x_numerical.device
        B = x_numerical.shape[0]
        feature_mask = torch.cat([torch.ones((B, self.n_numerical), device=dev), 
                                  torch.zeros((B, self.n_categorical), device=dev)], dim=1)

        # 3. Symmetric Expansion
        meta_num = x_numerical[:, :, 1:] # [P, Gamma]
        
        cat_freq = x_categorical_meta[:, :, 0:1]
        cat_card = x_categorical_meta[:, :, 1:2]
        sign = (x_categorical_idx.unsqueeze(-1) % 2) * 2 - 1
        transformed_freq = 0.5 + sign * 0.5 * (1.0 - cat_freq)
        
        unified_metadata = torch.cat([meta_num, torch.cat([transformed_freq, cat_card], dim=-1)], dim=1)
        
        return h_0, raw_values, feature_mask, unified_metadata.detach()