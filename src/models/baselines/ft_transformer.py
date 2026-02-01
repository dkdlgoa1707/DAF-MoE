import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer for FT-Transformer.
    Converts numerical scalar values and categorical indices into d-dimensional embeddings.
    """
    def __init__(self, n_num, n_cat, total_cats, d_emb):
        super().__init__()
        # Numerical: Independent linear projection for each feature
        self.num_proj = nn.ModuleList([nn.Linear(1, d_emb) for _ in range(n_num)])
        # Categorical: Shared embedding table
        self.cat_embed = nn.Embedding(total_cats + 1, d_emb)
    
    def forward(self, x_num, x_cat):
        # x_num: [B, N_num, 3] -> use only value channel [:, :, 0:1]
        x_num_val = x_num[:, :, 0:1]
        
        # Project each numerical feature independently
        x_n_emb = [proj(x_num_val[:, i]) for i, proj in enumerate(self.num_proj)]
        x_n_emb = torch.stack(x_n_emb, dim=1) # [B, N_num, d_emb]
        
        # Look up categorical embeddings
        x_c_emb = self.cat_embed(x_cat.long()) # [B, N_cat, d_emb]
        
        return torch.cat([x_n_emb, x_c_emb], dim=1)

class FTTransformerWrapper(nn.Module):
    """
    FT-Transformer Baseline (Gorishniy et al., 2021).
    Standard Transformer Encoder applied to tabular data.
    """
    def __init__(self, config):
        super().__init__()
        
        # Compatibility: Support both 'd_token' (HPO) and 'd_emb' (Default)
        self.d_emb = config.d_token if config.d_token is not None else config.d_emb
        
        # Feed-Forward Dimension
        d_ff = int(self.d_emb * config.d_ff_factor)
        
        self.tokenizer = FeatureTokenizer(
            config.n_numerical, config.n_categorical, config.total_cats, self.d_emb
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_emb,
            nhead=config.n_heads,
            dim_feedforward=d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_emb))
        self.ln_f = nn.LayerNorm(self.d_emb)
        self.head = nn.Linear(self.d_emb, config.out_dim)

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        B = x_numerical.shape[0]
        
        # Tokenization
        x = self.tokenizer(x_numerical, x_categorical_idx)
        
        # Append [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer Encoding
        x = self.transformer(x)
        
        # Classification head on [CLS] token
        x_cls = self.ln_f(x[:, 0])
        return {"logits": self.head(x_cls), "aux_loss": None}