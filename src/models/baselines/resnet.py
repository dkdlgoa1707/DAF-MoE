import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    Standard ResNet Block: x + Linear(Dropout(ReLU(Linear(Norm(x)))))
    """
    def __init__(self, d, d_hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return x + out

class TabularResNet(nn.Module):
    """
    ResNet Baseline for Tabular Data (Gorishniy et al., 2021).
    Embeds features, flattens them, and processes via Residual Blocks.
    """
    def __init__(self, config):
        super().__init__()
        # Compatibility: Support both 'd_token' (HPO) and 'd_emb' (Default)
        self.d_main = config.d_token if config.d_token is not None else config.d_emb
        d_hidden = int(self.d_main * config.d_hidden_factor)
        
        # 1. Feature Embedding
        self.num_proj = nn.ModuleList([nn.Linear(1, self.d_main) for _ in range(config.n_numerical)])
        self.cat_embed = nn.Embedding(config.total_cats + 1, self.d_main)
        
        # 2. Input Projection (Flatten -> d_main)
        total_features = config.n_numerical + config.n_categorical
        self.first_linear = nn.Linear(total_features * self.d_main, self.d_main)
        
        # 3. ResNet Blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(self.d_main, d_hidden, config.dropout) 
            for _ in range(config.n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(self.d_main)
        self.head = nn.Linear(self.d_main, config.out_dim)

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        # A. Feature Embedding
        x_num_val = x_numerical[:, :, 0:1]
        x_n_emb = torch.stack([proj(x_num_val[:, i]) for i, proj in enumerate(self.num_proj)], dim=1)
        x_c_emb = self.cat_embed(x_categorical_idx.long())
        
        # B. Flatten & Initial Project
        # [B, N_features, d_main] -> [B, N_features * d_main] -> [B, d_main]
        x = torch.cat([x_n_emb, x_c_emb], dim=1)
        x = x.flatten(1) 
        x = self.first_linear(x) 
        
        # C. ResNet Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm_f(x)
        return {"logits": self.head(x), "aux_loss": None}