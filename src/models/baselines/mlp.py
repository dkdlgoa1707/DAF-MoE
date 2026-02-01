import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) Baseline.
    Concatenates all embeddings into a single vector and passes through dense layers.
    """
    def __init__(self, config):
        super().__init__()
        # Compatibility: Support both 'd_token' (HPO) and 'd_emb' (Default)
        self.d_emb = config.d_token if config.d_token is not None else config.d_emb
        
        # Independent projections for numerical features
        self.num_proj = nn.ModuleList([nn.Linear(1, self.d_emb) for _ in range(config.n_numerical)])
        self.cat_embed = nn.Embedding(config.total_cats + 1, self.d_emb)
        
        # Flattened Input Dimension
        input_dim = (config.n_numerical + config.n_categorical) * self.d_emb
        
        layers = []
        curr_dim = input_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(curr_dim, self.d_emb))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            curr_dim = self.d_emb
            
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(self.d_emb, config.out_dim)

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        B = x_numerical.shape[0]
        
        # Numerical Embedding
        x_num_val = x_numerical[:, :, 0:1]
        x_n_emb = torch.stack([proj(x_num_val[:, i]) for i, proj in enumerate(self.num_proj)], dim=1)
        
        # Categorical Embedding
        x_c_emb = self.cat_embed(x_categorical_idx.long())
        
        # Flatten and Forward
        x = torch.cat([x_n_emb, x_c_emb], dim=1).view(B, -1)
        x = self.body(x)
        
        return {"logits": self.head(x), "aux_loss": None}