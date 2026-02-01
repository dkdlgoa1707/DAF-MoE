import torch
import torch.nn as nn

class DAFEmbedding(nn.Module):
    """
    Distribution-Embedded Tokenization Layer.
    
    Constructs the input representation z_j and raw preservation vector r_j as defined in Section 3.1.
    
    Output:
        h_0 (Tensor): Initial semantic projection (z_j projected).
        r_j (Tensor): Raw value vector for the preservation path.
        feature_mask (Tensor): Mask distinguishing numerical vs categorical features.
        psi_x (Tensor): Distributional position vector [Psi(x_j)] for specialization loss.
    """
    def __init__(self, config):
        super().__init__()
        self.n_numerical = config.n_numerical
        self.n_categorical = config.n_categorical
        self.d_emb = config.d_emb
        
        # 3 Channels for Numerical: [Value, P, Gamma] -> z_j projection
        self.numerical_proj = nn.Linear(3, self.d_emb)
        
        # Categorical: [Embedding + Meta] -> z_j projection
        self.categorical_embed = nn.Embedding(config.total_cats, self.d_emb)
        self.categorical_meta_proj = nn.Linear(2, self.d_emb)
        
        self.feature_identity = nn.Parameter(torch.randn(1, config.n_features, self.d_emb))
        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.gelu = nn.GELU()

    def forward(self, x_numerical, x_categorical_idx, x_categorical_meta):
        # 1. Unified Latent Projection (Equation 3)
        # Numerical Features
        h_num = self.gelu(self.numerical_proj(x_numerical))
        
        # Categorical Features
        h_cat = self.gelu(self.categorical_embed(x_categorical_idx) + 
                          self.categorical_meta_proj(x_categorical_meta))
        
        h_merged = torch.cat([h_num, h_cat], dim=1)
        h_0 = self.layer_norm(h_merged + self.feature_identity)

        # 2. Raw Value Vector r_j (Equation 1 & 2)
        # r_j = x_j (numerical) OR e_id(x_j) (categorical)
        # Note: For categorical, we pass indices to be looked up later in the expert layer
        r_j = torch.cat([x_numerical[:, :, 0:1], x_categorical_idx.unsqueeze(-1).float()], dim=1)
        
        dev = x_numerical.device
        B = x_numerical.shape[0]
        feature_mask = torch.cat([torch.ones((B, self.n_numerical), device=dev), 
                                  torch.zeros((B, self.n_categorical), device=dev)], dim=1)

        # 3. Distributional Position Psi(x_j) (Section 3.5.1)
        # Psi(x_j) = Phi(x_j) for numerical, Tilde(F_j) for categorical
        
        # Numerical metadata: [P, Gamma]
        # We use P (Percentile) as the primary positional info
        meta_num = x_numerical[:, :, 1:] 
        
        # Categorical Symmetric Rareness Expansion (Equation 6)
        cat_freq = x_categorical_meta[:, :, 0:1]
        cat_card = x_categorical_meta[:, :, 1:2]
        
        sign = (x_categorical_idx.unsqueeze(-1) % 2) * 2 - 1 # Simple hash function psi(.)
        tilde_F_j = 0.5 + sign * 0.5 * (1.0 - cat_freq)
        
        # psi_x will be used for both Router Input (q_j) and Specialization Loss (L_spec)
        psi_x = torch.cat([meta_num, torch.cat([tilde_F_j, cat_card], dim=-1)], dim=1)
        
        return h_0, r_j, feature_mask, psi_x.detach()