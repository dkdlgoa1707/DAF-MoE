import torch
import torch.nn as nn
from .embedding import DAFEmbedding
from .transformer_block import TransformerBlock

class DAFMoETransformer(nn.Module):
    """
    DAF-MoE: Distribution-Aware Feature-level Mixture of Experts
    ============================================================
    Main architecture class implementing the framework described in Section 3.
    
    Inputs:
        x_numerical: [Batch, Features, 3] (Value, P, Gamma)
        x_categorical_idx: [Batch, Features] (Indices)
        x_categorical_meta: [Batch, Features, 2] (Freq, Card)
        
    Outputs:
        logits: Final prediction
        history: Routing weights and mu history for loss calculation
        psi_x: Distributional position metadata for specialization loss
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
        
        # Global Prediction Head (Equation 11)
        self.head = nn.Sequential(
            nn.Linear(config.d_emb * config.n_features, config.d_emb),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_emb, config.out_dim)
        )

    def forward(self, x_numerical, x_categorical_idx, x_categorical_meta):
        # 1. Distribution-Embedded Tokenization
        h, r_j, feature_mask, psi_x = self.embedding(
            x_numerical, x_categorical_idx, x_categorical_meta
        )
        
        routing_history = []
        for block in self.blocks:
            # 2. Contextual Interaction & 3. MoE Layer
            h, weights, mu = block(h, psi_x, r_j, feature_mask)
            routing_history.append({'weights': weights, 'mu': mu})
            
        # 4. Global Prediction
        h = self.ln_f(h)
        h = torch.flatten(h, start_dim=1)
        logits = self.head(h)
        
        return {
            "logits": logits,
            "history": routing_history,  # For Bal & Repel Loss
            "psi_x": psi_x,              # For Spec Loss (Equation 12)
            "aux_loss": None             
        }