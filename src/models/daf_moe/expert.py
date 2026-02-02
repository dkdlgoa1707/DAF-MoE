import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DAFMoELayer(nn.Module):
    """
    Distribution-Adaptive MoE Layer with Dual-Pathway Experts.
    
    Implements Equation 8 (Dual Paths) and Equation 9 (Adaptive Centroid Gate).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.d_emb = config.d_emb
        self.d_ff = config.d_ff
        self.total_cats = config.total_cats

        # 1. Transformation Path (Standard FFN)
        self.w1 = nn.Parameter(torch.empty(self.n_experts, self.d_emb, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, self.d_ff, self.d_emb))
        
        # 2. Preservation Path (Omega Adapter)
        # Omega(r_j) for Numerical Features
        self.omega_num_w = nn.Parameter(torch.empty(self.n_experts, 1, self.d_emb))
        self.omega_num_b = nn.Parameter(torch.empty(self.n_experts, self.d_emb))

        # Omega(r_j) for Categorical Features (Identity Bypass Embedding)
        self.omega_cat_emb = nn.Parameter(torch.empty(self.n_experts, self.total_cats, self.d_emb))

        # Adaptive Gate Parameters (Equation 9)
        self.gate_steepness = nn.Parameter(torch.full((self.n_experts,), 1.0)) # kappa
        self.gate_threshold = nn.Parameter(torch.full((self.n_experts,), 0.3)) # tau

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w1, self.w2]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        
        nn.init.xavier_uniform_(self.omega_num_w)
        nn.init.zeros_(self.omega_num_b)
        nn.init.normal_(self.omega_cat_emb, std=0.02)

    def forward(self, h, gating_weights, selected_indices, mu, r_j, feature_mask):
        """
        Args:
            h (Tensor): Semantic input [B, S, D]
            mu (Tensor): Expert centroids [n_experts]
            r_j (Tensor): Raw value vector [B, S, 1]
        """
        B, S, D = h.shape
        E = self.n_experts
        
        # Flatten inputs
        h_flat = h.view(-1, D)            # [N, D]
        r_flat = r_j.view(-1, 1)          # [N, 1]
        mask_flat = feature_mask.view(-1, 1) # [N, 1]
        
        # -----------------------------------------------------------
        # Path 1: Transformation Path (FFN_k)
        # -----------------------------------------------------------
        if self.config.use_deep_path:
            # [N, D] x [E, D, d_ff] -> [N, E, d_ff]
            ff_1 = torch.einsum('nd, edf -> nef', h_flat, self.w1)
            ff_1 = F.gelu(ff_1)
            
            # [N, E, d_ff] x [E, d_ff, D] -> [N, E, D]
            expert_outputs = torch.einsum('nef, efd -> ned', ff_1, self.w2)
        else:
            expert_outputs = torch.zeros(B*S, E, D, device=h.device)

        # -----------------------------------------------------------
        # Path 2: Preservation Path (Omega(r_j) * alpha_k)
        # -----------------------------------------------------------
        if self.config.use_raw_path and (r_flat is not None):
            # (1) Compute Adaptive Centroid Gate alpha_k (Equation 9)
            centroid_pos = torch.sigmoid(mu)       # sigma(mu_k) [E]
            dist_to_center = torch.abs(centroid_pos - 0.5) # |sigma(mu) - 0.5|
            
            # alpha_k = sigmoid(kappa * (dist - tau))
            alpha_k = torch.sigmoid(self.gate_steepness * (dist_to_center - self.gate_threshold))
            alpha_k = alpha_k.view(1, E, 1)      # [1, E, 1] Broadcastable

            # (2) Numerical Preservation
            # Omega_num(r_j)
            num_val = r_flat * mask_flat
            preservation_num = torch.tanh(
                 torch.einsum('nf, efd -> ned', num_val, self.omega_num_w) 
                 + self.omega_num_b
            )
            
            # Add weighted preservation: + alpha_k * Omega(r_j)
            expert_outputs = expert_outputs + (alpha_k * preservation_num * mask_flat.unsqueeze(1))

            # (3) Categorical Preservation
            if self.total_cats > 0:
                cat_mask = (1 - mask_flat).long()                         # [N,1]
                if cat_mask.sum().item() > 0:                             # cat이 실제로 있을 때만
                    cat_indices = (r_flat.long() * cat_mask).squeeze(-1)  # [N]

                    device = h.device
                    expert_offsets = torch.arange(E, device=device) * self.total_cats  # [E]
                    lookup_indices = cat_indices.unsqueeze(1) + expert_offsets.unsqueeze(0)  # [N,E]

                    flat_embeddings = self.omega_cat_emb.view(-1, D)      # [E*total_cats, D]
                    preservation_cat = F.embedding(lookup_indices, flat_embeddings)     # [N,E,D]

                    expert_outputs = expert_outputs + (alpha_k * preservation_cat * cat_mask.unsqueeze(1))

        # Aggregation (Equation 10)
        mask = gating_weights.view(B*S, E) # [N, E]
        final_output = torch.einsum('ned, ne -> nd', expert_outputs, mask)
        
        return final_output.view(B, S, D)