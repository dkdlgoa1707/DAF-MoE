import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAFMoELayerV15(nn.Module):
    """
    DAF-MoE v1.5 expert layer with optional lightweight preservation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.d_emb = config.d_emb
        self.d_ff = config.d_ff
        self.total_cats = config.total_cats
        self.use_lightweight_preservation = getattr(config, 'use_lightweight_preservation', False)

        self.w1 = nn.Parameter(torch.empty(self.n_experts, self.d_emb, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, self.d_ff, self.d_emb))

        if self.use_lightweight_preservation:
            self.omega_shared = nn.Parameter(torch.randn(self.d_emb) / math.sqrt(self.d_emb))
        else:
            self.omega_num_w = nn.Parameter(torch.empty(self.n_experts, 1, self.d_emb))
            self.omega_num_b = nn.Parameter(torch.empty(self.n_experts, self.d_emb))
            self.omega_cat_emb = nn.Parameter(torch.empty(self.n_experts, self.total_cats, self.d_emb))

        self.gate_steepness = nn.Parameter(torch.full((self.n_experts,), 1.0))
        self.gate_threshold = nn.Parameter(torch.full((self.n_experts,), 0.3))

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w1, self.w2]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if not self.use_lightweight_preservation:
            nn.init.xavier_uniform_(self.omega_num_w)
            nn.init.zeros_(self.omega_num_b)
            nn.init.normal_(self.omega_cat_emb, std=0.02)

    def forward(self, h, gating_weights, selected_indices, mu, r_j, feature_mask):
        B, S, D = h.shape
        E = self.n_experts

        h_flat = h.view(-1, D)
        r_flat = r_j.view(-1, 1)
        mask_flat = feature_mask.view(-1, 1)

        if self.config.use_deep_path:
            ff_1 = torch.einsum('nd, edf -> nef', h_flat, self.w1)
            ff_1 = F.gelu(ff_1)
            expert_outputs = torch.einsum('nef, efd -> ned', ff_1, self.w2)
        else:
            expert_outputs = torch.zeros(B * S, E, D, device=h.device)

        if self.config.use_raw_path and (r_flat is not None):
            centroid_pos = torch.sigmoid(mu)
            dist_to_center = torch.abs(centroid_pos - 0.5)

            alpha_k = torch.sigmoid(self.gate_steepness * (dist_to_center - self.gate_threshold))
            alpha_k = alpha_k.view(1, E, 1)

            if self.use_lightweight_preservation:
                preservation = r_flat * self.omega_shared.view(1, D)
                expert_outputs = expert_outputs + (
                    alpha_k * preservation.unsqueeze(1) * mask_flat.unsqueeze(1)
                )
            else:
                num_val = r_flat * mask_flat
                preservation_num = torch.tanh(
                    torch.einsum('nf, efd -> ned', num_val, self.omega_num_w)
                    + self.omega_num_b
                )

                expert_outputs = expert_outputs + (alpha_k * preservation_num * mask_flat.unsqueeze(1))

                if self.total_cats > 0:
                    cat_mask = (1 - mask_flat).long()
                    if cat_mask.sum().item() > 0:
                        cat_indices = (r_flat.long() * cat_mask).squeeze(-1)

                        device = h.device
                        expert_offsets = torch.arange(E, device=device) * self.total_cats
                        lookup_indices = cat_indices.unsqueeze(1) + expert_offsets.unsqueeze(0)

                        flat_embeddings = self.omega_cat_emb.view(-1, D)
                        preservation_cat = F.embedding(lookup_indices, flat_embeddings)

                        expert_outputs = expert_outputs + (alpha_k * preservation_cat * cat_mask.unsqueeze(1))

        mask = gating_weights.view(B * S, E)
        final_output = torch.einsum('ned, ne -> nd', expert_outputs, mask)

        return final_output.view(B, S, D)
