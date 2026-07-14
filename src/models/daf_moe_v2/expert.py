import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAFMoELayerV2(nn.Module):
    """Dual-pathway expert layer with lightweight preservation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.d_emb = config.d_emb
        self.d_ff = config.d_ff

        self.w1 = nn.Parameter(
            torch.empty(self.n_experts, self.d_emb, self.d_ff)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.n_experts, self.d_ff, self.d_emb)
        )

        self.omega_shared = nn.Parameter(
            torch.randn(self.d_emb) / math.sqrt(self.d_emb)
        )

        self.gate_steepness = nn.Parameter(
            torch.full((self.n_experts,), 1.0)
        )
        self.gate_threshold = nn.Parameter(
            torch.full((self.n_experts,), 0.3)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.w1, self.w2]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, h, gating_weights, selected_indices, mu, r_j, feature_mask):
        batch_size, sequence_length, d_emb = h.shape
        n_experts = self.n_experts

        h_flat = h.view(-1, d_emb)
        r_flat = r_j.view(-1, 1)
        mask_flat = feature_mask.view(-1, 1)

        if self.config.use_deep_path:
            ff_1 = torch.einsum('nd, edf -> nef', h_flat, self.w1)
            ff_1 = F.gelu(ff_1)
            expert_outputs = torch.einsum('nef, efd -> ned', ff_1, self.w2)
        else:
            expert_outputs = torch.zeros(
                batch_size * sequence_length,
                n_experts,
                d_emb,
                device=h.device,
            )

        if self.config.use_raw_path and r_flat is not None:
            centroid_pos = torch.sigmoid(mu)
            dist_to_center = torch.abs(centroid_pos - 0.5)
            alpha_k = torch.sigmoid(
                self.gate_steepness * (dist_to_center - self.gate_threshold)
            )
            alpha_k = alpha_k.view(1, n_experts, 1)

            preservation = r_flat * self.omega_shared.view(1, d_emb)
            expert_outputs = expert_outputs + (
                alpha_k * preservation.unsqueeze(1) * mask_flat.unsqueeze(1)
            )

        mask = gating_weights.view(batch_size * sequence_length, n_experts)
        final_output = torch.einsum('ned, ne -> nd', expert_outputs, mask)
        return final_output.view(batch_size, sequence_length, d_emb)
