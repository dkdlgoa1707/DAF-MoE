import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_emb = config.d_emb
        self.n_experts = config.n_experts
        self.k = config.top_k
        self.noise_std = config.router_noise_std

        # mu 초기화 유연화
        if config.mu_init_strategy == 'custom' and config.initial_mu:
            init_mu = torch.tensor(config.initial_mu)
        else:
            init_mu = torch.linspace(0, 1, self.n_experts)
        self.mu = nn.Parameter(init_mu) # Raw logits

        # 메타데이터 투영
        self.metadata_proj = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 8)
        )
        self.gate = nn.Linear(self.d_emb + 8, self.n_experts)

    def forward(self, h, metadata):
        h_stopped = h.detach() # Stop-Gradient
        m_emb = self.metadata_proj(metadata)
        logits = self.gate(torch.cat([h_stopped, m_emb], dim=-1))
        
        if self.training:
            logits += torch.randn_like(logits) * self.noise_std

        top_k_logits, selected_indices = torch.topk(logits, self.k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, selected_indices, top_k_logits)
        gating_weights = F.softmax(mask, dim=-1)

        return gating_weights, selected_indices, self.mu