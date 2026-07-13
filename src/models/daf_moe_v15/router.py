import torch
import torch.nn as nn
import torch.nn.functional as F


class DAFRouterV15(nn.Module):
    """
    DAF-MoE v1.5 router.

    Flag-off behavior mirrors the v1 router, except for the Phase 0 linspace
    centroid fix used by v1.5 variants.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_emb = config.d_emb
        self.n_experts = config.n_experts
        self.k = config.top_k
        self.noise_std = config.router_noise_std
        self.use_film_gating = getattr(config, 'use_film_gating', False)
        self.use_loss_free_balancing = getattr(config, 'use_loss_free_balancing', False)

        strategy = getattr(config, 'mu_init_strategy', 'linspace')

        if strategy == 'custom' and hasattr(config, 'initial_mu'):
            init_mu = torch.tensor(config.initial_mu)
        elif strategy == 'random':
            init_mu = torch.rand(self.n_experts)
        elif strategy == 'normal':
            init_mu = torch.randn(self.n_experts)
        else:
            init_mu = torch.linspace(-3, 3, self.n_experts)

        self.mu = nn.Parameter(init_mu)

        if self.use_film_gating:
            hidden = max(1, self.d_emb // 4)
            self.film_generator = nn.Sequential(
                nn.Linear(2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * self.d_emb),
            )
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
            self.gate = nn.Linear(self.d_emb, self.n_experts)
        else:
            self.meta_proj = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 8)
            )
            self.gate = nn.Linear(self.d_emb + 8, self.n_experts)

        if self.use_loss_free_balancing:
            self.register_buffer('expert_bias', torch.zeros(self.n_experts))

    def forward(self, h, psi_x):
        h_stopped = h.detach()

        if self.use_film_gating:
            film_input = psi_x if self.config.use_dist_token else torch.zeros_like(psi_x)
            gamma, beta = self.film_generator(film_input).chunk(2, dim=-1)
            router_input = (1 + gamma) * h_stopped + beta
        else:
            if self.config.use_dist_token:
                m_emb = self.meta_proj(psi_x)
                router_input = torch.cat([h_stopped, m_emb], dim=-1)
            else:
                dummy_meta = torch.zeros(h.size(0), h.size(1), 8, device=h.device)
                router_input = torch.cat([h_stopped, dummy_meta], dim=-1)

        logits = self.gate(router_input)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        selection_logits = logits
        if self.use_loss_free_balancing:
            selection_logits = logits + self.expert_bias.view(1, 1, -1)

        _, selected_indices = torch.topk(selection_logits, self.k, dim=-1)
        top_k_logits = logits.gather(-1, selected_indices)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, selected_indices, top_k_logits)
        gating_weights = F.softmax(mask, dim=-1)

        if self.training and self.use_loss_free_balancing:
            with torch.no_grad():
                selected_flat = selected_indices.reshape(-1)
                counts = torch.bincount(selected_flat, minlength=self.n_experts).to(logits.dtype)
                total = counts.sum().clamp_min(1.0)
                f_k = counts / total
                target = torch.full_like(f_k, 1.0 / self.n_experts)
                self.expert_bias += self.config.bias_update_rate * torch.sign(target - f_k)

        return gating_weights, selected_indices, self.mu
