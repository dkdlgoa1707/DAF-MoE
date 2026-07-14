import torch
import torch.nn as nn
import torch.nn.functional as F


class DAFRouterV2(nn.Module):
    """Distribution-guided gating with FiLM-based conditioning."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_emb = config.d_emb
        self.n_experts = config.n_experts
        self.k = config.top_k
        self.noise_std = config.router_noise_std

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

        hidden = max(1, self.d_emb // 4)
        self.film_generator = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.d_emb),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.gate = nn.Linear(self.d_emb, self.n_experts)

    def forward(self, h, psi_x):
        h_stopped = h.detach()

        film_input = psi_x if self.config.use_dist_token else torch.zeros_like(psi_x)
        gamma, beta = self.film_generator(film_input).chunk(2, dim=-1)
        router_input = (1 + gamma) * h_stopped + beta

        logits = self.gate(router_input)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        _, selected_indices = torch.topk(logits, self.k, dim=-1)
        top_k_logits = logits.gather(-1, selected_indices)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, selected_indices, top_k_logits)
        gating_weights = F.softmax(mask, dim=-1)

        return gating_weights, selected_indices, self.mu
