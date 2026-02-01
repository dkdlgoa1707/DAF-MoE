import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFRouter(nn.Module):
    """
    Distribution-Guided Gating (DGG) Mechanism.
    
    Computes gating scores based on feature distribution metadata (Psi(x)).
    Section 3.3.1 in the paper.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_emb = config.d_emb
        self.n_experts = config.n_experts
        self.k = config.top_k
        self.noise_std = config.router_noise_std

        # Learnable Centroids mu_k (Section 3.5.1)
        strategy = getattr(config, 'mu_init_strategy', 'linspace')
        
        if strategy == 'custom' and hasattr(config, 'initial_mu'):
            init_mu = torch.tensor(config.initial_mu)
        elif strategy == 'random':
            init_mu = torch.rand(self.n_experts)
        elif strategy == 'normal':
            init_mu = torch.randn(self.n_experts)
        else: # 'linspace' (Default)
            init_mu = torch.linspace(0, 1, self.n_experts)

        self.mu = nn.Parameter(init_mu) 

        # Projection for Metadata m_j
        self.meta_proj = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 8)
        )
        self.gate = nn.Linear(self.d_emb + 8, self.n_experts)

    def forward(self, h, psi_x):
        """
        Args:
            h (Tensor): Feature representation (semantic).
            psi_x (Tensor): Distributional position metadata (Psi(x)).
        Returns:
            gating_weights: Softmax scores.
            selected_indices: Indices of top-k experts.
            mu: Centroid parameters.
        """
        # Equation 5: q_j = [sg(h) || Proj(m_j)]
        h_stopped = h.detach() # Stop-Gradient to prevent collapse
        
        if self.config.use_dist_token:
            m_emb = self.meta_proj(psi_x)
            router_input = torch.cat([h_stopped, m_emb], dim=-1)
        else:
            # Ablation fallback
            dummy_meta = torch.zeros(h.size(0), h.size(1), 8, device=h.device)
            router_input = torch.cat([h_stopped, dummy_meta], dim=-1)

        # Equation 7: TopK Gating
        logits = self.gate(router_input)
        
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std

        top_k_logits, selected_indices = torch.topk(logits, self.k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, selected_indices, top_k_logits)
        gating_weights = F.softmax(mask, dim=-1)

        return gating_weights, selected_indices, self.mu