import torch
import torch.nn as nn

class DAFExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 하드코딩 제거: config.d_emb, config.d_ff 사용
        self.ffn = nn.Sequential(
            nn.Linear(config.d_emb, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_emb)
        )
        
        # Lite Version: alpha, beta 제거됨
        self.magnitude_bp = nn.Linear(1, config.d_emb)
        self.identity_bp = nn.Embedding(config.total_cats, config.d_emb)
        
        # 학습 가능 게이트 파라미터 (초기값은 경험적 고정값 유지)
        self.gate_steepness = nn.Parameter(torch.tensor(10.0))
        self.gate_margin = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, mu_j_raw, raw_val, feature_mask_subset):
        out = self.ffn(x)
        if raw_val is None: return out

        # [CRITICAL FIX] Raw Logit -> Sigmoid -> 거리 계산
        mu_prob = torch.sigmoid(mu_j_raw)
        dist = torch.abs(mu_prob - 0.5)
        lambda_j = torch.sigmoid(self.gate_steepness * (dist - self.gate_margin))
        
        is_num = (feature_mask_subset == 1)
        is_cat = ~is_num

        if is_num.any():
            bp_num = torch.tanh(self.magnitude_bp(raw_val[is_num]))
            out[is_num] = out[is_num] + (lambda_j * bp_num)

        if is_cat.any():
            # squeeze(-1) 처리로 차원 맞춤
            bp_cat = self.identity_bp(raw_val[is_cat].long().squeeze(-1))
            out[is_cat] = out[is_cat] + (lambda_j * bp_cat)
            
        return out

class DAFMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.experts = nn.ModuleList([DAFExpert(config) for _ in range(self.n_experts)])

    def forward(self, h, gating_weights, selected_indices, mu_raw_vector, raw_values, feature_mask):
        from einops import rearrange
        B, L, D = h.shape
        combined_output = torch.zeros_like(h)
        
        flat_h = rearrange(h, 'b l d -> (b l) d')
        flat_raw = rearrange(raw_values, 'b l c -> (b l) c')
        flat_mask = rearrange(feature_mask, 'b l -> (b l)')

        for k in range(selected_indices.shape[-1]):
            indices = rearrange(selected_indices[:, :, k], 'b l -> (b l)')
            weights = gating_weights.gather(-1, selected_indices[:, :, k:k+1]).view(-1, 1)
            
            for i, expert in enumerate(self.experts):
                token_indices = (indices == i).nonzero(as_tuple=True)[0]
                if token_indices.numel() == 0: continue
                
                exp_out = expert(
                    flat_h[token_indices], 
                    mu_raw_vector[i], 
                    flat_raw[token_indices], 
                    feature_mask_subset=flat_mask[token_indices]
                )
                combined_output.view(-1, D).index_add_(0, token_indices, exp_out * weights[token_indices])
                
        return combined_output