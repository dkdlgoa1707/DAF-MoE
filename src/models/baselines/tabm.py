import torch
import torch.nn as nn

from .realmlp import _PLREmbedding


class LinearBatchEnsemble(nn.Module):
    """
    BatchEnsemble Linear Layer (Wen et al., ICLR 2020).

    Implements k implicit ensemble members sharing one weight matrix W,
    each with independent rank-1 adapters (r_i, s_i) and bias b_i:

        l_i(x) = s_i ⊙ W(r_i ⊙ x) + b_i

    Vectorized over k members simultaneously:
        Y = (X_expanded ⊙ R) @ W.T  ⊙ S + B

    where:
        R  [k, in_features]  — per-member input scaling
        S  [k, out_features] — per-member output scaling
        B  [k, out_features] — per-member bias
        W  [out_features, in_features] — shared weight (single copy)

    Input:  [B, k, in_features]
    Output: [B, k, out_features]
    """

    def __init__(self, in_features: int, out_features: int, k: int, is_first=False):
        super().__init__()
        self.k = k
        self.is_first = is_first
        # Shared weight matrix (one copy for all k members)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Per-member rank-1 adapters
        self.r = nn.Parameter(torch.empty(k, in_features))   # input scale
        self.s = nn.Parameter(torch.empty(k, out_features))  # output scale
        self.b = nn.Parameter(torch.zeros(k, out_features))  # per-member bias

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.is_first:
            nn.init.normal_(self.r, mean=0.0, std=0.5)
            nn.init.normal_(self.s, mean=0.0, std=0.5)
        else:
            nn.init.ones_(self.r)
            nn.init.ones_(self.s)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, k, in_features]
        # Step 1: scale input by r_i for each member
        x_scaled = x * self.r.unsqueeze(0)           # [B, k, in_features]

        # Step 2: apply shared W — reshape to [B*k, in] for linear op
        B = x.shape[0]
        x_flat = x_scaled.reshape(B * self.k, -1)    # [B*k, in_features]
        out = x_flat @ self.weight.t()                # [B*k, out_features]
        out = out.reshape(B, self.k, -1)              # [B, k, out_features]

        # Step 3: scale output by s_i and add per-member bias
        out = out * self.s.unsqueeze(0) + self.b.unsqueeze(0)
        return out


class TabMWrapper(nn.Module):
    """
    TabM Baseline (Gorishniy et al., ICLR 2025).

    MLP backbone where every Linear layer is replaced with LinearBatchEnsemble,
    producing k implicit submodels in a single forward pass. Final logits are
    the mean of all k predictions (ensemble averaging).

    Paper: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
            Ensembling" (https://arxiv.org/abs/2410.24210)

    Mapping to paper concepts:
        LinearBatchEnsemble  — "BatchEnsemble" layers (Section 3)
        k                    — ensemble size (number of implicit members)
        mean over k dim      — ensemble prediction averaging (Section 3.3)
    """

    def __init__(self, config):
        super().__init__()
        # d_token (HPO alias) takes precedence over d_emb (default), matching other baselines
        self.d_emb = config.d_token if config.d_token is not None else config.d_emb
        self.k = getattr(config, 'k', 32)

        # ── Feature Embedding (identical to mlp.py) ──────────────────────────
        # PLR numerical embeddings used by the recommended TabM setup.
        self.num_embedding = (
            _PLREmbedding(config.n_numerical, out_dim=self.d_emb)
            if config.n_numerical
            else None
        )
        # Categorical features: index → d_emb (shared embedding table)
        self.cat_embed = nn.Embedding(config.total_cats + 1, self.d_emb)

        # ── BatchEnsemble MLP Body ────────────────────────────────────────────
        # Replaces every Linear(d, d) in the MLP with LinearBatchEnsemble.
        input_dim = (config.n_numerical + config.n_categorical) * self.d_emb

        self.input_proj = LinearBatchEnsemble(input_dim, self.d_emb, self.k, is_first=True)

        self.layers = nn.ModuleList([
            LinearBatchEnsemble(self.d_emb, self.d_emb, self.k, is_first=False)
            for _ in range(config.n_layers - 1)
            ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        # ── Head ─────────────────────────────────────────────────────────────
        # Final projection is also a BatchEnsemble layer so each member gets
        # its own output weights, then predictions are averaged (Section 3.3).
        self.head = LinearBatchEnsemble(self.d_emb, config.out_dim, self.k, is_first=False)

    def forward(self, x_numerical, x_categorical_idx, **kwargs):
        B = x_numerical.shape[0]

        # 1. Embed features (shared across all k members, same as mlp.py)
        x_num_val = x_numerical[:, :, 0]  # drop P and gamma channels
        x_n_emb = (
            self.num_embedding(x_num_val)
            if self.num_embedding is not None
            else x_numerical.new_empty(B, 0, self.d_emb)
        )
        x_c_emb = self.cat_embed(x_categorical_idx.long())  # [B, n_categorical, d_emb]

        # 2. Flatten into single vector and expand across k members
        x = torch.cat([x_n_emb, x_c_emb], dim=1).view(B, -1)  # [B, input_dim]
        x = x.unsqueeze(1).expand(-1, self.k, -1)               # [B, k, input_dim]

        # 3. BatchEnsemble MLP forward
        x = self.input_proj(x)                  # [B, k, d_emb]
        x = self.activation(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)                        # [B, k, d_emb]
            x = self.activation(x)
            x = self.dropout(x)

        # 4. Per-member logits then average across k (ensemble prediction)
        logits_k = self.head(x)                 # [B, k, out_dim]
        logits = logits_k.mean(dim=1)           # [B, out_dim]

        # logits_k exposed so trainer can compute mean loss (not loss of mean) for classification
        return {"logits": logits, "logits_k": logits_k, "aux_loss": None}
