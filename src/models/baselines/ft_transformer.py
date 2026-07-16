"""Canonical FT-Transformer baseline with independent dropout controls."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import categorical_cardinalities, resolve_width


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.missing_weight = nn.Parameter(torch.empty(n_features, d_token))
        bound = d_token ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.missing_weight, -bound, bound)

    def forward(self, values, missing):
        return (
            values.unsqueeze(-1) * self.weight
            + self.bias
            + missing.unsqueeze(-1) * self.missing_weight
        )


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities, d_token):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cardinalities]
        )
        self.bias = nn.Parameter(torch.empty(len(cardinalities), d_token))
        bound = d_token ** -0.5
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values):
        tokens = torch.stack(
            [embedding(values[:, i]) for i, embedding in enumerate(self.embeddings)],
            dim=1,
        )
        return tokens + self.bias


class MultiheadAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout):
        super().__init__()
        if d_token % n_heads:
            raise ValueError("FT d_token must be divisible by n_heads.")
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_token, d_token)
        self.W_k = nn.Linear(d_token, d_token)
        self.W_v = nn.Linear(d_token, d_token)
        self.W_out = nn.Linear(d_token, d_token) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout)
        for layer in (self.W_q, self.W_k, self.W_v, self.W_out):
            if layer is not None:
                nn.init.zeros_(layer.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d_token = x.shape
        d_head = d_token // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv):
        q = self._reshape(self.W_q(x_q))
        k = self._reshape(self.W_k(x_kv))
        v = self._reshape(self.W_v(x_kv))
        probabilities = F.softmax(q @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)
        probabilities = self.dropout(probabilities)
        x = probabilities @ v
        batch_size = x_q.shape[0]
        x = (
            x.reshape(batch_size, self.n_heads, x_q.shape[1], -1)
            .transpose(1, 2)
            .reshape(batch_size, x_q.shape[1], -1)
        )
        return self.W_out(x) if self.W_out is not None else x


class ReGLU(nn.Module):
    def forward(self, x):
        left, gate = x.chunk(2, dim=-1)
        return left * F.relu(gate)


class FTBlock(nn.Module):
    def __init__(
        self,
        d_token,
        n_heads,
        d_hidden,
        attention_dropout,
        ffn_dropout,
        residual_dropout,
        first,
    ):
        super().__init__()
        self.attention_normalization = None if first else nn.LayerNorm(d_token)
        self.attention = MultiheadAttention(d_token, n_heads, attention_dropout)
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_normalization = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 2 * d_hidden),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_hidden, d_token),
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x, query_cls_only=False):
        identity = x
        attention_input = (
            x if self.attention_normalization is None else self.attention_normalization(x)
        )
        query = attention_input[:, :1] if query_cls_only else attention_input
        x = identity + self.attention_residual_dropout(
            self.attention(query, attention_input)
        )
        identity = x
        x = identity + self.ffn_residual_dropout(
            self.ffn(self.ffn_normalization(x))
        )
        return x


class FTTransformerWrapper(nn.Module):
    """RTDL FT-Transformer: PreNorm, ReGLU, CLS head, and three dropouts."""

    def __init__(self, config):
        super().__init__()
        d_token = resolve_width(config)
        n_heads = int(config.n_heads)
        if n_heads != 8:
            raise ValueError("Phase 2 FT-Transformer fixes n_heads=8.")
        n_layers = int(config.n_layers)
        if n_layers <= 0:
            raise ValueError("FT n_layers must be positive.")
        d_hidden = int(d_token * float(config.d_ff_factor))
        if d_hidden <= 0:
            raise ValueError("FT feed-forward width must be positive.")

        self.numerical_tokenizer = (
            NumericalFeatureTokenizer(config.n_numerical, d_token)
            if config.n_numerical
            else None
        )
        cardinalities = categorical_cardinalities(config)
        self.categorical_tokenizer = (
            CategoricalFeatureTokenizer(cardinalities, d_token)
            if cardinalities
            else None
        )
        self.cls_token = nn.Parameter(torch.empty(d_token))
        nn.init.uniform_(self.cls_token, -d_token ** -0.5, d_token ** -0.5)
        self.blocks = nn.ModuleList(
            [
                FTBlock(
                    d_token,
                    n_heads,
                    d_hidden,
                    float(config.attention_dropout),
                    float(config.ffn_dropout),
                    float(config.residual_dropout),
                    first=index == 0,
                )
                for index in range(n_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, config.out_dim),
        )

    def forward(
        self,
        x_numerical_values,
        x_numerical_missing,
        x_categorical_idx,
        **kwargs,
    ):
        tokens = [self.cls_token.expand(x_numerical_values.shape[0], 1, -1)]
        if self.numerical_tokenizer is not None:
            tokens.append(
                self.numerical_tokenizer(
                    x_numerical_values,
                    x_numerical_missing,
                )
            )
        if self.categorical_tokenizer is not None:
            tokens.append(self.categorical_tokenizer(x_categorical_idx.long()))
        x = torch.cat(tokens, dim=1)
        for index, block in enumerate(self.blocks):
            x = block(x, query_cls_only=index + 1 == len(self.blocks))
        return {"logits": self.output(x[:, 0]), "aux_loss": None}
