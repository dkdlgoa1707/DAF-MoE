"""
Regression check: v15 with all Phase 1 flags off should match v1.

This uses a non-linspace mu initialization because v1.5 intentionally changes
linspace centroids from [0, 1] to [-3, 3] as a Phase 0 fix.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.models.daf_moe.daf_moe_transformer import DAFMoETransformer
from src.models.daf_moe_v15.daf_moe_transformer import DAFMoETransformerV15


def make_config():
    config = DAFConfig()
    config.model_name = "daf_moe"
    config.n_numerical = 3
    config.n_categorical = 2
    config.n_features = 5
    config.total_cats = 7
    config.task_type = "classification"
    config.out_dim = 1
    config.d_emb = 16
    config.d_ff_factor = 2.0
    config.n_heads = 4
    config.n_layers = 2
    config.n_experts = 4
    config.top_k = 2
    config.dropout = 0.0
    config.router_noise_std = 0.0
    config.mu_init_strategy = "normal"
    config.use_loss_free_balancing = False
    config.use_film_gating = False
    config.use_lightweight_preservation = False
    config.use_ple_embedding = False
    return config


def main():
    seed = 1234
    config_v1 = make_config()
    torch.manual_seed(seed)
    model_v1 = DAFMoETransformer(config_v1).eval()

    config_v15 = make_config()
    torch.manual_seed(seed)
    model_v15 = DAFMoETransformerV15(config_v15).eval()

    torch.manual_seed(seed + 1)
    batch_size = 8
    x_numerical = torch.randn(batch_size, config_v1.n_numerical, 3)
    x_numerical[:, :, 1] = torch.rand(batch_size, config_v1.n_numerical)
    x_categorical_idx = torch.randint(0, config_v1.total_cats, (batch_size, config_v1.n_categorical))
    x_categorical_meta = torch.rand(batch_size, config_v1.n_categorical, 2)

    with torch.no_grad():
        out_v1 = model_v1(x_numerical, x_categorical_idx, x_categorical_meta)
        out_v15 = model_v15(x_numerical, x_categorical_idx, x_categorical_meta)

    logits_v1 = out_v1["logits"]
    logits_v15 = out_v15["logits"]
    max_abs_diff = (logits_v1 - logits_v15).abs().max().item()

    if not torch.allclose(logits_v1, logits_v15, rtol=1e-4, atol=1e-6):
        raise AssertionError(f"v15 equivalence failed. max_abs_diff={max_abs_diff:.8f}")

    print(f"v15 equivalence passed. max_abs_diff={max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
