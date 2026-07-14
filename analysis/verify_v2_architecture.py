"""Verify DAF-MoE v2 construction, forward pass, and architecture choices."""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.data.ple_utils import compute_ple_boundaries
from src.models.daf_moe_v2.daf_moe_transformer import DAFMoETransformerV2
from src.models.daf_moe_v2.router import DAFRouterV2
from src.models.factory import create_model


DATASETS = ['california', 'adult', 'mimic4']
SEED = 42


def make_synthetic_config(dataset):
    """Load a base config and populate dimensions for a synthetic smoke test."""
    with open(
        f'configs/experiments/{dataset}_daf_moe_best.yaml',
        encoding='utf-8',
    ) as source:
        experiment = yaml.safe_load(source)
    with open(experiment['data_config_path'], encoding='utf-8') as source:
        data = yaml.safe_load(source)

    config = DAFConfig()
    for key, value in experiment.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.model_name = 'daf_moe_v2'
    config.n_numerical = len(data.get('num_cols', []))
    config.n_categorical = len(data.get('cat_cols', []))
    config.n_features = config.n_numerical + config.n_categorical
    config.total_cats = max(2, 3 * config.n_categorical) if config.n_categorical else 0
    config.d_ff = int(config.d_emb * config.d_ff_factor)

    rng = np.random.RandomState(SEED)
    x_num = rng.randn(1000, config.n_numerical)
    config.ple_boundaries = compute_ple_boundaries(x_num, config.ple_n_bins)
    return config


def synthetic_inputs(config, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    batch_size = 4
    x_num = torch.randn(
        batch_size, config.n_numerical, 3, generator=generator
    )
    x_num[:, :, 1] = torch.rand(
        batch_size, config.n_numerical, generator=generator
    )
    if config.n_categorical:
        x_cat_idx = torch.randint(
            0,
            config.total_cats,
            (batch_size, config.n_categorical),
            generator=generator,
        )
        x_cat_meta = torch.rand(
            batch_size, config.n_categorical, 2, generator=generator
        )
    else:
        x_cat_idx = torch.empty(batch_size, 0, dtype=torch.long)
        x_cat_meta = torch.empty(batch_size, 0, 2)
    return x_num, x_cat_idx, x_cat_meta


def component_param_breakdown(model):
    """Return parameter counts grouped by the first two name components."""
    breakdown = {}
    for name, parameter in model.named_parameters():
        parts = name.split('.')
        top_key = parts[0] + ('.' + parts[1] if len(parts) > 1 else '')
        breakdown[top_key] = breakdown.get(top_key, 0) + parameter.numel()
    return breakdown


def main():
    print('=' * 60)
    print('DAF-MoE v2 Architecture Verification')
    print('=' * 60)

    linspace_config = make_synthetic_config(DATASETS[0])
    linspace_config.mu_init_strategy = 'linspace'
    router = DAFRouterV2(linspace_config)
    expected_mu = torch.linspace(-3, 3, linspace_config.n_experts)
    assert torch.equal(router.mu.detach(), expected_mu)
    print('[PASS] linspace mu initialization covers [-3, 3]')

    for dataset in DATASETS:
        print(f'\n[{dataset}]')
        torch.manual_seed(SEED)
        config = make_synthetic_config(dataset)
        model = create_model(config).eval()
        assert isinstance(model, DAFMoETransformerV2)

        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(f'  Total parameters: {total_params:,}')
        print(
            f'  n_numerical={config.n_numerical}, '
            f'n_categorical={config.n_categorical}, d_emb={config.d_emb}, '
            f'n_experts={config.n_experts}'
        )
        print('  Component breakdown:')
        for component, count in component_param_breakdown(model).items():
            print(f'    {component}: {count:,}')

        inputs = synthetic_inputs(config)
        with torch.no_grad():
            output = model(*inputs)
        assert output['logits'].shape == (4, config.out_dim)
        print(f"  Forward pass OK. logits shape: {output['logits'].shape}")

        parameter_names = [name for name, _ in model.named_parameters()]
        assert not any('categorical_meta_proj' in name for name in parameter_names), (
            'Option B violated: categorical_meta_proj exists in v2'
        )
        print('  [PASS] Option B: categorical_meta_proj absent')

        assert any('omega_shared' in name for name in parameter_names), (
            'omega_shared missing'
        )
        assert not any(
            legacy_name in name
            for name in parameter_names
            for legacy_name in ('omega_num_w', 'omega_num_b', 'omega_cat_emb')
        ), 'Lightweight preservation violated: legacy omega params found'
        print('  [PASS] Lightweight preservation')

        assert any('film_generator' in name for name in parameter_names), (
            'film_generator missing'
        )
        assert not any('meta_proj' in name for name in parameter_names), (
            'FiLM violated: legacy meta_proj found'
        )
        print('  [PASS] FiLM gating')

        buffer_names = [name for name, _ in model.named_buffers()]
        assert not any('expert_bias' in name for name in buffer_names), (
            'Loss-free unexpectedly adopted: expert_bias buffer found'
        )
        print('  [PASS] Loss-free balancing not adopted')

    print('\n' + '=' * 60)
    print('ALL v2 ARCHITECTURE CHECKS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
