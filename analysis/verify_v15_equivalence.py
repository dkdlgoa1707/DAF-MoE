"""Verify v1.5 flag-off equivalence for all Phase 1 datasets."""

import copy
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.models.daf_moe.daf_moe_transformer import DAFMoETransformer
from src.models.daf_moe_v15.daf_moe_transformer import DAFMoETransformerV15


DATASETS = ['california', 'adult', 'mimic4']
SEED = 42
ATOL = 1e-5
RTOL = 1e-4
REPORT_PATH = Path("results/phase1_v15/equivalence_report.md")


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as source:
        return yaml.safe_load(source)


def make_config(dataset, strategy):
    experiment = load_yaml(f"configs/experiments/{dataset}_daf_moe_best.yaml")
    data = load_yaml(experiment['data_config_path'])
    config = DAFConfig()
    for key, value in experiment.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.seed = SEED
    config.mu_init_strategy = strategy
    config.n_numerical = len(data.get('num_cols', []))
    config.n_categorical = len(data.get('cat_cols', []))
    config.n_features = config.n_numerical + config.n_categorical
    config.total_cats = max(2, 3 * config.n_categorical) if config.n_categorical else 0
    config.dropout = 0.0
    config.router_noise_std = 0.0
    config.use_loss_free_balancing = False
    config.use_film_gating = False
    config.use_lightweight_preservation = False
    config.use_ple_embedding = False
    return config


def make_inputs(config, input_seed):
    generator = torch.Generator().manual_seed(input_seed)
    batch_size = 4
    x_numerical = torch.randn(batch_size, config.n_numerical, 3, generator=generator)
    x_numerical[:, :, 1] = torch.rand(
        batch_size, config.n_numerical, generator=generator
    )
    if config.n_categorical:
        x_categorical_idx = torch.randint(
            0,
            config.total_cats,
            (batch_size, config.n_categorical),
            generator=generator,
        )
        x_categorical_meta = torch.rand(
            batch_size, config.n_categorical, 2, generator=generator
        )
    else:
        x_categorical_idx = torch.empty(batch_size, 0, dtype=torch.long)
        x_categorical_meta = torch.empty(batch_size, 0, 2)
    return x_numerical, x_categorical_idx, x_categorical_meta


def build_pair(config):
    torch.manual_seed(SEED)
    model_v1 = DAFMoETransformer(copy.deepcopy(config)).eval()
    torch.manual_seed(SEED)
    model_v15 = DAFMoETransformerV15(copy.deepcopy(config)).eval()
    return model_v1, model_v15


def logits(model, inputs):
    with torch.no_grad():
        return model(*inputs)['logits']


def max_diff(left, right):
    return float((left - right).abs().max().item())


def verify_non_linspace(dataset, strategy):
    config = make_config(dataset, strategy)
    model_v1, model_v15 = build_pair(config)
    inputs = make_inputs(config, SEED + 1)
    output_v1 = logits(model_v1, inputs)
    output_v15 = logits(model_v15, inputs)
    difference = max_diff(output_v1, output_v15)
    passed = torch.allclose(output_v1, output_v15, atol=ATOL, rtol=RTOL)
    return passed, difference


def verify_linspace(dataset):
    config = make_config(dataset, 'linspace')
    model_v1, model_v15 = build_pair(config)
    inputs = make_inputs(config, SEED + 1)

    output_v1 = logits(model_v1, inputs)
    output_changed = logits(model_v15, inputs)
    changed_diff = max_diff(output_v1, output_changed)
    changed = not torch.allclose(output_v1, output_changed, atol=ATOL, rtol=RTOL)

    with torch.no_grad():
        for block_v1, block_v15 in zip(model_v1.blocks, model_v15.blocks):
            block_v15.router.mu.copy_(block_v1.router.mu)
    output_restored = logits(model_v15, inputs)
    restored_diff = max_diff(output_v1, output_restored)
    restored = torch.allclose(output_v1, output_restored, atol=ATOL, rtol=RTOL)
    return changed and restored, changed_diff, restored_diff


def main():
    rows = []
    all_passed = True
    for dataset in DATASETS:
        experiment = load_yaml(f"configs/experiments/{dataset}_daf_moe_best.yaml")
        native_strategy = experiment.get('mu_init_strategy', 'linspace')
        scenario_a_strategy = native_strategy if native_strategy != 'linspace' else 'normal'

        passed_a, diff_a = verify_non_linspace(dataset, scenario_a_strategy)
        passed_b, changed_diff, restored_diff = verify_linspace(dataset)
        all_passed = all_passed and passed_a and passed_b
        rows.append(
            (dataset, scenario_a_strategy, passed_a, diff_a, passed_b, changed_diff, restored_diff)
        )
        print(
            f"{dataset}: A={passed_a} (strategy={scenario_a_strategy}, diff={diff_a:.8g}) | "
            f"B={passed_b} (changed={changed_diff:.8g}, restored={restored_diff:.8g})"
        )

    report = [
        "# DAF-MoE v1.5 Equivalence Report",
        "",
        f"Seed: {SEED}; tolerance: atol={ATOL}, rtol={RTOL}",
        "",
        "| Dataset | Scenario A strategy | A pass | A max diff | B pass | B changed diff | B restored diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        dataset, strategy, passed_a, diff_a, passed_b, changed_diff, restored_diff = row
        report.append(
            f"| {dataset} | {strategy} | {passed_a} | {diff_a:.8g} | "
            f"{passed_b} | {changed_diff:.8g} | {restored_diff:.8g} |"
        )
    report.extend([
        "",
        "Scenario A requires full flag-off equivalence for a non-linspace strategy.",
        "Scenario B requires natural linspace outputs to differ and mu-restored outputs to match.",
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding='utf-8')
    print(f"Report: {REPORT_PATH}")

    if not all_passed:
        raise SystemExit("v1.5 equivalence verification failed.")


if __name__ == "__main__":
    main()
