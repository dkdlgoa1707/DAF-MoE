"""Aggregate DAF-MoE v1.5 Phase 1 results into a Markdown report."""

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.data.loader import get_dataloaders
from src.models.factory import create_model


VARIANTS = [f"M{idx}" for idx in range(7)]
PAIRS = [
    ("M0", "M1", "Preservation only"),
    ("M0", "M2", "Loss-free router only"),
    ("M2", "M3", "FiLM addition"),
    ("M3", "M4", "Preservation on revised router"),
    ("M4", "M5", "PLE on combined changes"),
    ("M0", "M6", "PLE only"),
]
MINIMIZE_METRICS = {'loss', 'rmse', 'mse', 'mae'}


def parse_config_value(value):
    if not isinstance(value, str):
        return value
    if value == 'None':
        return None
    if value in {'True', 'False'}:
        return value == 'True'
    if value.startswith('[') or value.startswith('{'):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def config_from_result(result):
    config = DAFConfig()
    for key, value in result.get('config', {}).items():
        if hasattr(config, key):
            setattr(config, key, parse_config_value(value))
    return config


def infer_variant(result):
    model = result.get('model', '')
    config = result.get('config', {})
    flag = lambda name: parse_config_value(config.get(name, False))
    if not model.startswith('daf_moe_v15'):
        return 'M0'
    if flag('use_ple_embedding'):
        return 'M5' if flag('use_lightweight_preservation') else 'M6'
    if flag('use_lightweight_preservation') and flag('use_film_gating'):
        return 'M4'
    if flag('use_film_gating'):
        return 'M3'
    if flag('use_loss_free_balancing'):
        return 'M2'
    if flag('use_lightweight_preservation'):
        return 'M1'
    return 'unknown'


def target_metric(result):
    config = result.get('config', {})
    metrics = result.get('metrics', {})
    task_type = parse_config_value(config.get('task_type', 'classification'))
    metric = parse_config_value(config.get('optimize_metric'))
    metric = metric or ('rmse' if task_type == 'regression' else 'acc')
    return metric, metrics.get(metric)


def load_results(result_dir):
    root = Path(result_dir)
    grouped = defaultdict(list)
    for path in root.rglob('*.json'):
        with open(path, 'r', encoding='utf-8') as source:
            result = json.load(source)
        relative_parts = path.relative_to(root).parts
        variant = next((part for part in relative_parts if part in VARIANTS), None)
        variant = variant or infer_variant(result)
        dataset = result.get('dataset', 'unknown')
        result['_path'] = str(path)
        grouped[(dataset, variant)].append(result)
    return grouped


def metric_table(grouped):
    lines = [
        "| Dataset | Variant | Metric | Mean | Std | Seeds |",
        "|---|---|---|---:|---:|---:|",
    ]
    stats = {}
    for key in sorted(grouped):
        values_by_seed = {}
        metric_name = None
        for result in grouped[key]:
            metric_name, value = target_metric(result)
            if value is not None:
                values_by_seed[int(result['seed'])] = float(value)
        if not values_by_seed:
            continue
        values = list(values_by_seed.values())
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats[key] = {
            'metric': metric_name,
            'values_by_seed': values_by_seed,
            'mean': mean,
            'std': std,
        }
        lines.append(
            f"| {key[0]} | {key[1]} | {metric_name} | {mean:.6f} | {std:.6f} | {len(values)} |"
        )
    return lines, stats


def paired_statistics(left, right):
    common_seeds = sorted(set(left['values_by_seed']) & set(right['values_by_seed']))
    if not common_seeds:
        return None, None, 0
    differences = np.array([
        right['values_by_seed'][seed] - left['values_by_seed'][seed]
        for seed in common_seeds
    ])
    mean_diff = float(differences.mean())
    if len(differences) < 2:
        return mean_diff, None, len(differences)
    std = differences.std(ddof=1)
    if std == 0:
        t_like = math.inf if mean_diff else 0.0
    else:
        t_like = float(mean_diff / (std / math.sqrt(len(differences))))
    return mean_diff, t_like, len(differences)


def effect_direction(metric, difference):
    if abs(difference) < 1e-12:
        return 'flat'
    improved = difference < 0 if metric in MINIMIZE_METRICS else difference > 0
    return 'improved' if improved else 'degraded'


def comparison_table(stats):
    lines = [
        "| Dataset | Pair | Purpose | Mean diff (right-left) | Direction | t-like | Paired seeds |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    datasets = sorted({dataset for dataset, _ in stats})
    for dataset in datasets:
        for left_name, right_name, purpose in PAIRS:
            left = stats.get((dataset, left_name))
            right = stats.get((dataset, right_name))
            if not left or not right:
                continue
            difference, t_like, count = paired_statistics(left, right)
            t_text = '' if t_like is None else f"{t_like:.3f}"
            direction = effect_direction(left['metric'], difference)
            lines.append(
                f"| {dataset} | {left_name} vs {right_name} | {purpose} | "
                f"{difference:.6f} | {direction} | {t_text} | {count} |"
            )
    return lines


def independence_table(stats):
    lines = [
        "| Dataset | Check | First effect | Second effect | Direction aligned |",
        "|---|---|---:|---:|---|",
    ]
    checks = [
        ('A', ('M0', 'M1'), ('M3', 'M4')),
        ('B', ('M0', 'M6'), ('M4', 'M5')),
    ]
    datasets = sorted({dataset for dataset, _ in stats})
    for dataset in datasets:
        for label, first_pair, second_pair in checks:
            entries = [stats.get((dataset, variant)) for variant in (*first_pair, *second_pair)]
            if not all(entries):
                continue
            first_diff, _, _ = paired_statistics(entries[0], entries[1])
            second_diff, _, _ = paired_statistics(entries[2], entries[3])
            aligned = np.sign(first_diff) == np.sign(second_diff)
            lines.append(
                f"| {dataset} | {label} | {first_diff:.6f} | {second_diff:.6f} | {aligned} |"
            )
    return lines


def parameter_table(grouped):
    lines = [
        "| Dataset | Variant | Parameters | Delta vs M0 |",
        "|---|---|---:|---:|",
    ]
    for dataset in sorted({dataset for dataset, _ in grouped}):
        counts = {}
        for variant in ('M0', 'M1', 'M4', 'M6'):
            results = grouped.get((dataset, variant), [])
            if not results:
                continue
            config = config_from_result(results[0])
            with torch.no_grad():
                counts[variant] = sum(p.numel() for p in create_model(config).parameters())
        baseline = counts.get('M0')
        for variant in ('M0', 'M1', 'M4', 'M6'):
            if variant not in counts:
                continue
            delta = counts[variant] - baseline if baseline is not None else 0
            lines.append(f"| {dataset} | {variant} | {counts[variant]} | {delta:+d} |")
    return lines


def expert_utilization_table(grouped, result_dir):
    lines = [
        "| Dataset | Variant | Expert | Selection ratio | Checkpoints |",
        "|---|---|---:|---:|---:|",
    ]
    root = Path(result_dir)
    found_any = False
    for (dataset, variant), results in sorted(grouped.items()):
        counts = None
        total = 0.0
        checkpoint_count = 0
        for result in results:
            config = config_from_result(result)
            slug = Path(config.data_config_path).stem
            checkpoint = root / variant / 'checkpoints' / f"{slug}_seed{config.seed}_best.pth"
            if not checkpoint.exists():
                continue
            data_cfg = load_yaml(config.data_config_path)
            _, _, test_loader = get_dataloaders(config, data_cfg)
            model = create_model(config)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
            model.eval()

            if counts is None:
                counts = torch.zeros(config.n_experts, dtype=torch.float64)
            with torch.no_grad():
                for inputs, _ in test_loader:
                    output = model(**inputs)
                    for history in output['history']:
                        selected = history['weights'] > 0
                        counts += selected.sum(dim=(0, 1)).to(torch.float64)
                        total += float(selected.sum().item())
            checkpoint_count += 1

        if checkpoint_count and total:
            found_any = True
            ratios = counts / total
            for expert_idx, ratio in enumerate(ratios.tolist()):
                lines.append(
                    f"| {dataset} | {variant} | {expert_idx} | {ratio:.6f} | {checkpoint_count} |"
                )
    if not found_any:
        lines.append("| - | - | - | No archived Phase 1 checkpoints found | 0 |")
    return lines


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as source:
        return yaml.safe_load(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default='results/phase1_v15')
    parser.add_argument('--skip-expert-utilization', action='store_true')
    args = parser.parse_args()

    grouped = load_results(args.result_dir)
    metric_lines, stats = metric_table(grouped)
    utilization_lines = (
        ["Skipped by --skip-expert-utilization."]
        if args.skip_expert_utilization
        else expert_utilization_table(grouped, args.result_dir)
    )
    report = [
        "# DAF-MoE v1.5 Phase 1 Report",
        "",
        "## Performance",
        *metric_lines,
        "",
        "## Pairwise Effects",
        *comparison_table(stats),
        "",
        "## Independence Cross-checks",
        *independence_table(stats),
        "",
        "## Parameter Count",
        *parameter_table(grouped),
        "",
        "## Expert Utilization",
        *utilization_lines,
        "",
    ]

    output_path = Path(args.result_dir) / 'PHASE1_REPORT.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding='utf-8')
    print(f"Saved report: {output_path}")


if __name__ == '__main__':
    main()
