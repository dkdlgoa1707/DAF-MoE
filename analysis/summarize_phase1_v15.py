"""
Summarize DAF-MoE v1.5 Phase 1 JSON results.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.models.factory import create_model


PAIRS = [("M0", "M1"), ("M0", "M2"), ("M2", "M3"), ("M3", "M4"), ("M4", "M5")]


def infer_variant(result):
    model = result.get("model", "")
    config = result.get("config", {})
    if model == "daf_moe":
        return "M0"
    if config.get("use_ple_embedding") == "True":
        return "M5"
    if config.get("use_lightweight_preservation") == "True" and config.get("use_film_gating") == "True":
        return "M4"
    if config.get("use_film_gating") == "True":
        return "M3"
    if config.get("use_loss_free_balancing") == "True":
        return "M2"
    if config.get("use_lightweight_preservation") == "True":
        return "M1"
    match = re.search(r"_M([1-5])", model)
    return f"M{match.group(1)}" if match else "unknown"


def target_metric(result):
    config = result.get("config", {})
    metrics = result.get("metrics", {})
    metric = config.get("optimize_metric") or ("rmse" if config.get("task_type") == "regression" else "acc")
    return metric, metrics.get(metric)


def paired_t_like(a_values, b_values):
    if len(a_values) != len(b_values) or len(a_values) < 2:
        return None
    diff = np.array(b_values, dtype=float) - np.array(a_values, dtype=float)
    std = diff.std(ddof=1)
    if std == 0:
        return math.inf if diff.mean() != 0 else 0.0
    return float(diff.mean() / (std / math.sqrt(len(diff))))


def load_results(result_dir):
    grouped = defaultdict(list)
    for path in Path(result_dir).glob("*.json"):
        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        dataset = result.get("dataset", "unknown")
        variant = infer_variant(result)
        grouped[(dataset, variant)].append(result)
    return grouped


def metric_table(grouped):
    lines = ["| Dataset | Variant | Metric | Mean | Std | Seeds |",
             "|---|---:|---|---:|---:|---:|"]
    stats = {}
    for key in sorted(grouped):
        values = []
        metric_name = None
        seeds = []
        for result in grouped[key]:
            metric_name, value = target_metric(result)
            if value is not None:
                values.append(float(value))
                seeds.append(result.get("seed"))
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats[key] = {"metric": metric_name, "values": values, "seeds": seeds, "mean": mean, "std": std}
        lines.append(f"| {key[0]} | {key[1]} | {metric_name} | {mean:.6f} | {std:.6f} | {len(values)} |")
    return lines, stats


def comparison_table(stats):
    lines = ["| Dataset | Pair | Mean Diff | t-like | Note |",
             "|---|---|---:|---:|---|"]
    datasets = sorted({dataset for dataset, _ in stats})
    for dataset in datasets:
        for left, right in PAIRS:
            a = stats.get((dataset, left))
            b = stats.get((dataset, right))
            if not a or not b:
                continue
            n = min(len(a["values"]), len(b["values"]))
            t_value = paired_t_like(a["values"][:n], b["values"][:n])
            diff = float(np.mean(b["values"][:n]) - np.mean(a["values"][:n]))
            note = "rough paired check, |t|>=2" if t_value is not None and abs(t_value) >= 2 else "rough paired check"
            t_text = "" if t_value is None else f"{t_value:.3f}"
            lines.append(f"| {dataset} | {left} vs {right} | {diff:.6f} | {t_text} | {note} |")
    return lines


def count_params(config_updates):
    config = DAFConfig()
    for key, value in config_updates.items():
        if hasattr(config, key):
            current = getattr(config, key)
            if isinstance(current, bool):
                value = str(value) == "True"
            elif isinstance(current, int):
                value = int(float(value))
            elif isinstance(current, float):
                value = float(value)
            setattr(config, key, value)
    return sum(p.numel() for p in create_model(config).parameters())


def param_table(grouped):
    lines = ["| Dataset | M0 Params | M4 Params | Delta |",
             "|---|---:|---:|---:|"]
    datasets = sorted({dataset for dataset, _ in grouped})
    for dataset in datasets:
        m0 = grouped.get((dataset, "M0"), [None])[0]
        m4 = grouped.get((dataset, "M4"), [None])[0]
        if not m0 or not m4:
            continue
        with torch.no_grad():
            m0_params = count_params(m0.get("config", {}))
            m4_params = count_params(m4.get("config", {}))
        lines.append(f"| {dataset} | {m0_params} | {m4_params} | {m4_params - m0_params} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results/phase1_v15")
    args = parser.parse_args()

    grouped = load_results(args.result_dir)
    metric_lines, stats = metric_table(grouped)
    compare_lines = comparison_table(stats)
    params_lines = param_table(grouped)

    report = [
        "# DAF-MoE v1.5 Phase 1 Report",
        "",
        "## Performance",
        *metric_lines,
        "",
        "## Pairwise Checks",
        *compare_lines,
        "",
        "## Parameter Count",
        *params_lines,
        "",
        "## Expert Utilization",
        "Routing histories are not persisted in the current JSON result format. Reuse checkpoints with analysis/analyze_expert.py for per-expert utilization.",
        "",
    ]

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = Path(args.result_dir) / "PHASE1_REPORT.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
