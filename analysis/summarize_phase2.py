#!/usr/bin/env python
"""Summarize Phase 2 availability and ranks without imputing missing results."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase2_protocol import (  # noqa: E402
    DATASETS,
    FINAL_EVALUATION_SEEDS,
    MAIN_METHODS,
    PROTOCOL_VERSION,
    SECONDARY_METHODS,
)


ALL_METHODS = MAIN_METHODS + SECONDARY_METHODS


def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def dataset_metadata():
    metadata = {}
    aliases = {}
    for dataset in DATASETS:
        base = _read_yaml(ROOT / f"configs/experiments/phase2/base/{dataset}.yaml")
        data = _read_yaml(ROOT / base["data_config_path"])
        name = data["dataset_name"]
        metadata[dataset] = {
            "name": name,
            "metric": base["optimize_metric"],
            "direction": (
                "minimize" if base["optimize_metric"] == "rmse" else "maximize"
            ),
        }
        aliases[dataset] = dataset
        aliases[name] = dataset
        aliases[name.lower().replace(" ", "_").replace("-", "_")] = dataset
    return metadata, aliases


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def collect_final_results(result_root, aliases):
    records = defaultdict(list)
    for path in Path(result_root).glob("**/seed*.json"):
        payload = _load_json(path)
        if not payload or payload.get("protocol_version") != PROTOCOL_VERSION:
            continue
        dataset = aliases.get(payload.get("dataset"))
        model = payload.get("model")
        seed = payload.get("seed")
        metrics = payload.get("metrics", {})
        if dataset not in DATASETS or model not in ALL_METHODS:
            continue
        if seed not in FINAL_EVALUATION_SEEDS:
            continue
        records[(dataset, model)].append(
            {"seed": int(seed), "metrics": metrics, "path": str(path)}
        )
    return records


def collect_hpo_statuses(artifact_root, aliases):
    statuses = {}
    for path in Path(artifact_root).glob("**/hpo_run_status.json"):
        payload = _load_json(path)
        if not payload:
            continue
        dataset = aliases.get(payload.get("dataset"))
        model = payload.get("model_name")
        if dataset in DATASETS and model in ALL_METHODS:
            statuses[(dataset, model)] = {**payload, "path": str(path)}
    return statuses


def build_availability(records, hpo_statuses):
    rows = []
    required_seeds = set(FINAL_EVALUATION_SEEDS)
    for dataset in DATASETS:
        for model in ALL_METHODS:
            final_records = records.get((dataset, model), [])
            seeds = {item["seed"] for item in final_records}
            hpo = hpo_statuses.get((dataset, model), {})
            if hpo.get("status") == "COMPUTE_LIMIT":
                status = "COMPUTE_LIMIT"
                reason = hpo.get("reason", "compute ceiling reached")
            elif seeds == required_seeds:
                status = "COMPLETE"
                reason = None
            elif seeds:
                status = "PARTIAL_FINAL"
                reason = f"{len(seeds)}/{len(required_seeds)} final seeds"
            elif hpo.get("status") == "COMPLETE":
                status = "HPO_COMPLETE"
                reason = "final evaluation not complete"
            else:
                status = "MISSING"
                reason = "no complete final result"
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "status": status,
                    "reason": reason,
                    "final_seed_count": len(seeds),
                    "valid_complete_trials": hpo.get("valid_complete"),
                    "gpu_hours_lower_bound": hpo.get("gpu_hours_lower_bound"),
                    "peak_memory_bytes": hpo.get("max_cuda_memory_allocated"),
                    "metric": None,
                    "rank": None,
                }
            )
    return rows


def _average_tie_ranks(values, minimize):
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=not minimize)
    ranks = [None] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for index, _ in ordered[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def build_common_subset_ranking(records, metadata):
    required_seeds = set(FINAL_EVALUATION_SEEDS)
    means = {}
    for dataset in DATASETS:
        metric = metadata[dataset]["metric"]
        for model in MAIN_METHODS:
            items = records.get((dataset, model), [])
            seeds = {item["seed"] for item in items}
            values = [item["metrics"].get(metric) for item in items]
            if (
                seeds == required_seeds
                and len(values) == len(required_seeds)
                and all(
                    value is not None and math.isfinite(float(value))
                    for value in values
                )
            ):
                means[(dataset, model)] = sum(map(float, values)) / len(values)

    common_datasets = [
        dataset
        for dataset in DATASETS
        if all((dataset, model) in means for model in MAIN_METHODS)
    ]
    rank_rows = []
    model_ranks = defaultdict(list)
    for dataset in common_datasets:
        values = [means[(dataset, model)] for model in MAIN_METHODS]
        ranks = _average_tie_ranks(
            values, minimize=metadata[dataset]["direction"] == "minimize"
        )
        for model, value, rank in zip(MAIN_METHODS, values, ranks):
            rank_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_metric": value,
                    "rank": rank,
                }
            )
            model_ranks[model].append(rank)
    average_ranks = {
        model: sum(values) / len(values)
        for model, values in model_ranks.items()
        if values
    }

    omnibus = {
        "test": "friedmanchisquare",
        "dataset_subset": common_datasets,
        "statistic": None,
        "p_value": None,
        "reason": None,
    }
    if len(common_datasets) < 2:
        omnibus["reason"] = "At least two common datasets are required."
    else:
        from scipy.stats import friedmanchisquare

        samples = [
            [means[(dataset, model)] for dataset in common_datasets]
            for model in MAIN_METHODS
        ]
        statistic, p_value = friedmanchisquare(*samples)
        omnibus["statistic"] = float(statistic)
        omnibus["p_value"] = float(p_value)
    return {
        "included_models": list(MAIN_METHODS),
        "common_dataset_subset": common_datasets,
        "rows": rank_rows,
        "average_ranks": average_ranks,
        "omnibus": omnibus,
    }


def build_summary(result_root, artifact_root):
    metadata, aliases = dataset_metadata()
    records = collect_final_results(result_root, aliases)
    hpo_statuses = collect_hpo_statuses(artifact_root, aliases)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "missing_metric_policy": "never_impute_or_rank",
        "availability": build_availability(records, hpo_statuses),
        "ranking": build_common_subset_ranking(records, metadata),
    }


def write_summary(summary, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    availability = {
        (row["dataset"], row["model"]): row for row in summary["availability"]
    }
    lines = [
        "# Phase 2 Availability and Ranking",
        "",
        "`COMPUTE_LIMIT` and missing results are never converted to metrics or ranks.",
        "",
        "## Availability",
        "",
        "| Dataset | " + " | ".join(ALL_METHODS) + " |",
        "|---|" + "---|" * len(ALL_METHODS),
    ]
    for dataset in DATASETS:
        cells = []
        for model in ALL_METHODS:
            row = availability[(dataset, model)]
            cell = row["status"]
            if row["status"] == "COMPUTE_LIMIT":
                cell = "— (COMPUTE_LIMIT)"
            cells.append(cell)
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    ranking = summary["ranking"]
    lines.extend(
        [
            "",
            "## Common Dataset Ranking",
            "",
            "Common dataset subset: "
            + (", ".join(ranking["common_dataset_subset"]) or "none"),
            "",
            "| Model | Average rank |",
            "|---|---:|",
        ]
    )
    for model, rank in sorted(
        ranking["average_ranks"].items(), key=lambda item: item[1]
    ):
        lines.append(f"| {model} | {rank:.4f} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="results/phase2/final")
    parser.add_argument("--artifact-root", default="results/phase2/trials")
    parser.add_argument("--output-json", default="results/phase2/PHASE2_SUMMARY.json")
    parser.add_argument(
        "--output-markdown", default="results/phase2/PHASE2_AVAILABILITY.md"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = build_summary(args.result_root, args.artifact_root)
    write_summary(summary, args.output_json, args.output_markdown)
    print(f"Summary JSON: {args.output_json}")
    print(f"Availability: {args.output_markdown}")
    print(
        "Common dataset subset: "
        + (", ".join(summary["ranking"]["common_dataset_subset"]) or "none")
    )


if __name__ == "__main__":
    main()
