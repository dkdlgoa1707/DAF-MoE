"""Main native/official estimator entrypoint for Phase 2.

This runner intentionally has separate HPO and final data constructors. HPO
never constructs a test partition. Full experiment orchestration is added by
Prompt 5; this module is the model-faithful execution layer.
"""

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.provenance import stable_hash
from src.data.splits import HPO_SEED, load_raw_dataset
from src.native.data import prepare_native_final, prepare_native_hpo
from src.native.runner import (
    evaluate_hpo_estimator,
    fit_hpo_estimator,
    run_native_final,
)


def _load_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    return value or {}


def run_hpo_validation(args, raw_dataset, params):
    if args.seed != HPO_SEED:
        raise ValueError(f"HPO seed is fixed at {HPO_SEED}, got {args.seed}.")
    if args.model == "tabicl":
        raise ValueError("TabICLv2 has no HPO entrypoint.")
    data = prepare_native_hpo(
        raw_dataset, args.model, args.task_type, seed=HPO_SEED
    )
    estimator, dependency, resolved = fit_hpo_estimator(
        args.model, args.task_type, data, HPO_SEED, params=params
    )
    value = evaluate_hpo_estimator(estimator, data, args.task_type, args.metric)
    manifest = dict(data.manifest)
    manifest.update({"dependency": dependency, "resolved_config": resolved})
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    return {
        "mode": "hpo-validation",
        "model": args.model,
        "dataset": raw_dataset.dataset_name,
        "seed": HPO_SEED,
        "metric": args.metric,
        "value": value,
        "manifest": manifest,
    }


def run_final_seed(args, raw_dataset, params):
    data = prepare_native_final(
        raw_dataset, args.model, args.task_type, seed=args.seed
    )
    result = run_native_final(
        args.model,
        data,
        raw_dataset.dataset_name,
        args.task_type,
        args.metric,
        args.seed,
        params=params,
        device=args.device,
    )
    return {
        "mode": "final",
        "model": result.model_name,
        "dataset": raw_dataset.dataset_name,
        "seed": args.seed,
        "metric": result.metric_name,
        "value": result.metric_value,
        "elapsed_seconds": result.elapsed_seconds,
        "manifest": result.manifest,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hpo-validation", "final"), required=True)
    parser.add_argument(
        "--model", choices=("xgboost", "catboost", "realmlp", "tabicl"), required=True
    )
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument(
        "--task-type", choices=("classification", "regression"), required=True
    )
    parser.add_argument(
        "--metric", choices=("rmse", "acc", "auprc", "auroc"), required=True
    )
    parser.add_argument("--seed", type=int, default=HPO_SEED)
    parser.add_argument("--params-yaml")
    parser.add_argument("--device")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_config = _load_yaml(args.dataset_config)
    raw_dataset = load_raw_dataset(data_config)
    params = _load_yaml(args.params_yaml) if args.params_yaml else {}
    if args.mode == "hpo-validation":
        result = run_hpo_validation(args, raw_dataset, params)
    else:
        result = run_final_seed(args, raw_dataset, params)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
