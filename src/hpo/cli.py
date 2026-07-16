"""Command-line orchestration for Phase 2 HPO and final evaluation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import yaml

from src.data.phase2_loader import get_phase2_dataloaders
from src.data.splits import load_raw_dataset
from src.hpo.engine import (
    TrialArtifactWriter,
    create_phase2_study,
    make_guarded_objective,
    phase2_study_name,
    run_until_valid_complete,
    valid_complete_count,
)
from src.hpo.schema import load_search_space
from src.native.data import prepare_native_final
from src.phase2_execution import (
    NATIVE_MODELS,
    execute_final_seed,
    execute_hpo_trial,
    materialize_neural_config,
)
from src.phase2_protocol import (
    FINAL_EVALUATION_SEEDS,
    HPO_SEED,
    N_HPO_COMPLETE_TRIALS,
    PROTOCOL_VERSION,
)
from src.phase2_results import (
    build_execution_manifest,
    reusable_result,
    write_result,
)
from src.utils.common import seed_everything


def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def _load_inputs(base_config_path, search_space_path):
    base_config = _read_yaml(base_config_path)
    search_space = load_search_space(search_space_path)
    base_config.setdefault("model_name", search_space.model_name)
    if base_config.get("model_name") != search_space.model_name:
        raise ValueError("Base config and search-space model_name differ.")
    required = ("data_config_path", "task_type", "optimize_metric")
    missing = [key for key in required if key not in base_config]
    if missing:
        raise ValueError(f"Base config is missing required fields: {missing}")
    data_config = _read_yaml(base_config["data_config_path"])
    raw_dataset = load_raw_dataset(data_config)
    return base_config, search_space, raw_dataset


def _device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _best_config_payload(base_config, search_space, study, n_rows):
    best_trial = study.best_trial
    resolved = best_trial.user_attrs.get("resolved_config")
    if resolved is None:
        raise RuntimeError("Best trial has no resolved_config artifact.")
    search_space.validate_resolved(resolved, n_rows=n_rows, task_type=base_config["task_type"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_name": search_space.model_name,
        "search_space_hash": search_space.schema_hash,
        "study_name": study.study_name,
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "resolved_config": resolved,
        "base_config": base_config,
    }


def run_hpo(args):
    base, space, raw = _load_inputs(args.base_config, args.search_space)
    if space.model_name == "tabicl":
        raise ValueError("TabICLv2 is fixed and has no HPO study.")
    if args.complete_trials != N_HPO_COMPLETE_TRIALS and not args.smoke:
        raise ValueError("Non-50 HPO targets require explicit --smoke.")
    if args.seed != HPO_SEED:
        raise ValueError(f"HPO seed is fixed at {HPO_SEED}.")
    seed_everything(HPO_SEED)
    study_name = phase2_study_name(raw.dataset_name, space.model_name)
    storage = args.storage or f"sqlite:///results/phase2/hpo/{study_name}.db"
    if storage.startswith("sqlite:///"):
        Path(storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    direction = "minimize" if base["optimize_metric"] == "rmse" else "maximize"
    study = create_phase2_study(raw.dataset_name, space.model_name, direction, storage)
    artifact_writer = TrialArtifactWriter(args.artifact_root)
    device = _device(args.device)

    def evaluator(resolved, trial):
        checkpoint = (
            Path(args.artifact_root)
            / study_name
            / f"trial_{trial.number:05d}_best.pth"
        )
        return execute_hpo_trial(
            raw,
            base,
            space,
            resolved,
            base["task_type"],
            base["optimize_metric"],
            checkpoint_path=checkpoint,
            device=device,
        )

    objective = make_guarded_objective(
        space,
        evaluator,
        n_rows=len(raw.features),
        artifact_writer=artifact_writer,
        study_name=study_name,
        task_type=base["task_type"],
    )
    run_until_valid_complete(study, objective, target=args.complete_trials)
    payload = _best_config_payload(base, space, study, len(raw.features))
    output = args.best_output or (
        f"configs/experiments/phase2/{space.model_name}/"
        f"{raw.dataset_name.lower().replace(' ', '_')}_best.yaml"
    )
    _write_yaml(output, payload)
    print(
        f"HPO_COMPLETE study={study.study_name} "
        f"valid_complete={valid_complete_count(study)} best={output}"
    )


def _resolved_for_final(args, space, task_type, n_rows):
    if space.model_name == "tabicl":
        if args.best_config:
            raise ValueError("TabICLv2 must not receive an HPO best config.")
        resolved = dict(space.fixed)
        space.validate_resolved(resolved, n_rows=n_rows, task_type=task_type)
        return resolved
    if not args.best_config:
        raise ValueError("Tunable methods require --best-config for final evaluation.")
    payload = _read_yaml(args.best_config)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Best config protocol_version mismatch.")
    if payload.get("search_space_hash") != space.schema_hash:
        raise ValueError("Best config search-space hash mismatch.")
    if payload.get("model_name") != space.model_name:
        raise ValueError("Best config model_name mismatch.")
    resolved = payload.get("resolved_config", {})
    space.validate_resolved(resolved, n_rows=n_rows, task_type=task_type)
    return resolved


def _resume_manifest(raw, base, space, resolved, seed):
    if space.model_name in NATIVE_MODELS:
        prepared = prepare_native_final(
            raw, space.model_name, base["task_type"], seed=seed
        )
        data_manifest = prepared.manifest
    else:
        config = materialize_neural_config(
            base,
            space,
            resolved,
            base["task_type"],
            base["optimize_metric"],
            seed,
        )
        get_phase2_dataloaders(config, raw)
        data_manifest = config.phase2_manifest
    return build_execution_manifest(
        data_manifest,
        space.model_name,
        resolved,
        space.schema_hash,
        seed,
    )


def run_final(args):
    base, space, raw = _load_inputs(args.base_config, args.search_space)
    resolved = _resolved_for_final(args, space, base["task_type"], len(raw.features))
    seeds = tuple(args.seeds or FINAL_EVALUATION_SEEDS)
    invalid = sorted(set(seeds).difference(FINAL_EVALUATION_SEEDS))
    if invalid:
        raise ValueError(f"Final seeds must be a subset of 43..57, got {invalid}.")
    device = _device(args.device)
    dataset_slug = raw.dataset_name.lower().replace(" ", "_").replace("-", "_")
    for seed in seeds:
        seed_everything(seed)
        result_path = (
            Path(args.result_root)
            / dataset_slug
            / space.model_name
            / f"seed{seed}.json"
        )
        resume_manifest = _resume_manifest(raw, base, space, resolved, seed)
        if reusable_result(result_path, resume_manifest):
            print(f"RESUME seed={seed} path={result_path}")
            continue
        checkpoint = result_path.with_suffix(".best.pth")
        outcome = execute_final_seed(
            raw,
            base,
            space,
            resolved,
            base["task_type"],
            base["optimize_metric"],
            seed,
            checkpoint_path=checkpoint,
            device=device,
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": raw.dataset_name,
            "model": space.model_name,
            "seed": seed,
            "metrics": outcome.metrics,
            "best_epoch_or_iteration": outcome.best_epoch_or_iteration,
            "epochs_completed": outcome.epochs_completed,
            "manifest": outcome.manifest,
            "resume_manifest": resume_manifest,
        }
        write_result(result_path, payload)
        print(
            f"FINAL_COMPLETE seed={seed} {base['optimize_metric']}="
            f"{outcome.metric_value:.8g} path={result_path}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hpo = subparsers.add_parser("hpo", help="Run/resume sealed HPO.")
    hpo.add_argument("--base-config", required=True)
    hpo.add_argument("--search-space", required=True)
    hpo.add_argument("--storage")
    hpo.add_argument("--artifact-root", default="results/phase2/trials")
    hpo.add_argument("--best-output")
    hpo.add_argument("--complete-trials", type=int, default=N_HPO_COMPLETE_TRIALS)
    hpo.add_argument("--smoke", action="store_true")
    hpo.add_argument("--seed", type=int, default=HPO_SEED)
    hpo.add_argument("--device")
    hpo.set_defaults(func=run_hpo)

    final = subparsers.add_parser("final", help="Run hash-gated final seeds.")
    final.add_argument("--base-config", required=True)
    final.add_argument("--search-space", required=True)
    final.add_argument("--best-config")
    final.add_argument("--seeds", type=int, nargs="+")
    final.add_argument("--result-root", default="results/phase2/final")
    final.add_argument("--device")
    final.set_defaults(func=run_final)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
