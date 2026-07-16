"""Optuna engine with valid-COMPLETE counting and structured trial artifacts."""

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import resource
import time

from src.data.provenance import current_git_sha, stable_hash
from src.phase2_protocol import HPO_SEED, N_HPO_COMPLETE_TRIALS, PROTOCOL_VERSION


def _state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", str(state))


def valid_complete_trials(study):
    valid = []
    for trial in study.trials:
        value = getattr(trial, "value", None)
        if _state_name(trial) == "COMPLETE" and value is not None and math.isfinite(float(value)):
            valid.append(trial)
    return valid


def valid_complete_count(study):
    return len(valid_complete_trials(study))


def phase2_study_name(dataset_name, model_name):
    dataset = str(dataset_name).lower().replace(" ", "_").replace("-", "_")
    return f"{dataset}__{model_name}__{PROTOCOL_VERSION}"


def create_phase2_study(dataset_name, model_name, direction, storage_url):
    try:
        import optuna
        from optuna.pruners import NopPruner
        from optuna.samplers import TPESampler
        from optuna.storages import RDBStorage
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for Phase 2 HPO. Install requirements.txt."
        ) from exc

    engine_kwargs = (
        {"connect_args": {"timeout": 60}}
        if storage_url.startswith("sqlite:///")
        else {}
    )
    storage = RDBStorage(
        url=storage_url,
        engine_kwargs=engine_kwargs,
    )
    return optuna.create_study(
        study_name=phase2_study_name(dataset_name, model_name),
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=TPESampler(seed=HPO_SEED),
        pruner=NopPruner(),
    )


def run_until_valid_complete(
    study,
    objective,
    target=N_HPO_COMPLETE_TRIALS,
    max_attempts=None,
):
    """Continue one trial at a time until finite COMPLETE count reaches target."""
    attempts = 0
    while valid_complete_count(study) < int(target):
        if max_attempts is not None and attempts >= int(max_attempts):
            raise RuntimeError(
                f"Reached max_attempts={max_attempts} with "
                f"{valid_complete_count(study)}/{target} valid COMPLETE trials."
            )
        study.optimize(objective, n_trials=1, catch=(Exception,))
        attempts += 1
    return valid_complete_trials(study)


def _outcome_dict(outcome):
    if is_dataclass(outcome):
        return asdict(outcome)
    if isinstance(outcome, dict):
        return dict(outcome)
    return {"metric_value": float(outcome)}


class TrialArtifactWriter:
    def __init__(self, root):
        self.root = Path(root)

    def write(self, study_name, trial_number, payload):
        directory = self.root / study_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"trial_{int(trial_number):05d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path


def make_guarded_objective(search_space, evaluator, n_rows, artifact_writer, study_name, task_type=None):
    """Raise on every invalid trial so Optuna records FAIL, never fake COMPLETE."""
    def objective(trial):
        started = time.perf_counter()
        base = {
            "protocol_version": PROTOCOL_VERSION,
            "git_sha": current_git_sha(),
            "study_name": study_name,
            "trial_number": int(trial.number),
            "model_name": search_space.model_name,
            "search_space_hash": search_space.schema_hash,
            "params": dict(getattr(trial, "params", {})),
            "resolved_config": None,
        }
        try:
            resolved = search_space.sample(trial, n_rows=n_rows, task_type=task_type)
            base["params"] = dict(getattr(trial, "params", {}))
            base["resolved_config"] = resolved
            outcome = _outcome_dict(evaluator(resolved, trial))
            value = float(outcome["metric_value"])
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation metric: {value}")
            base.update(outcome)
            base.update(
                {
                    "status": "COMPLETE",
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            base["artifact_hash"] = stable_hash(base)
            artifact_writer.write(study_name, trial.number, base)
            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("artifact_hash", base["artifact_hash"])
                trial.set_user_attr("resolved_config", resolved)
                trial.set_user_attr("resolved_config_hash", stable_hash(resolved))
            return value
        except Exception as exc:
            base.update(
                {
                    "status": "FAIL",
                    "failure_type": exc.__class__.__name__,
                    "failure_reason": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            base["artifact_hash"] = stable_hash(base)
            artifact_writer.write(study_name, trial.number, base)
            raise

    return objective
