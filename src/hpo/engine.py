"""Optuna engine with bounded retries and structured trial artifacts."""

from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import resource
import time

from src.data.provenance import current_git_sha, stable_hash
from src.hpo.identity import StudyIdentity, resolve_study_storage
from src.phase2_protocol import HPO_SEED, N_HPO_COMPLETE_TRIALS


DEFAULT_MAX_ATTEMPTS = 200
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10


class HpoRunStopped(RuntimeError):
    """Base class for a bounded HPO run that stopped before its target."""

    def __init__(self, summary):
        self.summary = dict(summary)
        super().__init__(self.summary["message"])


class HpoAttemptLimitError(HpoRunStopped):
    """Attempt or consecutive-failure ceiling reached."""


class HpoComputeLimitError(HpoRunStopped):
    """User-provided cumulative compute ceiling reached."""


def _state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", str(state))


def _expected_trial_attributes(study, identity=None):
    signature = (
        identity.signature if identity else study.user_attrs.get("study_signature")
    )
    search_hash = (
        identity.components["search_space_hash"]
        if identity
        else study.user_attrs.get("search_space_hash")
    )
    if not signature or not search_hash:
        raise RuntimeError("Study is missing canonical signature attributes.")
    return signature, search_hash


def valid_complete_trials(study, identity=None):
    signature, search_hash = _expected_trial_attributes(study, identity)
    valid = []
    for trial in study.trials:
        value = getattr(trial, "value", None)
        attrs = getattr(trial, "user_attrs", {})
        if (
            _state_name(trial) == "COMPLETE"
            and value is not None
            and math.isfinite(float(value))
            and attrs.get("study_signature") == signature
            and attrs.get("search_space_hash") == search_hash
        ):
            valid.append(trial)
    return valid


def valid_complete_count(study, identity=None):
    return len(valid_complete_trials(study, identity))


def study_state_counts(study):
    counts = Counter(_state_name(trial) for trial in study.trials)
    return dict(sorted(counts.items()))


def recent_failure_summaries(study, limit=5):
    failures = []
    for trial in reversed(study.trials):
        if _state_name(trial) != "FAIL":
            continue
        attrs = getattr(trial, "user_attrs", {})
        failures.append(
            {
                "trial_number": int(trial.number),
                "failure_type": attrs.get("failure_type", "UnknownFailure"),
                "failure_reason": attrs.get("failure_reason", "not recorded"),
            }
        )
        if len(failures) >= int(limit):
            break
    return failures


def create_phase2_study(identity: StudyIdentity, direction, storage_url=None):
    try:
        import optuna
        from optuna.pruners import NopPruner
        from optuna.samplers import TPESampler
        from optuna.storages import RDBStorage
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for Phase 2 HPO. Install requirements.txt."
        ) from exc

    storage_url = resolve_study_storage(identity, storage_url)
    engine_kwargs = (
        {"connect_args": {"timeout": 60}}
        if storage_url.startswith("sqlite:///")
        else {}
    )
    storage = RDBStorage(url=storage_url, engine_kwargs=engine_kwargs)
    study = optuna.create_study(
        study_name=identity.study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=TPESampler(seed=HPO_SEED),
        pruner=NopPruner(),
    )
    expected = identity.study_attributes
    if study.user_attrs:
        mismatched = {
            key: (study.user_attrs.get(key), value)
            for key, value in expected.items()
            if study.user_attrs.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"Refusing mismatched study resume: {mismatched}")
    else:
        for key, value in expected.items():
            study.set_user_attr(key, value)
    return study


def _run_summary(
    study,
    identity,
    target,
    attempts,
    consecutive_failures,
    cumulative_seconds,
    status,
    reason,
):
    signature, _ = _expected_trial_attributes(study, identity)
    valid = valid_complete_count(study, identity)
    summary = {
        "status": status,
        "study_name": study.study_name,
        "study_signature": signature,
        "target_valid_complete": int(target),
        "valid_complete": valid,
        "attempts_this_run": int(attempts),
        "state_counts": study_state_counts(study),
        "consecutive_failures": int(consecutive_failures),
        "recent_failures": recent_failure_summaries(study),
        "cumulative_elapsed_seconds": float(cumulative_seconds),
        "gpu_hours_lower_bound": float(cumulative_seconds) / 3600.0,
        "reason": reason,
    }
    summary["message"] = (
        f"study={study.study_name} signature={signature} "
        f"valid_complete={valid}/{target} attempts={attempts} "
        f"states={summary['state_counts']} consecutive_failures="
        f"{consecutive_failures} status={status} reason={reason}"
    )
    study.set_user_attr("hpo_cumulative_elapsed_seconds", float(cumulative_seconds))
    study.set_user_attr("last_hpo_run_summary", summary)
    return summary


def run_until_valid_complete(
    study,
    objective,
    target=N_HPO_COMPLETE_TRIALS,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    max_consecutive_failures=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    compute_ceiling_hours=None,
    identity=None,
):
    """Continue one trial at a time with explicit retry and compute ceilings."""
    target = int(target)
    if target <= 0:
        raise ValueError("target must be positive.")
    if max_attempts is not None and int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive.")
    if max_consecutive_failures is not None and int(max_consecutive_failures) <= 0:
        raise ValueError("max_consecutive_failures must be positive.")
    if compute_ceiling_hours is not None and float(compute_ceiling_hours) <= 0:
        raise ValueError("compute_ceiling_hours must be positive when provided.")

    attempts = 0
    consecutive_failures = 0
    run_started = time.perf_counter()
    cumulative_before = float(
        study.user_attrs.get("hpo_cumulative_elapsed_seconds", 0.0)
    )

    def cumulative_seconds():
        return cumulative_before + (time.perf_counter() - run_started)

    def stop(exception_class, status, reason):
        summary = _run_summary(
            study,
            identity,
            target,
            attempts,
            consecutive_failures,
            cumulative_seconds(),
            status,
            reason,
        )
        raise exception_class(summary)

    while valid_complete_count(study, identity) < target:
        if max_attempts is not None and attempts >= int(max_attempts):
            stop(
                HpoAttemptLimitError,
                "ATTEMPT_LIMIT",
                f"Reached max_attempts={int(max_attempts)}.",
            )
        if (
            compute_ceiling_hours is not None
            and cumulative_seconds() >= float(compute_ceiling_hours) * 3600.0
        ):
            stop(
                HpoComputeLimitError,
                "COMPUTE_LIMIT",
                f"Reached compute_ceiling_hours={float(compute_ceiling_hours)}.",
            )

        existing_numbers = {trial.number for trial in study.trials}
        study.optimize(objective, n_trials=1, catch=(Exception,))
        new_trials = [
            trial for trial in study.trials if trial.number not in existing_numbers
        ]
        if len(new_trials) != 1:
            raise RuntimeError(
                f"Expected exactly one new trial, observed {len(new_trials)}."
            )
        attempts += 1
        if _state_name(new_trials[0]) == "FAIL":
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        study.set_user_attr(
            "hpo_cumulative_elapsed_seconds", float(cumulative_seconds())
        )
        if (
            max_consecutive_failures is not None
            and consecutive_failures >= int(max_consecutive_failures)
            and valid_complete_count(study, identity) < target
        ):
            stop(
                HpoAttemptLimitError,
                "CONSECUTIVE_FAILURE_LIMIT",
                f"Reached max_consecutive_failures={int(max_consecutive_failures)}.",
            )

    return _run_summary(
        study,
        identity,
        target,
        attempts,
        consecutive_failures,
        cumulative_seconds(),
        "COMPLETE",
        "Target valid COMPLETE count reached.",
    )


def _outcome_dict(outcome):
    if is_dataclass(outcome):
        return asdict(outcome)
    if isinstance(outcome, dict):
        return dict(outcome)
    return {"metric_value": float(outcome)}


class TrialArtifactWriter:
    def __init__(self, root):
        self.root = Path(root)

    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def write(self, study_name, trial_number, payload):
        path = self.root / study_name / f"trial_{int(trial_number):05d}.json"
        return self._write(path, payload)

    def write_run_status(self, study_name, payload):
        return self._write(self.root / study_name / "hpo_run_status.json", payload)


def make_guarded_objective(
    search_space,
    evaluator,
    n_rows,
    artifact_writer,
    study_identity,
    task_type=None,
):
    """Raise on every invalid trial so Optuna records FAIL, never fake COMPLETE."""

    def objective(trial):
        started = time.perf_counter()
        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("study_signature", study_identity.signature)
            trial.set_user_attr("search_space_hash", search_space.schema_hash)
        base = {
            "protocol_version": study_identity.components["protocol_version"],
            "git_sha": current_git_sha(),
            "study_name": study_identity.study_name,
            "study_signature": study_identity.signature,
            "study_signature_components": dict(study_identity.components),
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
            elapsed = time.perf_counter() - started
            base.update(outcome)
            base.update(
                {
                    "status": "COMPLETE",
                    "elapsed_seconds": elapsed,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            base["artifact_hash"] = stable_hash(base)
            artifact_writer.write(study_identity.study_name, trial.number, base)
            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("artifact_hash", base["artifact_hash"])
                trial.set_user_attr("resolved_config", resolved)
                trial.set_user_attr("resolved_config_hash", stable_hash(resolved))
                trial.set_user_attr("elapsed_seconds", elapsed)
            return value
        except Exception as exc:
            elapsed = time.perf_counter() - started
            base.update(
                {
                    "status": "FAIL",
                    "failure_type": exc.__class__.__name__,
                    "failure_reason": str(exc),
                    "elapsed_seconds": elapsed,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            base["artifact_hash"] = stable_hash(base)
            artifact_writer.write(study_identity.study_name, trial.number, base)
            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("failure_type", exc.__class__.__name__)
                trial.set_user_attr("failure_reason", str(exc))
                trial.set_user_attr("elapsed_seconds", elapsed)
                trial.set_user_attr("artifact_hash", base["artifact_hash"])
            raise

    return objective
