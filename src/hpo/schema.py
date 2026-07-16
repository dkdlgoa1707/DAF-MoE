"""Validated YAML search-space schema for the fixed Phase 2 protocol."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.provenance import stable_hash
from src.phase2_protocol import MAIN_RANK_INCLUDED, PROTOCOL_VERSION


SUPPORTED_DISTRIBUTIONS = {
    "uniform",
    "log_uniform",
    "int",
    "stepped_int",
    "categorical",
    "zero_mixture",
    "weighted_categorical",
    "conditional_by_n",
    "log_uniform_int",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_distribution(name, spec, nested=False):
    _require(isinstance(spec, Mapping), f"{name} distribution must be a mapping.")
    kind = spec.get("type")
    _require(kind in SUPPORTED_DISTRIBUTIONS, f"{name} has unsupported type: {kind}")
    if nested:
        _require(
            kind not in {"conditional_by_n", "zero_mixture"},
            f"{name} contains an unsupported nested {kind} distribution.",
        )

    when_task = spec.get("when_task")
    if when_task is not None:
        _require(
            when_task in {"classification", "regression"},
            f"{name}.when_task must be classification or regression.",
        )
        _require("otherwise" in spec, f"{name}.otherwise is required with when_task.")

    if kind in {"uniform", "log_uniform"}:
        low, high = spec.get("low"), spec.get("high")
        _require(isinstance(low, (int, float)), f"{name}.low must be numeric.")
        _require(isinstance(high, (int, float)), f"{name}.high must be numeric.")
        _require(low < high, f"{name} requires low < high.")
        if kind == "log_uniform":
            _require(low > 0, f"{name} log_uniform requires low > 0.")
    elif kind in {"int", "stepped_int", "log_uniform_int"}:
        low, high = spec.get("low"), spec.get("high")
        _require(isinstance(low, int) and isinstance(high, int), f"{name} bounds must be int.")
        _require(low <= high, f"{name} requires low <= high.")
        if kind == "stepped_int":
            step = spec.get("step")
            _require(isinstance(step, int) and step > 0, f"{name}.step must be positive int.")
            _require((high - low) % step == 0, f"{name} high must lie on its step grid.")
        if kind == "log_uniform_int":
            _require(low > 0, f"{name} log_uniform_int requires low > 0.")
    elif kind == "categorical":
        choices = spec.get("choices")
        _require(isinstance(choices, list) and choices, f"{name}.choices must be non-empty.")
    elif kind == "weighted_categorical":
        choices, weights = spec.get("choices"), spec.get("weights")
        _require(isinstance(choices, list) and choices, f"{name}.choices must be non-empty.")
        _require(isinstance(weights, list), f"{name}.weights must be a list.")
        _require(len(choices) == len(weights), f"{name} choices/weights length mismatch.")
        _require(all(isinstance(x, (int, float)) and x > 0 for x in weights), f"{name} weights must be positive.")
    elif kind == "zero_mixture":
        probability = spec.get("zero_probability", 0.5)
        _require(0.0 < probability < 1.0, f"{name}.zero_probability must be in (0,1).")
        _validate_distribution(f"{name}.nonzero", spec.get("nonzero"), nested=True)
    elif kind == "conditional_by_n":
        threshold = spec.get("threshold")
        _require(isinstance(threshold, int) and threshold > 0, f"{name}.threshold must be positive int.")
        _validate_distribution(f"{name}.at_most", spec.get("at_most"), nested=True)
        _validate_distribution(f"{name}.above", spec.get("above"), nested=True)


def _sample_distribution(trial, name, spec, n_rows, task_type=None):
    if spec.get("when_task") is not None and task_type != spec["when_task"]:
        return spec["otherwise"]
    kind = spec["type"]
    if kind == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "log_uniform":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]), log=True
        )
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "stepped_int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=int(spec["step"])
        )
    if kind == "log_uniform_int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), log=True
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "zero_mixture":
        zero_probability = float(spec.get("zero_probability", 0.5))
        selector = trial.suggest_float(f"{name}__zero_selector", 0.0, 1.0)
        if selector < zero_probability:
            return 0
        return _sample_distribution(trial, name, spec["nonzero"], n_rows, task_type)
    if kind == "weighted_categorical":
        selector = trial.suggest_float(f"{name}__weighted_selector", 0.0, 1.0)
        weights = [float(value) for value in spec["weights"]]
        total = sum(weights)
        cumulative = 0.0
        for choice, weight in zip(spec["choices"], weights):
            cumulative += weight / total
            if selector <= cumulative:
                return choice
        return spec["choices"][-1]
    if kind == "conditional_by_n":
        if n_rows is None:
            raise ValueError(f"Sampling {name} requires n_rows.")
        branch = spec["at_most"] if n_rows <= int(spec["threshold"]) else spec["above"]
        return _sample_distribution(trial, name, branch, n_rows, task_type)
    raise AssertionError(f"Unhandled distribution: {kind}")


def _value_matches_distribution(spec, value, n_rows=None, task_type=None):
    if spec.get("when_task") is not None and task_type != spec["when_task"]:
        return value == spec["otherwise"]
    kind = spec["type"]
    if kind in {"uniform", "log_uniform"}:
        return isinstance(value, (int, float)) and (spec["low"] <= value <= spec["high"] or math.isclose(value, spec["low"]) or math.isclose(value, spec["high"]))
    if kind in {"int", "stepped_int", "log_uniform_int"}:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if not spec["low"] <= value <= spec["high"]:
            return False
        return kind != "stepped_int" or (value - spec["low"]) % spec["step"] == 0
    if kind in {"categorical", "weighted_categorical"}:
        return any(value == choice for choice in spec["choices"])
    if kind == "zero_mixture":
        return value == 0 or _value_matches_distribution(
            spec["nonzero"], value, n_rows=n_rows, task_type=task_type
        )
    if kind == "conditional_by_n":
        if n_rows is None:
            return _value_matches_distribution(
                spec["at_most"], value, task_type=task_type
            ) or _value_matches_distribution(
                spec["above"], value, task_type=task_type
            )
        branch = spec["at_most"] if n_rows <= spec["threshold"] else spec["above"]
        return _value_matches_distribution(
            branch, value, n_rows=n_rows, task_type=task_type
        )
    return False


@dataclass(frozen=True)
class SearchSpace:
    model_name: str
    protocol_version: str
    rank_included: bool
    fixed: Mapping[str, Any]
    search: Mapping[str, Mapping[str, Any]]
    forbidden: tuple
    source_path: str = ""

    @property
    def schema_hash(self):
        return stable_hash(self.as_dict())

    def as_dict(self):
        return {
            "model_name": self.model_name,
            "protocol_version": self.protocol_version,
            "rank_included": self.rank_included,
            "fixed": dict(self.fixed),
            "search": dict(self.search),
            "forbidden": list(self.forbidden),
        }

    def sample(self, trial, n_rows=None, task_type=None):
        resolved = dict(self.fixed)
        for name, spec in self.search.items():
            resolved[name] = _sample_distribution(trial, name, spec, n_rows, task_type)
        forbidden_present = sorted(set(resolved).intersection(self.forbidden))
        if forbidden_present:
            raise ValueError(f"Forbidden fields resolved for {self.model_name}: {forbidden_present}")
        return resolved

    def validate_resolved(self, resolved, n_rows=None, task_type=None):
        expected = set(self.fixed).union(self.search)
        missing = sorted(expected.difference(resolved))
        extra = sorted(set(resolved).difference(expected))
        if missing or extra:
            raise ValueError(f"Resolved config mismatch: missing={missing}, extra={extra}")
        forbidden_present = sorted(set(resolved).intersection(self.forbidden))
        if forbidden_present:
            raise ValueError(f"Forbidden fields present: {forbidden_present}")
        for name, spec in self.search.items():
            if not _value_matches_distribution(
                spec, resolved[name], n_rows=n_rows, task_type=task_type
            ):
                raise ValueError(f"Resolved value is outside {name} distribution: {resolved[name]}")
        for key, value in self.fixed.items():
            if resolved[key] != value:
                raise ValueError(f"Fixed field {key} changed: {resolved[key]} != {value}")


def parse_search_space(payload, source_path=""):
    _require(isinstance(payload, Mapping), "Search-space YAML must contain a mapping.")
    model_name = payload.get("model_name")
    _require(model_name in MAIN_RANK_INCLUDED, f"Unknown Phase 2 model: {model_name}")
    protocol_version = payload.get("protocol_version")
    _require(protocol_version == PROTOCOL_VERSION, f"Expected protocol_version={PROTOCOL_VERSION}.")
    rank_included = payload.get("rank_included")
    _require(
        rank_included is MAIN_RANK_INCLUDED[model_name],
        f"rank_included mismatch for {model_name}.",
    )
    fixed = payload.get("fixed", {})
    search = payload.get("search", {})
    forbidden = tuple(payload.get("forbidden", ()))
    _require(isinstance(fixed, Mapping), "fixed must be a mapping.")
    _require(isinstance(search, Mapping), "search must be a mapping.")
    collision = sorted(set(fixed).intersection(search))
    _require(not collision, f"fixed/search field collision: {collision}")
    forbidden_collision = sorted(set(fixed).union(search).intersection(forbidden))
    _require(not forbidden_collision, f"forbidden fields are configured: {forbidden_collision}")
    for name, spec in search.items():
        _validate_distribution(name, spec)
    return SearchSpace(
        model_name=model_name,
        protocol_version=protocol_version,
        rank_included=rank_included,
        fixed=dict(fixed),
        search=dict(search),
        forbidden=forbidden,
        source_path=str(source_path),
    )


def load_search_space(path):
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return parse_search_space(payload, source_path=path)
