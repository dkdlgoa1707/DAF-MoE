"""Canonical Phase 2 study identity and storage-path construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.data.provenance import stable_hash
from src.data.splits import RawSplitRegistry
from src.phase2_protocol import (
    HPO_SEED,
    PROTOCOL_VERSION,
    model_implementation_version,
    resolve_target_policy,
)


SIGNATURE_COMPONENT_KEYS = (
    "protocol_version",
    "dataset_name",
    "dataset_schema_hash",
    "dataset_schema_version",
    "model_name",
    "model_implementation_version",
    "task_type",
    "optimize_metric",
    "search_space_hash",
    "base_experiment_config_hash",
    "effective_regression_target_policy",
)


def _slug(value):
    return str(value).lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class StudyIdentity:
    components: Mapping[str, object]
    signature: str

    @classmethod
    def from_components(cls, components):
        components = dict(components)
        missing = [key for key in SIGNATURE_COMPONENT_KEYS if key not in components]
        extra = sorted(set(components).difference(SIGNATURE_COMPONENT_KEYS))
        if missing or extra:
            raise ValueError(
                f"Study signature components mismatch: missing={missing}, extra={extra}"
            )
        return cls(components=components, signature=stable_hash(components))

    @property
    def prefix(self):
        return self.signature[:12]

    @property
    def study_name(self):
        return (
            f"{_slug(self.components['dataset_name'])}__"
            f"{self.components['model_name']}__{PROTOCOL_VERSION}__{self.prefix}"
        )

    @property
    def study_attributes(self):
        return {
            "study_signature": self.signature,
            "study_signature_components": dict(self.components),
            "protocol_version": self.components["protocol_version"],
            "search_space_hash": self.components["search_space_hash"],
            "target_policy": self.components["effective_regression_target_policy"],
            "model_implementation_version": self.components[
                "model_implementation_version"
            ],
        }

    def default_storage_url(self, root="results/phase2/hpo"):
        return f"sqlite:///{Path(root) / (self.study_name + '.db')}"


def build_study_identity(raw_dataset, base_config, search_space):
    model_name = search_space.model_name
    task_type = base_config["task_type"]
    split_registry = RawSplitRegistry(raw_dataset, task_type, HPO_SEED)
    components = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": raw_dataset.dataset_name,
        "dataset_schema_hash": split_registry.dataset_schema_hash,
        "dataset_schema_version": split_registry.dataset_schema_version,
        "model_name": model_name,
        "model_implementation_version": model_implementation_version(model_name),
        "task_type": task_type,
        "optimize_metric": base_config["optimize_metric"],
        "search_space_hash": search_space.schema_hash,
        "base_experiment_config_hash": stable_hash(dict(base_config)),
        "effective_regression_target_policy": resolve_target_policy(
            model_name, task_type
        ),
    }
    return StudyIdentity.from_components(components)


def resolve_study_storage(identity, storage_url=None, root="results/phase2/hpo"):
    storage_url = storage_url or identity.default_storage_url(root)
    if storage_url.startswith("sqlite:///"):
        path = Path(storage_url[len("sqlite:///"):])
        expected_name = identity.study_name + ".db"
        if path.name != expected_name:
            raise ValueError(
                "Custom SQLite storage must be signature-isolated; expected file "
                f"name {expected_name}, got {path.name}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    return storage_url
