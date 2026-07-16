"""Pinned dependency checks for official/native Phase 2 estimators."""

from dataclasses import dataclass
from importlib import import_module, metadata
import sys


@dataclass(frozen=True)
class DependencySpec:
    distribution: str
    module: str
    version: str
    install_extra: str = ""
    source: str = ""
    source_revision: str = ""
    license: str = ""

    @property
    def requirement(self):
        suffix = f"[{self.install_extra}]" if self.install_extra else ""
        return f"{self.distribution}{suffix}=={self.version}"


DEPENDENCIES = {
    "xgboost": DependencySpec(
        distribution="xgboost",
        module="xgboost",
        version="2.1.4",
        source="https://github.com/dmlc/xgboost",
        license="Apache-2.0",
    ),
    "catboost": DependencySpec(
        distribution="catboost",
        module="catboost",
        version="1.2.10",
        source="https://github.com/catboost/catboost",
        license="Apache-2.0",
    ),
    "realmlp": DependencySpec(
        distribution="pytabkit",
        module="pytabkit",
        version="1.7.3",
        install_extra="models",
        source="https://github.com/dholzmueller/pytabkit",
        source_revision="c126ea51187c5080b91f28d352481dbd3b2194b0",
        license="Apache-2.0",
    ),
    "tabicl": DependencySpec(
        distribution="tabicl",
        module="tabicl",
        version="2.1.1",
        source="https://github.com/soda-inria/tabicl",
        source_revision="46b91961db4f8873dd049ec09990698a435e1e29",
        license="BSD-3-Clause",
    ),
}


class DependencyCompatibilityError(RuntimeError):
    """Raised when an official dependency is absent or not protocol-compatible."""


def dependency_report(model_name):
    key = model_name.lower()
    spec = DEPENDENCIES[key]
    try:
        installed = metadata.version(spec.distribution)
    except metadata.PackageNotFoundError:
        installed = None
    minimum_python = {"realmlp": (3, 9), "tabicl": (3, 10)}.get(key, (3, 8))
    python_compatible = sys.version_info[:2] >= minimum_python
    return {
        "distribution": spec.distribution,
        "required_version": spec.version,
        "installed_version": installed,
        "compatible": installed == spec.version and python_compatible,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "minimum_python": ".".join(map(str, minimum_python)),
        "python_compatible": python_compatible,
        "install_command": f"python -m pip install '{spec.requirement}'",
        "source": spec.source,
        "source_revision": spec.source_revision,
        "license": spec.license,
    }


def require_dependency(model_name):
    key = model_name.lower()
    if key not in DEPENDENCIES:
        raise ValueError(f"Unknown native model dependency: {model_name}")
    report = dependency_report(key)
    if not report["python_compatible"]:
        raise DependencyCompatibilityError(
            f"{report['distribution']}=={report['required_version']} requires Python "
            f">={report['minimum_python']}; current Python is {report['python_version']}. "
            "Use a compatible Phase 2 environment; no fallback is permitted."
        )
    if report["installed_version"] is None:
        raise DependencyCompatibilityError(
            f"{report['distribution']} is not installed. "
            f"Required: {report['install_command']}"
        )
    if not report["compatible"]:
        raise DependencyCompatibilityError(
            f"{report['distribution']}=={report['installed_version']} is incompatible "
            f"with this protocol; require =={report['required_version']}. "
            f"Run: {report['install_command']}"
        )
    return import_module(DEPENDENCIES[key].module), report
