"""Dependency status checks for supervised model algorithms."""

from __future__ import annotations

from importlib import import_module

from quant_research_platform._version import __version__ as qrp_version
from quant_research_platform.contracts.models import _model_algorithm_facts

_PREFLIGHTED_PACKAGES: set[str] = set()


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = _model_algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return {"status": "available", "package": "internal", "version": qrp_version}
    try:
        module = import_module(package)
    except ImportError:
        return {"status": "missing", "package": package, "version": "unavailable"}
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("model_dependency_version_unavailable")
    return {"status": "available", "package": package, "version": version}


def require_model_dependency(algorithm: str) -> dict[str, str]:
    status = model_dependency_status(algorithm)
    if status["status"] == "missing":
        raise ImportError(f"model_dependency_unavailable: {status['package']}")
    return status


def model_dependency_preflighted(algorithm: str) -> bool:
    package = _model_algorithm_facts(algorithm).dependency_package
    return package is None or package in _PREFLIGHTED_PACKAGES


def preflight_model_dependency(algorithm: str) -> None:
    facts = _model_algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return
    import_module(package)
    _PREFLIGHTED_PACKAGES.add(package)


__all__ = ["model_dependency_status", "require_model_dependency"]
