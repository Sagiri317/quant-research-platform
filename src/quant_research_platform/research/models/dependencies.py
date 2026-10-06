"""Dependency status checks for supervised model algorithms."""

from __future__ import annotations

from importlib import import_module

from quant_research_platform.contracts.models import _model_algorithm_facts


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = _model_algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return {"status": "available", "package": "internal"}
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
