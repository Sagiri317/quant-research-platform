"""Dependency status checks for supervised model algorithms."""

from __future__ import annotations

from importlib import import_module

from quant_research_platform.research.models import algorithm_facts


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = algorithm_facts.algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return {"status": "available", "package": "internal"}
    try:
        import_module(package)
    except ImportError:
        return {"status": "missing", "package": package}
    return {"status": "available", "package": package}


def require_model_dependency(algorithm: str) -> dict[str, str]:
    status = model_dependency_status(algorithm)
    if status["status"] == "missing":
        raise ImportError(f"model_dependency_unavailable: {status['package']}")
    return status


__all__ = ["model_dependency_status", "require_model_dependency"]
