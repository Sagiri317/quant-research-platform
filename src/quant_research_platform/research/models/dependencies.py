"""Dependency status checks for supervised model algorithms."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import psutil

import quant_research_platform.research.models.algorithm_facts as algorithm_facts
from quant_research_platform._version import __version__ as qrp_version

_NATIVE_IMAGE_BYTES: dict[str, int] = {}


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = algorithm_facts.algorithm_facts(algorithm)
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
    package = algorithm_facts.algorithm_facts(algorithm).dependency_package
    return package is None or package in _NATIVE_IMAGE_BYTES


def preflight_model_dependency_native_image(algorithm: str) -> int:
    facts = algorithm_facts.algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return 0
    module = import_module(package)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("model_dependency_location_unavailable")
    module_root = Path(module_file).resolve().parent
    package_token = package.casefold().replace("-", "_")
    native_files: dict[str, int] = {}
    try:
        mappings = psutil.Process().memory_maps(grouped=True)
        for mapping in mappings:
            raw_path = mapping.path
            if not raw_path or raw_path.startswith("["):
                continue
            path = Path(raw_path)
            name = path.name.casefold()
            if not (
                name.endswith((".dll", ".pyd", ".dylib")) or ".so" in name
            ):
                continue
            resolved = path.resolve()
            if module_root not in resolved.parents and package_token not in name:
                continue
            native_files[str(resolved).casefold()] = resolved.stat().st_size
    except (OSError, psutil.Error) as exc:
        raise RuntimeError("model_native_image_inventory_unavailable") from exc
    total = sum(native_files.values())
    if total <= 0:
        raise RuntimeError("model_native_image_inventory_empty")
    _NATIVE_IMAGE_BYTES[package] = max(_NATIVE_IMAGE_BYTES.get(package, 0), total)
    return _NATIVE_IMAGE_BYTES[package]


def model_dependency_native_image_bytes(algorithm: str) -> int:
    package = algorithm_facts.algorithm_facts(algorithm).dependency_package
    if package is None:
        return 0
    try:
        return _NATIVE_IMAGE_BYTES[package]
    except KeyError as exc:
        raise RuntimeError("model_dependency_not_preflighted") from exc


__all__ = ["model_dependency_status", "require_model_dependency"]
