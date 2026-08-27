from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tests.helpers.root_config_contract import python_minor_from_requires

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
ENVIRONMENT_YML = PROJECT_ROOT / "environment.yml"

RUNTIME_DEPENDENCIES = {
    "fastapi",
    "httpx",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "psutil",
    "scikit-learn",
    "threadpoolctl",
    "xgboost",
    "lightgbm",
    "typer",
    "websockets",
}
DEV_DEPENDENCIES = {
    "build",
    "pytest",
    "ruff",
}
BUILD_SYSTEM_TO_CONDA_PACKAGES = {
    "build": "python-build",
    "setuptools": "setuptools",
    "wheel": "wheel",
}


def test_python_and_ruff_contracts_follow_root_config() -> None:
    pyproject = _load_pyproject()
    requires_python = pyproject["project"]["requires-python"]
    python_minor = python_minor_from_requires(requires_python)

    assert python_minor
    assert pyproject["tool"]["ruff"]["target-version"] == (
        "py" + python_minor.replace(".", "")
    )


def test_core_dependencies_have_version_boundaries() -> None:
    pyproject = _load_pyproject()
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])

    assert set(runtime) == RUNTIME_DEPENDENCIES
    assert set(dev) == DEV_DEPENDENCIES
    assert _unbounded(runtime) == []
    assert _unbounded(dev) == []


def test_tooling_environment_dependencies_match_pyproject_dependency_specs() -> None:
    pyproject = _load_pyproject()
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])
    build_system_requires = _dependency_specs(pyproject["build-system"]["requires"])
    conda_specs = _environment_dependency_specs(environment)
    expected = {
        "build": ("python-build", dev["build"]),
        "setuptools": ("setuptools", build_system_requires["setuptools"]),
        "wheel": ("wheel", build_system_requires["wheel"]),
        "pyarrow": ("pyarrow", runtime["pyarrow"]),
        "threadpoolctl": ("threadpoolctl", runtime["threadpoolctl"]),
        "pytest": ("pytest", dev["pytest"]),
        "ruff": ("ruff", dev["ruff"]),
    }
    mismatches = {
        pyproject_name: conda_name
        for pyproject_name, (conda_name, pyproject_spec) in expected.items()
        if conda_name not in conda_specs
        or not _conda_spec_is_same_or_stricter(conda_specs[conda_name], pyproject_spec)
    }

    assert mismatches == {}


def test_environment_python_pin_matches_requires_python_lower_bound() -> None:
    pyproject = _load_pyproject()
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    expected_minor = python_minor_from_requires(pyproject["project"]["requires-python"])

    assert _environment_python_pin(environment) == expected_minor


def test_build_system_requires_conda_managed_frontend_and_backend() -> None:
    pyproject = _load_pyproject()
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    conda_specs = _environment_dependency_specs(environment)
    build_system_requires = _dependency_specs(pyproject["build-system"]["requires"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])

    missing = {}
    if "build" not in dev or "python-build" not in conda_specs:
        missing["build"] = "python-build"
    missing.update(
        {
            pyproject_name: conda_name
            for pyproject_name, conda_name in BUILD_SYSTEM_TO_CONDA_PACKAGES.items()
            if pyproject_name != "build"
            and (
                pyproject_name not in build_system_requires
                or conda_name not in conda_specs
            )
        }
    )
    loose_bounds = {
        pyproject_name: conda_name
        for pyproject_name, conda_name in BUILD_SYSTEM_TO_CONDA_PACKAGES.items()
        if pyproject_name in build_system_requires
        and conda_name in conda_specs
        and not _conda_spec_is_same_or_stricter(
            conda_specs[conda_name], build_system_requires[pyproject_name]
        )
    }

    assert "build" in dev
    assert missing == {}
    assert loose_bounds == {}


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _environment_dependency_specs(text: str) -> dict[str, str]:
    specs: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        spec = stripped[2:].strip()
        if not spec or spec in {"conda-forge", "nodefaults"}:
            continue
        name = re.split(r"[<>=!~; ]", spec, maxsplit=1)[0]
        specs[name] = spec
    return specs


def _environment_python_pin(text: str) -> str:
    specs = _environment_dependency_specs(text)
    python_spec = specs.get("python", "")
    match = re.fullmatch(r"python=(\d+\.\d+)", python_spec)
    return match.group(1) if match else ""


def _dependency_specs(items: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items:
        name = re.split(r"[<>=!~; ]", item, maxsplit=1)[0]
        specs[name] = item
    return specs


def _unbounded(specs: dict[str, str]) -> list[str]:
    return sorted(
        name for name, spec in specs.items() if ">=" not in spec or "<" not in spec
    )


def _conda_spec_is_same_or_stricter(conda_spec: str, pyproject_spec: str) -> bool:
    pyproject_bounds = _bounds(pyproject_spec)
    conda_bounds = _bounds(conda_spec)
    if not pyproject_bounds:
        return False
    for operator, version in pyproject_bounds:
        if (operator, version) not in conda_bounds:
            return False
    return True


def _bounds(spec: str) -> set[tuple[str, str]]:
    return {
        (operator, version)
        for operator, version in re.findall(r"(>=|<=|<|>|==)\s*([0-9][^,;\s]*)", spec)
    }
