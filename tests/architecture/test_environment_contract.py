from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
ENVIRONMENT_YML = PROJECT_ROOT / "environment.yml"
GITIGNORE = PROJECT_ROOT / ".gitignore"

CONDA_NAMES = {
    "arch": "arch-py",
    "matplotlib": "matplotlib-base",
}


def test_python_and_ruff_contracts_follow_root_config() -> None:
    pyproject = _load_pyproject()
    requires_python = pyproject["project"]["requires-python"]
    python_minor = _python_minor_from_requires(requires_python)

    assert python_minor
    assert pyproject["tool"]["ruff"]["target-version"] == (
        "py" + python_minor.replace(".", "")
    )

    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    assert _environment_python_pin(environment) == python_minor


def test_repo_output_ignore_does_not_hide_source_packages() -> None:
    rules = GITIGNORE.read_text(encoding="utf-8").splitlines()

    assert "outputs/" not in rules
    assert "/outputs/" in rules


def test_core_dependencies_have_version_boundaries() -> None:
    pyproject = _load_pyproject()
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])

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
        name: (CONDA_NAMES.get(name, name), spec)
        for name, spec in (runtime | dev | build_system_requires).items()
    }
    mismatches = {
        pyproject_name: conda_name
        for pyproject_name, (conda_name, pyproject_spec) in expected.items()
        if conda_name not in conda_specs
        or not _conda_spec_has_same_bounds(conda_specs[conda_name], pyproject_spec)
    }

    assert mismatches == {}


def _python_minor_from_requires(spec: str) -> str:
    match = re.search(r">=\s*(\d+\.\d+)", spec)
    return match.group(1) if match else ""


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


def _conda_spec_has_same_bounds(conda_spec: str, pyproject_spec: str) -> bool:
    pyproject_bounds = _bounds(pyproject_spec)
    conda_bounds = _bounds(conda_spec)
    return bool(pyproject_bounds) and conda_bounds == pyproject_bounds


def _bounds(spec: str) -> set[tuple[str, str]]:
    return {
        (operator, version)
        for operator, version in re.findall(r"(>=|<=|<|>|==)\s*([0-9][^,;\s]*)", spec)
    }
