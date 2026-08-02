from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_pyproject(repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    with (repo_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def project_requires_python(repo_root: Path = PROJECT_ROOT) -> str:
    return str(load_pyproject(repo_root)["project"]["requires-python"])


def python_minor_from_requires(spec: str) -> str:
    match = re.search(r">=\s*(\d+\.\d+)", spec)
    return match.group(1) if match else ""


def dependency_constraint(spec: str) -> str:
    match = re.search(r"(>=|<=|<|>|==)\s*[0-9]", spec)
    return spec[match.start() :] if match else ""


def build_system_requirement(name: str, repo_root: Path = PROJECT_ROOT) -> str:
    return _dependency_specs(load_pyproject(repo_root)["build-system"]["requires"])[
        name
    ]


def project_dependency_requirement(name: str, repo_root: Path = PROJECT_ROOT) -> str:
    return _dependency_specs(load_pyproject(repo_root)["project"]["dependencies"])[name]


def dev_dependency_requirement(name: str, repo_root: Path = PROJECT_ROOT) -> str:
    return _dependency_specs(
        load_pyproject(repo_root)["project"]["optional-dependencies"]["dev"]
    )[name]


def render_minimal_pyproject_from_root_contract(
    repo_root: Path = PROJECT_ROOT,
    *,
    include_project_deps: bool = True,
    include_dev_deps: bool = True,
    package_name: str = "quant-research-platform",
) -> str:
    pyproject = load_pyproject(repo_root)
    build_requires = pyproject["build-system"]["requires"]
    requires_python = pyproject["project"]["requires-python"]
    lines = [
        "[build-system]",
        f"requires = {_toml_inline_array(build_requires)}",
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{package_name}"',
        'version = "0.0.0"',
        f'requires-python = "{requires_python}"',
    ]
    if include_project_deps:
        project_deps = [project_dependency_requirement("pyarrow", repo_root)]
        lines.append(
            f"dependencies = {_toml_inline_array(project_deps)}"
        )
    if include_dev_deps:
        lines.extend(
            [
                "",
                "[project.optional-dependencies]",
                "dev = "
                + _toml_inline_array(
                    [
                        dev_dependency_requirement("build", repo_root),
                        dev_dependency_requirement("pytest", repo_root),
                    ]
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _dependency_specs(items: list[str]) -> dict[str, str]:
    return {re.split(r"[<>=!~; ]", item, maxsplit=1)[0]: item for item in items}


def _toml_inline_array(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"
