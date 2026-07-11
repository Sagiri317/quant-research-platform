from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tests.helpers.root_config_contract import python_minor_from_requires

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
ENVIRONMENT_YML = PROJECT_ROOT / "environment.yml"
ENGINEERING_CONTRACT = PROJECT_ROOT / "docs/architecture/engineering-contract.md"
README = PROJECT_ROOT / "README.md"
VERIFICATION_AND_GATES = (
    PROJECT_ROOT
    / ".agents/skills/quant-research-platform-maintenance/references"
    / "verification-and-gates.md"
)
TOOLING_AND_PORTABILITY = (
    PROJECT_ROOT
    / ".agents/skills/quant-research-platform-maintenance/references"
    / "tooling-and-portability.md"
)

RUNTIME_DEPENDENCIES = {
    "fastapi",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "psutil",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "typer",
    "uvicorn",
}
DEV_DEPENDENCIES = {
    "build",
    "coverage",
    "httpx",
    "hypothesis",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "ruff",
}
PYPROJECT_TO_CONDA_DEV_PACKAGES = {
    "build": "python-build",
    "pytest-xdist": "pytest-xdist",
}
BUILD_SYSTEM_TO_CONDA_PACKAGES = {
    "build": "python-build",
    "setuptools": "setuptools",
    "wheel": "wheel",
}
PRIVATE_DEPENDENCY_TOKENS = (
    ".run/",
    ".run\\",
    "handoff",
    "data/raw",
    "data\\raw",
    "data/interim",
    "data\\interim",
    "data/processed",
    "data\\processed",
    "data/research",
    "data\\research",
)
LOCAL_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?!path[\\/])"),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
)


def test_python_and_ruff_contracts_follow_root_config() -> None:
    pyproject = _load_pyproject()
    requires_python = pyproject["project"]["requires-python"]

    assert python_minor_from_requires(requires_python)
    assert pyproject["tool"]["ruff"]["target-version"] == _ruff_target_from_requires(
        requires_python
    )


def test_core_dependencies_have_version_boundaries() -> None:
    pyproject = _load_pyproject()
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])

    assert set(runtime) == RUNTIME_DEPENDENCIES
    assert set(dev) == DEV_DEPENDENCIES
    assert _unbounded(runtime) == []
    assert _unbounded(dev) == []


def test_pyproject_dev_dependencies_have_conda_environment_packages() -> None:
    pyproject = _load_pyproject()
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])
    conda_packages = _environment_dependency_names(environment)

    missing = {
        pyproject_name: conda_name
        for pyproject_name, conda_name in PYPROJECT_TO_CONDA_DEV_PACKAGES.items()
        if pyproject_name in dev and conda_name not in conda_packages
    }

    assert missing == {}


def test_pytest_xdist_is_dev_only_speed_gate_dependency() -> None:
    pyproject = _load_pyproject()
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])
    build_system_requires = _dependency_specs(pyproject["build-system"]["requires"])
    conda_packages = _environment_dependency_specs(environment)

    assert "pytest-xdist" in dev
    assert conda_packages["pytest-xdist"] == dev["pytest-xdist"]
    assert "pytest-xdist" not in runtime
    assert "pytest-xdist" not in build_system_requires


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
        "pytest": ("pytest", dev["pytest"]),
        "pytest-xdist": ("pytest-xdist", dev["pytest-xdist"]),
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
    expected_minor = python_minor_from_requires(
        pyproject["project"]["requires-python"]
    )

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


def test_engineering_contract_points_to_executable_environment_contracts() -> None:
    text = ENGINEERING_CONTRACT.read_text(encoding="utf-8")
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")
    environment_name = _environment_name(environment)
    pyproject = _load_pyproject()
    requires_python = pyproject["project"]["requires-python"]
    python_pin = _environment_python_pin(environment)

    assert "pyproject.toml" in text
    assert "environment.yml" in text
    assert f'requires-python = "{requires_python}"' in PYPROJECT.read_text(
        encoding="utf-8"
    )
    assert f"python={python_pin}" in environment
    assert environment_name


def test_engineering_contract_includes_current_formal_commands() -> None:
    text = ENGINEERING_CONTRACT.read_text(encoding="utf-8")

    assert "```bash" in text
    assert "git diff --check" in text
    assert "python tests/tools/run_profile.py --profile fast" in text
    assert (
        "python tests/tools/run_profile.py --profile fast "
        "--run-dir .run/<purpose>/fast"
    ) in text
    assert (
        "python tests/tools/run_profile.py --profile all "
        "--run-dir .run/<purpose>/all -- --durations=100"
    ) in text
    assert (
        "python tests/tools/run_profile.py --profile all "
        "--run-dir .run/<purpose>/all-xdist -- -n 4 --dist loadfile "
        "--durations=100"
    ) in text
    assert "pytest-xdist" in text
    assert "speed gate" in text
    assert " --run-dir .run/<purpose>/fast" in text
    assert (
        "verify_governance.py --governance-only --scan-scope active-docs" in text
    )
    assert "verify_code_map_flow.py --repo-root ." in text


def test_default_environment_update_commands_keep_declared_env() -> None:
    checked = {
        "README.md": README,
        "docs/architecture/engineering-contract.md": ENGINEERING_CONTRACT,
        "verification-and-gates.md": VERIFICATION_AND_GATES,
        "tooling-and-portability.md": TOOLING_AND_PORTABILITY,
    }
    unsafe_defaults: list[str] = []
    for label, path in checked.items():
        text = path.read_text(encoding="utf-8")
        for block in _code_blocks(text):
            if "conda env update -f environment.yml --prune" in block:
                unsafe_defaults.append(label)

    assert unsafe_defaults == []


def test_prune_mentions_are_marked_as_non_default_qrp_minimal_rebuild() -> None:
    checked = {
        "README.md": README,
        "docs/architecture/engineering-contract.md": ENGINEERING_CONTRACT,
        "verification-and-gates.md": VERIFICATION_AND_GATES,
        "tooling-and-portability.md": TOOLING_AND_PORTABILITY,
    }
    missing_context: list[str] = []
    for label, path in checked.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "--prune" not in line:
                continue
            nearby = " ".join(lines[max(0, index - 3) : index + 4]).lower()
            has_default_warning = (
                "not default" in nearby
                or ("do not use" in nearby and "default" in nearby)
            )
            has_qrp_minimal_rebuild_context = (
                "qrp-minimal" in nearby and "intentionally rebuilding" in nearby
            )
            if not (has_default_warning and has_qrp_minimal_rebuild_context):
                missing_context.append(f"{label}:{index + 1}")

    assert missing_context == []


def test_formal_verification_commands_do_not_depend_on_private_artifacts() -> None:
    text = ENGINEERING_CONTRACT.read_text(encoding="utf-8")
    command_sections = "\n".join(
        line
        for block in _code_blocks(text)
        for line in block.splitlines()
        if "tests/tools/run_profile.py" not in line
    )

    hits = [
        token
        for token in PRIVATE_DEPENDENCY_TOKENS
        if token.lower() in command_sections.lower()
    ]

    assert hits == []


def test_environment_contract_docs_do_not_hardcode_local_absolute_paths() -> None:
    checked = {
        "docs/architecture/engineering-contract.md": ENGINEERING_CONTRACT,
        "README.md": README,
    }
    hits: list[str] = []
    for label, path in checked.items():
        text = path.read_text(encoding="utf-8")
        for pattern in LOCAL_ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                hits.append(label)

    assert hits == []


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _environment_name(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return ""


def _environment_dependency_names(text: str) -> set[str]:
    return set(_environment_dependency_specs(text))


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
        name
        for name, spec in specs.items()
        if ">=" not in spec or "<" not in spec
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


def _ruff_target_from_requires(spec: str) -> str:
    minor = python_minor_from_requires(spec)
    return "py" + minor.replace(".", "")


def _code_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if in_block:
                blocks.append("\n".join(current))
                current = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            current.append(line)
    return blocks
