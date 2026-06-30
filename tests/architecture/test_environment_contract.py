from __future__ import annotations

import re
import tomllib
from pathlib import Path

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

EXPECTED_PYTHON_SPEC = ">=3.13,<3.14"
EXPECTED_RUFF_TARGET = "py313"
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
    "coverage",
    "httpx",
    "hypothesis",
    "pytest",
    "pytest-cov",
    "ruff",
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


def test_python_and_ruff_contracts_are_locked_to_python_313() -> None:
    pyproject = _load_pyproject()

    assert pyproject["project"]["requires-python"] == EXPECTED_PYTHON_SPEC
    assert pyproject["tool"]["ruff"]["target-version"] == EXPECTED_RUFF_TARGET


def test_core_dependencies_have_version_boundaries() -> None:
    pyproject = _load_pyproject()
    runtime = _dependency_specs(pyproject["project"]["dependencies"])
    dev = _dependency_specs(pyproject["project"]["optional-dependencies"]["dev"])

    assert set(runtime) == RUNTIME_DEPENDENCIES
    assert set(dev) == DEV_DEPENDENCIES
    assert _unbounded(runtime) == []
    assert _unbounded(dev) == []


def test_engineering_contract_points_to_executable_environment_contracts() -> None:
    text = ENGINEERING_CONTRACT.read_text(encoding="utf-8")
    environment = ENVIRONMENT_YML.read_text(encoding="utf-8")

    assert "pyproject.toml" in text
    assert "environment.yml" in text
    assert f'requires-python = "{EXPECTED_PYTHON_SPEC}"' in PYPROJECT.read_text(
        encoding="utf-8"
    )
    assert "python=3.13" in environment
    assert "quantlab313" in environment


def test_engineering_contract_includes_current_formal_commands() -> None:
    text = ENGINEERING_CONTRACT.read_text(encoding="utf-8")

    assert "```bash" in text
    assert "git diff --check" in text
    assert "python tests/tools/run_profile.py --profile fast" in text
    assert (
        "python tests/tools/run_profile.py --profile fast "
        "--run-dir .run/<purpose>/fast"
    ) in text
    assert " --run-dir .run/<purpose>/fast" in text
    assert (
        "verify_governance.py --governance-only --scan-scope active-docs" in text
    )
    assert "verify_code_map_flow.py --repo-root ." in text


def test_default_environment_update_commands_do_not_prune_quantlab313() -> None:
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
