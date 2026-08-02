"""Minimal structural audit for QRP maintenance tooling."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

HANDOFF_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    findings: tuple[str, ...] = ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    checks = run_audit(Path(args.repo_root).resolve())
    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    payload = {
        "schema_id": "qrp-structure-audit",
        "status": status,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "findings": list(check.findings),
            }
            for check in checks
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"{check.status} {check.name}")
            for finding in check.findings:
                print(f"  {finding}")
    return 0 if status == "PASS" else 1


def run_audit(repo_root: Path) -> tuple[AuditCheck, ...]:
    return (
        _tooling_environment_dependency_sync(repo_root),
        _agents_scripts_self_contained_boundary(repo_root),
        _no_duplicate_validation_or_command_ids(repo_root),
    )


def _tooling_environment_dependency_sync(repo_root: Path) -> AuditCheck:
    pyproject_path = repo_root / "pyproject.toml"
    environment_path = repo_root / "environment.yml"
    if not pyproject_path.exists():
        return _check(
            "tooling_environment_dependency_sync",
            ("pyproject.toml:missing",),
        )
    if not environment_path.exists():
        return _check(
            "tooling_environment_dependency_sync",
            ("environment.yml:missing",),
        )
    pyproject = _load_pyproject(repo_root)
    environment_specs = _environment_dependency_specs(
        environment_path.read_text(encoding="utf-8")
    )
    project = pyproject.get("project", {})
    build_system = pyproject.get("build-system", {})
    tool = pyproject.get("tool", {})
    runtime = _dependency_specs(
        project.get("dependencies", []) if isinstance(project, Mapping) else []
    )
    dev = _dependency_specs(
        project.get("optional-dependencies", {}).get("dev", [])
        if isinstance(project, Mapping)
        and isinstance(project.get("optional-dependencies"), Mapping)
        else []
    )
    build_requires = _dependency_specs(
        build_system.get("requires", []) if isinstance(build_system, Mapping) else []
    )
    expected_specs = {**runtime, **dev, **build_requires}
    conda_names = {"build": "python-build"}
    findings: list[str] = []
    for pyproject_name, expected_spec in expected_specs.items():
        conda_name = conda_names.get(pyproject_name, pyproject_name)
        conda_spec = environment_specs.get(conda_name, "")
        if not conda_spec:
            findings.append(
                f"environment.yml:dependency_missing:{pyproject_name}->{conda_name}"
            )
        elif not _conda_spec_is_same_or_stricter(conda_spec, expected_spec):
            findings.append(
                f"environment.yml:dependency_mismatch:{pyproject_name}->{conda_name}"
            )

    requires_python = (
        str(project.get("requires-python", "")) if isinstance(project, Mapping) else ""
    )
    python_match = re.search(r">=\s*(\d+\.\d+)", requires_python)
    expected_python = python_match.group(1) if python_match else ""
    if (
        expected_python
        and _environment_python_pin(environment_specs) != expected_python
    ):
        findings.append(
            f"environment.yml:python_pin_mismatch:expected={expected_python}"
        )
    ruff_config = tool.get("ruff", {}) if isinstance(tool, Mapping) else {}
    actual_ruff_target = (
        str(ruff_config.get("target-version", ""))
        if isinstance(ruff_config, Mapping)
        else ""
    )
    expected_ruff_target = (
        "py" + expected_python.replace(".", "") if expected_python else ""
    )
    if expected_ruff_target and actual_ruff_target != expected_ruff_target:
        findings.append(
            f"pyproject.toml:ruff_target_mismatch:expected={expected_ruff_target}"
        )
    return _check("tooling_environment_dependency_sync", findings)


def _agents_scripts_self_contained_boundary(repo_root: Path) -> AuditCheck:
    scripts_root = (
        repo_root
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
    )
    if not scripts_root.exists():
        return _check(
            "agents_scripts_self_contained_boundary",
            ("scripts_root:missing",),
        )
    findings = [
        f"{path.relative_to(repo_root).as_posix()}:{imported}"
        for path in sorted(scripts_root.rglob("*.py"))
        for imported in _imported_modules(path)
        if imported == "tests" or imported.startswith("tests.")
    ]
    return _check("agents_scripts_self_contained_boundary", findings)


def _no_duplicate_validation_or_command_ids(repo_root: Path) -> AuditCheck:
    module_names = ("handoff_contract", "receipt_contract")
    previous_path = list(sys.path)
    previous_modules = {
        name: sys.modules[name] for name in module_names if name in sys.modules
    }
    loaded = {}
    try:
        handoff_root = repo_root / HANDOFF_ROOT
        sys.path[:0] = [str(handoff_root), str(repo_root / "src")]
        for name in module_names:
            path = handoff_root / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"handoff module unavailable: {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        sys.path[:] = previous_path
        for name in module_names:
            if name in previous_modules:
                sys.modules[name] = previous_modules[name]
            else:
                sys.modules.pop(name, None)
    handoff_contract = loaded["handoff_contract"]
    receipt_contract = loaded["receipt_contract"]
    row_counts = Counter(handoff_contract.MANDATORY_VALIDATION_ROWS)
    command_counts = Counter(receipt_contract.REQUIRED_CHECK_IDS)
    findings = [
        f"validation_row:{name}" for name, count in row_counts.items() if count > 1
    ]
    findings.extend(
        f"command_id:{name}" for name, count in command_counts.items() if count > 1
    )
    row_names = set(handoff_contract.MANDATORY_VALIDATION_ROWS)
    if set(handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT) != row_names:
        findings.append("validation_row_source_contract_mismatch")
    command_names = set(receipt_contract.REQUIRED_CHECK_IDS)
    if set(handoff_contract.REPLAY_COMMAND_CONTRACT) != command_names:
        findings.append("replay_command_contract_mismatch")
    findings.extend(
        f"replay_command_id_mismatch:{name}"
        for name, spec in handoff_contract.REPLAY_COMMAND_CONTRACT.items()
        if spec.command_id != name
    )
    return _check("no_duplicate_validation_or_command_ids", findings)


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


def _load_pyproject(repo_root: Path) -> dict[str, object]:
    try:
        with (repo_root / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _dependency_specs(items: Iterable[str]) -> dict[str, str]:
    return {
        re.split(r"[<>=!~; ]", str(item), maxsplit=1)[0]: str(item) for item in items
    }


def _conda_spec_is_same_or_stricter(conda_spec: str, pyproject_spec: str) -> bool:
    pyproject_bounds = _bounds(pyproject_spec)
    return bool(pyproject_bounds) and pyproject_bounds <= _bounds(conda_spec)


def _bounds(spec: str) -> set[tuple[str, str]]:
    return set(re.findall(r"(>=|<=|<|>|==)\s*([0-9][^,;\s]*)", spec))


def _environment_python_pin(environment_specs: Mapping[str, str]) -> str:
    match = re.fullmatch(r"python=(\d+\.\d+)", environment_specs.get("python", ""))
    return match.group(1) if match else ""


def _parse_python_file(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return ast.Module(body=[], type_ignores=[])


def _imported_modules(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_parse_python_file(path)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def _check(name: str, findings: Iterable[str]) -> AuditCheck:
    items = tuple(findings)
    return AuditCheck(
        name=name,
        status="PASS" if not items else "FAIL",
        findings=items,
    )


if __name__ == "__main__":
    raise SystemExit(main())
