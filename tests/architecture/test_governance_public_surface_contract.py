from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture.governance_scan_helpers import (
    APPROVED_API_LINES,
    APPROVED_CLI_LINES,
    APPROVED_FIELD_SPEC_FILES,
    FUTURE_PUBLIC_TODO_TERMS,
    PRIVATE_LOCAL_CLI_LINES,
    PUBLIC_CONTRACT_DOCS,
    RETIRED_PUBLIC_CONTRACT_TEXT,
    call_name,
    future_data_hit,
    iter_governance_python_files,
    iter_public_surface_files,
    literal_string_items,
    public_surface_scan_text,
)


def test_public_surfaces_do_not_expose_internal_provider_markers() -> None:
    forbidden_patterns = (
        ("csmar", re.compile(r"(?i)\bcsmar\b")),
        ("source_id", re.compile(r"\bsource_id\b")),
        ("provider_id", re.compile(r"\bprovider_id\b")),
        ("implementation_package", re.compile(r"\bimplementation_package\b")),
        ("--private-real", re.compile(r"--private-real\b")),
        ("--source-root", re.compile(r"--source-root\b")),
        ("--input-root", re.compile(r"--input-root\b")),
        ("--data-root", re.compile(r"--data-root\b")),
        ("--provider-path", re.compile(r"--provider-path\b")),
        ("public source route", re.compile(r"/api/0\.1/sources\b")),
        ("sources/csmar", re.compile(r"sources[/\\]csmar")),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in iter_public_surface_files():
        if path.resolve() == current_file:
            continue
        try:
            text = public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []

def test_no_retired_top_level_source_directories_exist() -> None:
    retired = {
        "datasets",
        "pipelines",
        "kernels",
        "core",
        "data",
        "data_layers",
        "data_artifacts",
        "research_datasets",
    }
    root = Path("src/quant_research_platform")
    existing = {path.name for path in root.iterdir() if path.is_dir()}

    assert existing.isdisjoint(retired)

def test_public_surfaces_do_not_expose_retired_public_cli_commands() -> None:
    pattern = re.compile(r"\bbuild-interim\b")
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in iter_public_surface_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            hits.append(str(path))

    assert hits == []

def test_public_docs_record_exact_final_api_and_cli_surface() -> None:
    docs = {
        path: path.read_text(encoding="utf-8")
        for path in (
            Path("README.md"),
            Path("docs/api/README.md"),
            Path("docs/code-map/platform-flow.md"),
        )
    }
    missing: list[str] = []
    stale: list[str] = []

    for path, text in docs.items():
        for line in APPROVED_API_LINES + APPROVED_CLI_LINES:
            if line not in text:
                missing.append(f"{path}: {line}")
        for retired in RETIRED_PUBLIC_CONTRACT_TEXT:
            if retired in text:
                stale.append(f"{path}: {retired}")

    assert missing == []
    assert stale == []

def test_public_docs_capture_naming_rules_and_future_only_items() -> None:
    api_doc = Path("docs/api/README.md").read_text(encoding="utf-8")
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in PUBLIC_CONTRACT_DOCS
    )
    required_naming_rules = (
        "API resource segment: kebab-case + plural",
        "API path parameter: snake_case",
        "ID value: snake_case",
        "JSON field: snake_case",
        "CLI group / command / option: kebab-case",
        "Python module/function: snake_case",
    )
    missing = [rule for rule in required_naming_rules if rule not in api_doc]
    stale_future = [term for term in FUTURE_PUBLIC_TODO_TERMS if term in combined]
    missing_private_cli = [
        line for line in PRIVATE_LOCAL_CLI_LINES if line not in combined
    ]

    assert missing == []
    assert stale_future == []
    assert missing_private_cli == []

def test_factor_and_label_definitions_do_not_use_future_data_token() -> None:
    checked_classes = {"FactorDefinition", "LabelDefinition"}
    hits: list[str] = []
    for path in (
        Path("src/quant_research_platform/contracts/factors.py"),
        Path("src/quant_research_platform/contracts/labels.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in checked_classes:
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == "uses_future_data":
                        hits.append(future_data_hit(path, child.lineno, node.name))
                    if (
                        isinstance(child, ast.Constant)
                        and child.value == "uses_future_data"
                    ):
                        hits.append(future_data_hit(path, child.lineno, node.name))

    assert hits == []

def test_concrete_field_specs_only_live_in_catalogs() -> None:
    hits: list[str] = []
    for path in iter_governance_python_files():
        if path.parts[:2] != ("src", "quant_research_platform"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append(f"{path}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node.func) == "FieldSpec":
                if path not in APPROVED_FIELD_SPEC_FILES:
                    hits.append(f"{path}:{node.lineno}: FieldSpec")

    assert hits == []

def test_compute_modules_do_not_export_private_helpers() -> None:
    hits: list[str] = []
    for path in Path("src/quant_research_platform/compute").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            for value in literal_string_items(node.value):
                if value.startswith("_"):
                    hits.append(f"{path}: {value}")

    assert hits == []
