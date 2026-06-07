from __future__ import annotations

import ast
import re
from pathlib import Path

SCAN_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("docs"),
    Path(".agents"),
    Path("README.md"),
    Path("data/README.md"),
    Path("AGENTS.md"),
    Path("PROJECT_STATUS.md"),
    Path("pyproject.toml"),
)

EXCLUDED_SCAN_PARTS = {
    ("docs", "archive"),
    ("__pycache__",),
}

PUBLIC_SURFACE_SCAN_ROOTS = (
    Path("src/quant_research_platform/interfaces"),
    Path("src/quant_research_platform/agents"),
    Path("src/quant_research_platform/reports"),
    Path("docs"),
    Path(".agents"),
    Path("README.md"),
    Path("data/README.md"),
    Path("AGENTS.md"),
    Path("PROJECT_STATUS.md"),
)

INTERNAL_PROVIDER_SCAN_EXCLUSIONS = {
    ("docs", "architecture", "open-issues-register.md"),
    ("docs", "code-map", "source-csmar-contracts.md"),
    ("src", "quant_research_platform", "reports", "schema_inventory.py"),
}

APPROVED_FIELD_SPEC_FILES = {
    Path("src/quant_research_platform/catalog/fields.py"),
    Path("src/quant_research_platform/markets/implementations/a_share/catalog.py"),
}

APPROVED_WRITER_MODULES = {
    Path("src/quant_research_platform/storage/atomic.py"),
    Path("src/quant_research_platform/storage/parquet.py"),
    Path("src/quant_research_platform/storage/manifests.py"),
    Path("src/quant_research_platform/reports/bundles.py"),
    Path("src/quant_research_platform/reports/manifests.py"),
    Path("src/quant_research_platform/reports/private_evidence.py"),
    Path("src/quant_research_platform/runtime/timing.py"),
}

PUBLIC_CONTRACT_DOCS = (
    Path("README.md"),
    Path("docs/api/README.md"),
    Path("docs/code-map/platform-flow.md"),
    Path(".agents/briefs/current-qrp-status.md"),
    Path("PROJECT_STATUS.md"),
)

APPROVED_API_LINES = (
    "GET  /api/0.1/catalog/data-products",
    "GET  /api/0.1/catalog/factors",
    "GET  /api/0.1/catalog/labels",
    "GET  /api/0.1/catalog/universes",
    "POST /api/0.1/data-products/{data_product_id}/interim-runs",
    "GET  /api/0.1/research-datasets/{dataset_id}/requirements",
)

APPROVED_CLI_LINES = (
    "qrp catalog data-products",
    "qrp catalog factors",
    "qrp catalog labels",
    "qrp catalog universes",
    "qrp data-products run-interim <data_product_id>",
    "qrp data-products run-interim <data_product_id> --dry-run",
    "qrp data-products run-interim <data_product_id> --run-id <run_id>",
    "qrp research-datasets requirements <dataset_id>",
)

RETIRED_PUBLIC_CONTRACT_TEXT = (
    "/api/0.1/data-products/a_share_daily_bar/interim",
    "/api/0.1/markets/{market_id}/processed-data",
    "POST /api/0.1/research-datasets",
    "POST /api/0.1/workflows",
    "GET  /api/0.1/reports/{run_id}",
    "qrp data-product interim",
    "qrp market build-processed",
    "qrp research build-dataset",
    "qrp workflow run",
    "qrp report show",
)

FUTURE_PUBLIC_TODO_TERMS = (
    "processed-data-runs",
    "workflow-runs",
    "report index/show",
    "RunRepository",
    "ReportRepository",
    "private plugin",
)
PRIVATE_LOCAL_CLI_LINES = (
    "qrp-private schema-inventory --project-root . --output <path>",
)

FORBIDDEN_WRITER_METHODS = {
    "to_parquet",
    "to_csv",
    "to_json",
    "write_parquet",
    "write_csv",
    "write_json",
    "write_text",
    "write_bytes",
}


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and not _is_excluded(path):
                files.append(path)
    return files


def _iter_python_files() -> list[Path]:
    return [path for path in _iter_text_files() if path.suffix == ".py"]


def _is_excluded(path: Path) -> bool:
    parts = path.parts
    return any(_contains_parts(parts, excluded) for excluded in EXCLUDED_SCAN_PARTS)


def _contains_parts(parts: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle for index in range(len(parts))
    )


def test_active_tree_has_no_retired_terms() -> None:
    forbidden_patterns = (
        ("raw_private", re.compile(r"raw_private")),
        ("interim_private", re.compile(r"interim_private")),
        ("processed_private", re.compile(r"processed_private")),
        ("processed_private/csmar", re.compile(r"processed_private[/\\]csmar")),
        ("factor_set_tier", re.compile(r"factor_set_tier")),
        ("core_long_period", re.compile(r"core_long_period")),
        ("smoke_minimal", re.compile(r"smoke_minimal")),
        ("factor_rebuild.py", re.compile(r"factor_rebuild\.py")),
        ("factor_rebuild.parquet", re.compile(r"factor_rebuild\.parquet")),
        ("prediction-handoff", re.compile(r"prediction-handoff")),
        ("evaluate-parquet", re.compile(r"evaluate-parquet")),
        ("/api/runs", re.compile(r"/api/runs\b")),
        ("/api/pipeline", re.compile(r"/api/pipeline\b")),
        ("/api/v0.1", re.compile(r"/api/v0\.1\b")),
        ("/api/v1", re.compile(r"/api/v1\b")),
        ("schema 0.7", re.compile(r"schema\s+`?0\.7`?")),
        ("schema 0.6", re.compile(r"schema\s+`?0\.6`?")),
        ("schema 1.0", re.compile(r"schema\s+`?1\.0`?")),
        ("panel_full", re.compile(r"panel_full")),
        ("price_volume_factors_full", re.compile(r"price_volume_factors_full")),
        (
            "_".join(("csmar", "daily", "market", "interim")),
            re.compile(r"csmar_" r"daily_market_interim"),
        ),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []


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
    for path in _iter_public_surface_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []


def _iter_public_surface_files() -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_SURFACE_SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and not _is_excluded(path)
                and not _is_internal_provider_doc(path)
            ):
                files.append(path)
    return files


def _is_internal_provider_doc(path: Path) -> bool:
    if path.parts[:2] == ("docs", "code-map") and path.name.startswith(
        "source-csmar-"
    ):
        return True
    parts = path.parts
    return any(
        _contains_parts(parts, excluded)
        for excluded in INTERNAL_PROVIDER_SCAN_EXCLUSIONS
    )


def test_no_retired_top_level_source_directories_exist() -> None:
    retired = {
        "artifacts",
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
    for path in _iter_public_surface_files():
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
    missing = [
        rule for rule in required_naming_rules if rule not in api_doc
    ]
    missing_future = [
        term for term in FUTURE_PUBLIC_TODO_TERMS if term not in combined
    ]
    missing_private_cli = [
        line for line in PRIVATE_LOCAL_CLI_LINES if line not in combined
    ]

    assert missing == []
    assert missing_future == []
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
                        hits.append(_future_data_hit(path, child.lineno, node.name))
                    if (
                        isinstance(child, ast.Constant)
                        and child.value == "uses_future_data"
                    ):
                        hits.append(_future_data_hit(path, child.lineno, node.name))

    assert hits == []


def _future_data_hit(path: Path, lineno: int, class_name: str) -> str:
    return f"{path}:{lineno}: {class_name}.uses_future_data"


def test_concrete_field_specs_only_live_in_catalogs() -> None:
    hits: list[str] = []
    for path in _iter_python_files():
        if path.parts[:2] != ("src", "quant_research_platform"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append(f"{path}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == "FieldSpec":
                if path not in APPROVED_FIELD_SPEC_FILES:
                    hits.append(f"{path}:{node.lineno}: FieldSpec")

    assert hits == []


def test_business_logic_does_not_bypass_persistent_writers() -> None:
    hits: list[str] = []
    for path in _iter_python_files():
        if _is_approved_writer_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append(f"{path}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_WRITER_METHODS:
                hits.append(f"{path}:{node.lineno}: {call_name}")
                continue
            if _is_open_write_call(node):
                hits.append(f"{path}:{node.lineno}: open(...).write")
                continue
            if _is_open_call(node.func):
                mode = _open_mode(node)
                if mode is None:
                    hits.append(f"{path}:{node.lineno}: open with dynamic mode")
                elif _is_write_mode(mode):
                    hits.append(f"{path}:{node.lineno}: open mode={mode!r}")

    assert hits == []


def _is_approved_writer_path(path: Path) -> bool:
    return path.parts[0] == "tests" or path in APPROVED_WRITER_MODULES


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_open_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "open"
    return isinstance(func, ast.Attribute) and func.attr == "open"


def _is_open_write_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "write"
        and isinstance(call.func.value, ast.Call)
        and _is_open_call(call.func.value.func)
    )


def _open_mode(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return _literal_string(keyword.value)

    mode_index = 1 if isinstance(call.func, ast.Name) else 0
    if len(call.args) <= mode_index:
        return "r"
    return _literal_string(call.args[mode_index])


def _literal_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def test_direct_write_gate_detects_chained_open_write() -> None:
    tree = ast.parse("open('artifact.txt', 'w').write('unsafe')\n")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(_is_open_write_call(call) for call in calls)


def test_approved_writer_signatures_do_not_accept_arbitrary_output_roots() -> None:
    import inspect

    from quant_research_platform.reports.bundles import (
        write_evaluation_report_files,
        write_prediction_bundle_files,
        write_stage_profile_bundle,
    )
    from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler

    writer_signatures = (
        inspect.signature(write_stage_profile_bundle),
        inspect.signature(write_prediction_bundle_files),
        inspect.signature(write_evaluation_report_files),
        inspect.signature(StageProfiler.write_artifacts),
    )

    assert "output_dir" not in ProfileConfig.__dataclass_fields__
    for signature in writer_signatures:
        assert "output_root" not in signature.parameters
        assert "output_dir" not in signature.parameters
