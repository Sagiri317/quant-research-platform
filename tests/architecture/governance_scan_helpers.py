from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

GOVERNANCE_SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
)
if str(GOVERNANCE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GOVERNANCE_SCRIPT_DIR))
import governance_contract as _governance_contract  # noqa: E402

CONTRACT_ACTIVE_GOVERNANCE_DOC_ROOTS = (
    _governance_contract.ACTIVE_GOVERNANCE_DOC_ROOTS
)
PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES = (
    _governance_contract.PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES
)
RETIRED_NAME_RULES = _governance_contract.RETIRED_NAME_RULES
active_doc_forbidden_patterns = _governance_contract.active_doc_forbidden_patterns
is_retired_path_policy_line = _governance_contract.is_retired_path_policy_line

SCAN_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("docs"),
    Path(".agents"),
    Path("README.md"),
    Path("data/README.md"),
    Path("AGENTS.md"),
    Path("pyproject.toml"),
)
GOVERNANCE_SCAN_TEST_FILES = {
    Path("tests/architecture/governance_scan_helpers.py"),
    Path("tests/architecture/test_governance_scan_boundaries.py"),
    Path("tests/architecture/test_governance_active_docs.py"),
    Path("tests/architecture/test_governance_profile_runner_contract.py"),
    Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
    Path("tests/architecture/test_governance_local_verification_contract.py"),
    Path("tests/architecture/test_governance_public_surface_contract.py"),
    Path("tests/architecture/test_governance_writer_boundary_contract.py"),
    Path("tests/architecture/test_governance_import_independence.py"),
}
RUNTIME_WORKSPACE_TEST_ALLOWLIST = {
    Path("tests/agents/handoff/test_build_handoff_package_integration.py"),
    Path("tests/agents/handoff/conftest.py"),
    Path("tests/agents/handoff/test_handoff_package_verifier.py"),
    Path("tests/architecture/test_environment_contract.py"),
    Path("tests/architecture/test_factor_contract_terms.py"),
    Path("tests/architecture/test_handoff_contract_single_source.py"),
    Path("tests/architecture/test_governance_scan_boundaries.py"),
    Path("tests/policy/test_share_safety.py"),
    Path("tests/policy/test_share_safety_runtime_edges.py"),
    Path("tests/tools/run_profile.py"),
    Path("tests/tools/test_run_profile.py"),
    Path("tests/workflows/test_local_verification_artifacts.py"),
    Path("tests/workflows/test_local_verification_config.py"),
    Path("tests/workflows/test_local_verification_outputs.py"),
    Path("tests/workflows/test_local_verification_processed_inputs.py"),
    Path("tests/workflows/test_local_verification_records.py"),
    Path("tests/workflows/test_local_verification_contract.py"),
} | GOVERNANCE_SCAN_TEST_FILES | {
    Path(path) for path in PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES
}
RUNTIME_WORKSPACE_SOURCE_ALLOWLIST_REASONS = {
    Path("src/quant_research_platform/workflows/local_verification.py"): (
        "Owner: workflows. Rationale: local verification run layout. "
        "Lifecycle: current workflow contract. "
        "Test coverage: workflow local verification tests."
    ),
    Path("src/quant_research_platform/policy/share_safety.py"): (
        "Owner: policy share safety. "
        "Rationale: detect local run paths in share-safe text. "
        "Lifecycle: permanent policy contract. "
        "Test coverage: policy share safety tests."
    ),
}
RUNTIME_WORKSPACE_SOURCE_ALLOWLIST = set(RUNTIME_WORKSPACE_SOURCE_ALLOWLIST_REASONS)
ACTIVE_GOVERNANCE_DOC_ROOTS = (
    Path(root.rstrip("/")) for root in CONTRACT_ACTIVE_GOVERNANCE_DOC_ROOTS
)
ACTIVE_DOC_FORBIDDEN_PATTERNS = active_doc_forbidden_patterns()
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_OUTPUT_ARTIFACTS = Path("src/quant_research_platform/artifacts/outputs.py")
RUNTIME_PROFILING = Path("src/quant_research_platform/runtime/profiling.py")
GOVERNANCE_NEGATIVE_FIXTURE_FILES = (
    Path("tests/agents/test_verify_governance_active_docs.py"),
    Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
    Path("tests/architecture/test_governance_version_surface_contract.py"),
    Path("tests/architecture/test_handoff_contract_single_source.py"),
    Path("tests/architecture/test_retired_terms.py"),
    *tuple(sorted(GOVERNANCE_SCAN_TEST_FILES)),
    Path("tests/helpers/governance_fixtures.py"),
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
)
INTERNAL_PROVIDER_SCAN_EXCLUSIONS = {
    ("docs", "code-map", "source-csmar-contracts.md"),
    ("src", "quant_research_platform", "reports", "schema_inventory.py"),
}
PUBLIC_SURFACE_LINE_ALLOWLIST = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "verify_governance.py"
    ): ("src/quant_research_platform/sources/csmar/interim.py",),
}
ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST = {
    "retired_term": {
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_code_map_flow.py"
        ): {
            "raw_private",
            "interim_private",
            "processed_private",
            "/api/v0.1",
            "/api/runs",
            "/api/pipeline",
        },
        Path("tests/contracts/test_platform_contracts.py"): {
            "core_long_period",
            "smoke_minimal",
        },
        Path("tests/architecture/test_governance_version_surface_contract.py"): {
            "/api/v0.1",
        },
    },
    "stale_version_token": {
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_code_map_flow.py"
        ): {"/api/v0.1"},
        Path("tests/architecture/test_governance_version_surface_contract.py"): {
            "/api/v0.1"
        },
    },
}
APPROVED_FIELD_SPEC_FILES = {
    Path("src/quant_research_platform/catalog/fields.py"),
    Path("src/quant_research_platform/markets/implementations/a_share/fields.py"),
}
APPROVED_WRITER_MODULES = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    ),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "replay_evidence.py"
    ),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "self_test.py"
    ),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "verify_handoff_package.py"
    ),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "compare_run_results.py"
    ),
    Path("src/quant_research_platform/storage/atomic.py"),
    Path("src/quant_research_platform/storage/_dataset_writing.py"),
    Path("src/quant_research_platform/storage/datasets.py"),
    Path("src/quant_research_platform/storage/manifests.py"),
    Path("src/quant_research_platform/artifacts/evidence.py"),
    Path("src/quant_research_platform/artifacts/execution_metrics.py"),
    Path("src/quant_research_platform/artifacts/_evaluation_outputs.py"),
    Path("src/quant_research_platform/artifacts/_portfolio_outputs.py"),
    Path("src/quant_research_platform/artifacts/_parquet_writers.py"),
    Path("src/quant_research_platform/artifacts/_prediction_outputs.py"),
    Path("src/quant_research_platform/artifacts/_stage_profile_outputs.py"),
    Path("src/quant_research_platform/artifacts/outputs.py"),
    Path("src/quant_research_platform/runtime/execution_metrics.py"),
}
PUBLIC_CONTRACT_DOCS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("docs/api/README.md"),
    Path("docs/code-map/platform-flow.md"),
    Path(
        ".agents/skills/quant-research-platform-maintenance/references/"
        "current-contract.md"
    ),
    Path("docs/architecture/open-issues.md"),
)
APPROVED_API_LINES = (
    "GET  /api/0.1/catalog/factors",
    "GET  /api/0.1/catalog/labels",
    "GET  /api/0.1/research-datasets/{dataset_id}/requirements",
)
APPROVED_CLI_LINES = (
    "qrp catalog factors",
    "qrp catalog labels",
    "qrp research-datasets requirements <dataset_id>",
)
RETIRED_DATA_PRODUCT_ID = "data_product" + "_id"
RETIRED_PUBLIC_CONTRACT_TEXT = (
    "/api/0.1/markets/{market_id}/processed-data",
    "POST /api/0.1/research-datasets",
    "POST /api/0.1/workflows",
    "GET  /api/0.1/reports/{run_id}",
    "qrp data-product interim",
    "qrp market build-processed",
    "qrp research build-dataset",
    "qrp workflow run",
    "qrp report show",
    "GET  /api/0.1/catalog/data-products",
    "GET  /api/0.1/catalog/universes",
    "POST /api/0.1/data-products/{" + RETIRED_DATA_PRODUCT_ID + "}/interim-runs",
    "qrp catalog data-products",
    "qrp catalog universes",
    "qrp data-products run-interim <" + RETIRED_DATA_PRODUCT_ID + ">",
)
RETIRED_DATA_PRODUCT_ALLOWED_NEGATIVE_TESTS = {
    Path("tests/architecture/test_governance_active_docs.py"): (
        "POST /api/0.1/data-products/{" + RETIRED_DATA_PRODUCT_ID + "}/interim-runs",
        "qrp data-products run-interim <" + RETIRED_DATA_PRODUCT_ID + ">",
        RETIRED_DATA_PRODUCT_ID,
    ),
    Path("tests/interfaces/test_api_cli_01_contract.py"): (
        "/api/0.1/data-products/{" + RETIRED_DATA_PRODUCT_ID + "}/interim-runs",
    ),
}
DATA_PRODUCT_CURRENT_VOCAB_ALLOWED_NEGATIVE_TESTS = {
    Path("tests/architecture/test_governance_active_docs.py"): (
        "GET  /api/0.1/catalog/data-products",
        "qrp catalog data-products",
        "DATA_PRODUCT",
        "DataProductId",
        "data product id",
        "data product",
    ),
    Path("tests/architecture/test_data_layer_boundaries.py"): (
        'SOURCE_ROOT / "catalog/data_products.py"',
    ),
    Path("tests/architecture/test_registration_allowlist.py"): (
        '"src/quant_research_platform/catalog/data_products.py"',
    ),
    Path("tests/interfaces/test_api_cli_01_contract.py"): (
        "/api/0.1/catalog/data-products",
        "data-products",
    ),
    Path("tests/catalog/test_profiles_composition_root.py"): ('"data_products"',),
}
RETIRED_DATA_LAYER_STRUCTURE_PATTERNS = (
    (RETIRED_DATA_PRODUCT_ID, re.compile(r"\b" + RETIRED_DATA_PRODUCT_ID + r"\b")),
    ("DATA_PRODUCT_ID", re.compile(r"\bDATA_PRODUCT_ID\b")),
    ("DataProductId", re.compile(r"\bDataProductId\b")),
    ("data product id", re.compile(r"\bdata product id\b", re.IGNORECASE)),
    ("asset_batches", re.compile(r"\basset_batches\b")),
    ("year_filter", re.compile(r"\byear_filter\b")),
    ("partition_filter", re.compile(r"\bpartition_filter\b")),
    ("ChunkAxis.YEAR", re.compile(r"\bChunkAxis\.YEAR\b")),
    ("ChunkAxis.ENTITY", re.compile(r"\bChunkAxis\.ENTITY\b")),
    ("source_bindings", re.compile(r"\bsource_bindings\b")),
    ("MaskCatalog", re.compile(r"\bMaskCatalog\b")),
    ("UniverseCatalog", re.compile(r"\bUniverseCatalog\b")),
    ("ProviderBinding", re.compile(r"\bProviderBinding\b")),
    ("data_products", re.compile(r"\bdata_products\b")),
)
ACTIVE_STRUCTURE_SCAN_ROOTS = (
    Path("src"),
    Path("docs"),
    Path(".agents"),
    Path("README.md"),
    Path("data/README.md"),
    Path("AGENTS.md"),
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
    "qrp-private local-verification run",
    "private local verification harness",
    "not a public CLI",
    "not a public execution mode",
    "not a public evaluate, predict, or report command",
    "not simulated or real-money order execution",
    "does not promise returns",
    "does not output row-level private data",
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
def iter_governance_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            raise AssertionError(f"scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path not in GOVERNANCE_SCAN_TEST_FILES
                and not _is_excluded(path)
            ):
                files.append(path)
    return files
def iter_active_governance_doc_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in ACTIVE_GOVERNANCE_DOC_ROOTS:
        if not root.exists():
            raise AssertionError(f"active_governance_doc_root_missing: {root}")
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
    return tuple(sorted(files))
def iter_governance_python_files() -> list[Path]:
    return [path for path in iter_governance_text_files() if path.suffix == ".py"]
def _is_excluded(path: Path) -> bool:
    parts = path.parts
    return any(_contains_parts(parts, excluded) for excluded in EXCLUDED_SCAN_PARTS)
def _contains_parts(parts: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle for index in range(len(parts))
    )
def iter_active_structure_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_STRUCTURE_SCAN_ROOTS:
        if not root.exists():
            raise AssertionError(f"active_structure_scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and not _is_excluded(path):
                files.append(path)
    return files
def iter_public_surface_files() -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_SURFACE_SCAN_ROOTS:
        if not root.exists():
            raise AssertionError(f"public_surface_scan_root_missing: {root}")
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
def public_surface_scan_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path == Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "governance_contract.py"
    ):
        return ""
    allowed_fragments = PUBLIC_SURFACE_LINE_ALLOWLIST.get(path, ())
    if allowed_fragments:
        text = "\n".join(
            line
            for line in text.splitlines()
            if not any(fragment in line for fragment in allowed_fragments)
        )
    return text
def _is_internal_provider_doc(path: Path) -> bool:
    if path.parts[:2] == ("docs", "code-map") and path.name.startswith("source-csmar-"):
        return True
    parts = path.parts
    return any(
        _contains_parts(parts, excluded)
        for excluded in INTERNAL_PROVIDER_SCAN_EXCLUSIONS
    )
def future_data_hit(path: Path, lineno: int, class_name: str) -> str:
    return f"{path}:{lineno}: {class_name}.uses_future_data"
def is_approved_writer_path(path: Path) -> bool:
    return path.parts[0] == "tests" or path in APPROVED_WRITER_MODULES
def call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""
def is_open_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "open"
    return isinstance(func, ast.Attribute) and func.attr == "open"
def is_open_write_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "write"
        and isinstance(call.func.value, ast.Call)
        and is_open_call(call.func.value.func)
    )
def open_mode(call: ast.Call) -> str | None:
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
def literal_string_items(node: ast.expr) -> tuple[str, ...]:
    if not isinstance(node, ast.List | ast.Tuple):
        return ()
    values: list[str] = []
    for element in node.elts:
        value = _literal_string(element)
        if value is not None:
            values.append(value)
    return tuple(values)
def function_arg_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {arg.arg for arg in node.args.args}
    raise AssertionError(f"function not found: {function_name}")
def _class_method_arg_names(path: Path, class_name: str, method_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                return {arg.arg for arg in child.args.args}
    raise AssertionError(f"method not found: {class_name}.{method_name}")
def class_field_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    field_names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                field_names.add(child.target.id)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        field_names.add(target.id)
        return field_names
    raise AssertionError(f"class not found: {class_name}")
def is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))
