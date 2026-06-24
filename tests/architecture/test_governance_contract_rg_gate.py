from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from tests.helpers import frequency_ids
from tests.helpers.governance_fixtures import (
    representative_name_forms,
    source_tree_sensitive_rule_summary,
    text_rule_example,
)

GOVERNANCE_SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
)
if str(GOVERNANCE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GOVERNANCE_SCRIPT_DIR))

from governance_contract import (  # noqa: E402
    ACTIVE_GOVERNANCE_DOC_ROOTS as CONTRACT_ACTIVE_GOVERNANCE_DOC_ROOTS,
)
from governance_contract import (  # noqa: E402
    PRIVACY_NEGATIVE_TEST_PATHS,
    RETIRED_NAME_RULES,
    active_doc_forbidden_patterns,
    is_retired_path_policy_line,
)

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
RUNTIME_WORKSPACE_TEST_ALLOWLIST = {
    Path("tests/agents/handoff/test_build_handoff_package_integration.py"),
    Path("tests/agents/handoff/test_handoff_tools.py"),
    Path("tests/agents/handoff/test_package_safety_check.py"),
    Path("tests/architecture/test_environment_contract.py"),
    Path("tests/architecture/test_factor_contract_terms.py"),
    Path("tests/architecture/test_handoff_package_contract.py"),
    Path("tests/workflows/test_verification_evidence.py"),
    Path("tests/workflows/test_verification_evidence_contract.py"),
} | {Path(path) for path in PRIVACY_NEGATIVE_TEST_PATHS}
RUNTIME_WORKSPACE_SOURCE_ALLOWLIST_REASONS = {
    Path("src/quant_research_platform/workflows/verification_evidence.py"): (
        "Owner: workflows. Rationale: verification output layout. "
        "Lifecycle: current workflow contract. Test coverage: workflow evidence tests."
    ),
    Path("src/quant_research_platform/runtime/privacy.py"): (
        "Owner: runtime privacy. Rationale: detect private run paths in public text. "
        "Lifecycle: permanent privacy contract. Test coverage: runtime privacy tests."
    ),
}
RUNTIME_WORKSPACE_SOURCE_ALLOWLIST = set(
    RUNTIME_WORKSPACE_SOURCE_ALLOWLIST_REASONS
)
ACTIVE_GOVERNANCE_DOC_ROOTS = (
    Path(root.rstrip("/")) for root in CONTRACT_ACTIVE_GOVERNANCE_DOC_ROOTS
)
ACTIVE_DOC_FORBIDDEN_PATTERNS = active_doc_forbidden_patterns()
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ARTIFACTS = Path("src/quant_research_platform/reports/report_artifacts.py")
RUNTIME_PROFILING = Path("src/quant_research_platform/runtime/profiling.py")
GOVERNANCE_NEGATIVE_FIXTURE_FILES = (
    Path("tests/agents/test_verify_governance.py"),
    Path("tests/architecture/test_handoff_package_contract.py"),
    Path("tests/architecture/test_retired_surface_absent.py"),
    Path("tests/architecture/test_governance_contract_rg_gate.py"),
    Path("tests/helpers/governance_fixtures.py"),
)


def test_formal_scan_roots_are_clean_checkout_stable_roots() -> None:
    runtime_workspace = Path(".run")

    assert Path("reports") not in SCAN_ROOTS
    assert runtime_workspace not in SCAN_ROOTS
    assert all(runtime_workspace.name not in root.parts for root in SCAN_ROOTS)


def test_tests_do_not_depend_on_runtime_workspace_path() -> None:
    runtime_workspace_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape('.run')}(?![A-Za-z0-9_])"
    )
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for path in Path("tests").rglob("*.py"):
        if path.resolve() == current_file or path in RUNTIME_WORKSPACE_TEST_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if runtime_workspace_pattern.search(line):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_source_and_tests_do_not_depend_on_runtime_workspace_path() -> None:
    runtime_workspace_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape('.run')}(?![A-Za-z0-9_])"
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            if (
                path.resolve() == current_file
                or (
                    root == Path("tests")
                    and path in RUNTIME_WORKSPACE_TEST_ALLOWLIST
                )
                or (
                    root == Path("src")
                    and path in RUNTIME_WORKSPACE_SOURCE_ALLOWLIST
                )
            ):
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if runtime_workspace_pattern.search(line):
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_active_governance_docs_do_not_publish_retired_state_truth() -> None:
    hits: list[str] = []
    for path in _iter_active_governance_doc_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if is_retired_path_policy_line(path, line):
                continue
            for name, pattern in ACTIVE_DOC_FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}")

    assert hits == []


def test_active_doc_forbidden_patterns_cover_required_examples() -> None:
    from governance_contract import (
        RETIRED_PUBLIC_SURFACE_RULES,
        STALE_GOVERNANCE_TRUTH_RULES,
    )

    examples = {
        rule.name: text_rule_example(
            STALE_GOVERNANCE_TRUTH_RULES,
            rule.name,
        )
        for rule in STALE_GOVERNANCE_TRUTH_RULES
    }
    examples.update(representative_name_forms(RETIRED_NAME_RULES))
    examples.update(representative_name_forms(RETIRED_PUBLIC_SURFACE_RULES))

    for name, text in examples.items():
        assert ACTIVE_DOC_FORBIDDEN_PATTERNS[name].search(text)


def test_active_doc_contract_covers_historical_narrative_examples() -> None:
    from governance_contract import ACTIVE_DOC_HISTORY_NARRATIVE_RULES

    assert ACTIVE_DOC_HISTORY_NARRATIVE_RULES
    for rule in ACTIVE_DOC_HISTORY_NARRATIVE_RULES:
        assert rule.example
        assert rule.compile().search(rule.example)


def test_governance_negative_fixtures_do_not_hardcode_rule_inventory() -> None:
    from governance_contract import (
        RETIRED_PUBLIC_SURFACE_RULES,
        STALE_GOVERNANCE_TRUTH_RULES,
        retired_name_patterns,
        stale_governance_truth_patterns,
    )

    rule_names = {
        rule.name
        for rule in (*RETIRED_NAME_RULES, *RETIRED_PUBLIC_SURFACE_RULES)
    } | {rule.name for rule in STALE_GOVERNANCE_TRUTH_RULES}
    patterns = {
        **retired_name_patterns(),
        **stale_governance_truth_patterns(),
    }
    hits: list[str] = []
    for path in GOVERNANCE_NEGATIVE_FIXTURE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if value in rule_names:
                continue
            matched = sorted(
                rule_name
                for rule_name, pattern in patterns.items()
                if pattern.search(value)
            )
            if matched:
                hits.append(f"{path}:{node.lineno}: {','.join(matched)}")

    assert hits == [], "hardcoded_governance_fixture: " + "; ".join(hits)


def test_source_tree_sensitive_negative_rules_are_contract_owned() -> None:
    from governance_contract import (
        SOURCE_TREE_SENSITIVE_NEGATIVE_RULES,
        source_tree_sensitive_negative_patterns,
        source_tree_sensitive_negative_rules,
    )

    rules = source_tree_sensitive_negative_rules()
    patterns = source_tree_sensitive_negative_patterns()
    summary = source_tree_sensitive_rule_summary(rules)

    assert rules == SOURCE_TREE_SENSITIVE_NEGATIVE_RULES
    assert rules
    assert set(patterns) == {rule.name for rule in rules}
    assert {item[0] for item in summary} == set(patterns)
    assert {rule.scan_group for rule in rules} == {"sensitive_negative_literals"}
    assert all(rule.category for rule in rules)
    assert all(rule.allowed_sources for rule in rules)


def test_business_chunk_consumers_do_not_concat_unbounded_research_chunks() -> None:
    hot_paths = (
        Path("src/quant_research_platform/research/portfolios/runner.py"),
    )
    forbidden_patterns = (
        re.compile(r"pd\.concat\(\s*chunks\b"),
        re.compile(r"pd\.concat\(\s*frames\b"),
    )

    hits: list[str] = []
    for path in hot_paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_prediction_artifact_builder_does_not_prederive_oos_end() -> None:
    text = Path(
        "src/quant_research_platform/research/predictions/artifacts.py"
    ).read_text(encoding="utf-8")

    assert '"oos_end"' not in text


def test_verification_evidence_replaces_retired_public_names() -> None:
    expected_files = {
        Path("src/quant_research_platform/workflows/verification_evidence.py"),
        Path("src/quant_research_platform/reports/verification_evidence.py"),
    }
    retired_files = {
        Path(
            "src/quant_research_platform/workflows/"
            + "private"
            + "_evidence.py"
        ),
        Path("src/quant_research_platform/reports/" + "private" + "_evidence.py"),
        Path("src/quant_research_platform/reports/evidence" + "_rendering.py"),
    }
    forbidden_terms = (
        "private" + "_evidence",
        "Private" + "Evidence",
        "run_" + "private" + "_evidence",
        "build_" + "private" + "_evidence_summary",
        "write_" + "private" + "_evidence_summary",
        "output_root_must_be_under_" + "run",
    )
    missing = sorted(str(path) for path in expected_files if not path.is_file())
    present = sorted(str(path) for path in retired_files if path.exists())
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(term in line for term in forbidden_terms):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert missing == []
    assert present == []
    assert hits == []

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
    ): (
        "src/quant_research_platform/sources/csmar/interim.py",
    ),
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
        Path("tests/agents/test_verify_governance.py"): {
            "/api/v0.1",
        },
    },
    "stale_version_token": {
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_code_map_flow.py"
        ): {"/api/v0.1"},
        Path("tests/agents/test_verify_governance.py"): {"/api/v0.1"},
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
        "handoff_safety.py"
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
        "compare_verification_evidence.py"
    ),
    Path("src/quant_research_platform/storage/atomic.py"),
    Path("src/quant_research_platform/storage/datasets.py"),
    Path("src/quant_research_platform/storage/manifests.py"),
    Path("src/quant_research_platform/reports/report_artifacts.py"),
    Path("src/quant_research_platform/reports/verification_evidence.py"),
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
    "POST /api/0.1/data-products/{"
    + RETIRED_DATA_PRODUCT_ID
    + "}/interim-runs",
    "qrp catalog data-products",
    "qrp catalog universes",
    "qrp data-products run-interim <" + RETIRED_DATA_PRODUCT_ID + ">",
)

RETIRED_DATA_PRODUCT_ALLOWED_NEGATIVE_TESTS = {
    Path("tests/architecture/test_governance_contract_rg_gate.py"): (
        "POST /api/0.1/data-products/{"
        + RETIRED_DATA_PRODUCT_ID
        + "}/interim-runs",
        "qrp data-products run-interim <" + RETIRED_DATA_PRODUCT_ID + ">",
        RETIRED_DATA_PRODUCT_ID,
    ),
    Path("tests/interfaces/test_api_cli_01_contract.py"): (
        "/api/0.1/data-products/{" + RETIRED_DATA_PRODUCT_ID + "}/interim-runs",
    ),
}

DATA_PRODUCT_CURRENT_VOCAB_ALLOWED_NEGATIVE_TESTS = {
    Path("tests/architecture/test_governance_contract_rg_gate.py"): (
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
    Path("tests/catalog/test_catalog_profiles.py"): (
        '"data_products"',
    ),
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
    "qrp-private evidence run",
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


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            raise AssertionError(f"scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and not _is_excluded(path):
                files.append(path)
    return files


def _iter_active_governance_doc_files() -> tuple[Path, ...]:
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
        ("/api/" + "v" + "1", re.compile(r"/api/" r"v" r"1\b")),
        ("schema 0.7", re.compile(r"schema\s+`?0\.7`?")),
        ("schema 0.6", re.compile(r"schema\s+`?0\.6`?")),
        ("schema 1.0", re.compile(r"schema\s+`?1\.0`?")),
        ("panel_full", re.compile(r"panel_full")),
        ("price_volume_factors_full", re.compile(r"price_volume_factors_full")),
        (
            "_".join(("csmar", "daily", "market", "interim")),
            re.compile(
                re.escape(f"csmar_{frequency_ids.MARKET_BARS_DATA_ID}_interim")
            ),
        ),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if name in ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST[
                "retired_term"
            ].get(path, set()):
                continue
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []


def test_active_tree_has_no_stale_qrp_owned_version_tokens() -> None:
    forbidden_patterns = (
        (
            "handoff-v1",
            re.compile(
                r"handoff-(?:sidecar|task-metadata|artifact-manifest|"
                r"validation-evidence|verification-matrix)/v1"
            ),
        ),
        ("/api/v0.1", re.compile(r"/api/v0\.1\b")),
        ("/api/v1", re.compile(r"/api/v1\b")),
        ("incidental-v2-label", re.compile(r"\b[a-z][a-z0-9_]*_v[2-9]\b")),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if name in ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST[
                "stale_version_token"
            ].get(path, set()):
                continue
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []


def test_active_tree_has_no_retired_data_product_vocabulary() -> None:
    current_file = Path(__file__).resolve()
    forbidden_patterns = (
        re.compile(r"\b" + RETIRED_DATA_PRODUCT_ID + r"\b"),
        re.compile(r"\bDataProductId\b"),
        re.compile(r"\bdata product id\b", re.IGNORECASE),
    )
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = RETIRED_DATA_PRODUCT_ALLOWED_NEGATIVE_TESTS.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(item in line for item in allowed):
                continue
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_active_tree_has_no_data_product_current_vocabulary() -> None:
    current_file = Path(__file__).resolve()
    forbidden_patterns = (
        re.compile(r"\bdata product\b", re.IGNORECASE),
        re.compile(r"\bdata products\b", re.IGNORECASE),
        re.compile(r"\bDATA_PRODUCT[A-Z0-9_]*\b"),
        re.compile(r"catalog/data_products\.py"),
    )
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = DATA_PRODUCT_CURRENT_VOCAB_ALLOWED_NEGATIVE_TESTS.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(item in line for item in allowed):
                continue
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_active_tree_has_no_retired_data_layer_structure_vocabulary() -> None:
    hits: list[str] = []
    for path in _iter_active_structure_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in RETIRED_DATA_LAYER_STRUCTURE_PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}: {line.strip()}")

    assert hits == []


def test_research_foundation_semantic_closure_terms_are_retired() -> None:
    forbidden_terms = (
        "Model" + "Family",
        "EvaluationSpec." + "feature_columns",
        "ComputePanelSpec." + "feature_columns",
        "evaluate" + "_factor_frame",
        "feature" + "_ic",
        "feature" + "_rankic",
        "grouped" + "_feature",
        "fit_linear" + "_baseline_result",
        "_fit_linear" + "_baseline_result_from_frames",
        "linear" + "_baseline",
        "research/models/" + "base" + "line.py",
        "base" + "line.py",
        "Prediction" + "Bundle",
        "PREDICTION" + "_BUNDLE",
        "bundle" + "_private",
        "bundle" + "_columns",
        "bundle" + "_schema",
        "bundle" + "_metadata",
        "compute/" + "config.py",
        "quant_research_platform.compute." + "config",
        "from quant_research_platform.compute." + "config import",
        "PortfolioSpec." + "label_column",
        "portfolio." + "label_column",
        "WeightConfig." + "cost_bps",
        "weight_config." + "cost_bps",
        "cost_bps in weight " + "config",
        "SignalSpec(" + "column=",
        'direction="' + "higher" + '"',
        'direction="' + "lower" + '"',
        "Backtest" + "Engine",
        "open" + "_backtest",
        "paper" + "_trading",
        "live" + "_trading",
        "br" + "oker",
    )
    negative_assertion_allowlist = {
        Path("tests/compute/test_cost_config_semantic_ownership.py"): (
            "weight_config." + "cost_bps",
        ),
        Path("tests/contracts/test_research_foundation_semantic_contracts.py"): (
            'direction="' + "higher" + '"',
        ),
        Path("docs/architecture/open-issues.md"): (
            "Prediction" + "Bundle",
        ),
        Path("AGENTS.md"): ("br" + "oker",),
        Path("README.md"): ("br" + "oker",),
        Path(".agents/briefs/user-principles.md"): ("br" + "oker",),
        Path(".agents/skills/quant-research-platform-maintenance/SKILL.md"): (
            "br" + "oker",
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/references/"
            "current-contract.md"
        ): ("br" + "oker",),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "governance_contract.py"
        ): (
            "Prediction" + "Bundle",
            "Model" + "Family",
        ),
    }

    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = negative_assertion_allowlist.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            for term in forbidden_terms:
                if term not in line:
                    continue
                if any(allowed_term in line for allowed_term in allowed):
                    continue
                hits.append(f"{path}:{line_number}: {term}: {line.strip()}")

    assert hits == []


def _iter_active_structure_files() -> list[Path]:
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


def test_a_share_planning_hot_paths_do_not_use_fingerprint_dataset() -> None:
    checked = (
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/materializer.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/panels.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "factors/calculations.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "labels/calculations.py"
        ),
    )
    missing = [str(path) for path in checked if not path.is_file()]
    assert missing == []
    hits = [
        str(path)
        for path in checked
        if "fingerprint_dataset" in path.read_text(encoding="utf-8")
    ]

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
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []


def _iter_public_surface_files() -> list[Path]:
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


def _public_surface_scan_text(path: Path) -> str:
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
    stale_future = [
        term for term in FUTURE_PUBLIC_TODO_TERMS if term in combined
    ]
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
            for value in _literal_string_items(node.value):
                if value.startswith("_"):
                    hits.append(f"{path}: {value}")

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


def _literal_string_items(node: ast.expr) -> tuple[str, ...]:
    if not isinstance(node, ast.List | ast.Tuple):
        return ()
    values: list[str] = []
    for element in node.elts:
        value = _literal_string(element)
        if value is not None:
            values.append(value)
    return tuple(values)


def _function_arg_names(path: Path, function_name: str) -> set[str]:
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


def _class_field_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fields: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                fields.add(child.target.id)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        fields.add(target.id)
        return fields
    raise AssertionError(f"class not found: {class_name}")


def _is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def test_direct_write_gate_detects_chained_open_write() -> None:
    tree = ast.parse("open('artifact.txt', 'w').write('unsafe')\n")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(_is_open_write_call(call) for call in calls)


def test_approved_writer_signatures_do_not_accept_arbitrary_output_roots() -> None:
    writer_arg_sets = (
        _function_arg_names(REPORT_ARTIFACTS, "write_stage_profile_report_artifacts"),
        _function_arg_names(REPORT_ARTIFACTS, "write_prediction_report_artifacts"),
        _function_arg_names(REPORT_ARTIFACTS, "write_evaluation_report_artifacts"),
    )

    assert "output_dir" not in _class_field_names(RUNTIME_PROFILING, "ProfileConfig")
    for arg_names in writer_arg_sets:
        assert "output_root" not in arg_names
        assert "output_dir" not in arg_names


def test_governance_contract_gate_imports_without_pyarrow() -> None:
    script = """
import importlib
import importlib.abc
import sys

class PyArrowBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("blocked pyarrow import")
        return None

sys.meta_path.insert(0, PyArrowBlocker())
importlib.import_module("tests.architecture.test_governance_contract_rg_gate")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
