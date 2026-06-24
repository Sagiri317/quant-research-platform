from __future__ import annotations

import ast
from pathlib import Path

RUNTIME_PRIVACY = Path("src/quant_research_platform/runtime/privacy.py")
REPORT_SAFETY = Path(
    "src/quant_research_platform/reports/verification_evidence.py"
)
HANDOFF_SAFETY = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "handoff/handoff_safety.py"
)


def test_report_and_handoff_safety_do_not_own_text_privacy_regexes() -> None:
    for path in (REPORT_SAFETY, HANDOFF_SAFETY):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        re_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(alias.name == "re" for alias in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and node.module == "re")
        ]
        regex_compilers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.func.attr == "compile"
        ]

        assert re_imports == [], f"text_privacy_regex_import: {path}"
        assert regex_compilers == [], f"text_privacy_regex_owner: {path}"
        assert "public_text_unsafe_reason" in source


def test_runtime_privacy_owns_required_text_safety_semantics() -> None:
    source = RUNTIME_PRIVACY.read_text(encoding="utf-8")

    for semantic in (
        "raw_export_filename",
        "source_local_key",
        "private_runtime_payload",
        "private_run_path",
    ):
        assert semantic in source


def test_handoff_member_path_rules_are_derived_from_governance_contract() -> None:
    source = HANDOFF_SAFETY.read_text(encoding="utf-8")

    assert "package_member_name_failures" in source
