from __future__ import annotations

import ast
from pathlib import Path

POLICY_SHARE_SAFETY = Path("src/quant_research_platform/policy/share_safety.py")
REPORT_SAFETY = Path("src/quant_research_platform/reports/verification_evidence.py")
HANDOFF_SAFETY = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "handoff/verify_handoff_package.py"
)


def test_report_and_handoff_safety_do_not_own_text_privacy_regexes() -> None:
    for path in (REPORT_SAFETY, HANDOFF_SAFETY):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        local_unsafe_regex_owners = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.startswith("_UNSAFE")
        ]

        assert local_unsafe_regex_owners == [], f"text_privacy_regex_owner: {path}"
        expected_delegate = (
            "evidence_text_unsafe_reason"
            if path == REPORT_SAFETY
            else "unsafe_for_share_reason"
        )
        assert expected_delegate in source


def test_policy_owns_required_text_safety_semantics() -> None:
    source = POLICY_SHARE_SAFETY.read_text(encoding="utf-8")

    for semantic in (
        "raw_export_filename",
        "source_local_key",
        "runtime_payload",
        "local_run_path",
    ):
        assert semantic in source


def test_handoff_member_path_rules_are_derived_from_governance_contract() -> None:
    source = HANDOFF_SAFETY.read_text(encoding="utf-8")

    assert "member_name_failures" in source
    assert "package_member_name_failures" in source
