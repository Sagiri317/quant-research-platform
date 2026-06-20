from __future__ import annotations

from pathlib import Path

HANDOFF_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff"
)


def test_handoff_package_contract_uses_evidence_mode_without_diff_or_pr_body() -> None:
    builder = (HANDOFF_ROOT / "build_handoff_package.py").read_text(encoding="utf-8")
    verifier = (HANDOFF_ROOT / "verify_handoff_package.py").read_text(encoding="utf-8")

    assert "handoff_evidence/" in builder
    assert "handoff_evidence/" in verifier
    old_evidence_dir = "review_" + "evidence/"
    assert old_evidence_dir not in builder
    assert old_evidence_dir not in verifier
    patch_entry = '"diff' + '.patch"'
    assert patch_entry not in builder
    assert patch_entry not in verifier
    assert '"' + "." + "run/handoffs" + '"' in builder
    assert '"evidence"' in builder
    assert '"pr_closure"' in builder


def test_pr_closure_only_entries_are_not_required_for_evidence_mode() -> None:
    verifier = (HANDOFF_ROOT / "verify_handoff_package.py").read_text(encoding="utf-8")

    required_block = verifier.split("REQUIRED_ZIP_ENTRIES", maxsplit=1)[1].split(
        "}",
        maxsplit=1,
    )[0]
    assert "review_threads.json" not in required_block
    assert "pr_body.md" not in required_block


def test_public_safe_text_checks_use_runtime_privacy_single_source() -> None:
    agent_safety = Path("src/quant_research_platform/agents/safety.py").read_text(
        encoding="utf-8"
    )
    handoff_safety = (HANDOFF_ROOT / "handoff_safety.py").read_text(encoding="utf-8")

    assert "public_text_unsafe_reason" in agent_safety
    assert "public_text_unsafe_reason" in handoff_safety
    forbidden_local_fact_sources = {
        "_UNSAFE_TEXT_PATTERNS",
        "DRIVE_ABSOLUTE_RE",
        "POSIX_USER_ABSOLUTE_RE",
        "ABSOLUTE_REPO_PRIVATE_PATH_RE",
        "WINDOWS_USER_ABSOLUTE_RE",
        "STOCK_CODE_RE",
        "CREDENTIAL_RE",
        "VENDOR_MARKER_RE",
        "RAW_TRACEBACK_RE",
    }
    combined = agent_safety + "\n" + handoff_safety

    assert [name for name in forbidden_local_fact_sources if name in combined] == []
