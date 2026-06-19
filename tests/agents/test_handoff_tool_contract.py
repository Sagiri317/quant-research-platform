from __future__ import annotations

from pathlib import Path

HANDOFF_TOOL_ROOT = Path("." + "run/tools/handoff")


def test_handoff_self_test_uses_current_tool_allowlist_only() -> None:
    source = (HANDOFF_TOOL_ROOT / "self_test.py").read_text(encoding="utf-8")
    allowed = {
        "README.md",
        "build_handoff_package.py",
        "handoff_safety.py",
        "self_test.py",
        "verify_handoff_package.py",
    }

    for name in allowed:
        assert name in source
        assert (HANDOFF_TOOL_ROOT / name).exists()
    assert 'glob("*.py")' not in source
    for stale_name in (
        "factor_chunk_diagnostic.py",
        "n2_artifact_validation.py",
        "smoke_compare.py",
    ):
        assert stale_name not in source
        assert not (HANDOFF_TOOL_ROOT / stale_name).exists()


def test_handoff_builder_requires_final_review_artifacts() -> None:
    source = (HANDOFF_TOOL_ROOT / "build_handoff_package.py").read_text(
        encoding="utf-8"
    )

    assert 'parser.add_argument("--base-commit-before-task", required=True)' in source
    assert 'parser.add_argument("--pr-body", required=True)' in source
    assert 'parser.add_argument("--review-threads", required=True)' in source
