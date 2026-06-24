from __future__ import annotations

from quant_research_platform.policy.share_safety import (
    public_text_unsafe_reason,
    sanitize_public_text,
    zip_member_unsafe_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert public_text_unsafe_reason(text) in {"local_path", "private_path"}
    assert "C:/Users" not in sanitize_public_text(text)


def test_share_safety_rejects_forbidden_zip_member() -> None:
    assert zip_member_unsafe_reason("data/raw/source/file.parquet") == "private_path"


def test_share_safety_allows_review_safe_handoff_evidence_name() -> None:
    assert zip_member_unsafe_reason("handoff_evidence/generation.json") is None
