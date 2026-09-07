from __future__ import annotations

from quant_research_platform.policy.share_safety import (
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert unsafe_for_share_reason(text) in {"local_path", "local_only_path"}
    assert "C:/Users" not in sanitize_shared_text(text)
