from __future__ import annotations

from pathlib import Path


def test_clean_rewrite_public_safe_parity_evidence_exists() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.md")
    text = path.read_text(encoding="utf-8")

    required_terms = [
        "processed panel parity",
        "factor value parity",
        "label value parity",
        "masks / universes parity",
        "research dataset parity",
        "evaluation parity",
        "prediction smoke parity",
        "supporting layers migration 2014-2025",
        "full historical daily_panel",
        "accepted explained-delta",
        "unexplained_regression | 0",
    ]
    for term in required_terms:
        assert term in text


def test_clean_rewrite_public_safe_parity_evidence_has_no_private_samples() -> None:
    text = Path("docs/reports/clean-rewrite-0.1-parity-evidence.md").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "C:/",
        "C:\\",
        "/mnt/",
        "/home/",
        "000001",
        "600000",
        "raw row",
        "row sample:",
        "token=",
    ]
    for term in forbidden:
        assert term not in text
