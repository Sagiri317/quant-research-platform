"""Report summary helpers."""

from __future__ import annotations


def summarize_status(status: str) -> dict[str, str]:
    return {"status": status}


__all__ = ["summarize_status"]
