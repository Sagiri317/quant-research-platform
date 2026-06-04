"""Manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.storage.privacy import sanitize_public_object


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    text = (
        json.dumps(sanitize_public_object(payload), ensure_ascii=False, indent=2)
        + "\n"
    )
    return atomic_write_text(path, text, encoding="utf-8")


__all__ = ["read_manifest", "write_manifest"]
