"""Schema inventory Markdown renderer."""

from __future__ import annotations

from typing import Any


def render_schema_inventory_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Schema Inventory",
        "",
        "| Layer | Kind | Artifact | Rows | Columns |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in payload["datasets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["layer"]),
                    str(item["kind"]),
                    str(item["artifact_id"]),
                    str(item["row_count"]),
                    str(len(item["columns"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["render_schema_inventory_markdown"]
