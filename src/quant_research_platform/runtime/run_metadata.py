"""定义跨模块共享的 run metadata 合同。"""

from __future__ import annotations

from typing import Any

MINIMUM_RUN_METADATA_FIELDS = (
    "project",
    "kind",
    "domain",
    "data_stage",
    "uses_external_data",
)


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    source: str = "metadata",
) -> None:
    """校验 artifact consumers 需要的 provenance fields。"""

    missing = [
        field for field in MINIMUM_RUN_METADATA_FIELDS if field not in metadata
    ]
    if missing:
        raise ValueError(
            f"{source} is missing required fields: {', '.join(missing)}."
        )
    for metadata_field in ("project", "kind", "domain", "data_stage"):
        value = metadata[metadata_field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{source} {metadata_field} must be a non-empty string."
            )
    if not isinstance(metadata["uses_external_data"], bool):
        raise TypeError(f"{source} uses_external_data must be bool.")


__all__ = ["MINIMUM_RUN_METADATA_FIELDS", "validate_run_metadata"]
