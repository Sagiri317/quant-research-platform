"""定义跨模块共享的 run metadata 合同。"""

from __future__ import annotations

import re
from typing import Any

MINIMUM_RUN_METADATA_FIELDS = (
    "project",
    "kind",
    "domain",
    "data_stage",
    "uses_external_data",
)
PREDICTION_HANDOFF_METADATA_FIELDS = (
    "prediction_handoff_version",
    "prediction_columns",
    "label_columns",
    "prediction_time_semantics",
    "source_ref",
    "model_ref",
    "training_scope",
    "prediction_boundary",
)
PREDICTION_TRAINING_SCOPE = "external_or_upstream_not_run_by_qrp"
_PRIVATE_PATH_MARKERS = (
    "raw_private",
    "interim_private",
    "processed_private",
    "private_real",
)
_PREDICTION_BOUNDARY_TERMS = (
    "not",
    "trading",
    "risk",
    "capacity",
    "simulation",
    "paper trading",
    "live trading",
)
_ROW_LEVEL_SAMPLE_KEY_MARKERS = (
    "row_sample",
    "stock_sample",
    "asset_sample",
    "sample_rows",
    "sample_stocks",
    "sample_assets",
)


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    source: str = "metadata",
) -> None:
    """校验 artifact consumers 需要的 provenance fields。"""

    missing = [field for field in MINIMUM_RUN_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"{source} is missing required fields: {', '.join(missing)}.")
    for metadata_field in ("project", "kind", "domain", "data_stage"):
        value = metadata[metadata_field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source} {metadata_field} must be a non-empty string.")
    if not isinstance(metadata["uses_external_data"], bool):
        raise TypeError(f"{source} uses_external_data must be bool.")
    if metadata["kind"] == "prediction":
        validate_prediction_handoff_metadata(metadata, source=source)


def validate_prediction_handoff_metadata(
    metadata: dict[str, Any],
    *,
    source: str = "metadata",
) -> None:
    """Validate N6 prediction handoff metadata without affecting factor runs."""

    missing = [
        field
        for field in PREDICTION_HANDOFF_METADATA_FIELDS
        if field not in metadata
    ]
    if missing:
        raise ValueError(
            f"{source} prediction metadata is missing required fields: "
            f"{', '.join(missing)}."
        )
    _validate_string_field(metadata, "prediction_handoff_version", source=source)
    _validate_string_sequence(metadata, "prediction_columns", source=source)
    _validate_string_sequence(metadata, "label_columns", source=source)
    if not isinstance(metadata["prediction_time_semantics"], dict) or not metadata[
        "prediction_time_semantics"
    ]:
        raise ValueError(
            f"{source} prediction_time_semantics must be a non-empty object."
        )
    _validate_portable_alias(metadata, "source_ref", source=source)
    _validate_portable_alias(metadata, "model_ref", source=source)
    if metadata["training_scope"] != PREDICTION_TRAINING_SCOPE:
        raise ValueError(
            f"{source} training_scope must be {PREDICTION_TRAINING_SCOPE!r} "
            "for prediction handoff runs."
        )
    boundary = _validate_string_field(metadata, "prediction_boundary", source=source)
    lowered_boundary = boundary.casefold()
    missing_terms = [
        term for term in _PREDICTION_BOUNDARY_TERMS if term not in lowered_boundary
    ]
    if missing_terms:
        raise ValueError(
            f"{source} prediction_boundary must explicitly say prediction evidence "
            "is not trading, risk, capacity, execution simulation, paper trading, "
            "or live trading evidence."
        )
    _validate_no_prediction_private_payload(metadata, source=source)


def _validate_string_field(
    metadata: dict[str, Any],
    field: str,
    *,
    source: str,
) -> str:
    value = metadata[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} {field} must be a non-empty string.")
    return value


def _validate_string_sequence(
    metadata: dict[str, Any],
    field: str,
    *,
    source: str,
) -> None:
    value = metadata[field]
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{source} {field} must contain non-empty strings.")


def _validate_portable_alias(
    metadata: dict[str, Any],
    field: str,
    *,
    source: str,
) -> None:
    value = _validate_string_field(metadata, field, source=source)
    if _looks_like_private_or_local_path(value):
        raise ValueError(
            f"{source} {field} must be a portable alias, not a private or "
            "machine-local path."
        )


def _looks_like_private_or_local_path(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    if "/" in normalized:
        return True
    if re.match(r"^[a-z]:", normalized):
        return True
    if ".." in normalized.split():
        return True
    return any(marker in normalized for marker in _PRIVATE_PATH_MARKERS)


def _validate_no_prediction_private_payload(
    metadata: dict[str, Any],
    *,
    source: str,
) -> None:
    for key, value in metadata.items():
        normalized_key = str(key).casefold()
        if any(marker in normalized_key for marker in _ROW_LEVEL_SAMPLE_KEY_MARKERS):
            raise ValueError(
                f"{source} prediction metadata must not include row-level sample "
                f"fields: {key}."
            )
        _validate_no_private_path_value(value, source=source, key=str(key))


def _validate_no_private_path_value(value: Any, *, source: str, key: str) -> None:
    if isinstance(value, str):
        if _looks_like_private_or_local_path(value):
            raise ValueError(
                f"{source} prediction metadata field {key} must not contain "
                "private or machine-local paths."
            )
        return
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            _validate_no_private_path_value(
                nested_value,
                source=source,
                key=f"{key}.{nested_key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_private_path_value(
                item,
                source=source,
                key=f"{key}[{index}]",
            )


__all__ = [
    "MINIMUM_RUN_METADATA_FIELDS",
    "PREDICTION_HANDOFF_METADATA_FIELDS",
    "PREDICTION_TRAINING_SCOPE",
    "validate_prediction_handoff_metadata",
    "validate_run_metadata",
]
