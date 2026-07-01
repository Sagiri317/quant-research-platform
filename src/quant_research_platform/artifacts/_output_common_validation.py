"""Common output validation and destination checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.layout import output_bundle_root
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.policy.share_safety import sanitize_shared_object
from quant_research_platform.storage.layout import StorageLayout


def _required_frame_metadata(
    frame: pd.DataFrame,
    *,
    key: str,
    error_name: str,
) -> dict[str, object]:
    if key not in frame.attrs:
        raise ValueError(error_name)
    return _sanitize_required_metadata(frame.attrs[key], error_name=error_name)


def _sanitize_required_metadata(
    value: object,
    *,
    error_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(error_name)
    sanitized = sanitize_shared_object(dict(value))
    if not isinstance(sanitized, dict) or not sanitized:
        raise ValueError(error_name)
    return sanitized


def _validate_metadata_row_count_chain(
    metadata: dict[str, object],
    keys: tuple[str, str, str],
    error_name: str,
) -> None:
    first, second, third = (int(metadata[key]) for key in keys)
    if not (0 <= third <= second <= first):
        raise ValueError(error_name)


def _validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if list(columns) != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def _require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if not isinstance(value, str) or not value:
        raise ValueError(error_name)
    return value


def _require_optional_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str | None:
    value = metadata[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(error_name)
    return value


def _require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if not isinstance(value, list):
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(error_name)
    return value


def _require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(error_name)
    return value


def _validate_output_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    error_name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{error_name}_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_{error_name}_columns: {', '.join(extra)}")


def _derive_output_bundle_root(
    *,
    output_spec: OutputSpec,
    layout: StorageLayout,
    expected_definition_id: str,
) -> Path:
    try:
        definition = output_catalog().get(output_spec.output_definition_id)
    except KeyError as exc:
        raise ValueError("output_spec_mismatch") from exc
    if definition.output_definition_id != expected_definition_id:
        raise ValueError("output_spec_mismatch")
    if definition.run_type != output_spec.run_type:
        raise ValueError("output_spec_mismatch")
    if definition.scope != output_spec.scope:
        raise ValueError("output_spec_mismatch")
    if output_spec.scope != OutputScope.LOCAL_ONLY:
        raise ValueError("output_scope_must_be_local_only")
    output_bundle = output_bundle_root(layout.project_root, output_spec.output_id)
    if any(
        f"data/{layer}" in output_bundle.as_posix()
        for layer in ("raw", "interim", "processed", "research")
    ):
        raise ValueError("output_bundle_must_not_be_dataset_root")
    return output_bundle
