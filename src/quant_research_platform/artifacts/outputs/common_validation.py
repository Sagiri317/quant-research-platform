"""Common output validation and destination checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.storage.identity import require_unaliased_path
from quant_research_platform.storage.layout import StorageLayout

_SIGNAL_METADATA_KEYS = frozenset(
    {"signal_id", "source_kind", "source_column", "direction", "transform"}
)


def validate_metadata_row_count_chain(
    metadata: dict[str, object],
    keys: tuple[str, str, str],
    error_name: str,
) -> None:
    first, second, third = (
        require_non_negative_metadata_int(metadata, key, error_name) for key in keys
    )
    if not (third <= second <= first):
        raise ValueError(error_name)


def validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if type(columns) is not list or columns != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_optional_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str | None:
    value = metadata[key]
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if type(value) is not list:
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(error_name)
    if len(value) != len(set(value)):
        raise ValueError(error_name)
    return value


def require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if type(value) is not int or value < 0:
        raise ValueError(error_name)
    return value


def require_signal_metadata_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> tuple[SignalSpec, ...]:
    value = metadata[key]
    if type(value) is not list or not value:
        raise ValueError(error_name)
    signals: list[SignalSpec] = []
    for signal in value:
        if type(signal) is not dict or set(signal) != _SIGNAL_METADATA_KEYS:
            raise ValueError(error_name)
        for field in _SIGNAL_METADATA_KEYS:
            require_metadata_string(signal, field, error_name)
        try:
            signals.append(
                SignalSpec(
                    signal_id=signal["signal_id"],
                    source_kind=signal["source_kind"],
                    source_column=signal["source_column"],
                    direction=signal["direction"],
                    transform=signal["transform"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(error_name) from exc
    return tuple(signals)


def validate_output_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    error_name: str,
) -> None:
    observed = tuple(frame.columns)
    if any(type(column) is not str or not column for column in observed):
        raise ValueError(f"invalid_{error_name}_columns")
    if len(observed) != len(set(observed)):
        raise ValueError(f"duplicate_{error_name}_columns")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{error_name}_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_{error_name}_columns: {', '.join(extra)}")
    if observed != required:
        raise ValueError(f"misordered_{error_name}_columns")


def derive_output_bundle_root(
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
    output_bundle = Path(layout.project_root) / "outputs" / output_spec.output_id
    require_unaliased_path(
        output_bundle, error="output_bundle_must_not_be_dataset_root"
    )
    if output_bundle.resolve().is_relative_to((layout.project_root / "data").resolve()):
        raise ValueError("output_bundle_must_not_be_dataset_root")
    return output_bundle
