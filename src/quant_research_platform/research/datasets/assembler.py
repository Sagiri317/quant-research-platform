"""Research dataset assembly from processed datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.columns import (
    ColumnRole,
    _validate_asset_time_key,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.ids import MARKET_ID
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.research.features.transforms import (
    _industry_group_codes,
    _mad_winsorize,
    _neutralize_cross_section,
    _winsorize,
    apply_feature_transforms,
    canonical_feature_validity_columns,
)
from quant_research_platform.research.filtering import (
    a_share_eligibility,
    a_share_eligibility_columns,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.writing import (
    write_dataset,
)
from quant_research_platform.storage.identity import require_artifact_root


def _assemble_research_dataset_from_sessions(
    *,
    output_root: Path,
    spec: ResearchDatasetSpec,
    time_window: TimeWindow,
    input_sessions: Mapping[str, DatasetSession],
    metadata: Mapping[str, object] | None,
) -> Path:
    artifact_root = require_artifact_root(output_root)
    metadata = dict(metadata or {})
    label_semantics = {}
    for session in input_sessions.values():
        semantics = session.manifest.get("producer_diagnostics", {}).get(
            "label_semantics", {}
        )
        for label, facts in semantics.items():
            if (
                label not in spec.label_availability_columns
                or label not in session.logical_columns
            ):
                continue
            if label in label_semantics and label_semantics[label] != facts:
                raise ValueError("research_label_semantics_conflict")
            label_semantics[label] = facts
    if label_semantics:
        metadata["producer_diagnostics"] = {
            **metadata.get("producer_diagnostics", {}),
            "label_semantics": label_semantics,
        }
    diagnostics = None
    if spec.feature_transforms:
        metadata["producer_diagnostics"] = dict(
            metadata.get("producer_diagnostics", {})
        )
        diagnostics = {}
        metadata["producer_diagnostics"]["canonical_features"] = diagnostics
    chunks = _iter_joined_processed_chunks(
        dataset_spec=spec,
        sessions=input_sessions,
        time_window=time_window,
        diagnostics=diagnostics,
    )

    write_dataset(
        artifact_root,
        chunks,
        spec=spec,
        metadata=metadata,
    )
    return artifact_root


def _iter_joined_processed_chunks(
    *,
    dataset_spec: ResearchDatasetSpec,
    sessions: Mapping[str, DatasetSession],
    time_window: TimeWindow,
    diagnostics: dict | None = None,
) -> Iterator[pd.DataFrame]:
    base_session = sessions[dataset_spec.base_input_data_id]
    physical_order = (
        "timestamp",
        *(key for key in dataset_spec.primary_key if key != "timestamp"),
    )
    _, stop, closed = time_window_timestamp_interval(time_window)
    stop = pd.Timestamp(stop)
    if not closed:
        stop -= pd.Timedelta(1, "ns")
    pending = None
    with closing_iterator(
        iter_complete_time_frames(
            iter_dataset_chunks(base_session), timestamp_column="timestamp"
        )
    ) as chunks:
        auxiliary = {}
        for data_id in dataset_spec.input_data_ids:
            if data_id == dataset_spec.base_input_data_id:
                continue
            # Physical partitions may be asset-major. Read selected rows once,
            # then order their actual timestamps before slicing base domains.
            frame = materialize_dataset_session(sessions[data_id])
            timestamps = strict_timestamp_series(
                frame.timestamp, error="invalid_time_chunk_timestamp_values"
            )
            order = timestamps.argsort(kind="mergesort")
            auxiliary[data_id] = (
                frame.iloc[order].reset_index(drop=True),
                timestamps.iloc[order].reset_index(drop=True),
            )

        def auxiliary_through(end):
            inputs = {}
            for data_id, (frame, timestamps) in auxiliary.items():
                split = int(timestamps.searchsorted(end, side="right"))
                inputs[data_id] = frame.iloc[:split]
                auxiliary[data_id] = (frame.iloc[split:], timestamps.iloc[split:])
            return inputs

        for frame in chunks:
            if frame.empty:
                continue
            if pending is not None:
                end = pd.Timestamp(pending.timestamp.iloc[-1])
                chunk = _join_processed_inputs(
                    dataset_spec=dataset_spec,
                    input_frames=auxiliary_through(end),
                    base_frame=pending,
                    diagnostics=diagnostics,
                )
                validate_research_dataset(chunk, dataset_spec)
                yield chunk.sort_values(
                    list(physical_order), kind="mergesort", ignore_index=True
                )
            pending = frame
        if pending is None:
            # An empty base still has no authority to hide auxiliary extra keys.
            for data_id, auxiliary in auxiliary_through(stop).items():
                if len(auxiliary):
                    raise ValueError(
                        f"extra_research_input_keys: {data_id}: {len(auxiliary)}"
                    )
            raise ValueError("no_research_dataset_sessions")
        chunk = _join_processed_inputs(
            dataset_spec=dataset_spec,
            input_frames=auxiliary_through(stop),
            base_frame=pending,
            diagnostics=diagnostics,
        )
        validate_research_dataset(chunk, dataset_spec)
        yield chunk.sort_values(
            list(physical_order), kind="mergesort", ignore_index=True
        )


def _join_processed_inputs(
    *,
    dataset_spec: ResearchDatasetSpec,
    input_frames: Mapping[str, pd.DataFrame],
    base_frame: pd.DataFrame,
    diagnostics: dict | None = None,
) -> pd.DataFrame:
    base = dataset_spec.base_input_data_id
    assembled = base_frame.reset_index(drop=True)
    validate_research_dataset(assembled, dataset_spec, require_all_columns=False)
    base_keys = _scalar_keys(assembled, dataset_spec.primary_key)
    for input_data_id in dataset_spec.input_data_ids:
        if input_data_id == base:
            continue
        input_frame = input_frames[input_data_id]
        validate_research_dataset(input_frame, dataset_spec, require_all_columns=False)
        positions = _input_key_positions(
            base_keys,
            input_frame,
            dataset_spec=dataset_spec,
            input_data_id=input_data_id,
        )
        payload = input_frame.drop(columns=list(dataset_spec.primary_key))
        if -1 in positions:
            # Adding a missing row must not round observed discrete values.
            payload = payload.copy()
            for column, dtype in payload.dtypes.items():
                if isinstance(dtype, np.dtype) and dtype.kind in "iub":
                    prefix = "UInt" if dtype.kind == "u" else "Int"
                    nullable = (
                        "boolean"
                        if dtype.kind == "b"
                        else f"{prefix}{dtype.itemsize * 8}"
                    )
                    payload[column] = payload[column].astype(nullable)
            payload = payload.reset_index(drop=True).reindex(positions)
        else:
            payload = payload.iloc[positions]
        assembled = pd.concat([assembled, payload.reset_index(drop=True)], axis=1)
    assembled = _add_a_share_semantic_columns(assembled, dataset_spec)
    if dataset_spec.feature_transforms:
        assembled = apply_feature_transforms(
            assembled, dataset_spec.feature_transforms, diagnostics=diagnostics
        )
    return assembled.loc[:, list(_ordered_output_columns(dataset_spec))]


def _ordered_input_data_ids(
    dataset_spec: ResearchDatasetSpec,
) -> tuple[str, ...]:
    base = dataset_spec.base_input_data_id
    return (
        base,
        *(data_id for data_id in dataset_spec.input_data_ids if data_id != base),
    )


def _scalar_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple]:
    # Extract columns independently; NumPy row coercion can change key identity.
    return list(
        zip(
            *(
                [
                    value.item()
                    if isinstance(value, (np.integer, np.floating, np.bool_))
                    else value
                    for value in frame[column].tolist()
                ]
                for column in columns
            ),
            strict=True,
        )
    )


def _input_key_positions(
    base_keys: list[tuple],
    input_frame: pd.DataFrame,
    *,
    dataset_spec: ResearchDatasetSpec,
    input_data_id: str,
) -> list[int]:
    input_positions = {
        key: position
        for position, key in enumerate(
            _scalar_keys(input_frame, dataset_spec.primary_key)
        )
    }
    extra = input_positions.keys() - set(base_keys)
    if extra:
        raise ValueError(f"extra_research_input_keys: {input_data_id}: {len(extra)}")
    positions = [input_positions.get(key, -1) for key in base_keys]
    policy = dataset_spec.key_coverage.get(input_data_id, "strict")
    if policy == "sparse":
        return positions
    if policy != "strict":
        raise ValueError(f"invalid_research_key_coverage: {input_data_id}={policy}")
    missing = positions.count(-1)
    if missing:
        raise ValueError(f"missing_research_input_keys: {input_data_id}: {missing}")
    return positions


def _required_input_columns(spec: ResearchDatasetSpec) -> tuple[str, ...]:
    columns = list(_ordered_output_columns(spec))
    for transform in spec.feature_transforms:
        columns.extend(transform.support_columns)
        columns.extend(transform.source_columns)
    generated = _generated_semantic_columns(spec)
    if "eligible" in generated:
        columns.extend(
            a_share_eligibility_columns(
                minimum_listing_sessions=spec.minimum_listing_sessions,
                exclude_st=spec.exclude_st,
                exclude_pt=spec.exclude_pt,
            )
        )
        if spec.eligibility_recipe in {"a_share_nonfinancial", "a_share_fundamental"}:
            columns.extend(
                (
                    "is_nonfinancial_industry",
                    "industry_standard_id",
                    "industry_level1_code",
                    "industry_level1_name",
                )
            )
        if spec.eligibility_recipe == "a_share_fundamental":
            columns.append("has_legal_fundamental_pit")
    if "model_target" in generated:
        if spec.target_transform == "industry_size_residual":
            columns.extend(
                ("industry_standard_id", "industry_level1_code", "log_total_market_cap")
            )
        columns.extend(
            (
                "eligible",
                "forward_return_20d",
                spec.label_availability_columns.get(
                    "forward_return_20d", "forward_return_20d_available_at"
                ),
            )
        )
    return tuple(dict.fromkeys(columns))


def _processed_input_columns(
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    *,
    existing_columns: tuple[str, ...],
) -> tuple[str, ...]:
    existing = set(existing_columns)
    output_columns = set(_required_input_columns(dataset_spec)).difference(
        _generated_semantic_columns(dataset_spec)
    )
    columns: list[str] = []
    for column in dataset_spec.primary_key:
        columns.append(column)
    for column in processed_spec.required_columns:
        if column in dataset_spec.primary_key:
            continue
        if column not in output_columns:
            continue
        if column in existing:
            continue
        columns.append(column)
    return tuple(columns)


def _validate_research_input_specs(
    dataset_spec: ResearchDatasetSpec, processed_specs: Mapping[str, ProcessedDataSpec]
) -> None:
    for data_id, processed_spec in processed_specs.items():
        missing_key_columns = [
            column
            for column in dataset_spec.primary_key
            if column not in processed_spec.primary_key
        ]
        if missing_key_columns:
            raise ValueError(
                f"missing_research_join_key: {data_id}: "
                + ", ".join(missing_key_columns)
            )
        if set(processed_spec.primary_key) != set(dataset_spec.primary_key):
            raise ValueError(f"incompatible_research_input_keys: {data_id}")
        _validate_asset_time_key(
            processed_spec.primary_key, processed_spec.column_roles
        )
    _require_no_duplicate_output_columns(dataset_spec, processed_specs)
    _require_research_required_columns_available(dataset_spec, processed_specs)
    role_by_column: dict[str, set[ColumnRole]] = {}
    kind_by_column: dict[str, set[ProcessedDataKind]] = {}
    for processed_spec in processed_specs.values():
        for column, role in processed_spec.column_roles.items():
            role_by_column.setdefault(column, set()).add(role)
            kind_by_column.setdefault(column, set()).add(processed_spec.kind)
    _require_columns_from_roles_or_kinds(
        dataset_spec.factor_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FACTOR},
        allowed_kinds={ProcessedDataKind.FACTOR},
        error_name="invalid_research_factor_columns",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            dict.fromkeys(
                column
                for transform in dataset_spec.feature_transforms
                for column in transform.source_columns
            )
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FACTOR, ColumnRole.CONTEXT},
        allowed_kinds=set(),
        error_name="invalid_feature_transform_sources",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            column
            for column in dataset_spec.feature_columns
            if column not in _generated_feature_columns(dataset_spec)
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FEATURE},
        allowed_kinds=set(),
        error_name="invalid_research_feature_columns",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            column
            for column in dataset_spec.label_columns
            if column not in _generated_semantic_columns(dataset_spec)
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.LABEL},
        allowed_kinds=set(),
        error_name="invalid_research_label_columns",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            column
            for column in dataset_spec.filter_columns
            if column not in _generated_semantic_columns(dataset_spec)
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.MASK, ColumnRole.UNIVERSE},
        allowed_kinds=set(),
        error_name="invalid_research_filter_columns",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            column
            for column in dataset_spec.context_columns
            if column not in _generated_semantic_columns(dataset_spec)
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.CONTEXT, ColumnRole.TIMESTAMP},
        allowed_kinds=set(),
        error_name="invalid_research_context_columns",
    )


def _require_columns_from_roles_or_kinds(
    columns: tuple[str, ...],
    *,
    role_by_column: dict[str, set[ColumnRole]],
    kind_by_column: dict[str, set[ProcessedDataKind]],
    allowed_roles: set[ColumnRole],
    allowed_kinds: set[ProcessedDataKind],
    error_name: str,
) -> None:
    invalid = [
        column
        for column in columns
        if not role_by_column.get(column, set()).intersection(allowed_roles)
        and (not kind_by_column.get(column, set()).intersection(allowed_kinds))
    ]
    if invalid:
        raise ValueError(f"{error_name}: {', '.join(invalid)}")


def _require_no_duplicate_output_columns(
    dataset_spec: ResearchDatasetSpec, processed_specs: dict[str, ProcessedDataSpec]
) -> None:
    requested = (
        set(_required_input_columns(dataset_spec))
        .difference(dataset_spec.primary_key)
        .difference(_generated_feature_columns(dataset_spec))
        .difference(_generated_semantic_columns(dataset_spec))
    )
    producers: dict[str, list[str]] = {}
    for data_id, processed_spec in processed_specs.items():
        for column in processed_spec.required_columns:
            if column in dataset_spec.primary_key or column not in requested:
                continue
            producers.setdefault(column, []).append(data_id)
    duplicates = sorted(
        (column for column, data_ids in producers.items() if len(data_ids) > 1)
    )
    if duplicates:
        raise ValueError("duplicate_research_output_columns: " + ", ".join(duplicates))


def _require_research_required_columns_available(
    dataset_spec: ResearchDatasetSpec, processed_specs: dict[str, ProcessedDataSpec]
) -> None:
    available = {
        column
        for processed_spec in processed_specs.values()
        for column in processed_spec.required_columns
    }
    missing = sorted(
        set(_required_input_columns(dataset_spec))
        .difference(available)
        .difference(
            _generated_feature_columns(dataset_spec)
            + _generated_semantic_columns(dataset_spec)
        )
    )
    if missing:
        raise ValueError("missing_research_required_columns: " + ", ".join(missing))


def _ordered_output_columns(dataset_spec: ResearchDatasetSpec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *dataset_spec.factor_columns,
                *dataset_spec.feature_columns,
                *dataset_spec.label_columns,
                *dataset_spec.filter_columns,
                *dataset_spec.context_columns,
            )
        )
    )


def _generated_feature_columns(spec: ResearchDatasetSpec) -> tuple[str, ...]:
    return tuple(
        column
        for transform in spec.feature_transforms
        for column in transform.output_columns
    )


def _generated_semantic_columns(dataset_spec: ResearchDatasetSpec) -> tuple[str, ...]:
    if dataset_spec.market_id != MARKET_ID:
        return ()
    needed = set(_ordered_output_columns(dataset_spec))
    if dataset_spec.feature_transforms or "model_target" in dataset_spec.label_columns:
        needed.add("eligible")
    generated = []
    if dataset_spec.eligibility_recipe is not None and "eligible" in needed:
        generated.append("eligible")
    if "model_target" in dataset_spec.label_columns:
        generated.extend(("model_target", "model_target_available_at"))
    return tuple(generated)


def _add_a_share_semantic_columns(
    frame: pd.DataFrame, dataset_spec: ResearchDatasetSpec
) -> pd.DataFrame:
    generated = _generated_semantic_columns(dataset_spec)
    if not generated:
        return frame
    result = frame.copy()
    if "eligible" in generated:
        result["eligible"] = a_share_eligibility(
            result,
            dataset_spec.eligibility_recipe,
            minimum_listing_sessions=dataset_spec.minimum_listing_sessions,
            exclude_st=dataset_spec.exclude_st,
            exclude_pt=dataset_spec.exclude_pt,
        )
    if "model_target" in generated:
        if "forward_return_20d" not in result.columns:
            raise ValueError("a_share_model_target_label_missing")
        maturity = dataset_spec.label_availability_columns.get("forward_return_20d")
        if maturity is None:
            raise ValueError("model_target_label_availability_required")
        result["model_target"] = _a_share_model_target(
            result,
            transform=dataset_spec.target_transform,
            clip=dataset_spec.target_clip,
        )
        participating = result.model_target.notna()
        clocks = strict_timestamp_series(
            result[maturity].where(participating),
            error="invalid_model_target_label_availability",
            allow_null=True,
        )
        if (
            dataset_spec.target_transform == "raw"
            and dataset_spec.target_clip == "none"
        ):
            result["model_target_available_at"] = clocks
        else:
            unknown = (
                (participating & clocks.isna())
                .groupby(result.timestamp, sort=False)
                .transform("any")
            )
            result["model_target_available_at"] = (
                clocks.groupby(result.timestamp, sort=False)
                .transform("max")
                .where(participating & ~unknown)
            )
    return result


def _a_share_model_target(
    frame: pd.DataFrame, *, transform: str = "date_zscore", clip: str = "none"
) -> pd.Series:
    values = strict_finite_numeric_series(
        frame["forward_return_20d"],
        error="invalid_a_share_model_target_label",
        allow_null=True,
    ).astype("float64")
    eligible = strict_bool_series(
        frame["eligible"], error="invalid_a_share_eligibility"
    )
    scoped = frame.loc[eligible].copy()
    scoped["forward_return_20d"] = values.loc[eligible]
    if transform == "industry_size_residual":
        sizes = strict_finite_numeric_series(
            scoped["log_total_market_cap"],
            error="invalid_neutralization_size",
            allow_null=True,
        )
        scoped = scoped.loc[
            _industry_group_codes(scoped).notna() & sizes.notna()
        ].copy()
    groups = scoped.groupby("timestamp", sort=False).indices
    if clip == "mad":
        scoped["forward_return_20d"] = _mad_winsorize(scoped.forward_return_20d, groups)
    elif clip == "quantile_1_99":
        scoped["forward_return_20d"] = _winsorize(scoped.forward_return_20d, groups)
    if transform == "industry_size_residual":
        scoped["industry_level1_code"] = _industry_group_codes(scoped)
        scoped = _neutralize_cross_section(
            scoped,
            factor_columns=("forward_return_20d",),
            industry_column="industry_level1_code",
            size_column="log_total_market_cap",
            rescale=False,
            time_groups=groups,
        )
    target = pd.Series(np.nan, index=frame.index, dtype="float64")
    for positions in groups.values():
        group = scoped.forward_return_20d.iloc[positions]
        valid = group.dropna()
        if transform in {"raw", "industry_size_residual"}:
            target.loc[group.index] = group
        elif len(valid) >= 2:
            target.loc[group.index] = (
                _stable_zscore(group)
                if transform == "date_zscore"
                else group.rank(method="average", pct=True)
            )
    return target


def assemble(config: Mapping) -> Path:
    """Assemble selected columns from actual input declarations and optional recipes."""
    from quant_research_platform.contracts.features import FeatureTransformSpec
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.storage.datasets.reading import _dataset_input_options

    config = qrp_json_mapping_snapshot(config)
    required = {"market_id", "dataset_id", "inputs", "time_window", "output_root"}
    optional = {
        "factor_columns",
        "feature_columns",
        "label_columns",
        "feature_recipe",
        "feature_transform",
        "feature_groups",
        "target_transform",
        "target_clip",
        "base_input_data_id",
        "sparse_input_data_ids",
        "context_columns",
        "filter_columns",
        "eligibility_recipe",
        "minimum_listing_sessions",
        "exclude_st",
        "exclude_pt",
    }
    if (
        not isinstance(config, Mapping)
        or not required <= set(config)
        or set(config) - required - optional
    ):
        raise ValueError("assemble_fields_invalid")
    if any(
        type(config[key]) is not str or not config[key]
        for key in ("market_id", "dataset_id", "output_root")
    ):
        raise ValueError("assemble_value_invalid")
    inputs = config["inputs"]
    if (
        not isinstance(inputs, Mapping)
        or not inputs
        or any(type(key) is not str or not key for key in inputs)
    ):
        raise ValueError("assemble_input_paths_invalid")
    bounds = config["time_window"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"start", "end"}:
        raise ValueError("assemble_window_invalid")
    window = TimeWindow(**bounds)
    if config["market_id"] == "a_share":
        from quant_research_platform.research.protocols.a_share import (
            research_dataset_catalog,
        )
    elif config["market_id"] == "crypto":
        from quant_research_platform.research.protocols.crypto_hft import (
            build_hft_research_dataset_catalog as research_dataset_catalog,
        )
    else:
        research_dataset_catalog = None
    defaults = (
        next(
            (
                item
                for item in research_dataset_catalog().list()
                if item.data_id == config["dataset_id"]
            ),
            None,
        )
        if research_dataset_catalog is not None
        else None
    )
    base_input = config.get(
        "base_input_data_id",
        defaults.base_input_data_id if defaults else next(iter(inputs)),
    )
    if type(base_input) is not str or base_input not in inputs:
        raise ValueError("assemble_base_input_invalid")
    sessions = {
        data_id: open_dataset_session(
            **_dataset_input_options(value, time_window=window)
        )
        for data_id, value in inputs.items()
    }
    actual_specs = {}
    for data_id, session in sessions.items():
        declared = session.spec
        if declared.get("market_id", config["market_id"]) != config["market_id"]:
            raise ValueError("research_input_semantics_mismatch")
        actual_specs[data_id] = ProcessedDataSpec(
            data_id=declared["data_id"],
            market_id=config["market_id"],
            kind=declared.get("kind", "panel"),
            frequency=declared["frequency"],
            grain=declared["grain"],
            primary_key=tuple(declared["primary_key"]),
            partition_columns=tuple(declared["partition_columns"]),
            column_roles={
                column: declared["column_roles"][column]
                for column in declared["required_columns"]
            },
            column_units=declared.get("column_units", {}),
            label_availability_columns=declared.get("label_availability_columns", {}),
        )
    selected_roles = {
        column: role
        for data_id, item in actual_specs.items()
        for column, role in item.column_roles.items()
        if column in sessions[data_id].logical_columns
    }
    selected_factors = {
        column
        for data_id, item in actual_specs.items()
        for column, role in item.column_roles.items()
        if role is ColumnRole.FACTOR and column in sessions[data_id].logical_columns
    }
    selected_labels = {
        column
        for data_id, item in actual_specs.items()
        for column, role in item.column_roles.items()
        if role is ColumnRole.LABEL and column in sessions[data_id].logical_columns
    }
    selected_features = {
        column for column, role in selected_roles.items() if role is ColumnRole.FEATURE
    }
    reused_features = config.get(
        "feature_columns",
        [column for column in selected_roles if column in selected_features],
    )
    factors = config.get(
        "factor_columns",
        [
            column
            for column in (defaults.factor_columns if defaults else selected_roles)
            if column in selected_factors
        ],
    )
    labels = config.get(
        "label_columns",
        [
            column
            for column in (defaults.label_columns if defaults else selected_roles)
            if column in selected_labels
            or (column == "model_target" and "forward_return_20d" in selected_labels)
        ],
    )
    filters = config.get(
        "filter_columns", list(defaults.filter_columns) if defaults else []
    )
    context = config.get("context_columns", [])
    sparse_inputs = config.get("sparse_input_data_ids", [])
    for name, values in (
        ("factor_columns", factors),
        ("feature_columns", reused_features),
        ("label_columns", labels),
        ("filter_columns", filters),
        ("context_columns", context),
        ("sparse_inputs", sparse_inputs),
    ):
        if (
            type(values) is not list
            or any(type(value) is not str or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("assemble_" + name + "_invalid")
    if set(factors) - {
        column
        for item in actual_specs.values()
        for column, role in item.column_roles.items()
        if role is ColumnRole.FACTOR
    }:
        raise ValueError("assemble_factor_not_available")
    if set(factors) - selected_factors:
        raise ValueError("assemble_factor_not_selected")
    if set(reused_features) - selected_features:
        raise ValueError("assemble_feature_not_selected")
    if not factors and not reused_features:
        raise ValueError("research_factor_or_feature_columns_required")
    # A selected feature and its declared companions remain one producer's facts.
    feature_producers = {}
    for column in tuple(reused_features):
        candidates = [
            name
            for name, session in sessions.items()
            if column in session.logical_columns
        ]
        if len(candidates) != 1:
            raise ValueError("duplicate_research_output_columns: " + column)
        name = candidates[0]
        declared = actual_specs[name].required_columns
        companions = (column, *canonical_feature_validity_columns((column,), declared))
        for companion in companions:
            if companion in feature_producers and feature_producers[companion] != name:
                raise ValueError("duplicate_research_output_columns: " + companion)
            feature_producers[companion] = name
            if companion not in reused_features:
                reused_features.append(companion)
    feature_lineage = {}
    for name, session in sessions.items():
        columns = {
            column for column, producer in feature_producers.items() if producer == name
        }
        if not columns:
            continue
        inherited = session.manifest.get("producer_diagnostics", {}).get(
            "reused_feature_transforms", {}
        )
        feature_lineage.update(
            {column: inherited[column] for column in columns if column in inherited}
        )
        for transform in session.spec.get("feature_transforms", []):
            outputs = FeatureTransformSpec(
                **{
                    **transform,
                    "source_columns": tuple(transform["source_columns"]),
                    "output_groups": tuple(transform["output_groups"]),
                }
            ).output_columns
            feature_lineage.update(
                {column: transform for column in outputs if column in columns}
            )
    generated_labels = {"model_target"} if config["market_id"] == "a_share" else set()
    if (
        set(labels)
        - {
            column
            for item in actual_specs.values()
            for column, role in item.column_roles.items()
            if role is ColumnRole.LABEL
        }
        - generated_labels
    ):
        raise ValueError("assemble_label_not_available")
    if set(labels) - selected_labels - generated_labels or (
        "model_target" in labels and "forward_return_20d" not in selected_labels
    ):
        raise ValueError("assemble_label_not_selected")
    generated_context = (
        {"model_target_available_at"} if "model_target" in labels else set()
    )
    if set(context) - set(selected_roles) - generated_context:
        raise ValueError("assemble_context_not_selected")
    if {"target_transform", "target_clip"}.intersection(config) and (
        config["market_id"] != "a_share" or "model_target" not in labels
    ):
        raise ValueError("assemble_target_transform_requires_model_target")
    business_columns = set((*factors, *labels, *filters, *context))
    recipe = config.get(
        "feature_recipe",
        "a_share_canonical" if config["market_id"] == "a_share" and factors else None,
    )
    groups = config.get("feature_groups", ["raw", "processed"] if recipe else ["raw"])
    if (
        type(groups) is not list
        or not groups
        or any(type(group) is not str for group in groups)
        or len(set(groups)) != len(groups)
        or any(group not in {"raw", "processed", "pre_neutral"} for group in groups)
    ):
        raise ValueError("assemble_feature_groups_invalid")
    if "feature_groups" in config:
        reused_groups = dict.fromkeys(reused_features, "processed")
        for column, lineage in feature_lineage.items():
            transform = FeatureTransformSpec(
                **{
                    **lineage,
                    "source_columns": tuple(lineage["source_columns"]),
                    "output_groups": tuple(lineage["output_groups"]),
                }
            )
            for group in transform.output_groups:
                if column in replace(transform, output_groups=(group,)).output_columns:
                    reused_groups[column] = group
        if "feature_columns" in config and any(
            reused_groups[column] not in groups for column in config["feature_columns"]
        ):
            raise ValueError("assemble_feature_group_conflict")
        reused_features = [
            column for column in reused_features if reused_groups[column] in groups
        ]
        feature_producers = {
            column: name
            for column, name in feature_producers.items()
            if column in reused_features
        }
        feature_lineage = {
            column: value
            for column, value in feature_lineage.items()
            if column in reused_features
        }
    processed_groups = tuple(group for group in groups if group != "raw")
    if not processed_groups:
        recipe = None
    elif recipe is None and not reused_features:
        raise ValueError("assemble_feature_groups_requires_recipe")
    options = config.get("feature_transform", {})
    if not isinstance(options, Mapping) or set(options) - {
        "clip",
        "industry_neutralize",
        "size_neutralize",
        "size_column",
        "center",
        "scale",
        "impute_missing",
    }:
        raise ValueError("assemble_feature_transform_invalid")
    if config["market_id"] != "a_share" and (recipe is not None or options):
        raise ValueError("assemble_feature_recipe_not_supported")
    if recipe is None and options:
        raise ValueError("assemble_feature_transform_requires_recipe")
    transforms = (
        (
            FeatureTransformSpec(
                recipe, tuple(factors), output_groups=processed_groups, **options
            ),
        )
        if recipe is not None
        else ()
    )
    if transforms and "feature_recipe" not in config and not options:
        transform = transforms[0]
        pending_sources = tuple(
            factor
            for factor in factors
            if not set(
                replace(transform, source_columns=(factor,)).output_columns
            ).issubset(reused_features)
        )
        transforms = (
            (replace(transform, source_columns=pending_sources),)
            if pending_sources
            else ()
        )
        if not transforms:
            recipe = None
    generated_features = tuple(
        column for transform in transforms for column in transform.output_columns
    )
    if set(generated_features).intersection(reused_features):
        raise ValueError("assemble_feature_output_conflict")
    features = (*reused_features, *generated_features)
    labels, context = list(labels), list(context)
    availability = {}
    maturity_producers = {}
    for label in (
        *labels,
        *(("forward_return_20d",) if "model_target" in labels else ()),
    ):
        candidates = [
            name
            for name, session in sessions.items()
            if label in session.logical_columns
        ]
        if len(candidates) > 1:
            raise ValueError("duplicate_research_output_columns: " + label)
        if candidates:
            maturity = actual_specs[candidates[0]].label_availability_columns.get(label)
            if maturity:
                availability[label] = maturity
                if (
                    maturity in maturity_producers
                    and maturity_producers[maturity] != candidates[0]
                ):
                    raise ValueError("duplicate_research_output_columns: " + maturity)
                maturity_producers[maturity] = candidates[0]
    if "model_target" in labels:
        if "forward_return_20d" not in availability:
            raise ValueError("model_target_label_availability_required")
        labels.append("forward_return_20d")
        context.append("model_target_available_at")
        availability["model_target"] = "model_target_available_at"
    for label in tuple(labels):
        maturity = availability.get(label)
        if maturity and maturity not in labels and maturity not in context:
            producer = actual_specs[maturity_producers[maturity]]
            (
                labels
                if producer.column_roles[maturity] is ColumnRole.LABEL
                else context
            ).append(maturity)
    labels = tuple(dict.fromkeys(labels))
    context = tuple(dict.fromkeys(context))
    if set(sparse_inputs) - set(inputs) or base_input in sparse_inputs:
        raise ValueError("assemble_sparse_inputs_invalid")
    output_factors = factors if "raw" in groups else []
    source = actual_specs[base_input]
    output_roles = {
        **{column: source.column_roles[column] for column in source.primary_key},
        **dict.fromkeys(output_factors, ColumnRole.FACTOR),
        **dict.fromkeys(features, ColumnRole.FEATURE),
        **dict.fromkeys(labels, ColumnRole.LABEL),
        **{
            column: selected_roles.get(column, ColumnRole.UNIVERSE)
            for column in filters
        },
        **dict.fromkeys(context, ColumnRole.CONTEXT),
    }
    spec = ResearchDatasetSpec(
        data_id=config["dataset_id"],
        market_id=config["market_id"],
        frequency=source.frequency,
        grain="research_row",
        primary_key=source.primary_key,
        partition_columns=("year",),
        input_data_ids=tuple(inputs),
        base_input_data_id=base_input,
        factor_columns=tuple(output_factors),
        feature_columns=features,
        feature_transforms=transforms,
        target_transform=config.get("target_transform", "date_zscore"),
        target_clip=config.get("target_clip", "none"),
        label_columns=labels,
        filter_columns=tuple(filters),
        context_columns=context,
        eligibility_recipe=config.get(
            "eligibility_recipe",
            defaults.eligibility_recipe if defaults else None,
        ),
        minimum_listing_sessions=config.get("minimum_listing_sessions", 252),
        exclude_st=config.get("exclude_st", True),
        exclude_pt=config.get("exclude_pt", True),
        column_roles=output_roles,
        label_availability_columns={
            label: maturity
            for label, maturity in availability.items()
            if label in labels and maturity in output_roles
        },
        key_coverage={
            key: "sparse" if key in sparse_inputs else "strict" for key in inputs
        },
    )
    # Explicit input columns bound business candidates; dependency-only columns
    # can still support the declared calculation without leaking into output X.
    required_inputs = set(_required_input_columns(spec))
    # Explicit projections select the authority for shared dependencies. A dependency
    # omitted from every projection may be recovered only from a unique producer.
    producers = {}
    for column in required_inputs.difference(spec.primary_key).difference(
        (*_generated_feature_columns(spec), *_generated_semantic_columns(spec))
    ):
        if column in feature_producers or column in maturity_producers:
            producers[column] = feature_producers.get(
                column, maturity_producers.get(column)
            )
            continue
        candidates = [
            data_id
            for data_id, session in sessions.items()
            if column in session.logical_columns
        ]
        if not candidates and column not in business_columns:
            candidates = [
                data_id
                for data_id, item in actual_specs.items()
                if column in item.required_columns
            ]
        if len(candidates) > 1:
            raise ValueError("duplicate_research_output_columns: " + column)
        if candidates:
            producers[column] = candidates[0]
    consumed_specs = {}
    for data_id, item in actual_specs.items():
        consumed = {
            column: role
            for column, role in item.column_roles.items()
            if column in item.primary_key or producers.get(column) == data_id
        }
        consumed_specs[data_id] = replace(
            item,
            column_roles=consumed,
            column_units={
                column: unit
                for column, unit in item.column_units.items()
                if column in consumed
            },
            label_availability_columns={
                label: maturity
                for label, maturity in item.label_availability_columns.items()
                if label in consumed and maturity in consumed
            },
        )
    _validate_research_input_specs(spec, consumed_specs)
    units = {
        column: unit
        for item in consumed_specs.values()
        for column, unit in item.column_units.items()
        if column in output_roles
    }
    for transform in transforms:
        for group in transform.output_groups:
            for factor in transform.source_columns:
                suffix = ("pre_neutral_" if group == "pre_neutral" else "") + factor
                for prefix in ("feature_missing_", "feature_observed_"):
                    units[prefix + suffix] = "dimensionless"
                source_unit = consumed_specs[producers[factor]].column_units.get(factor)
                if transform.scale:
                    units["feature_" + suffix] = "dimensionless"
                elif source_unit is not None:
                    units["feature_" + suffix] = source_unit
    if "model_target" in labels:
        if spec.target_transform in {"raw", "industry_size_residual"}:
            source_unit = consumed_specs[
                producers["forward_return_20d"]
            ].column_units.get("forward_return_20d")
            if source_unit is not None:
                units["model_target"] = source_unit
        else:
            units["model_target"] = "dimensionless"
    spec = replace(spec, column_units=units)
    existing = ()
    projected = {}
    for data_id in _ordered_input_data_ids(spec):
        columns = _processed_input_columns(
            spec, consumed_specs[data_id], existing_columns=existing
        )
        projected[data_id] = sessions[data_id].projected_view(columns)
        existing = tuple(dict.fromkeys((*existing, *columns)))
    return _assemble_research_dataset_from_sessions(
        output_root=Path(config["output_root"]),
        spec=spec,
        time_window=window,
        input_sessions=projected,
        metadata={
            "inputs": inputs,
            "producer_diagnostics": {"reused_feature_transforms": feature_lineage},
            "parameters": {
                "base_input_data_id": base_input,
                "input_data_ids": {
                    name: session.spec["data_id"] for name, session in sessions.items()
                },
                "input_columns": {
                    name: list(session.logical_columns)
                    for name, session in projected.items()
                },
                "feature_columns": list(reused_features),
                "filter_columns": list(filters),
                "context_columns": list(context),
                "factor_columns": list(factors),
                "label_columns": list(labels),
                "feature_recipe": recipe,
                "feature_groups": groups,
                "target_transform": spec.target_transform,
                "target_clip": spec.target_clip,
                "eligibility_recipe": spec.eligibility_recipe,
                "minimum_listing_sessions": spec.minimum_listing_sessions,
                "exclude_st": spec.exclude_st,
                "exclude_pt": spec.exclude_pt,
                "feature_transform": (
                    {
                        name: getattr(transforms[0], name)
                        for name in (
                            "clip",
                            "industry_neutralize",
                            "size_neutralize",
                            "size_column",
                            "center",
                            "scale",
                            "impute_missing",
                        )
                    }
                    if transforms
                    else None
                ),
                "sparse_input_data_ids": sparse_inputs,
                "time_window": dict(bounds),
            },
        },
    )
