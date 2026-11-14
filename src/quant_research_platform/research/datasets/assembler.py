"""Join declared processed inputs and materialize selected research features and
labels.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.columns import (
    ColumnRole,
    _validate_asset_time_key,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.risk import FactorRiskModel
from quant_research_platform.contracts.time import (
    DECISION_CONTEXT_COLUMNS,
    TimeWindow,
)
from quant_research_platform.contracts.validation import (
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.ids import MARKET_ID
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    a_share_eligibility,
    a_share_eligibility_columns,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.risk import (
    prepare_asof_risk_sessions,
    read_asof_risk_facts,
    risk_exposure_columns,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.research.labels.targets import a_share_model_target
from quant_research_platform.research.risk import FactorRiskState
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_time_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.writing import (
    write_dataset,
)


def _assemble_research_dataset_from_sessions(
    *,
    output_root: Path,
    spec: ResearchDatasetSpec,
    time_window: TimeWindow,
    input_sessions: Mapping[str, DatasetSession],
    metadata: Mapping[str, object] | None,
    risk_exposure_input: Mapping | None = None,
) -> Path:
    artifact_root = Path(output_root)
    metadata = dict(metadata or {})
    label_semantics = {}
    for session in input_sessions.values():
        semantics = session.manifest.get("producer_diagnostics", {}).get(
            "label_semantics", {}
        )
        for label, facts in semantics.items():
            if label not in spec.label_columns or label not in session.logical_columns:
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
        risk_exposure_input=risk_exposure_input,
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
    risk_exposure_input: Mapping | None = None,
) -> Iterator[pd.DataFrame]:
    physical_order = (
        "timestamp",
        *(key for key in dataset_spec.primary_key if key != "timestamp"),
    )
    risk_input_id = risk_exposure_input["data_id"] if risk_exposure_input else None
    risk_facts = (
        read_asof_risk_facts(
            sessions[risk_input_id],
            model=dataset_spec.risk_model,
            available_at_column=risk_exposure_input["available_at_column"],
            time_window=time_window,
        )
        if risk_input_id is not None
        else None
    )
    risk_cache = {}
    with ExitStack() as stack:
        streams = {
            data_id: stack.enter_context(
                closing_iterator(iter_dataset_time_chunks(session))
            )
            for data_id, session in sessions.items()
            if data_id != risk_input_id
        }

        def next_auxiliary(data_id):
            frame = next(streams[data_id], None)
            if frame is not None:
                frame = frame.assign(
                    timestamp=strict_timestamp_series(
                        frame.timestamp, error="invalid_time_chunk_timestamp_values"
                    )
                )
            return frame

        auxiliary = {
            data_id: next_auxiliary(data_id)
            for data_id in streams
            if data_id != dataset_spec.base_input_data_id
        }
        empty = {data_id: selected.iloc[:0] for data_id, selected in auxiliary.items()}
        yielded = False
        for frame in streams[dataset_spec.base_input_data_id]:
            if frame.empty:
                continue
            timestamps = strict_timestamp_series(
                frame.timestamp, error="invalid_time_chunk_timestamp_values"
            )
            frame = frame.assign(timestamp=timestamps)
            base_index = pd.MultiIndex.from_frame(
                frame.loc[:, list(dataset_spec.primary_key)]
            )
            inputs = {}
            for data_id, selected in auxiliary.items():
                pieces = []
                while selected is not None:
                    stop = selected.timestamp.searchsorted(
                        timestamps.iloc[-1], side="right"
                    )
                    if stop:
                        consumed = selected.iloc[:stop]
                        matched = pd.MultiIndex.from_frame(
                            consumed.loc[:, list(dataset_spec.primary_key)]
                        ).isin(base_index)
                        if matched.any():
                            pieces.append(consumed.loc[matched])
                    if stop < len(selected):
                        selected = selected.iloc[stop:]
                        break
                    selected = next_auxiliary(data_id)
                auxiliary[data_id] = selected
                inputs[data_id] = (
                    pd.concat(pieces, ignore_index=True) if pieces else empty[data_id]
                )
            chunk = _join_processed_inputs(
                dataset_spec=dataset_spec,
                input_frames=inputs,
                base_frame=frame,
                base_index=base_index,
                diagnostics=diagnostics,
                risk_exposure_input=risk_exposure_input,
                risk_facts=risk_facts,
                risk_cache=risk_cache,
            )
            yielded = True
            yield chunk.sort_values(
                list(physical_order), kind="mergesort", ignore_index=True
            )
        if not yielded:
            raise ValueError("no_research_dataset_sessions")


def _join_processed_inputs(
    *,
    dataset_spec: ResearchDatasetSpec,
    input_frames: Mapping[str, pd.DataFrame],
    base_frame: pd.DataFrame,
    base_index: pd.MultiIndex,
    diagnostics: dict | None = None,
    risk_exposure_input: Mapping | None = None,
    risk_facts: pd.DataFrame | None = None,
    risk_cache: dict | None = None,
) -> pd.DataFrame:
    assembled = base_frame.reset_index(drop=True)
    if (
        "market_id" in base_frame
        and not base_frame.market_id.eq(str(dataset_spec.market_id)).all()
    ):
        raise ValueError("research_dataset_market_mismatch")
    for input_data_id, input_frame in input_frames.items():
        positions = pd.MultiIndex.from_frame(
            input_frame.loc[:, list(dataset_spec.primary_key)]
        ).get_indexer(base_index)
        missing = np.count_nonzero(positions < 0)
        if (
            missing
            and dataset_spec.key_coverage.get(input_data_id, "strict") != "sparse"
        ):
            raise ValueError(f"missing_research_input_keys: {input_data_id}: {missing}")
        matched = input_frame.iloc[positions[positions >= 0]]
        if (
            "market_id" in matched
            and not matched.market_id.eq(str(dataset_spec.market_id)).all()
        ):
            raise ValueError("research_dataset_market_mismatch")
        payload = input_frame.drop(columns=list(dataset_spec.primary_key))
        if missing:
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
    risk_sessions = {}
    assembled = _add_a_share_semantic_columns(
        assembled,
        dataset_spec,
        risk_sessions=risk_sessions,
        risk_facts=risk_facts,
        risk_cache=risk_cache,
        risk_available_at=(
            risk_exposure_input["available_at_column"] if risk_exposure_input else None
        ),
    )
    if dataset_spec.feature_transforms:
        assembled = apply_feature_transforms(
            assembled,
            dataset_spec.feature_transforms,
            diagnostics=diagnostics,
            risk_sessions=risk_sessions,
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


def _required_input_columns(spec: ResearchDatasetSpec) -> tuple[str, ...]:
    columns = list(_ordered_output_columns(spec))
    for transform in spec.feature_transforms:
        columns.extend(transform.support_columns)
        columns.extend(transform.source_columns)
    generated = _generated_semantic_columns(spec)
    if "eligible" in generated and spec.eligibility_recipe is not None:
        columns.extend(
            a_share_eligibility_columns(
                recipe=spec.eligibility_recipe,
                minimum_listing_sessions=spec.minimum_listing_sessions,
                exclude_st=spec.exclude_st,
                exclude_pt=spec.exclude_pt,
            )
        )
    if "model_target" in generated:
        if spec.target_transform == "industry_size_residual":
            columns.extend(
                ("industry_standard_id", "industry_level1_code", "log_total_market_cap")
            )
        if spec.target_transform == "risk_residual":
            columns.extend(
                (
                    "industry_standard_id",
                    "industry_level1_code",
                    spec.risk_model.config.weight_column,
                    *spec.risk_model.style_columns,
                )
            )
        columns.extend(
            (
                "eligible",
                spec.target_label_column,
                spec.label_availability_columns.get(
                    spec.target_label_column, f"{spec.target_label_column}_available_at"
                ),
            )
        )
    return tuple(dict.fromkeys(columns))


def _processed_input_columns(
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    *,
    existing_columns: tuple[str, ...],
    extra_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    existing = set(existing_columns)
    output_columns = set(
        (*_required_input_columns(dataset_spec), *extra_columns)
    ).difference(_generated_semantic_columns(dataset_spec))
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
    dataset_spec: ResearchDatasetSpec,
    processed_specs: Mapping[str, ProcessedDataSpec],
    *,
    provided_columns: tuple[str, ...] = (),
    risk_input_data_id: str | None = None,
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
        # As-of risk facts can also key publication versions and listing lives.
        if data_id != risk_input_data_id and set(processed_spec.primary_key) != set(
            dataset_spec.primary_key
        ):
            raise ValueError(f"incompatible_research_input_keys: {data_id}")
        _validate_asset_time_key(
            processed_spec.primary_key, processed_spec.column_roles
        )
    _require_research_required_columns_available(
        dataset_spec, processed_specs, provided_columns=provided_columns
    )
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


def _require_research_required_columns_available(
    dataset_spec: ResearchDatasetSpec,
    processed_specs: dict[str, ProcessedDataSpec],
    *,
    provided_columns: tuple[str, ...] = (),
) -> None:
    available = {
        column
        for processed_spec in processed_specs.values()
        for column in processed_spec.required_columns
    }
    missing = sorted(
        set(_required_input_columns(dataset_spec))
        .difference(available)
        .difference(provided_columns)
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


def _unfiltered_hft_scope(spec: ResearchDatasetSpec) -> bool:
    return (
        spec.market_id == "a_share"
        and spec.base_input_data_id == "hft_panel"
        and spec.frequency == "tick"
        and spec.eligibility_recipe is None
    )


def _generated_semantic_columns(dataset_spec: ResearchDatasetSpec) -> tuple[str, ...]:
    if dataset_spec.market_id != MARKET_ID:
        return ()
    needed = set(_ordered_output_columns(dataset_spec))
    if dataset_spec.feature_transforms or "model_target" in dataset_spec.label_columns:
        needed.add("eligible")
    generated = []
    if "eligible" in needed and (
        dataset_spec.eligibility_recipe is not None
        or _unfiltered_hft_scope(dataset_spec)
    ):
        generated.append("eligible")
    if "model_target" in dataset_spec.label_columns:
        generated.extend(("model_target", "model_target_available_at"))
    return tuple(generated)


def _add_a_share_semantic_columns(
    frame: pd.DataFrame,
    dataset_spec: ResearchDatasetSpec,
    *,
    risk_sessions: dict | None = None,
    risk_facts: pd.DataFrame | None = None,
    risk_cache: dict | None = None,
    risk_available_at: str | None = None,
) -> pd.DataFrame:
    generated = _generated_semantic_columns(dataset_spec)
    result = frame.copy() if generated else frame
    if "eligible" in generated and _unfiltered_hft_scope(dataset_spec):
        result["eligible"] = True
    elif "eligible" in generated:
        result["eligible"] = a_share_eligibility(
            result,
            dataset_spec.eligibility_recipe,
            minimum_listing_sessions=dataset_spec.minimum_listing_sessions,
            exclude_st=dataset_spec.exclude_st,
            exclude_pt=dataset_spec.exclude_pt,
        )
    if risk_sessions is not None and (
        (
            "model_target" in generated
            and dataset_spec.target_transform == "risk_residual"
        )
        or any(
            transform.risk_model is not None and "processed" in transform.output_groups
            for transform in dataset_spec.feature_transforms
        )
    ):
        model = dataset_spec.risk_model
        if risk_facts is not None:
            risk_sessions.update(
                prepare_asof_risk_sessions(
                    result,
                    risk_facts,
                    model=model,
                    available_at_column=risk_available_at,
                    cache=risk_cache,
                )
            )
        else:
            state = FactorRiskState(
                style_columns=model.style_columns, config=model.config
            )
            for timestamp, cross in result.groupby("timestamp", sort=False):
                risk_sessions[pd.Timestamp(timestamp)] = state.prepare_exposure_session(
                    cross
                )
    if "model_target" in generated:
        if dataset_spec.target_label_column not in result.columns:
            raise ValueError("a_share_model_target_label_missing")
        maturity = dataset_spec.label_availability_columns.get(
            dataset_spec.target_label_column
        )
        if maturity is None:
            raise ValueError("model_target_label_availability_required")
        result[["model_target", "model_target_available_at"]] = a_share_model_target(
            result,
            transform=dataset_spec.target_transform,
            clip=dataset_spec.target_clip,
            label_column=dataset_spec.target_label_column,
            availability_column=maturity,
            risk_model=dataset_spec.risk_model,
            risk_sessions=risk_sessions,
        )
    return result


def assemble(config: Mapping) -> Path:
    """Assemble selected columns from actual input declarations and optional recipes."""
    from quant_research_platform.contracts.features import (
        FeatureTransformSpec,
        inherited_feature_transforms,
    )
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
        "target_label_column",
        "target_clip",
        "risk_model",
        "risk_exposure_input",
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

        if config["dataset_id"] == "a_share_hft":
            from quant_research_platform.research.protocols.a_share_hft import (
                research_dataset_catalog,
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
    risk_exposure_input = config.get("risk_exposure_input")
    if risk_exposure_input is not None and (
        not isinstance(risk_exposure_input, Mapping)
        or set(risk_exposure_input) != {"data_id", "available_at_column"}
        or any(
            type(value) is not str or not value
            for value in risk_exposure_input.values()
        )
        or risk_exposure_input["data_id"] not in inputs
        or risk_exposure_input["data_id"] == base_input
        or risk_exposure_input["available_at_column"]
        in {"timestamp", "asset_id", "market_id", "listing_date"}
    ):
        raise ValueError("assemble_risk_exposure_input_invalid")
    risk_input_id = risk_exposure_input["data_id"] if risk_exposure_input else None
    sessions = {
        data_id: open_dataset_session(
            **_dataset_input_options(
                value, time_window=None if data_id == risk_input_id else window
            )
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
    if risk_input_id is not None and actual_specs[risk_input_id].frequency != "1d":
        raise ValueError("assemble_risk_exposure_frequency_invalid")
    selected_roles = {
        column: role
        for data_id, item in actual_specs.items()
        if data_id != risk_input_id
        for column, role in item.column_roles.items()
        if column in sessions[data_id].logical_columns
    }
    selected_factors = {
        column
        for data_id, item in actual_specs.items()
        if data_id != risk_input_id
        for column, role in item.column_roles.items()
        if role is ColumnRole.FACTOR and column in sessions[data_id].logical_columns
    }
    selected_labels = {
        column
        for data_id, item in actual_specs.items()
        if data_id != risk_input_id
        for column, role in item.column_roles.items()
        if role is ColumnRole.LABEL and column in sessions[data_id].logical_columns
    }
    selected_features = {
        column for column, role in selected_roles.items() if role is ColumnRole.FEATURE
    }
    reused_features = config.get(
        "feature_columns",
        [
            column
            for column in (defaults.feature_columns if defaults else selected_roles)
            if column in selected_features
        ],
    )
    factors = config.get(
        "factor_columns",
        [
            column
            for column in (defaults.factor_columns if defaults else selected_roles)
            if column in selected_factors
        ],
    )
    target_label = config.get("target_label_column", "forward_return_20d")
    if type(target_label) is not str or not target_label:
        raise ValueError("target_label_column_required")
    labels = config.get(
        "label_columns",
        [
            column
            for column in (defaults.label_columns if defaults else selected_roles)
            if column in selected_labels
            or (column == "model_target" and target_label in selected_labels)
        ],
    )
    filters = config.get(
        "filter_columns", list(defaults.filter_columns) if defaults else []
    )
    context = config.get("context_columns", [])
    if "signal_session" in selected_roles:
        context = list(dict.fromkeys((*context, *DECISION_CONTEXT_COLUMNS)))
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
    # Each selected feature has exactly one producer.
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
        feature_producers[column] = name
    feature_lineage = {}
    for name, session in sessions.items():
        columns = {
            column for column, producer in feature_producers.items() if producer == name
        }
        if not columns:
            continue
        feature_lineage.update(
            inherited_feature_transforms(session.spec, session.manifest, columns)
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
        "model_target" in labels and target_label not in selected_labels
    ):
        raise ValueError("assemble_label_not_selected")
    generated_context = (
        {"model_target_available_at"} if "model_target" in labels else set()
    )
    if set(context) - set(selected_roles) - generated_context:
        raise ValueError("assemble_context_not_selected")
    if {"target_transform", "target_clip", "target_label_column"}.intersection(
        config
    ) and (config["market_id"] != "a_share" or "model_target" not in labels):
        raise ValueError("assemble_target_transform_requires_model_target")
    business_columns = set((*factors, *labels, *filters, *context))
    recipe = config.get(
        "feature_recipe",
        "a_share_canonical"
        if config["market_id"] == "a_share"
        and config["dataset_id"] != "a_share_hft"
        and factors
        else None,
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
    risk_model = (
        FactorRiskModel.from_mapping(config["risk_model"])
        if "risk_model" in config
        else None
    )
    options = config.get("feature_transform", {})
    if not isinstance(options, Mapping) or set(options) - {
        "clip",
        "industry_neutralize",
        "size_neutralize",
        "size_column",
        "center",
        "scale",
        "risk_neutralize",
    }:
        raise ValueError("assemble_feature_transform_invalid")
    if config["market_id"] != "a_share" and (recipe is not None or options):
        raise ValueError("assemble_feature_recipe_not_supported")
    if recipe is None and options:
        raise ValueError("assemble_feature_transform_requires_recipe")
    options = dict(options)
    risk_neutralize = options.pop("risk_neutralize", False)
    if type(risk_neutralize) is not bool:
        raise ValueError("assemble_feature_transform_invalid")
    if risk_neutralize:
        risk_model = risk_model or FactorRiskModel()
        options["risk_model"] = risk_model
        options.setdefault("industry_neutralize", False)
        options.setdefault("size_neutralize", False)
    from quant_research_platform.research.factors.macro_liquidity.definitions import (
        PRESERVE_AMPLITUDE_FACTORS as MACRO_AMPLITUDE_FACTORS,
    )
    from quant_research_platform.research.factors.market_structure import (
        PRESERVE_AMPLITUDE_FACTORS as STRUCTURE_AMPLITUDE_FACTORS,
    )

    preserve = (MACRO_AMPLITUDE_FACTORS | STRUCTURE_AMPLITUDE_FACTORS).intersection(
        factors
    )
    if preserve and "raw" not in groups:
        raise ValueError("amplitude_factors_require_raw_group")
    transform_factors = tuple(factor for factor in factors if factor not in preserve)
    transforms = (
        (
            FeatureTransformSpec(
                recipe, transform_factors, output_groups=processed_groups, **options
            ),
        )
        if recipe is not None and transform_factors
        else ()
    )
    if transforms and "feature_recipe" not in config and not options:
        transform = transforms[0]
        pending_sources = tuple(
            factor
            for factor in transform.source_columns
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
        *((target_label,) if "model_target" in labels else ()),
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
        if target_label not in availability:
            raise ValueError("model_target_label_availability_required")
        labels.append(target_label)
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
    if (
        "eligible" in filters
        and config["market_id"] == "a_share"
        and base_input == "hft_panel"
        and source.frequency == "tick"
        and config.get("eligibility_recipe") is None
    ):
        output_roles["eligible"] = ColumnRole.UNIVERSE
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
        target_label_column=target_label,
        target_clip=config.get("target_clip", "none"),
        risk_model=risk_model,
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
    risk_columns = (
        risk_exposure_columns(spec.risk_model)
        if risk_input_id is not None and spec.risk_model is not None
        else ()
    )
    if risk_input_id is not None and spec.risk_model is None:
        raise ValueError("assemble_risk_exposure_model_required")
    if risk_input_id is not None:
        required_inputs.add("listing_date")
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
            if data_id != risk_input_id and column in session.logical_columns
        ]
        if not candidates and column not in business_columns:
            candidates = [
                data_id
                for data_id, item in actual_specs.items()
                if data_id != risk_input_id and column in item.required_columns
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
    _validate_research_input_specs(
        spec,
        consumed_specs,
        provided_columns=risk_columns,
        risk_input_data_id=risk_input_id,
    )
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
                source_unit = consumed_specs[producers[factor]].column_units.get(factor)
                if transform.scale:
                    units["feature_" + suffix] = "dimensionless"
                elif source_unit is not None:
                    units["feature_" + suffix] = source_unit
    if "model_target" in labels:
        if spec.target_transform in {"raw", "industry_size_residual", "risk_residual"}:
            source_unit = consumed_specs[
                producers[spec.target_label_column]
            ].column_units.get(spec.target_label_column)
            if source_unit is not None:
                units["model_target"] = source_unit
        else:
            units["model_target"] = "dimensionless"
    spec = replace(spec, column_units=units)
    existing = ()
    projected = {}
    for data_id in _ordered_input_data_ids(spec):
        if data_id == risk_input_id:
            needed = {
                *spec.primary_key,
                "listing_date",
                risk_exposure_input["available_at_column"],
                *risk_columns,
            }
            columns = tuple(
                column
                for column in actual_specs[data_id].required_columns
                if column in needed
            )
        else:
            columns = _processed_input_columns(
                spec,
                consumed_specs[data_id],
                existing_columns=existing,
                extra_columns=("listing_date",) if risk_input_id is not None else (),
            )
        if data_id not in {base_input, risk_input_id} and not set(columns).difference(
            spec.primary_key
        ):
            continue
        projected[data_id] = sessions[data_id].projected_view(columns)
        if data_id != risk_input_id:
            existing = tuple(dict.fromkeys((*existing, *columns)))
    return _assemble_research_dataset_from_sessions(
        output_root=Path(config["output_root"]),
        spec=spec,
        time_window=window,
        input_sessions=projected,
        risk_exposure_input=risk_exposure_input,
        metadata={
            "inputs": inputs,
            "producer_diagnostics": {
                "reused_feature_transforms": feature_lineage,
                **(
                    {
                        "eligibility": {
                            "mode": "unfiltered",
                            "meaning": "no_stock_pool_filter",
                        }
                    }
                    if _unfiltered_hft_scope(spec)
                    else {}
                ),
            },
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
                "target_label_column": spec.target_label_column,
                "target_clip": spec.target_clip,
                **(
                    {"risk_exposure_input": dict(risk_exposure_input)}
                    if risk_exposure_input
                    else {}
                ),
                **(
                    {"risk_model": spec.risk_model.to_dict()}
                    if spec.risk_model is not None
                    else {}
                ),
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
                        )
                    }
                    | (
                        {"risk_neutralize": True}
                        if transforms[0].risk_model is not None
                        else {}
                    )
                    if transforms
                    else None
                ),
                "sparse_input_data_ids": sparse_inputs,
                "time_window": dict(bounds),
            },
        },
    )
