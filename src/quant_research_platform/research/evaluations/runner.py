"""Evaluate explicit signals against labels observable at the declared cutoff."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.evaluation import (
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
)
from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.evaluations import (
    EVALUATION_FORECAST_AGE_BOUNDS,
    EvaluationSpec,
)
from quant_research_platform.contracts.experiments import (
    label_maturity_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.features.transforms import (
    feature_observation_columns,
)
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def _input_roles(declared, spec, selected_columns):
    validate_input_market(declared, spec.market_id)
    consumed = {
        *spec.label_columns,
        *(
            signal.source_column
            for signal in spec.signals
            if signal.source_kind != "prediction"
        ),
    }
    if selected_columns is not None and consumed - set(selected_columns):
        raise ValueError("evaluation_input_projection_missing")
    roles = declared["column_roles"]
    _validate_asset_time_key(declared["primary_key"], roles)
    for signal in spec.signals:
        if signal.source_kind != "prediction" and (
            roles.get(signal.source_column) != signal.source_kind
        ):
            raise ValueError("evaluation_signal_role_mismatch")
    if any(roles.get(label) != "label" for label in spec.label_columns):
        raise ValueError("evaluation_label_role_mismatch")


def evaluate(config):
    """Publish one new numeric evaluation; no training or implicit input resolution."""
    config = _strict_config(
        config,
        {"inputs", "output", "evaluation", "window", "cutoff"},
        {
            "input_spec",
            "predictions",
            "sample_role",
            "sample_scope",
            "comparison_inputs",
            "strata",
            "response_diagnostics",
        },
    )
    reject_if_exists(Path(config["output"]))
    config.setdefault("sample_scope", "own")
    if config["sample_scope"] not in {"own", "common"} or (
        config["sample_scope"] == "own" and "comparison_inputs" in config
    ):
        raise ValueError("evaluation_sample_scope_invalid")
    config.setdefault("sample_role", "evaluation")
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("evaluation_sample_role_invalid")
    window = TimeWindow(**config["window"])
    if window.end is None or any(
        pd.isna(pd.Timestamp(value)) for value in (window.start, window.end)
    ):
        raise ValueError("evaluation_window_invalid")
    cutoff = strict_timestamp_series(
        pd.Series([config["cutoff"]]), error="evaluation_cutoff_invalid"
    ).iloc[0]
    source = config["inputs"]
    if config.get("input_spec") is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": config["input_spec"]}
    options = _dataset_input_options(source, time_window=window)
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    declared = session.spec
    values = dict(config["evaluation"])
    values["signals"] = tuple(SignalSpec(**item) for item in values["signals"])
    values["label_columns"] = tuple(values["label_columns"])
    supplied_availability = values.get("label_availability_columns", {})
    values["label_availability_columns"] = {
        label: supplied_availability.get(
            label,
            declared.get("label_availability_columns", {}).get(
                label, training_availability_columns(label)[0]
            ),
        )
        for label in values["label_columns"]
    }
    if set(supplied_availability) - set(values["label_columns"]):
        raise ValueError("label_availability_columns_invalid")
    spec = EvaluationSpec(**values)
    for label, column in supplied_availability.items():
        native_column = declared.get("label_availability_columns", {}).get(label)
        if native_column is not None and native_column != column:
            raise ValueError("evaluation_label_availability_mismatch")
    _input_roles(declared, spec, selected_columns)
    response = config.get("response_diagnostics")
    if response is not None:
        from quant_research_platform.contracts.evaluations import (
            validate_response_diagnostics,
        )

        validate_response_diagnostics(response, signals=spec.signals)
        if (
            str(spec.market_id) != "crypto"
            or spec.evaluation_axis != "single_asset_time_series"
            or config["sample_scope"] != "own"
            or config.get("strata")
            or any(signal.transform != "identity" for signal in spec.signals)
            or any(
                declared.get("column_units", {}).get(label) != "decimal_return"
                for label in spec.label_columns
            )
        ):
            raise ValueError("evaluation_response_domain_invalid")
    prediction_sources = {
        signal.source_column
        for signal in spec.signals
        if signal.source_kind == "prediction"
    }
    supplied_predictions = config.get("predictions", {})
    if (
        type(supplied_predictions) is not dict
        or set(supplied_predictions) != prediction_sources
    ):
        raise ValueError("evaluation_prediction_sources_mismatch")
    ordinary = tuple(
        signal.source_column
        for signal in spec.signals
        if signal.source_kind != "prediction"
    )
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *ordinary,
                *feature_observation_columns(
                    tuple(
                        signal.source_column
                        for signal in spec.signals
                        if signal.source_kind == "feature"
                    ),
                    declared["column_roles"],
                ),
                *spec.label_columns,
                *spec.label_availability_columns.values(),
                *required_filter_columns(spec.filter_mode),
            )
        )
    )
    strata, extra_columns = _resolved_strata(
        config,
        spec=spec,
        declared={
            **declared,
            "feature_transforms": [
                *declared.get("feature_transforms", []),
                *session.manifest.get("producer_diagnostics", {})
                .get("reused_feature_transforms", {})
                .values(),
            ],
        },
    )
    supports_daily = (
        str(spec.market_id) == "a_share"
        and spec.evaluation_axis == "cross_sectional"
        and spec.filter_mode == "eligible"
        and not prediction_sources
    )
    if config["sample_scope"] == "common" or strata or supports_daily:
        columns = tuple(
            dict.fromkeys(
                (
                    *columns,
                    *extra_columns,
                    *(
                        ("market_id",)
                        if "market_id" in declared["column_roles"]
                        else ()
                    ),
                )
            )
        )
        return _scoped_evaluate(
            config,
            spec=spec,
            session=session,
            source=source,
            columns=columns,
            cutoff=cutoff,
            window=window,
            strata=strata,
        )
    predictions = {}
    for column, path in supplied_predictions.items():
        frame, metadata = read_predictions(
            Path(path),
            columns=("timestamp", "asset_id", "prediction_score"),
            time_window=window,
        )
        validate_input_market(metadata["model_facts"], spec.market_id)
        # The prediction reader already owns uniqueness of these exact keys.
        frame.set_index(["timestamp", "asset_id"], inplace=True)
        predictions[column] = (frame["prediction_score"], metadata)
        del frame

    def prepared():
        # The reader owns the declared PK; only a wider PK needs this
        # additional, narrower evaluation-key invariant.
        seen = (
            ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
            if set(declared["primary_key"]) != {"timestamp", "asset_id"}
            else None
        )
        # Acquire the input in its consuming generator. A rejected destination or
        # failed initialization cannot leave an unstarted input session open.
        with closing_iterator(
            iter_dataset_chunks(session.projected_view(columns))
        ) as source:
            for frame in source:
                if seen is not None and seen.add_table(
                    pa.Table.from_pandas(
                        frame.loc[:, ["timestamp", "asset_id"]],
                        preserve_index=False,
                    )
                ):
                    raise ValueError("evaluation_input_duplicate_keys")
                frame = frame.copy()
                if predictions:
                    keys = pd.MultiIndex.from_frame(frame[["timestamp", "asset_id"]])
                    for column, (prediction, _metadata) in predictions.items():
                        # Assign positionally: input chunk indexes need not match
                        # the prediction key index or start at zero.
                        frame[column] = prediction.reindex(keys).to_numpy()
                if (
                    predictions
                    and not ordinary
                    and all(
                        metadata.get("asset_ids") is not None
                        for _, metadata in predictions.values()
                    )
                ):
                    # Explicit prediction asset scopes constrain prediction-only
                    # evaluations without requalifying historical model inputs.
                    assets = {
                        asset
                        for _, metadata in predictions.values()
                        for asset in metadata["asset_ids"]
                    }
                    frame = frame.loc[frame.asset_id.isin(assets)]
                for label, available in spec.label_availability_columns.items():
                    mature = label_maturity_mask(
                        frame,
                        fitting_cutoff=cutoff,
                        availability_column=available,
                        target_column=label,
                    )
                    frame.loc[~mature, label] = float("nan")
                yield frame

    with build_evaluation_result_from_frames(
        prepared(),
        spec=spec,
        asset_column="asset_id",
        declared_columns=declared["column_roles"],
        response_diagnostics=response,
        source_columns_by_kind={
            kind: tuple(
                signal.source_column
                for signal in spec.signals
                if signal.source_kind == kind
            )
            for kind in ("factor", "feature", "prediction")
        },
    ) as result:
        return write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=Path(config["output"]),
            metadata={
                **{
                    key: config[key]
                    for key in ("window", "cutoff", "sample_role", "sample_scope")
                },
                "input_facts": _evaluation_source_facts(
                    source,
                    session,
                    columns=tuple(
                        column
                        for column in columns
                        if column in declared["column_roles"]
                    ),
                    signals=ordinary,
                    availability=dict(spec.label_availability_columns),
                ),
                "comparison_inputs": [],
                "strata": {},
                "stratum_domains": {},
                "optional_member_rows": {}
                if response is None
                else {"response_metrics": 0, "response_groups": 0},
                **({"response_diagnostics": response} if response is not None else {}),
                "common_rule": "mask_only_no_retransform",
            },
        )


def _selected_transform_sources(transform, columns):
    selected = set(columns)
    parameters = dict(transform)
    if "output_groups" in parameters:
        parameters["output_groups"] = tuple(parameters["output_groups"])
    for source_column in transform["source_columns"]:
        outputs = FeatureTransformSpec(
            **{**parameters, "source_columns": (source_column,)}
        ).output_columns
        if selected.intersection(outputs):
            yield source_column


def _evaluation_source_facts(source, session, *, columns, signals, availability):
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot

    declared = session.spec
    selector = (
        {"path": str(source)} if not isinstance(source, Mapping) else dict(source)
    )
    selector["path"] = str(Path(selector["path"]).resolve())
    roles = declared["column_roles"]
    units = declared.get("column_units", {})
    reused = session.manifest.get("producer_diagnostics", {}).get(
        "reused_feature_transforms", {}
    )
    transforms = []
    for signal in sorted(signals):
        for transform in [
            *declared.get("feature_transforms", []),
            *([reused[signal]] if signal in reused else []),
        ]:
            for source_column in _selected_transform_sources(transform, (signal,)):
                fact = {**transform, "source_columns": [source_column]}
                if fact not in transforms:
                    transforms.append(fact)
    return qrp_json_mapping_snapshot(
        {
            "selector": selector,
            "columns": {column: roles[column] for column in columns},
            "column_units": {column: units.get(column) for column in columns},
            "observed_columns": {
                column: list(feature_observation_columns((column,), roles))
                for column in signals
            },
            "feature_transforms": transforms,
            "signal_units": {column: units.get(column) for column in signals},
            "label_units": {label: units.get(label) for label in availability},
            "label_availability_columns": availability,
            "label_semantics": {
                label: facts
                for label, facts in session.manifest.get("producer_diagnostics", {})
                .get("label_semantics", {})
                .items()
                if label in availability
            },
            "market_id": declared.get("market_id"),
        }
    )


def _resolved_strata(config, *, spec, declared):
    value = config.get("strata")
    if value is None:
        return {}, ()
    if (
        type(value) is not dict
        or set(value) - {"primary_label", "kinds", "forecast_freshness_days"}
        or not {"primary_label", "kinds"} <= set(value)
    ):
        raise ValueError("evaluation_strata_invalid")
    kinds = value["kinds"]
    if (
        type(kinds) is not list
        or not kinds
        or any(type(kind) is not str for kind in kinds)
        or len(set(kinds)) != len(kinds)
        or set(kinds) - {"industry", "size", "liquidity", "forecast_age"}
        or value["primary_label"] not in spec.label_columns
    ):
        raise ValueError("evaluation_strata_invalid")
    resolved = {
        **value,
        "population": "focal_eligible_primary_label_mature_before_common",
        "age_bounds": [],
    }
    columns = [
        {
            "industry": "industry_level1_code",
            "size": "total_market_cap",
            "liquidity": "amount",
        }[kind]
        for kind in kinds
        if kind != "forecast_age"
    ]
    if "forecast_age" in kinds:
        age = "median_forecast_age_days_fy1"
        freshness = value.get("forecast_freshness_days")
        if type(freshness) is not int or freshness not in {90, 180}:
            raise ValueError("evaluation_forecast_freshness_invalid")
        selected = {signal.source_column for signal in spec.signals}
        actual = age in selected and declared["column_roles"].get(age) == "factor"
        for recipe in declared.get("feature_transforms", []):
            if age in _selected_transform_sources(recipe, selected):
                actual = True
        if not actual or declared.get("column_units", {}).get(age) != "days":
            raise ValueError("evaluation_forecast_age_source_invalid")
        columns.append(age)
        resolved.update(
            age_column=age,
            age_bounds=[
                list(item)
                for item in EVALUATION_FORECAST_AGE_BOUNDS
                if item[2] <= freshness
            ],
        )
    elif "forecast_freshness_days" in value:
        raise ValueError("evaluation_forecast_freshness_not_applicable")
    if set(columns) - set(declared["column_roles"]):
        raise ValueError("evaluation_strata_columns_missing")
    return resolved, tuple(columns)


def _scoped_evaluation_frames(sources):
    from contextlib import ExitStack
    from dataclasses import replace

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.runtime.dataframes import iter_complete_session_frames

    grouped = {}
    for source in sources:
        key = canonical_json_hash(source["facts"]["selector"])
        source["scan_key"] = key
        if key not in grouped:
            grouped[key] = {
                "session": source["session"],
                "columns": [],
                "allow_infinite": [],
            }
        grouped[key]["columns"].extend(source["columns"])
        grouped[key]["allow_infinite"].extend(source.get("classification_columns", ()))

    def scan(entry):
        session = entry["session"].projected_view(
            tuple(dict.fromkeys(entry["columns"]))
        )
        session = replace(
            session,
            _allow_infinite_columns=tuple(dict.fromkeys(entry["allow_infinite"])),
        )
        with closing_iterator(
            iter_complete_session_frames(
                iter_dataset_chunks(session), timestamp_column="timestamp"
            )
        ) as frames:
            for frame in frames:
                # The dataset reader owns its declared PK. The join may use a
                # narrower key, so check assets only within this complete date.
                if frame.asset_id.isna().any() or frame.asset_id.duplicated().any():
                    raise ValueError("evaluation_input_duplicate_keys")
                if "market_id" in frame and not frame.market_id.eq("a_share").all():
                    raise ValueError("evaluation_market_mismatch")
                yield frame.reset_index(drop=True)

    with closing_iterator(ExitStack()) as stack:
        streams = {
            key: stack.enter_context(closing_iterator(scan(entry)))
            for key, entry in grouped.items()
        }
        focal_key = sources[0]["scan_key"]
        heads = {
            key: next(stream, None)
            for key, stream in streams.items()
            if key != focal_key
        }
        for raw in streams[focal_key]:
            if raw.empty:
                continue
            timestamp = raw.timestamp.iloc[0]
            daily = {focal_key: raw}
            for key, head in heads.items():
                while head is not None and (
                    head.empty or head.timestamp.iloc[0] < timestamp
                ):
                    head = next(streams[key], None)
                heads[key] = head
                daily[key] = (
                    head
                    if head is not None and head.timestamp.iloc[0] == timestamp
                    else None
                )
            yield raw, [daily[source["scan_key"]] for source in sources[1:]]


def _scoped_evaluate(config, *, spec, session, source, columns, cutoff, window, strata):
    from quant_research_platform.research.evaluations.engine import (
        build_scoped_evaluation_result,
    )

    if (
        str(spec.market_id) != "a_share"
        or spec.evaluation_axis != "cross_sectional"
        or spec.filter_mode != "eligible"
        or any(
            signal.source_kind not in {"factor", "feature"} for signal in spec.signals
        )
    ):
        raise ValueError("evaluation_scoped_requires_a_share_factors")
    focal = {
        "session": session,
        "declared": session.spec,
        "signal_columns": [signal.source_column for signal in spec.signals],
        "availability": dict(spec.label_availability_columns),
        "columns": columns,
    }
    focal["classification_columns"] = [
        column
        for kind, column in (
            ("size", "total_market_cap"),
            ("liquidity", "amount"),
            ("forecast_age", "median_forecast_age_days_fy1"),
        )
        if kind in strata.get("kinds", [])
    ]
    focal["facts"] = _evaluation_source_facts(
        source,
        session,
        columns=columns,
        signals=focal["signal_columns"],
        availability=focal["availability"],
    )
    sources = [focal]
    peers = config.get("comparison_inputs", [])
    if type(peers) is not list or (config["sample_scope"] == "own" and peers):
        raise ValueError("evaluation_comparison_inputs_invalid")
    for peer in peers:
        if type(peer) is not dict or set(peer) != {"inputs", "signal_columns"}:
            raise ValueError("evaluation_comparison_inputs_invalid")
        signals = peer["signal_columns"]
        if (
            type(signals) is not list
            or not signals
            or any(type(column) is not str for column in signals)
            or len(set(signals)) != len(signals)
        ):
            raise ValueError("evaluation_comparison_signals_invalid")
        options = _dataset_input_options(peer["inputs"], time_window=window)
        selected = options.get("columns")
        options.setdefault("columns", ("timestamp", "asset_id"))
        peer_session = open_dataset_session(**options)
        declared = peer_session.spec
        validate_input_market(declared, spec.market_id)
        _validate_asset_time_key(declared["primary_key"], declared["column_roles"])
        if any(
            declared["column_roles"].get(column) not in {"factor", "feature"}
            for column in signals
        ) or any(
            declared["column_roles"].get(label) != "label"
            for label in spec.label_columns
        ):
            raise ValueError("evaluation_comparison_domain_invalid")
        if selected is not None and set((*signals, *spec.label_columns)) - set(
            selected
        ):
            raise ValueError("evaluation_input_projection_missing")
        availability = {
            label: declared.get("label_availability_columns", {}).get(
                label, training_availability_columns(label)[0]
            )
            for label in spec.label_columns
        }
        peer_columns = tuple(
            dict.fromkeys(
                (
                    "timestamp",
                    "asset_id",
                    "eligible",
                    *signals,
                    *feature_observation_columns(signals, declared["column_roles"]),
                    *spec.label_columns,
                    *availability.values(),
                    *(
                        ("market_id",)
                        if "market_id" in declared["column_roles"]
                        else ()
                    ),
                )
            )
        )
        facts = _evaluation_source_facts(
            peer["inputs"],
            peer_session,
            columns=peer_columns,
            signals=signals,
            availability=availability,
        )
        if facts["label_units"] != focal["facts"]["label_units"]:
            raise ValueError("evaluation_label_units_mismatch")
        if facts["label_semantics"] != focal["facts"]["label_semantics"]:
            raise ValueError("evaluation_label_semantics_mismatch")
        sources.append(
            {
                "session": peer_session,
                "declared": declared,
                "signal_columns": signals,
                "availability": availability,
                "columns": peer_columns,
                "facts": facts,
            }
        )
    science = {
        key: config[key] for key in ("window", "cutoff", "sample_role", "sample_scope")
    }
    science.update(
        input_facts=focal["facts"],
        comparison_inputs=[source["facts"] for source in sources[1:]],
        strata=strata,
        stratum_domains={kind: [] for kind in strata.get("kinds", ())},
        optional_member_rows={"daily_coverage": 0, "daily_strata": 0},
        common_rule="mask_only_no_retransform",
    )
    with build_scoped_evaluation_result(
        _scoped_evaluation_frames(sources),
        spec=spec,
        sources=sources,
        cutoff=cutoff,
        sample_scope=config["sample_scope"],
        strata=strata,
    ) as result:
        return write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=Path(config["output"]),
            metadata=science,
        )
