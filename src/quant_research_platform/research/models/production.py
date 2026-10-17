"""Explicit model fitting and final refitting; never discover or build upstream work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    ModelSpec,
    _fit_windows,
    _model_spec,
    _resolve_candidate_parameters,
    _target_definition,
    model_spec_payload,
    validate_model_facts,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    _window_payload,
    time_value_isoformat,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.estimators import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
    _label_calibration,
    rank_model_parameters,
)
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)


def fit_model(
    frames,
    spec: ModelSpec,
    *,
    fit_windows,
    fit_cutoff,
    candidate_parameters=None,
    label_availability_column=None,
    declared_columns=(),
    trading_calendar=None,
):
    """Fit one candidate on its legal observed domain."""
    cutoff = strict_timestamp_series(
        pd.Series([fit_cutoff]), error="invalid_fit_cutoff"
    ).iloc[0]
    windows = _fit_windows(fit_windows, cutoff)

    def scoped():
        with closing_iterator(iter(frames)) as source:
            for frame in source:
                timestamps = strict_timestamp_series(
                    frame["timestamp"], error="invalid_training_timestamp"
                )
                mask = pd.Series(False, index=frame.index)
                for window in windows:
                    from quant_research_platform.contracts.time import (
                        time_window_timestamp_interval,
                    )

                    lower, upper, closed = time_window_timestamp_interval(window)
                    mask |= timestamps.ge(pd.Timestamp(lower)) & (
                        timestamps.le(pd.Timestamp(upper))
                        if closed
                        else timestamps.lt(pd.Timestamp(upper))
                    )
                yield frame.loc[mask]

    prepared = fitting._collect_training_frame(
        scoped(),
        spec,
        fitting_cutoff=cutoff,
        availability_column=label_availability_column,
        declared_columns=declared_columns,
        trading_calendar=trading_calendar,
    )
    if prepared.training.empty:
        raise ValueError("model_training_sample_empty")
    if candidate_parameters is not None:
        if spec.hyperparameters:
            raise ValueError("candidate_parameters_conflict_with_hyperparameters")
        spec = replace(
            spec,
            hyperparameters=_resolve_candidate_parameters(
                spec.model_algorithm, candidate_parameters, len(prepared.training)
            ),
        )
    result = fitting._fit_model_result(
        prepared, spec, estimator=build_estimator_from_model_spec(spec)
    )
    if isinstance(
        result.prediction_model, (RankEnsemblePredictor, ICWeightedRankEnsemble)
    ):
        calibration = _label_calibration(
            fitting.prediction_callable(spec.input_columns, result.prediction_model)(
                prepared.training
            ),
            prepared.training[spec.target_column].to_numpy(dtype="float64"),
            prepared.preprocessing[0],
        )
        # Rank stays rank. Economic consumers use this training-only calibration.
        result.prediction_model = replace(
            result.prediction_model, label_calibration=calibration
        )
        result.metadata["rank_parameters"] = rank_model_parameters(
            result.prediction_model
        )
    return result


def fit(config: Mapping) -> Path:
    """Fit and save one model with an append-only trial beside its output."""
    config = _strict_config(
        config,
        {
            "inputs",
            "output",
            "model",
            "fit_windows",
            "fit_cutoff",
            "market_id",
        },
        {
            "label_unit",
            "candidate_parameters",
            "selection",
            "asset_ids",
            "input_spec",
            "family_id",
            "trading_calendar",
        },
    )
    output = Path(config["output"])
    reject_if_exists(output)
    spec = _model_spec(config["model"])
    source = config["inputs"]
    if config.get("input_spec") is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": config["input_spec"]}
    options = _dataset_input_options(source)
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    declared = session.spec
    validate_model_input_roles(
        declared,
        spec,
        market_id=config["market_id"],
        include_target=True,
        selected_columns=selected_columns,
    )
    availability = declared.get("label_availability_columns", {}).get(
        spec.target_column
    )
    unit = declared.get("column_units", {}).get(spec.target_column)
    config.setdefault("label_unit", unit)
    if type(config["label_unit"]) is not str or not config["label_unit"]:
        raise ValueError("label_unit_required")
    if unit is not None and unit != config["label_unit"]:
        raise ValueError("model_label_unit_mismatch")

    from quant_research_platform.contracts.ids import validate_market_id

    market = validate_market_id(config["market_id"])
    if market == "a_share":
        from quant_research_platform.research.protocols.a_share import (
            validate_model_selection,
        )

        validate_model_selection([spec])
    cutoff = strict_timestamp_series(
        pd.Series([config["fit_cutoff"]]), error="invalid_fit_cutoff"
    ).iloc[0]
    windows = _fit_windows(config["fit_windows"], cutoff)
    selection = None
    if "selection" in config:
        from quant_research_platform.research.models.trial_selection import (
            read_selection,
        )

        selection = read_selection(Path(config["selection"]))
        if (
            spec.model_id
            not in {item["selected_config_id"] for item in selection["families"]}
            or pd.Timestamp(selection["cutoff"]) > cutoff
        ):
            raise ValueError("refit_selection_mismatch")
        selected = next(
            item
            for item in selection["candidates"]
            if item["model_id"] == spec.model_id
        )
        family = selected["family_id"]
        if (
            selection["refit_windows"][family]
            != [_window_payload(window) for window in windows]
            or pd.Timestamp(selection["refit_cutoff"]) != cutoff
        ):
            raise ValueError("refit_selection_scope_mismatch")
    family_id = config.get("family_id", spec.model_id)
    if type(family_id) is not str or not family_id or family_id == "champion":
        raise ValueError("model_family_id_invalid")
    facts = {
        "trading_calendar": config.get("trading_calendar"),
        "family_id": family_id,
        "asset_ids": config.get("asset_ids"),
        "market_id": str(market),
        "model": model_spec_payload(spec),
        "inputs": config["inputs"],
        "fit_windows": [_window_payload(w) for w in windows],
        "fit_cutoff": time_value_isoformat(cutoff, preserve_datetime=True),
        "candidate_parameters": config.get(
            "candidate_parameters", dict(spec.hyperparameters)
        ),
        "label_unit": config["label_unit"],
        "target_definition": _target_definition(
            declared, spec.target_column, session.manifest
        ),
        "input_units": {
            column: declared["column_units"][column]
            for column in spec.input_columns
            if column in declared.get("column_units", {})
        },
        "selection": config.get("selection"),
        "selection_cutoff": None if selection is None else selection["cutoff"],
        "input_spec": config.get("input_spec"),
    }
    validate_model_facts(facts, expected=selected if selection is not None else None)
    recorder = TrialRecorder(output.with_name(output.name + ".trial"), facts)
    recorder.begin()
    try:
        columns = fitting.model_training_read_columns(
            ("timestamp", "asset_id"),
            spec.input_columns,
            label_column=spec.target_column,
            sample_filter_mode=spec.sample_filter_mode,
            availability_column=availability,
            declared_columns=declared["column_roles"],
        )
        if spec.training_weighting.half_life_periods is not None:
            columns = (*columns, "market_type")
        assets = config.get("asset_ids")
        if assets is not None and (
            type(assets) is not list
            or not assets
            or len(assets) != len(set(assets))
            or any(type(item) is not str or not item for item in assets)
        ):
            raise ValueError("training_asset_ids_invalid")

        def checked_frames():
            import pyarrow as pa

            from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

            # The reader owns exact dataset keys. A wider declared key does not
            # establish uniqueness on the model's narrower time/asset axis.
            keys = (
                None
                if set(declared["primary_key"]) == {"timestamp", "asset_id"}
                else ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
            )
            for window in windows:
                frames = iter_dataset_chunks(
                    session.filtered_view(window).projected_view(columns)
                )
                with closing_iterator(iter(frames)) as source:
                    for frame in source:
                        if assets is not None:
                            frame = frame.loc[frame.asset_id.isin(assets)]
                        if keys is not None and (
                            frame.loc[:, ["timestamp", "asset_id"]]
                            .isna()
                            .any(axis=None)
                            or keys.add_table(
                                pa.Table.from_pandas(
                                    frame.loc[:, ["timestamp", "asset_id"]],
                                    preserve_index=False,
                                )
                            )
                        ):
                            raise ValueError("model_training_keys_invalid")
                        yield frame

        calendar = None
        if spec.training_weighting.half_life_periods is not None:
            if "trading_calendar" not in config:
                raise ValueError("training_decay_calendar_required")
            from quant_research_platform.storage.datasets.reading import (
                materialize_dataset_session,
            )

            calendar = materialize_dataset_session(
                open_dataset_session(
                    **_dataset_input_options(config["trading_calendar"])
                )
            )
        result = fit_model(
            checked_frames(),
            spec,
            fit_windows=windows,
            fit_cutoff=cutoff,
            candidate_parameters=config.get("candidate_parameters"),
            label_availability_column=availability,
            declared_columns=declared["column_roles"],
            trading_calendar=calendar,
        )
        from quant_research_platform.research.models.persisted import save_model

        terminal = recorder.finish(result={"model_metadata": result.metadata})
        save_model(
            output, result, facts={**facts, "trial": str(recorder.root)}, trial=terminal
        )
    except BaseException as exc:
        if not (recorder.root / "terminal.json").exists():
            try:
                recorder.finish(error=exc)
            except BaseException as recording_error:
                exc.add_note(
                    "trial failure record also failed: "
                    f"{type(recording_error).__name__}"
                )
        raise
    return output
