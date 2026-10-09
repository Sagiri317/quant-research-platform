from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pandas as pd

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.experiments import training_availability_columns
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.models.persisted import (
    read_quarterly_model_artifact,
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.production import _build_a_share_models
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
)
from quant_research_platform.research.models.trial_selection import ModelSampleCoverage
from quant_research_platform.research.protocols import a_share
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_specs import make_demo_research_dataset_spec


def _column_predictor(column: str, offset: float = 0.0) -> LinearScaledPredictor:
    return LinearScaledPredictor(
        (offset, 1.0), ModelMatrixScaler((column,), (0.0,), (0.0,), (1.0,))
    )


def _metadata(
    columns: tuple[str, ...],
    *,
    algorithm: str = "linear_regression",
    model_id: str = "champion",
    rank_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model_algorithm": algorithm,
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2014-01-01",
        "train_end": "2018-12-31",
        "input_factor_columns": list(columns),
        "input_feature_columns": [],
        "input_columns": list(columns),
        "input_count": len(columns),
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": 11,
        "max_train_rows": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
        },
        "training_weighting": {"method": "equal_timestamp", "half_life": None},
        "scaler": None,
        "feature_importance": {column: 1.0 for column in columns},
        "fit_cutoff": "2018-12-31",
    }
    coverage = ModelSampleCoverage(
        {"fixture_training": model_id, "inputs": list(columns)}
    )
    coverage.add(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [result["train_start"], result["train_end"]]
                ),
                "asset_id": ["asset_0", "asset_0"],
            }
        )
    )
    result["fit_coverage_id"] = coverage.semantic_identity
    if rank_parameters is not None:
        result["rank_parameters"] = rank_parameters
    return result


def _passthrough_session(**attributes):
    session = SimpleNamespace(**attributes)

    def filtered_view(_time_window):
        return session

    session.filtered_view = filtered_view
    return session


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "momentum_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)


def shared_model_protocol(specs, *, development_end="2019-06-30", configurations=None):
    protocol = a_share.protocol("pv_all").manifest()
    protocol.update(
        scenario_id="fixture",
        root_seed=11,
        formal_dataset_window={"start": "2014-01-01", "end": development_end},
        development_window={"start": "2019-01-01", "end": development_end},
        holdout_window={"start": "2024-01-01", "end": "2025-12-31"},
        model_chronology={
            "scoring_lookback_months": 12,
            "warmup_window": {"start": "2018-01-01", "end": "2018-12-31"},
        },
        research_configuration={
            "models": configurations
            if configurations is not None
            else [
                {
                    "model_id": spec.model_id,
                    "hyperparameters": dict(spec.hyperparameters),
                }
                for spec in specs
            ]
        },
    )
    return protocol


def shared_model_rows(specs, *, development_end="2019-06-30", assets=8):
    periods = pd.period_range("2018-01-01", development_end, freq="Q")
    dates = [
        pd.Timestamp("2014-01-02"),
        pd.Timestamp("2016-01-04"),
        pd.Timestamp("2017-12-29"),
        *(
            period.start_time + pd.offsets.BDay(day)
            for period in periods
            for day in (1, 2)
        ),
    ]
    inputs = tuple(
        dict.fromkeys(column for spec in specs for column in spec.input_columns)
    )
    label = specs[0].label_column
    return pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": f"asset_{asset}",
                "timestamp": date,
                "eligible": True,
                **{
                    column: float(asset + (index + 1) * 0.01 + number * 0.2)
                    for number, column in enumerate(inputs)
                },
                label: float(asset * 0.03 + index * 0.001),
                training_availability_columns(label)[0]: date + pd.offsets.BDay(1),
            }
            for index, date in enumerate(dates)
            for asset in range(assets)
        ]
    )


def shared_model_case(
    tmp_path,
    *,
    specs=None,
    frame=None,
    dataset=None,
    protocol=None,
    development_end="2019-06-30",
    chunk_size=5,
    requested_end=None,
    verify_source=lambda: None,
    model_results_wrapper=None,
    software_identity="c" * 64,
):
    """Exercise the actual shared candidate, selection, refit and persisted readers."""
    specs = specs or (
        ModelSpec(
            model_id="linear_regression",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
    )
    protocol = protocol or shared_model_protocol(specs, development_end=development_end)
    development_end = protocol["development_window"]["end"]
    requested_end = development_end if requested_end is None else requested_end
    frame = (
        shared_model_rows(specs, development_end=development_end)
        if frame is None
        else frame
    )
    if dataset is None:
        factors = tuple(
            dict.fromkeys(
                column for spec in specs for column in spec.input_factor_columns
            )
        )
        features = tuple(
            dict.fromkeys(
                column for spec in specs for column in spec.input_feature_columns
            )
        )
        dataset = make_demo_research_dataset_spec(
            data_id="fixture",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=factors,
            feature_columns=features,
            label_columns=(specs[0].label_column,),
            filter_columns=("eligible",),
            context_columns=(),
        )
    dataset = replace(dataset, data_id=protocol["scenario_id"])
    layout = StorageLayout(tmp_path).with_target_roots(
        {dataset.data_id: tmp_path / ("d" * 64)}
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity=canonical_json_hash(protocol),
        software_identity=software_identity,
        research_window={
            "start": "2018-01-01",
            "end": requested_end,
            "access": "development",
        },
        upstream_identities={dataset.data_id: "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol,
                "model_specs": [model_spec_payload(spec) for spec in specs],
            }
        },
    )

    def chunks(columns, *, input_columns, start, end):
        selected = frame.loc[frame.timestamp.le(pd.Timestamp(end))]
        if start is not None:
            selected = selected.loc[selected.timestamp.ge(pd.Timestamp(start))]
        for offset in range(0, len(selected), chunk_size):
            yield selected.iloc[offset : offset + chunk_size].loc[:, list(columns)]

    results = partial(
        _build_a_share_models,
        layout=layout,
        dataset_spec=dataset,
        model_catalog=ModelCatalog(specs),
        dataset_id=dataset.data_id,
        model_ids=tuple(spec.model_id for spec in specs),
        time_window=TimeWindow("2018-01-01", requested_end),
        chunks=chunks,
        sessions=pd.DatetimeIndex(sorted(frame.timestamp.unique())),
        chronology=protocol["model_chronology"],
        formal_dataset_window=protocol["formal_dataset_window"],
        development_window=protocol["development_window"],
        root_seed=protocol["root_seed"],
        run_id="run_fixture",
        research_protocol_identity=canonical_json_hash(protocol),
        model_target_id="model-artifact",
        backend_versions={"internal": "test"},
        holdout_window=protocol["holdout_window"],
        research_configuration=protocol["research_configuration"],
    )
    if model_results_wrapper is not None:
        results = model_results_wrapper(results)
    root = write_quarterly_model_artifact(
        tmp_path / "models",
        model_id="model-artifact",
        envelope=envelope,
        model_results=results,
        trial_layout=layout,
        verify_source=verify_source,
    )
    artifact = read_quarterly_model_artifact(
        root, expected_identity=envelope.instance_identity, trial_layout=layout
    )
    return SimpleNamespace(
        root=root,
        artifact=artifact,
        layout=layout,
        specs=specs,
        dataset=dataset,
        protocol=protocol,
        envelope=envelope,
        frame=frame,
    )
