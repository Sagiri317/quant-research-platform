from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
    default_prediction_column,
)
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.research.evaluations.engine import evaluate_factor_frame
from quant_research_platform.research.models import fit_linear_baseline_result
from quant_research_platform.research.predictions import (
    build_prediction_bundle,
    run_prediction_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout


def _filter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * 6,
            "asset_id": [f"asset_{idx}" for idx in range(6)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 6,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": [0.01, 0.02, 0.03, 0.04, 0.05, np.nan],
            "eligible": [True, True, False, True, True, True],
            "can_enter": [True, True, True, False, True, True],
            "can_exit": [True, True, True, True, False, True],
        }
    )


def _dataset_catalog() -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            data_id="demo_filter",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
            base_input_data_id="demo_annotation",
            feature_columns=("factor",),
            label_columns=("label",),
            filter_columns=("eligible", "can_enter", "can_exit"),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "factor": ColumnRole.FEATURE,
                "label": ColumnRole.LABEL,
                "eligible": ColumnRole.MASK,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def _processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_factor",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "factor": ColumnRole.FACTOR,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_label",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "label": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_annotation",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def test_evaluation_filter_modes_and_metadata_use_hand_calculated_values() -> None:
    frame = _filter_frame()
    eligible_spec = EvaluationSpec(
        run_id="eval_eligible",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        feature_columns=("factor",),
        label_columns=("label",),
    )
    tradable_spec = EvaluationSpec(
        run_id="eval_tradable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        feature_columns=("factor",),
        label_columns=("label",),
        filter_mode="tradable",
    )
    unfiltered_spec = EvaluationSpec(
        run_id="eval_unfiltered",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        feature_columns=("factor",),
        label_columns=("label",),
        filter_mode="unfiltered",
    )

    eligible = evaluate_factor_frame(frame, eligible_spec, n_groups=2)
    tradable = evaluate_factor_frame(frame, tradable_spec, n_groups=2)
    unfiltered = evaluate_factor_frame(frame, unfiltered_spec, n_groups=2)

    assert eligible.metadata == {
        "filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 6,
        "rows_after_filter_mode": 5,
        "rows_after_feature_label_dropna": 4,
    }
    assert tradable.metadata["filter_columns_used"] == [
        "eligible",
        "can_enter",
        "can_exit",
    ]
    assert tradable.metadata["rows_after_filter_mode"] == 3
    assert tradable.metadata["rows_after_feature_label_dropna"] == 2
    assert unfiltered.metadata["filter_columns_used"] == []
    assert unfiltered.metadata["rows_after_filter_mode"] == 6
    assert unfiltered.metadata["rows_after_feature_label_dropna"] == 5

    assert eligible.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert eligible.metrics.loc[0, "rankic"] == pytest.approx(1.0)
    assert eligible.grouped_returns["mean_return"].tolist() == pytest.approx(
        [0.015, 0.045]
    )
    assert tradable.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert tradable.grouped_returns["mean_return"].tolist() == pytest.approx(
        [0.01, 0.02]
    )
    assert unfiltered.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert unfiltered.grouped_returns["mean_return"].tolist() == pytest.approx(
        [0.02, 0.045]
    )


@pytest.mark.parametrize(
    ("spec", "frame", "match"),
    [
        (
            EvaluationSpec(
                run_id="missing_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                feature_columns=("factor",),
                label_columns=("label",),
                filter_mode="tradable",
            ),
            _filter_frame().drop(columns=["can_exit"]),
            "missing_filter_columns: can_exit",
        ),
        (
            EvaluationSpec(
                run_id="null_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                feature_columns=("factor",),
                label_columns=("label",),
            ),
            _filter_frame().assign(eligible=[True, None, True, True, True, True]),
            "null_filter_column_values: eligible",
        ),
        (
            EvaluationSpec(
                run_id="bad_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                feature_columns=("factor",),
                label_columns=("label",),
            ),
            _filter_frame().assign(eligible=["yes"] * 6),
            "invalid_filter_column_dtype: eligible",
        ),
    ],
)
def test_evaluation_filter_columns_fail_closed(
    spec: EvaluationSpec,
    frame: pd.DataFrame,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_factor_frame(frame, spec)


def test_filter_columns_cannot_be_features() -> None:
    with pytest.raises(ValueError, match="filter_column_in_feature_columns: eligible"):
        EvaluationSpec(
            run_id="bad_eval_features",
            market_id=MarketId("a_share"),
            research_dataset_id="demo_filter",
            feature_columns=("eligible",),
            label_columns=("label",),
        )
    with pytest.raises(ValueError, match="filter_column_in_feature_columns: can_exit"):
        ModelSpec(
            model_id="bad_model_features",
            model_family="linear",
            feature_columns=("can_exit",),
            label_column="label",
        )


def test_model_sample_filter_modes_and_metadata_ignore_tradability_masks() -> None:
    frame = _filter_frame()
    eligible_spec = ModelSpec(
        model_id="linear_eligible",
        model_family="linear",
        feature_columns=("factor",),
        label_column="label",
    )
    unfiltered_spec = ModelSpec(
        model_id="linear_unfiltered",
        model_family="linear",
        feature_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
    )

    eligible = fit_linear_baseline_result(frame, eligible_spec)
    unfiltered = fit_linear_baseline_result(frame, unfiltered_spec)

    assert eligible.metadata == {
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 6,
        "rows_after_filter_mode": 5,
        "rows_after_feature_label_dropna": 4,
    }
    assert unfiltered.metadata["rows_after_filter_mode"] == 6
    assert unfiltered.metadata["rows_after_feature_label_dropna"] == 5
    assert eligible.coefficients.tolist() == pytest.approx([0.0, 0.01])
    assert unfiltered.coefficients.tolist() == pytest.approx([0.0, 0.01])

    all_tradability_blocked = frame.assign(can_enter=False, can_exit=False)
    still_trains = fit_linear_baseline_result(all_tradability_blocked, eligible_spec)
    assert still_trains.metadata["rows_after_feature_label_dropna"] == 4


def test_model_rejects_tradable_sample_filter_mode() -> None:
    with pytest.raises(ValueError, match="unsupported_model_sample_filter_mode"):
        ModelSpec(
            model_id="linear_tradable",
            model_family="linear",
            feature_columns=("factor",),
            label_column="label",
            sample_filter_mode="tradable",
        )


def test_prediction_sample_filter_modes_metadata_and_label_free_generation(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        ],
        ignore_index=True,
    )
    spec = PredictionSpec(
        prediction_id="pred_eligible",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        prediction_column=default_prediction_column(),
        primary_key=platform_entity_time_key(),
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )
    prediction = build_prediction_bundle(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        feature_columns=("factor",),
    )

    assert prediction.attrs["metadata"] == {
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 12,
        "rows_after_filter_mode": 10,
        "rows_after_feature_dropna": 10,
    }
    assert len(prediction) == 5
    assert set(prediction.columns) == {
        "market_id",
        "asset_id",
        "timestamp",
        default_prediction_column(),
    }

    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_family="linear",
            feature_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(spec)

    run_root = run_prediction_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_eligible",
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_metadata"]["sample_filter_mode"] == "eligible"
    assert manifest["model_metadata"]["rows_after_feature_label_dropna"] == 4
    assert manifest["prediction_metadata"]["sample_filter_mode"] == "eligible"
    assert manifest["prediction_metadata"]["rows_after_feature_dropna"] == 10


def _write_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    frame: pd.DataFrame,
) -> None:
    columns_by_id = {
        "demo_factor": ["market_id", "asset_id", "timestamp", "factor"],
        "demo_label": ["market_id", "asset_id", "timestamp", "label"],
        "demo_annotation": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
        ],
    }
    for data_id, columns in columns_by_id.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame.loc[:, columns],
            spec=spec,
        )


def test_prediction_unfiltered_mode_and_tradable_rejection() -> None:
    spec = PredictionSpec(
        prediction_id="pred_unfiltered",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        prediction_column=default_prediction_column(),
        primary_key=platform_entity_time_key(),
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )
    prediction = build_prediction_bundle(
        _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        feature_columns=("factor",),
    )

    assert prediction.attrs["metadata"]["rows_after_filter_mode"] == 6
    assert prediction.attrs["metadata"]["rows_after_feature_dropna"] == 6

    with pytest.raises(ValueError, match="unsupported_prediction_sample_filter_mode"):
        PredictionSpec(
            prediction_id="pred_tradable",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            prediction_column=default_prediction_column(),
            primary_key=platform_entity_time_key(),
            train_end="2024-01-02",
            test_start="2024-01-03",
            sample_filter_mode="tradable",
        )


def test_prediction_bundle_rejects_filter_columns_as_direct_features() -> None:
    spec = PredictionSpec(
        prediction_id="pred_bad_feature",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        prediction_column=default_prediction_column(),
        primary_key=platform_entity_time_key(),
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )

    with pytest.raises(ValueError, match="filter_column_in_feature_columns: eligible"):
        build_prediction_bundle(
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            spec,
            coefficients=np.asarray([0.0, 0.01]),
            feature_columns=("eligible",),
        )
