"""A declared family may have no legal rows inside a nonempty prediction window."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.predictions.alignment import (
    _iter_prediction_joined_chunks,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_persisted_model,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.research_configuration import research_configuration
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.prediction_output_helpers import write_prediction_result
from tests.research.workflow_test_support import shared_model_case, shared_model_rows


@pytest.fixture(scope="module")
def empty_family_case(tmp_path_factory):
    config = research_configuration("a_share", "fusion_nf_core")
    factors = ("earnings_yield", "book_to_market")
    config["evaluation"] = {
        "factors": list(factors),
        "labels": ["forward_return_5d"],
        "primary_label": "forward_return_5d",
    }
    template = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    config["models"] = [
        {
            **template,
            "model_id": column,
            "input_factor_columns": [column],
            "input_feature_columns": [],
            "label_column": "forward_return_5d",
        }
        for column in factors
    ]
    config["comparison_model_id"] = factors[0]
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["prediction_selector"] == "champion"
    ]
    bound = bind_research_configuration(research_profile(), config)
    specs = bound.models.list()
    frame = shared_model_rows(specs, development_end="2019-03-31", assets=3)
    sessions = pd.bdate_range("2014-01-01", "2019-04-10")
    prices = pd.DataFrame(
        [
            {
                "asset_id": f"asset_{asset}",
                "timestamp": date,
                "adj_open": 10.0 + index * (asset + 1) / 100,
                "can_enter": True,
            }
            for index, date in enumerate(sessions)
            for asset in range(3)
        ]
    )
    labels = add_forward_return_labels(prices, horizons=(5,), sessions=sessions)
    frame = frame.drop(
        columns=["forward_return_5d", "forward_return_5d_available_at"]
    ).merge(
        labels[
            [
                "asset_id",
                "timestamp",
                "forward_return_5d",
                "forward_return_5d_available_at",
            ]
        ],
        on=["asset_id", "timestamp"],
        validate="one_to_one",
    )
    for date in frame.timestamp.unique():
        assert (
            frame.loc[frame.timestamp.eq(date), "forward_return_5d_available_at"]
            .eq(sessions[sessions.get_loc(date) + 6])
            .all()
        )
    frame.loc[frame.timestamp.eq("2019-01-03"), "earnings_yield"] = np.nan
    empty = frame.loc[frame.timestamp.eq("2019-01-03")].copy()
    empty["timestamp"] = pd.Timestamp("2019-01-04")
    empty[list(factors)] = np.nan
    endpoints = (
        frame.iloc[[0]].loc[frame.iloc[[0]].index.repeat(5)].reset_index(drop=True)
    )
    endpoints["timestamp"] = pd.to_datetime(
        ["2018-03-30", "2018-06-29", "2018-09-28", "2018-12-31", "2019-03-29"]
    )
    endpoints[list(factors)] = np.nan
    endpoints["eligible"] = False
    added = (
        pd.concat((empty, endpoints), ignore_index=True)
        .drop(columns=["forward_return_5d", "forward_return_5d_available_at"])
        .merge(
            labels[
                [
                    "asset_id",
                    "timestamp",
                    "forward_return_5d",
                    "forward_return_5d_available_at",
                ]
            ],
            on=["asset_id", "timestamp"],
            validate="one_to_one",
        )
    )
    frame = pd.concat((frame, added), ignore_index=True).sort_values(
        ["timestamp", "asset_id"]
    )
    frame["timestamp"] = frame.timestamp.dt.as_unit("ns")
    dataset = make_demo_research_dataset_spec(
        data_id="fixture",
        input_data_ids=("processed_demo",),
        base_input_data_id="processed_demo",
        factor_columns=factors,
        label_columns=("forward_return_5d", "forward_return_5d_available_at"),
        filter_columns=("eligible",),
        context_columns=(),
    )
    case = shared_model_case(
        tmp_path_factory.mktemp("empty_family"),
        specs=specs,
        frame=frame,
        dataset=dataset,
        development_end="2019-03-31",
    )
    native_root = case.layout.instance_root(
        case.layout.research_dataset_root("a_share", case.dataset.data_id), "d" * 64
    )
    case.layout = case.layout.with_target_roots({case.dataset.data_id: native_root})
    write_dataset(
        case.layout.target_root(
            case.dataset.data_id,
            case.layout.research_dataset_root("a_share", case.dataset.data_id),
        ),
        frame,
        spec=case.dataset,
    )
    assert len(case.artifact.candidate_models) == 10
    assert len(case.artifact.final_models) == 4
    return case


def _result(case, name, start, end, *, access="development"):
    spec = PredictionSpec(
        prediction_id=name,
        model_id="model-artifact",
        research_dataset_id=case.dataset.data_id,
        train_end=None,
        test_start=start,
        test_end=end,
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    return build_prediction_result_from_persisted_model(
        layout=case.layout,
        dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
        prediction_spec=spec,
        persisted_model=case.artifact,
        selection_group="global",
        access_window=access,
    )


def test_native_development_family_domain_wide_narrow_and_empty(empty_family_case):
    case = empty_family_case
    for name, start, counts in (
        ("wide", "2019-01-02", {"earnings_yield": 3, "book_to_market": 6}),
        ("narrow", "2019-01-03", {"book_to_market": 3}),
    ):
        result = _result(case, name, start, "2019-01-03")
        root = write_prediction_result(case.layout, result)
        saved = pd.read_parquet(root / "prediction.parquet")
        assert saved.groupby("family_id").size().to_dict() == counts
        assert not saved.duplicated(["timestamp", "asset_id", "family_id"]).any()
        assert str(saved.timestamp.dtype) == "datetime64[ns]"
        manifest, lookup = read_prediction_output_bundle(
            root, expected_output_id=name, family_selection="book_to_market"
        )
        assert manifest["metadata"]["prediction_metadata"]["prediction_rows"] == sum(
            counts.values()
        )
        requested = case.frame.loc[case.frame.timestamp.between(start, "2019-01-03")]
        joined = list(
            _iter_prediction_joined_chunks(
                (requested,),
                prediction_lookup=lookup,
                input_columns=("book_to_market",),
                model_algorithm="linear_regression",
                sample_filter_mode="eligible",
                test_start=start,
                test_end="2019-01-03",
            )
        )
        expected = saved.loc[saved.family_id.eq("book_to_market")].sort_values(
            ["timestamp", "asset_id"]
        )
        actual = pd.concat(joined).sort_values(["timestamp", "asset_id"])
        pd.testing.assert_frame_equal(
            actual[["timestamp", "asset_id"]].reset_index(drop=True),
            expected[["timestamp", "asset_id"]].reset_index(drop=True),
        )
        np.testing.assert_array_equal(
            actual[lookup.prediction_column], expected[lookup.prediction_column]
        )
        assert lookup.coverage_counts() == (0, 0)
        _, extra_lookup = read_prediction_output_bundle(
            root, expected_output_id=name, family_selection="book_to_market"
        )
        with pytest.raises(ValueError, match="misaligned_persisted_prediction_keys"):
            list(
                _iter_prediction_joined_chunks(
                    (requested.iloc[:-1],),
                    prediction_lookup=extra_lookup,
                    input_columns=("book_to_market",),
                    model_algorithm="linear_regression",
                    sample_filter_mode="eligible",
                    test_start=start,
                    test_end="2019-01-03",
                )
            )
        if name == "wide":
            continue
        _, empty_lookup = read_prediction_output_bundle(
            root,
            expected_output_id=name,
            family_selection="earnings_yield",
            time_window=TimeWindow("2019-01-03", "2019-01-03"),
        )
        assert empty_lookup.row_count == 0
        list(
            _iter_prediction_joined_chunks(
                (requested.loc[requested.timestamp.eq("2019-01-03")],),
                prediction_lookup=empty_lookup,
                input_columns=("earnings_yield",),
                model_algorithm="linear_regression",
                sample_filter_mode="eligible",
                test_start="2019-01-03",
                test_end="2019-01-03",
            )
        )
        assert empty_lookup.coverage_counts() == (0, 0)
        with pytest.raises(ValueError, match="empty_prediction_window"):
            read_prediction_output_bundle(
                root,
                expected_output_id=name,
                time_window=TimeWindow("2019-01-04", "2019-01-04"),
            )
    with pytest.raises(ValueError, match="empty_prediction_window"):
        write_prediction_result(
            case.layout, _result(case, "all_empty", "2019-01-04", "2019-01-04")
        )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "family"))
def test_nonempty_required_domain_still_rejects_bad_keys(empty_family_case, mutation):
    case = empty_family_case
    result = _result(case, f"bad_{mutation}", "2019-01-03", "2019-01-03")
    with result:
        frames = list(result.output_chunks)
        metadata = result.prediction_metadata_factory()
    assert len(frames) == 1
    frame = frames[0].copy()
    if mutation == "missing":
        frame = frame.iloc[:-1]
        from quant_research_platform.research.predictions.outputs import (
            _add_a_share_scores,
        )

        _add_a_share_scores(frame, frame.raw_prediction)
        metadata["prediction_rows"] = len(frame)
    elif mutation == "duplicate":
        frame = pd.concat((frame, frame.iloc[:1]), ignore_index=True)
    else:
        frame["family_id"] = "earnings_yield"
    altered = replace(
        result, output_chunks=(frame,), prediction_metadata_factory=lambda: metadata
    )
    if mutation != "missing":
        error = (
            "duplicate_prediction_keys"
            if mutation == "duplicate"
            else "prediction_family_model_misaligned"
        )
        with pytest.raises(ValueError, match=error):
            write_prediction_result(case.layout, altered)
        return
    root = write_prediction_result(case.layout, altered)
    _, lookup = read_prediction_output_bundle(
        root,
        expected_output_id=result.spec.prediction_id,
        family_selection="book_to_market",
    )
    requested = case.frame.loc[case.frame.timestamp.eq("2019-01-03")]
    with pytest.raises(ValueError, match="missing_required_prediction_keys"):
        list(
            _iter_prediction_joined_chunks(
                (requested,),
                prediction_lookup=lookup,
                input_columns=("book_to_market",),
                model_algorithm="linear_regression",
                sample_filter_mode="eligible",
                test_start="2019-01-03",
                test_end="2019-01-03",
            )
        )


def test_frozen_holdout_aggregates_empty_family_without_refit(
    empty_family_case, tmp_path
):
    from quant_research_platform.research.models.fitting import prediction_callable
    from quant_research_platform.research.models.persisted import (
        read_quarterly_predictor,
    )
    from quant_research_platform.storage.layout import StorageLayout

    case = empty_family_case
    sessions = pd.bdate_range("2024-01-01", "2024-01-16")
    prices = pd.DataFrame(
        [
            {
                "asset_id": f"asset_{asset}",
                "timestamp": date,
                "adj_open": 100.0 + index * (asset + 1),
                "can_enter": True,
            }
            for index, date in enumerate(sessions)
            for asset in range(3)
        ]
    )
    frame = add_forward_return_labels(prices, horizons=(5,), sessions=sessions)
    frame = frame.loc[
        frame.timestamp.isin(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    ].copy()
    frame["market_id"] = "a_share"
    frame["eligible"] = True
    frame["book_to_market"] = [float(int(asset[-1]) + 1) for asset in frame.asset_id]
    frame["earnings_yield"] = frame.book_to_market.where(
        frame.timestamp.eq("2024-01-02")
    )
    frame.loc[frame.timestamp.eq("2024-01-04"), "book_to_market"] = np.nan
    frame["timestamp"] = frame.timestamp.dt.as_unit("ns")
    frame = frame.loc[:, list(case.dataset.required_columns)].sort_values(
        ["timestamp", "asset_id"]
    )
    layout = StorageLayout(tmp_path)
    native_root = layout.instance_root(
        layout.research_dataset_root("a_share", case.dataset.data_id), "d" * 64
    )
    layout = layout.with_target_roots({case.dataset.data_id: native_root})
    write_dataset(
        layout.target_root(
            case.dataset.data_id,
            layout.research_dataset_root("a_share", case.dataset.data_id),
        ),
        frame,
        spec=case.dataset,
    )
    frozen_identity = case.envelope.instance_identity
    for name, start, counts in (
        ("holdout_wide", "2024-01-02", {"earnings_yield": 3, "book_to_market": 6}),
        ("holdout_narrow", "2024-01-03", {"book_to_market": 3}),
    ):
        spec = PredictionSpec(
            prediction_id=name,
            model_id="model-artifact",
            research_dataset_id=case.dataset.data_id,
            train_end=None,
            test_start=start,
            test_end="2024-01-03",
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        result = build_prediction_result_from_persisted_model(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
            prediction_spec=spec,
            persisted_model=case.artifact,
            selection_group="global",
            access_window="holdout",
        )
        root = write_prediction_result(layout, result)
        saved = pd.read_parquet(root / "prediction.parquet")
        assert saved.groupby("family_id").size().to_dict() == counts
        manifest, _ = read_prediction_output_bundle(root, expected_output_id=name)
        assert (
            manifest["metadata"]["model_metadata"]["model_artifact_identity"]
            == frozen_identity
        )
        predictor = read_quarterly_predictor(case.artifact, "global", "book_to_market")
        requested = frame.loc[frame.timestamp.between(start, "2024-01-03")]
        expected = prediction_callable(("book_to_market",), predictor)(requested)
        actual = saved.loc[saved.family_id.eq("book_to_market")].sort_values(
            ["timestamp", "asset_id"]
        )
        np.testing.assert_array_equal(actual.raw_prediction, expected)

    empty_spec = replace(
        spec,
        prediction_id="holdout_all_empty",
        test_start="2024-01-04",
        test_end="2024-01-04",
    )
    empty_result = build_prediction_result_from_persisted_model(
        layout=layout,
        dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
        prediction_spec=empty_spec,
        persisted_model=case.artifact,
        selection_group="global",
        access_window="holdout",
    )
    with pytest.raises(ValueError, match="empty_prediction_window"):
        write_prediction_result(layout, empty_result)
    from quant_research_platform.research.models.persisted import (
        read_quarterly_model_artifact,
    )

    reloaded = read_quarterly_model_artifact(
        case.root, expected_identity=frozen_identity, trial_layout=case.layout
    )
    assert reloaded.member_records == case.artifact.member_records
    assert reloaded.selections == case.artifact.selections
