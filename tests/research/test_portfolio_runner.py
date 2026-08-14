from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.portfolios.runner as portfolio_runner
from quant_research_platform.artifacts._output_bundle_validation import (
    _read_prediction_output_bundle,
)
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.runner import (
    build_portfolio_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.research_specs import (
    make_portfolio_research_dataset_spec,
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_output_and_read


def test_portfolio_read_columns_include_diagnostics_config_columns() -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_column="signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        diagnostics_config=PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
            numeric_exposure_columns=("beta",),
            categorical_exposure_columns=("industry",),
        ),
    )

    assert portfolio_runner._portfolio_read_columns(spec) == (
        "market_id",
        "asset_id",
        "timestamp",
        "signal",
        "forward_return",
        "amount",
        "float_market_cap",
        "beta",
        "industry",
    )


def test_portfolio_runner_accepts_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("Europe/Berlin"),
                ),
            ]
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timestamps = portfolio_runner._timestamp_set(frame)

    assert timestamps.equals(
        pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-15T11:00:00"),
                pd.Timestamp("2024-01-15T11:01:00"),
            ]
        )
    )


def test_portfolio_runner_joins_persisted_prediction_primary_key_with_explicit_gaps(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )

    expected_result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
    )
    expected_chunks = list(expected_result.output_chunks)
    expected_summary = expected_result.summary_factory()
    expected_result.close()

    original_to_dict = pd.DataFrame.to_dict

    def reject_return_row_graph(frame, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        orient = args[0] if args else kwargs.get("orient", "dict")
        if orient == "records" and "gross_portfolio_return" in frame.columns:
            raise AssertionError("portfolio returns converted to a Python row graph")
        return original_to_dict(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", reject_return_row_graph)
    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
    )
    chunks = list(result.output_chunks)
    summary = result.summary_factory()
    returns = pd.concat(
        [chunk["portfolio_returns"] for chunk in chunks],
        ignore_index=True,
    )

    assert len(chunks) == len(expected_chunks)
    assert returns["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert returns["gross_portfolio_return"].tolist() == pytest.approx([0.0, 0.20])
    assert summary.keys() == expected_summary.keys()
    for name in summary:
        pd.testing.assert_frame_equal(summary[name], expected_summary[name])
    result.close()


def test_portfolio_prediction_join_keeps_its_creation_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024**2,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            build_portfolio_result_from_research_dataset(
                layout=layout,
                dataset_catalog=dataset_catalog,
                portfolio_catalog=portfolio_catalog,
                portfolio_id="demo_prediction_portfolio",
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
                prediction_frame=prediction,
                prediction_metadata=manifest["metadata"],
            )
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
            resource_policy=policy,
        )

        monkeypatch.setattr(
            portfolio_runner,
            "pandas_frame_payload_bytes",
            lambda _frame: policy.task_memory_budget_bytes,
        )

        def reject_prediction_index(*_args, **_kwargs):
            raise AssertionError("prediction index allocated before admission")

        monkeypatch.setattr(
            pd.MultiIndex,
            "from_frame",
            classmethod(reject_prediction_index),
        )
        with pytest.raises(
            ValueError,
            match="portfolio_prediction_join_working_set_exceeds_task_memory_budget",
        ):
            list(result.output_chunks)


def test_portfolio_result_explicit_close_owns_operation_root(tmp_path) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)
    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
        resource_policy=policy,
    )
    assert portfolio_runner.current_runtime_resource_policy() is policy
    result.close()
    result.close()
    assert portfolio_runner.current_runtime_resource_policy() is None
    with pytest.raises(RuntimeError, match="portfolio_result_closed"):
        list(result.output_chunks)
    with pytest.raises(RuntimeError, match="portfolio_result_closed"):
        result.summary_factory()


def test_portfolio_result_context_preserves_body_primary() -> None:
    primary = ValueError("body_primary")
    closed: list[bool] = []
    policy = capture_test_policy(available_memory_bytes=1024, rss_bytes=100)
    scope = portfolio_runner.runtime_resource_policy_scope(policy)
    scope.__enter__()

    def close() -> None:
        closed.append(True)
        scope.__exit__(None, None, None)
        raise MemoryError("cleanup_secondary")

    result = portfolio_runner.PortfolioResult(
        spec=object(),  # type: ignore[arg-type]
        output_chunks=(),
        summary_factory=dict,
        _close=close,
    )
    with pytest.raises(ValueError) as raised:
        with result:
            raise primary

    assert raised.value is primary
    assert raised.value.__notes__ == [
        "portfolio result cleanup also failed: MemoryError"
    ]
    assert closed == [True]
    assert portfolio_runner.current_runtime_resource_policy() is None


def test_portfolio_lazy_result_keeps_final_rejection_terminal(monkeypatch) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024,
        rss_bytes=100,
        explicit_cap_bytes=600,
    )
    measurement = policy._measurement  # noqa: SLF001
    original_payload = measurement._payload  # type: ignore[attr-defined]  # noqa: SLF001

    def failed_payload() -> dict[str, int | float | str]:
        payload = original_payload()
        payload["sampled_peak_rss_bytes"] = 701
        payload["peak_incremental_rss_bytes"] = 601
        payload["status"] = "FAIL"
        return payload

    measurement._payload = failed_payload  # type: ignore[attr-defined]  # noqa: SLF001
    monkeypatch.setattr(
        portfolio_runner,
        "_build_portfolio_result_from_research_dataset",
        lambda **_kwargs: portfolio_runner.PortfolioResult(
            spec=object(),  # type: ignore[arg-type]
            output_chunks=({"returns": pd.DataFrame({"value": [1.0]})},),
            summary_factory=lambda: {"summary": pd.DataFrame({"value": [1.0]})},
        ),
    )
    result = build_portfolio_result_from_research_dataset(
        layout=object(),  # type: ignore[arg-type]
        dataset_catalog=object(),  # type: ignore[arg-type]
        portfolio_catalog=object(),  # type: ignore[arg-type]
        portfolio_id="portfolio",
        resource_policy=policy,
    )
    assert len(list(result.output_chunks)) == 1

    result.summary_factory()
    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded") as first:
        result.close()
    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded") as second:
        result.summary_factory()

    assert second.value is first.value
    assert portfolio_runner.current_runtime_resource_policy() is None


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("duplicate", "duplicate_persisted_prediction_keys"),
        ("missing", "missing_required_prediction_keys"),
        ("misaligned", "misaligned_persisted_prediction_keys"),
    ],
)
def test_portfolio_runner_rejects_invalid_persisted_prediction_coverage(
    tmp_path,
    corruption: str,
    error: str,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    if corruption == "duplicate":
        prediction = pd.concat([prediction, prediction.iloc[[0]]], ignore_index=True)
    elif corruption == "missing":
        prediction = prediction.iloc[1:].reset_index(drop=True)
    else:
        prediction = prediction.copy()
        prediction.loc[0, "asset_id"] = "misaligned"

    with pytest.raises(ValueError, match=error):
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
        )
        list(result.output_chunks)


def test_persisted_prediction_portfolio_uses_bounded_disk_lookup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, _prediction, _manifest, root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    def reject_full_read(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("persisted prediction was materialized as one DataFrame")

    monkeypatch.setattr(pd, "read_parquet", reject_full_read)
    with runtime_resource_policy_scope(policy):
        with _read_prediction_output_bundle(
            root,
            expected_output_id="demo_prediction",
        ) as (manifest, prediction_lookup):
            result = build_portfolio_result_from_research_dataset(
                layout=layout,
                dataset_catalog=dataset_catalog,
                portfolio_catalog=portfolio_catalog,
                portfolio_id="demo_prediction_portfolio",
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
                prediction_frame=prediction_lookup,
                prediction_metadata=manifest["metadata"],
                resource_policy=policy,
            )
            chunks = list(result.output_chunks)
            summary = result.summary_factory()
            result.close()

    returns = pd.concat(
        [chunk["portfolio_returns"] for chunk in chunks],
        ignore_index=True,
    )
    assert returns["gross_portfolio_return"].tolist() == pytest.approx([0.0, 0.20])
    assert summary["performance_summary"].iloc[0]["periods"] == 2


def test_prediction_bundle_rejects_duplicate_outside_selected_window(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "model_input": [1.0, 2.0],
            "eligible": [True, True],
        }
    )
    spec = PredictionSpec(
        prediction_id="windowed_prediction",
        model_id="linear_regression_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-03",
    )
    _prediction, root, _manifest = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        spec,
        np.array([0.0, 1.0]),
        ("model_input",),
    )
    path = root / "prediction.parquet"
    persisted = pd.read_parquet(path)
    excluded = persisted.loc[persisted["timestamp"] == pd.Timestamp("2024-01-02")].iloc[
        [0]
    ]
    pd.concat([persisted, excluded], ignore_index=True).to_parquet(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="duplicate_persisted_prediction_keys"):
            with _read_prediction_output_bundle(
                root,
                expected_output_id=spec.prediction_id,
                time_window=TimeWindow("2024-01-03", "2024-01-03"),
            ):
                pass


def _prediction_portfolio_inputs(tmp_path):
    layout = StorageLayout(project_root=tmp_path)
    dataset_spec = make_portfolio_research_dataset_spec(
        factor_columns=("model_input",),
        label_columns=("forward_return",),
        filter_columns=("eligible",),
    )
    dataset_catalog = make_research_dataset_catalog(dataset_spec)
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-01")] * 4
            + [pd.Timestamp("2024-01-02")] * 4,
            "model_input": [1.0, 2.0, 3.0, 4.0, 4.0, 1.0, np.nan, 3.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.5, 0.2],
            "eligible": [True] * 7 + [False],
        }
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), dataset_spec.data_id),
        frame,
        spec=dataset_spec,
    )
    prediction_spec = PredictionSpec(
        prediction_id="demo_prediction",
        model_id="linear_regression_demo",
        research_dataset_id=dataset_spec.data_id,
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    prediction, root, manifest = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        np.array([0.0, 1.0]),
        ("model_input",),
    )
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(
        PortfolioSpec(
            portfolio_id="demo_prediction_portfolio",
            market_id=MarketId("a_share"),
            research_dataset_id=dataset_spec.data_id,
            signals=(
                SignalSpec(
                    signal_id=prediction_spec.prediction_id,
                    source_column=prediction_spec.prediction_column,
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )
    )
    return layout, dataset_catalog, portfolio_catalog, prediction, manifest, root
