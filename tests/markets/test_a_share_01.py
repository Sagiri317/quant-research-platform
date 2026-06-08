from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_factor_catalog,
    a_share_price_volume_factor_columns,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    build_factors,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    build_labels,
)
from quant_research_platform.markets.implementations.a_share.masks import (
    build as build_masks,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    contexts,
    panels,
)
from quant_research_platform.markets.implementations.a_share.source_bindings import (
    csmar as csmar_binding,
)
from quant_research_platform.markets.implementations.a_share.universes import (
    build as build_universes,
)
from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
    largest_price_volume_lookback_days,
    price_volume_factor_columns,
    price_volume_factor_input_columns,
)
from quant_research_platform.research.labels import forward_returns
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.synthetic_a_share import (
    make_synthetic_daily_market,
    make_synthetic_market_cap,
    make_synthetic_stock_master,
    make_synthetic_trading_calendar,
)
from tests.storage.low_level_fixtures import write_low_level_parquet_artifact


def _read_artifact(
    root: Path,
    *,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    return read_dataset(root, partition_filter=_year_filter(years), columns=columns)


def _processed_artifact_root(
    processed_root: Path,
    category: str,
    processed_data_id: str,
) -> Path:
    return Path(processed_root) / category / processed_data_id


def _processed_manifest_payload(
    processed_root: Path,
    category: str,
    processed_data_id: str,
) -> dict[str, object]:
    return inspect_dataset(
        _processed_artifact_root(processed_root, category, processed_data_id)
    ).manifest


def _year_filter(years: tuple[int, ...] | None) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}


def test_a_share_calendar_helper_builds_unified_and_market_sessions() -> None:
    from quant_research_platform.markets.implementations.a_share import calendar

    frame = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-06",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-09",
                ]
            ),
            "source_market_type": [1, 4, 16, 32, 64, 1, 1],
            "is_trading_day": [True, True, False, True, True, False, False],
        }
    )

    unified = calendar.a_share_unified_trading_sessions(frame)
    market_1 = calendar.market_type_trading_sessions(frame, 1)
    market_64 = calendar.market_type_trading_sessions(frame, 64)

    assert isinstance(unified, pd.DatetimeIndex)
    assert unified.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-06"),
    ]
    assert market_1.tolist() == [pd.Timestamp("2024-01-02")]
    assert market_64.tolist() == [pd.Timestamp("2024-01-08")]


def test_a_share_calendar_helper_rejects_raw_calendar_columns() -> None:
    from quant_research_platform.markets.implementations.a_share import calendar

    raw_like = pd.DataFrame(
        {
            "Markettype": [1],
            "Clddt": ["2024-01-02"],
            "State": ["O"],
        }
    )

    with pytest.raises(ValueError, match="calendar_missing_normalized_columns"):
        calendar.a_share_unified_trading_sessions(raw_like)


def test_a_share_calendar_helper_calculates_warmup_start_from_open_sessions() -> None:
    from quant_research_platform.markets.implementations.a_share import calendar

    frame = _trading_calendar_rows(
        [
            ("2019-12-26", 1, True),
            ("2019-12-27", 1, True),
            ("2019-12-28", 1, False),
            ("2019-12-29", 1, True),
            ("2019-12-30", 1, True),
            ("2020-01-01", 1, False),
            ("2020-01-02", 1, True),
            ("2020-01-02", 64, True),
        ]
    )

    warmup_start = calendar.warmup_start_date_for_window(
        frame,
        start_date="2020-01-01",
        max_window=3,
    )

    assert warmup_start == pd.Timestamp("2019-12-27")


def test_a_share_calendar_helper_rejects_invalid_warmup_window() -> None:
    from quant_research_platform.markets.implementations.a_share import calendar

    with pytest.raises(ValueError, match="invalid_factor_warmup_window"):
        calendar.warmup_start_date_for_window(
            _trading_calendar_frame(["2020-01-02"], market_type=1),
            start_date="2020-01-02",
            max_window=-1,
        )


def test_price_volume_factor_lookback_infers_requested_factor_window() -> None:
    assert largest_price_volume_lookback_days(("mom_3d",)) == 3
    assert largest_price_volume_lookback_days(()) == 0


def _panel() -> pd.DataFrame:
    rows = []
    for day in range(1, 6):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": pd.Timestamp(f"2014-01-0{day}"),
                "session_date": pd.Timestamp(f"2014-01-0{day}"),
                "available_at": pd.Timestamp(f"2014-01-0{day}"),
                "adj_open": float(day + 9),
                "adj_close": float(day + 10),
                "return_1d": 0.01 * day,
                "amount": 1000.0 + day,
                "turnover": 0.1 + day / 100.0,
                "market_type": "A",
                "csrc_2012_industry_code": "I",
                "csrc_2012_industry_name": "industry",
                "total_market_cap": 10000.0 + day,
                "float_market_cap": 9000.0 + day,
                "is_st": False,
                "is_pt": False,
                "limit_status": 0,
                "listing_age_trading_days": 130 + day,
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
                "high": float(day + 11),
                "low": float(day + 9),
                "prev_close": float(day + 9),
            }
        )
    return pd.DataFrame(rows)


def test_a_share_price_volume_factor_contract_is_catalog_driven() -> None:
    columns = price_volume_factor_columns()

    assert columns == a_share_price_volume_factor_columns()
    assert "mom_20d" in columns
    assert "mom_252d" in columns
    assert "rev_252d" in columns
    assert "vol_252d" not in columns
    assert "forward_return_20d" not in columns
    for factor_id in columns:
        assert price_volume_factor_input_columns(factor_id)
    with pytest.raises(ValueError, match="invalid_price_volume_preset"):
        price_volume_factor_columns(preset="full")


def test_a_share_factor_and_label_helpers_use_past_and_forward_domains() -> None:
    panel = _panel()
    factored = calculate_price_volume_factors(
        panel,
        factor_columns=("mom_2d",),
        asset_column="asset_id",
        date_column="timestamp",
    )
    labeled = forward_returns.add_forward_return_labels(panel, horizons=(1,))

    assert "mom_2d" in factored.columns
    assert "forward_return_1d" not in factored.columns
    assert "forward_return_1d" in labeled.columns


def test_a_share_factor_and_label_builders_write_split_layers(tmp_path) -> None:
    panel = _panel()
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(panel["timestamp"], market_type=1),
    )
    factor_path, factor_rows = (
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2014,
            end_year=2014,
        )
    )
    label_path, label_rows = build_labels.calculate_forward_returns_asset_batches(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    factor_manifest = _processed_manifest_payload(
        processed_root, "factors", PRICE_VOLUME_PROCESSED_DATA_ID
    )
    label_manifest = _processed_manifest_payload(
        processed_root, "labels", FORWARD_RETURNS_PROCESSED_DATA_ID
    )
    factor_output = _read_artifact(factor_path, years=(2014,))
    label_output = _read_artifact(label_path, years=(2014,))

    assert factor_rows == len(panel)
    assert label_rows == len(panel)
    assert factor_path.as_posix().endswith("processed/a_share/factors/price_volume")
    assert label_path.as_posix().endswith(
        "processed/a_share/labels/forward_returns"
    )
    assert tuple(factor_output.columns) == (
        *platform_entity_time_key(),
        *a_share_price_volume_factor_columns(),
    )
    assert tuple(label_output.columns) == tuple(
        a_share_processed_catalog().get(FORWARD_RETURNS_PROCESSED_DATA_ID).required_columns
    )
    assert "forward_return_1d" in label_output.columns
    assert factor_manifest["schema_version"] == "0.1"
    assert label_manifest["schema_version"] == "0.1"


def test_a_share_factor_build_passes_spec_columns_to_price_volume_helper(
    tmp_path,
    monkeypatch,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5))
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(panel["timestamp"], market_type=1),
    )
    captured: dict[str, object] = {}

    def fake_price_volume_helper(frame, **kwargs):
        captured["factor_columns"] = kwargs.get("factor_columns")
        result = frame.copy()
        for column in a_share_price_volume_factor_columns():
            result[column] = 0.0
        return result

    monkeypatch.setattr(
        build_factors,
        "calculate_price_volume_factors",
        fake_price_volume_helper,
    )

    output_root, rows = build_factors.calculate_price_volume_factor_asset_batches(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = _read_artifact(output_root, years=(2014,))

    assert captured["factor_columns"] == build_factors._factor_output_columns()
    assert set(captured["factor_columns"]) == set(
        definition.output_column for definition in a_share_factor_catalog().list()
    )
    assert rows == len(output)
    assert set(a_share_price_volume_factor_columns()).issubset(output.columns)


def test_a_share_factor_builder_prunes_to_spec_columns_before_copy(
    tmp_path,
    monkeypatch,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5))
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(panel["timestamp"], market_type=1),
    )
    expected_columns = tuple(
        build_factors.a_share_processed_catalog()
        .get(PRICE_VOLUME_PROCESSED_DATA_ID)
        .required_columns
    )

    def fake_price_volume_helper(frame, **kwargs):
        result = frame.copy()
        result["extra_internal_work_column"] = 1.0
        for column in a_share_price_volume_factor_columns():
            result[column] = 0.0
        return result

    monkeypatch.setattr(
        build_factors,
        "calculate_price_volume_factors",
        fake_price_volume_helper,
    )

    output_root, rows = build_factors.calculate_price_volume_factor_asset_batches(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = _read_artifact(output_root, years=(2014,))

    assert rows == len(panel)
    assert tuple(output.columns) == expected_columns
    assert "extra_internal_work_column" not in output.columns


def test_price_volume_helper_prunes_contract_extra_outputs() -> None:
    contract_columns = price_volume_factor_columns()
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=260))

    result = calculate_price_volume_factors(
        panel,
        factor_columns=contract_columns,
    )
    extra_factor_columns = {
        column
        for column in result.columns
        if _looks_like_price_volume_factor_column(column)
        and column not in contract_columns
    }

    assert "mom_252d" in result.columns
    assert "rev_252d" in result.columns
    assert not {
        "vol_252d",
        "downside_vol_252d",
        "upside_vol_252d",
        "max_return_252d",
        "min_return_252d",
        "drawdown_252d",
        "skew_252d",
        "kurt_252d",
        "avg_amount_252d",
        "turnover_252d",
        "turnover_chg_252d",
        "turnover_vol_252d",
        "amihud_252d",
        "zero_volume_ratio_252d",
        "amplitude_252d",
        "intraday_range_252d",
    }.intersection(result.columns)
    assert extra_factor_columns == set()


def test_factor_builder_fails_closed_when_registered_input_column_missing(
    tmp_path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5)).drop(
        columns=["amount"]
    )
    write_low_level_parquet_artifact(
        _processed_artifact_root(processed_root, "panels", "daily_panel"),
        panel,
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(panel["timestamp"], market_type=1),
    )

    with pytest.raises(ValueError, match="missing_factor_panel_input_columns: amount"):
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2014,
            end_year=2014,
        )


def test_a_share_factor_builder_reads_calendar_warmup_and_writes_requested_window(
    tmp_path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    dates = pd.bdate_range("2019-01-14", periods=254)
    panel = _factor_panel_for_dates(dates)
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(dates, market_type=1),
    )

    output_root, rows = build_factors.calculate_price_volume_factor_asset_batches(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2020,
        end_year=2020,
    )
    output = _read_artifact(output_root, years=(2020,))

    assert rows == len(output)
    assert output["timestamp"].min() >= pd.Timestamp("2020-01-01")
    assert output["timestamp"].max() <= pd.Timestamp("2020-12-31")
    first_2020 = output.sort_values("timestamp").iloc[0]
    assert pd.notna(first_2020["mom_252d"])


def test_a_share_factor_builder_allows_missing_pre_start_warmup_years(
    tmp_path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    calendar_dates = pd.bdate_range("2012-12-31", "2014-12-31")
    panel = _factor_panel_for_dates(calendar_dates[calendar_dates.year == 2014])
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(calendar_dates, market_type=1),
    )

    output_root, rows = build_factors.calculate_price_volume_factor_asset_batches(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = _read_artifact(output_root, years=(2014,))

    assert rows == len(output)
    assert output["timestamp"].min() >= pd.Timestamp("2014-01-01")
    assert output["timestamp"].max() <= pd.Timestamp("2014-12-31")
    assert pd.isna(output.sort_values("timestamp").iloc[0]["mom_252d"])


def test_a_share_factor_builder_reads_available_warmup_years_when_present(
    tmp_path,
) -> None:
    calendar_dates = pd.bdate_range("2012-12-31", "2014-12-31")

    no_warmup_root = tmp_path / "no_warmup" / "data" / "processed" / "a_share"
    no_warmup_interim = tmp_path / "no_warmup" / "data" / "interim" / "csmar"
    no_warmup_panel = _factor_panel_for_dates(
        calendar_dates[calendar_dates.year == 2014]
    )
    _write_processed_panel_fixture(no_warmup_root, no_warmup_panel)
    _write_interim_fixture(
        no_warmup_interim,
        "trading_calendar",
        _trading_calendar_frame(calendar_dates, market_type=1),
    )
    no_warmup_output_root, _ = (
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=no_warmup_interim,
            processed_root=no_warmup_root,
            start_year=2014,
            end_year=2014,
        )
    )
    no_warmup_output = _read_artifact(
        no_warmup_output_root,
        years=(2014,),
    )

    warmup_root = tmp_path / "with_warmup" / "data" / "processed" / "a_share"
    warmup_interim = tmp_path / "with_warmup" / "data" / "interim" / "csmar"
    warmup_panel = _factor_panel_for_dates(
        calendar_dates[calendar_dates.year.isin([2013, 2014])]
    )
    _write_processed_panel_fixture(warmup_root, warmup_panel)
    _write_interim_fixture(
        warmup_interim,
        "trading_calendar",
        _trading_calendar_frame(calendar_dates, market_type=1),
    )
    warmup_output_root, _ = (
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=warmup_interim,
            processed_root=warmup_root,
            start_year=2014,
            end_year=2014,
        )
    )
    warmup_output = _read_artifact(
        warmup_output_root,
        years=(2014,),
    )

    assert warmup_output["timestamp"].min() >= pd.Timestamp("2014-01-01")
    assert warmup_output["timestamp"].max() <= pd.Timestamp("2014-12-31")
    assert (
        warmup_output["mom_252d"].isna().sum()
        < no_warmup_output["mom_252d"].isna().sum()
    )


def test_a_share_factor_builder_asset_batches_match_default_dataset_output(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2013-01-01", "2014-12-31")
    panel = pd.concat(
        [
            _factor_panel_for_dates(dates).assign(asset_id="asset_a"),
            _factor_panel_for_dates(dates).assign(asset_id="asset_b"),
        ],
        ignore_index=True,
    )
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(dates, market_type=1),
    )

    default_path, _ = build_factors.calculate_price_volume_factor_asset_batches(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    expected = _read_artifact(default_path, years=(2014,))
    chunked_root = tmp_path / "chunked" / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(chunked_root, panel)
    actual_path, actual_rows = (
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=chunked_root,
            start_year=2014,
            end_year=2014,
            asset_chunk_size=1,
        )
    )
    actual = _read_artifact(actual_path, years=(2014,))

    assert actual_rows == len(actual)
    pd.testing.assert_frame_equal(actual, expected)


def test_a_share_label_builder_asset_batches_keep_future_lookahead_boundary(
    tmp_path,
) -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-12-29", "2015-01-05"))
    chunked_root = tmp_path / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(chunked_root, panel)
    label_root, rows = build_labels.calculate_forward_returns_asset_batches(
        processed_root=chunked_root,
        start_year=2014,
        end_year=2014,
        asset_chunk_size=1,
    )

    labels = _read_artifact(label_root, years=(2014,))
    assert rows == len(labels)
    assert labels["timestamp"].dt.year.tolist() == [2014, 2014, 2014]
    assert labels["forward_return_1d"].notna().all()


def test_a_share_label_builder_dataset_path_fails_closed_for_missing_input(
    tmp_path,
) -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5)).drop(
        columns=["adj_open"]
    )
    processed_root = tmp_path / "data" / "processed" / "a_share"
    write_low_level_parquet_artifact(
        _processed_artifact_root(processed_root, "panels", "daily_panel"),
        panel,
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )

    with pytest.raises(ValueError, match="missing_label_panel_input_columns: adj_open"):
        build_labels.calculate_forward_returns_asset_batches(
            processed_root=processed_root,
            start_year=2014,
            end_year=2014,
        )


def test_a_share_factor_builder_requires_trading_calendar(tmp_path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    panel = _factor_panel_for_dates(pd.bdate_range("2020-01-01", periods=3))
    _write_processed_panel_fixture(processed_root, panel)

    with pytest.raises(ValueError, match="missing_trading_calendar_interim"):
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2020,
            end_year=2020,
        )


def test_a_share_factor_builder_requires_requested_output_partition(
    tmp_path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    dates = pd.bdate_range("2019-01-14", periods=254)
    panel = _factor_panel_for_dates(dates[dates.year == 2019])
    _write_processed_panel_fixture(processed_root, panel)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(dates, market_type=1),
    )

    with pytest.raises(ValueError, match="missing_requested_processed_input"):
        build_factors.calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2020,
            end_year=2020,
        )


def test_a_share_label_builder_allows_missing_future_lookahead_year(tmp_path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    panel = _factor_panel_for_dates(pd.bdate_range("2014-12-29", "2014-12-31"))
    _write_processed_panel_fixture(processed_root, panel)

    label_path, label_rows = build_labels.calculate_forward_returns_asset_batches(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = _read_artifact(label_path, years=(2014,))

    assert label_rows == len(output)
    assert output["timestamp"].dt.year.tolist() == [2014, 2014, 2014]
    assert output["forward_return_1d"].isna().sum() == 2


def test_a_share_label_builder_uses_future_lookahead_when_available(
    tmp_path,
) -> None:
    panel_without_future = _factor_panel_for_dates(
        pd.bdate_range("2014-12-29", "2014-12-31")
    )
    without_future_root = tmp_path / "without_future" / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(without_future_root, panel_without_future)
    labels_without_future_root, _ = (
        build_labels.calculate_forward_returns_asset_batches(
            processed_root=without_future_root,
            start_year=2014,
            end_year=2014,
        )
    )
    labels_without_future = _read_artifact(labels_without_future_root, years=(2014,))

    panel_with_future = _factor_panel_for_dates(
        pd.bdate_range("2014-12-29", "2015-01-02")
    )
    with_future_root = tmp_path / "with_future" / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(with_future_root, panel_with_future)
    labels_with_future_root, _ = build_labels.calculate_forward_returns_asset_batches(
        processed_root=with_future_root,
        start_year=2014,
        end_year=2014,
    )
    labels_with_future = _read_artifact(labels_with_future_root, years=(2014,))

    tail_without_future = labels_without_future.loc[
        labels_without_future["timestamp"].dt.year == 2014,
        "forward_return_1d",
    ]
    tail_with_future = labels_with_future.loc[
        labels_with_future["timestamp"].dt.year == 2014,
        "forward_return_1d",
    ]
    assert tail_with_future.isna().sum() < tail_without_future.isna().sum()


def test_a_share_label_builder_asset_batches_preserve_year_boundary_lookahead(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2014-12-01", "2015-06-30")
    panel = pd.concat(
        [
            _factor_panel_for_dates(dates).assign(asset_id="asset_a"),
            _factor_panel_for_dates(dates).assign(asset_id="asset_b"),
        ],
        ignore_index=True,
    )
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(processed_root, panel)

    default_path, _ = build_labels.calculate_forward_returns_asset_batches(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    expected = _read_artifact(default_path, years=(2014,))
    chunked_root = tmp_path / "chunked" / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(chunked_root, panel)
    actual_path, actual_rows = build_labels.calculate_forward_returns_asset_batches(
        processed_root=chunked_root,
        start_year=2014,
        end_year=2014,
        asset_chunk_size=1,
    )
    actual = _read_artifact(actual_path, years=(2014,))

    assert actual_rows == len(actual)
    assert actual["forward_return_1d"].notna().any()
    pd.testing.assert_frame_equal(actual, expected)


def test_a_share_factor_builder_does_not_depend_on_raw_calendar() -> None:
    source = Path(build_factors.__file__).read_text(encoding="utf-8")

    forbidden_terms = (
        "data/raw/csmar/trading_calendar",
        'a_share_csmar_normalize_private_export_frame(raw_frame, "trading_calendar")',
        "Markettype",
        "Clddt",
        "Daywk",
        "State",
    )
    assert not any(term in source for term in forbidden_terms)


def test_a_share_context_universe_and_mask_builders_write_split_layers(
    tmp_path,
) -> None:
    panel = _panel()
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(processed_root, panel)

    context_path, context_rows = contexts.write_contexts_from_panel_dataset(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    universe_path, universe_rows = build_universes.write_universe_from_panel_dataset(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    mask_outputs = build_masks.write_masks_from_panel_dataset(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    context_output = _read_artifact(context_path, years=(2014,))
    universe_output = _read_artifact(universe_path, years=(2014,))
    execution_mask = _read_artifact(mask_outputs["execution"][0], years=(2014,))

    assert context_rows == len(panel)
    assert universe_rows == len(panel)
    assert context_path.as_posix().endswith("processed/a_share/contexts/daily_context")
    assert universe_path.as_posix().endswith(
        "processed/a_share/universes/a_share_default"
    )
    assert sorted(mask_outputs) == ["as_of", "evaluation", "execution"]
    assert mask_outputs["execution"][0].as_posix().endswith(
        "processed/a_share/masks/execution/a_share_execution"
    )
    assert (
        _processed_manifest_payload(
            processed_root, "masks/execution", "a_share_execution"
        )["schema_version"]
        == "0.1"
    )
    assert tuple(context_output.columns) == tuple(
        a_share_processed_catalog().get("daily_context").required_columns
    )
    assert tuple(universe_output.columns) == tuple(
        a_share_processed_catalog().get("a_share_default").required_columns
    )
    assert tuple(execution_mask.columns) == tuple(
        a_share_processed_catalog().get("a_share_execution").required_columns
    )
    assert context_output["timestamp"].dt.year.tolist() == [2014] * len(panel)
    assert universe_output["eligible"].tolist() == [True] * len(panel)
    assert execution_mask[["can_enter", "can_exit"]].all().all()


def test_context_universe_and_mask_panel_dataset_writers_handle_chunked_panels(
    tmp_path,
) -> None:
    panel = pd.concat(
        [
            _panel().assign(asset_id="asset_a"),
            _panel().assign(asset_id="asset_b"),
        ],
        ignore_index=True,
    )
    chunked_root = tmp_path / "chunked" / "data" / "processed" / "a_share"
    _write_processed_panel_fixture(chunked_root, panel)

    actual_context_root, context_rows = contexts.write_contexts_from_panel_dataset(
        chunked_root,
        start_year=2014,
        end_year=2014,
    )
    (
        actual_universe_root,
        universe_rows,
    ) = build_universes.write_universe_from_panel_dataset(
        chunked_root,
        start_year=2014,
        end_year=2014,
    )
    actual_masks = build_masks.write_masks_from_panel_dataset(
        chunked_root,
        start_year=2014,
        end_year=2014,
    )
    context = _read_artifact(actual_context_root, years=(2014,))
    universe = _read_artifact(actual_universe_root, years=(2014,))

    assert context_rows == len(panel)
    assert universe_rows == len(panel)
    assert context["asset_id"].tolist() == sorted(context["asset_id"].tolist())
    assert universe["asset_id"].value_counts().to_dict() == {
        "asset_a": len(_panel()),
        "asset_b": len(_panel()),
    }
    for mask_kind, (mask_root, mask_rows) in sorted(actual_masks.items()):
        mask = _read_artifact(mask_root, years=(2014,))
        assert mask_rows == len(panel)
        assert mask["asset_id"].value_counts().to_dict() == {
            "asset_a": len(_panel()),
            "asset_b": len(_panel()),
        }, mask_kind


def test_a_share_panel_joins_stock_master_as_of_timestamp() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-06-30"),
            _daily_market_row("2020-07-02"),
        ]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            ),
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="special_status",
                industry_code="I2",
            ),
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row()]),
        trading_calendar=_basic_trading_calendar(),
        min_listing_trading_days=1,
    )

    assert len(panel) == len(daily)
    assert not panel.duplicated(["asset_id", "timestamp"]).any()
    by_date = panel.set_index("timestamp").sort_index()
    assert by_date.loc[pd.Timestamp("2020-06-30"), "company_status"] == "normal"
    assert (
        by_date.loc[pd.Timestamp("2020-06-30"), "status_change_date"]
        == pd.Timestamp("2020-01-01")
    )
    assert by_date.loc[pd.Timestamp("2020-07-02"), "company_status"] == (
        "special_status"
    )
    assert (
        by_date.loc[pd.Timestamp("2020-07-02"), "status_change_date"]
        == pd.Timestamp("2020-07-01")
    )
    assert by_date["listing_date"].eq(pd.Timestamp("2010-01-01")).all()


def test_a_share_panel_stock_master_exact_duplicate_effective_date_is_deduped() -> None:
    daily = pd.DataFrame([_daily_market_row("2020-07-02")])
    row = _stock_master_row(
        status_change_date="2020-07-01",
        company_status="normal_duplicate",
        industry_code="I1",
    )
    stock_master = pd.DataFrame([row, row.copy()])

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row()]),
        trading_calendar=_basic_trading_calendar(),
        min_listing_trading_days=1,
    )

    assert len(panel) == 1
    assert not panel.duplicated(["asset_id", "timestamp"]).any()
    assert panel.loc[0, "company_status"] == "normal_duplicate"
    assert panel.loc[0, "csrc_2012_industry_code"] == "I1"


def test_a_share_panel_stock_master_conflicting_duplicate_effective_date_fails_closed(
) -> None:
    daily = pd.DataFrame([_daily_market_row("2020-07-02")])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="first_duplicate",
                industry_code="I1",
            ),
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="second_duplicate",
                industry_code="I2",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="conflicting_panel_join_duplicate_keys: stock_master",
    ):
        panels.build_research_panel(
            daily,
            stock_master=stock_master,
            market_cap=pd.DataFrame([_market_cap_row()]),
            trading_calendar=_basic_trading_calendar(),
            min_listing_trading_days=1,
        )


def test_a_share_panel_market_cap_exact_duplicate_timestamp_is_deduped() -> None:
    daily = pd.DataFrame([_daily_market_row("2020-07-02")])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )
    row = _market_cap_row(share_change_date="2020-01-01")
    market_cap = pd.DataFrame([row, row.copy()])

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=_basic_trading_calendar(),
        min_listing_trading_days=1,
    )

    assert len(panel) == 1
    assert not panel.duplicated(["asset_id", "timestamp"]).any()
    assert panel.loc[0, "total_shares"] == 1000.0
    assert panel.loc[0, "a_float_shares"] == 500.0


def test_a_share_panel_market_cap_conflicting_duplicate_timestamp_fails_closed(
) -> None:
    daily = pd.DataFrame([_daily_market_row("2020-07-02")])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )
    first = _market_cap_row(share_change_date="2020-01-01")
    second = {**first, "a_float_shares": 600.0}
    market_cap = pd.DataFrame([first, second])

    with pytest.raises(
        ValueError,
        match="conflicting_panel_join_duplicate_keys: market_cap",
    ):
        panels.build_research_panel(
            daily,
            stock_master=stock_master,
            market_cap=market_cap,
            trading_calendar=_basic_trading_calendar(),
            min_listing_trading_days=1,
        )


def test_a_share_panel_stock_master_does_not_use_future_status() -> None:
    daily = pd.DataFrame([_daily_market_row("2020-01-01")])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="future_status",
                industry_code="I2",
            ),
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row()]),
        trading_calendar=_basic_trading_calendar(),
        min_listing_trading_days=1,
    )

    assert len(panel) == 1
    assert not panel.duplicated(["asset_id", "timestamp"]).any()
    assert pd.isna(panel.loc[0, "company_status"])
    assert pd.isna(panel.loc[0, "status_change_date"])
    assert pd.isna(panel.loc[0, "csrc_2012_industry_code"])
    assert panel.loc[0, "listing_date"] == pd.Timestamp("2010-01-01")


def test_a_share_panel_listing_age_uses_calendar_before_build_window() -> None:
    daily = pd.DataFrame([_daily_market_row("2014-01-02", market_type=1)])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                listing_date="2010-01-01",
                status_change_date="2010-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )
    calendar_dates = list(pd.bdate_range("2010-01-01", periods=130))
    calendar_dates.append(pd.Timestamp("2014-01-02"))

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row()]),
        trading_calendar=_trading_calendar_frame(calendar_dates, market_type=1),
        min_listing_trading_days=120,
    )

    assert panel.loc[0, "listing_age_trading_days"] > 120
    assert bool(panel.loc[0, "eligible_after_listing_age"]) is True


def test_a_share_panel_listing_age_new_stock_threshold() -> None:
    timestamps = pd.date_range("2020-01-01", periods=3, freq="D")
    daily = pd.DataFrame(
        [_daily_market_row(str(date.date()), market_type=1) for date in timestamps]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2020-01-01")]),
        trading_calendar=_trading_calendar_frame(timestamps, market_type=1),
        min_listing_trading_days=3,
    )

    assert panel["listing_age_trading_days"].tolist() == [1, 2, 3]
    assert panel["eligible_after_listing_age"].tolist() == [False, False, True]


def test_a_share_panel_masks_use_reference_asof_calendar_and_next_day_limits() -> None:
    panel = panels.build_research_panel(
        make_synthetic_daily_market(),
        stock_master=make_synthetic_stock_master(),
        market_cap=make_synthetic_market_cap(),
        trading_calendar=make_synthetic_trading_calendar(),
        min_listing_trading_days=4,
    )
    by_key = panel.set_index(["asset_id", "timestamp"]).sort_index()

    asset_a_early = by_key.loc[("asset_a", pd.Timestamp("2024-01-03"))]
    asset_a_after_change = by_key.loc[("asset_a", pd.Timestamp("2024-01-08"))]
    asset_a_before_limit_up = by_key.loc[("asset_a", pd.Timestamp("2024-01-02"))]
    asset_b_before_limit_down = by_key.loc[("asset_b", pd.Timestamp("2024-01-02"))]
    asset_c_zero_liquidity = by_key.loc[("asset_c", pd.Timestamp("2024-01-05"))]
    asset_a_last = by_key.loc[("asset_a", pd.Timestamp("2024-01-08"))]

    assert int(asset_a_early["listing_age_trading_days"]) == 3
    assert bool(asset_a_early["eligible_after_listing_age"]) is False
    assert bool(asset_a_after_change["eligible_after_listing_age"]) is True
    assert asset_a_early["company_status"] == "normal"
    assert asset_a_after_change["company_status"] == "watched"
    assert float(asset_a_early["total_shares"]) == 1000.0
    assert float(asset_a_after_change["total_shares"]) == 2000.0
    assert bool(asset_b_before_limit_down["eligible"]) is False
    assert bool(asset_c_zero_liquidity["eligible_liquidity"]) is False
    assert bool(asset_a_before_limit_up["can_enter"]) is False
    assert bool(asset_b_before_limit_down["can_exit"]) is True
    assert bool(asset_a_last["can_enter"]) is False
    assert bool(asset_a_last["can_exit"]) is False


def test_a_share_panel_masks_block_only_one_word_next_day_limits() -> None:
    timestamps = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
        ]
    )
    rows = [
        _daily_market_row("2020-01-01", market_type=1),
        {
            **_daily_market_row("2020-01-02", market_type=1),
            "limit_status": 1,
            "open": 11.0,
            "high": 11.0,
            "low": 11.0,
            "close": 11.0,
        },
        {
            **_daily_market_row("2020-01-03", market_type=1),
            "limit_status": 1,
            "open": 10.8,
            "high": 11.0,
            "low": 10.7,
            "close": 10.9,
        },
        {
            **_daily_market_row("2020-01-06", market_type=1),
            "limit_status": -1,
            "open": 9.0,
            "high": 9.0,
            "low": 9.0,
            "close": 9.0,
        },
        {
            **_daily_market_row("2020-01-07", market_type=1),
            "limit_status": -1,
            "open": 9.2,
            "high": 9.4,
            "low": 9.0,
            "close": 9.1,
        },
    ]

    panel = panels.build_research_panel(
        pd.DataFrame(rows),
        stock_master=pd.DataFrame(
            [
                _stock_master_row(
                    status_change_date="2010-01-01",
                    company_status="normal",
                    industry_code="I1",
                )
            ]
        ),
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2010-01-01")]),
        trading_calendar=_trading_calendar_frame(timestamps, market_type=1),
        min_listing_trading_days=1,
    ).set_index("timestamp")

    assert bool(panel.loc[pd.Timestamp("2020-01-01"), "can_enter"]) is False
    assert bool(panel.loc[pd.Timestamp("2020-01-02"), "can_enter"]) is True
    assert bool(panel.loc[pd.Timestamp("2020-01-03"), "can_exit"]) is False
    assert bool(panel.loc[pd.Timestamp("2020-01-06"), "can_exit"]) is True
    assert bool(panel.loc[pd.Timestamp("2020-01-07"), "can_enter"]) is False
    assert bool(panel.loc[pd.Timestamp("2020-01-07"), "can_exit"]) is False


def test_a_share_panel_masks_use_explicit_one_word_fields_when_available() -> None:
    daily = pd.DataFrame(
        [
            {
                **_daily_market_row("2020-01-01", market_type=1),
                "one_word_limit_up": False,
                "one_word_limit_down": False,
            },
            {
                **_daily_market_row("2020-01-02", market_type=1),
                "limit_status": 1,
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "one_word_limit_up": True,
                "one_word_limit_down": False,
            },
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=pd.DataFrame(
            [
                _stock_master_row(
                    status_change_date="2010-01-01",
                    company_status="normal",
                    industry_code="I1",
                )
            ]
        ),
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2010-01-01")]),
        trading_calendar=_trading_calendar_frame(
            pd.to_datetime(["2020-01-01", "2020-01-02"]),
            market_type=1,
        ),
        min_listing_trading_days=1,
    )

    assert bool(panel.loc[0, "can_enter"]) is False


def test_a_share_panel_missing_limit_status_does_not_imply_one_word_limit() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-01-01", market_type=1),
            {
                **_daily_market_row("2020-01-02", market_type=1),
                "limit_status": None,
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
            },
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=pd.DataFrame(
            [
                _stock_master_row(
                    status_change_date="2010-01-01",
                    company_status="normal",
                    industry_code="I1",
                )
            ]
        ),
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2010-01-01")]),
        trading_calendar=_trading_calendar_frame(
            pd.to_datetime(["2020-01-01", "2020-01-02"]),
            market_type=1,
        ),
        min_listing_trading_days=1,
    )

    assert bool(panel.loc[0, "can_enter"]) is True


def test_a_share_panel_shifted_masks_do_not_emit_futurewarning() -> None:
    daily = pd.DataFrame(
        [
            {
                **_daily_market_row("2020-01-01", market_type=1),
                "one_word_limit_up": 0,
                "one_word_limit_down": 0,
            },
            {
                **_daily_market_row("2020-01-02", market_type=1),
                "one_word_limit_up": 1,
                "one_word_limit_down": 0,
            },
        ]
    )

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        panels.build_research_panel(
            daily,
            stock_master=pd.DataFrame(
                [
                    _stock_master_row(
                        status_change_date="2010-01-01",
                        company_status="normal",
                        industry_code="I1",
                    )
                ]
            ),
            market_cap=pd.DataFrame([_market_cap_row(share_change_date="2010-01-01")]),
            trading_calendar=_trading_calendar_frame(
                pd.to_datetime(["2020-01-01", "2020-01-02"]),
                market_type=1,
            ),
            min_listing_trading_days=1,
        )

    future_warnings = [
        warning
        for warning in warnings_record
        if issubclass(warning.category, FutureWarning)
    ]
    assert future_warnings == []


def test_a_share_panel_masks_fail_closed_without_one_word_inference_columns() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-01-01", market_type=1),
            _daily_market_row("2020-01-02", market_type=1),
        ]
    ).drop(columns=["low"])

    with pytest.raises(
        ValueError,
        match="missing_one_word_limit_inference_columns: low",
    ):
        panels.build_research_panel(
            daily,
            stock_master=pd.DataFrame(
                [
                    _stock_master_row(
                        status_change_date="2010-01-01",
                        company_status="normal",
                        industry_code="I1",
                    )
                ]
            ),
            market_cap=pd.DataFrame([_market_cap_row(share_change_date="2010-01-01")]),
            trading_calendar=_trading_calendar_frame(
                pd.to_datetime(["2020-01-01", "2020-01-02"]),
                market_type=1,
            ),
            min_listing_trading_days=1,
        )


def test_a_share_panel_listing_age_starts_after_closed_listing_date() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-01-01", market_type=1),
            _daily_market_row("2020-01-02", market_type=1),
        ]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )
    calendar = _trading_calendar_rows(
        [
            ("2020-01-01", 1, False),
            ("2020-01-02", 1, True),
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2020-01-01")]),
        trading_calendar=calendar,
        min_listing_trading_days=1,
    )

    assert panel["listing_age_trading_days"].tolist() == [0, 1]
    assert panel["eligible_after_listing_age"].tolist() == [False, True]


def test_a_share_panel_listing_age_uses_market_specific_sessions() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-01-01", asset_id="000001", market_type=1),
            _daily_market_row("2020-01-01", asset_id="000002", market_type=32),
        ]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                asset_id="000001",
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            ),
            _stock_master_row(
                asset_id="000002",
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I2",
            ),
        ]
    )
    calendar = _trading_calendar_rows(
        [
            ("2020-01-01", 1, True),
            ("2020-01-01", 32, False),
            ("2020-01-02", 32, True),
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame(
            [
                _market_cap_row(asset_id="000001", share_change_date="2020-01-01"),
                _market_cap_row(asset_id="000002", share_change_date="2020-01-01"),
            ]
        ),
        trading_calendar=calendar,
        min_listing_trading_days=1,
    )

    ages = panel.set_index("asset_id")["listing_age_trading_days"].to_dict()
    assert ages == {"000001": 1, "000002": 0}


def test_a_share_panel_listing_age_fallback_unified_excludes_bjse() -> None:
    daily = pd.DataFrame(
        [
            _daily_market_row("2020-01-01", market_type=999),
            _daily_market_row("2020-01-02", market_type=999),
        ]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            )
        ]
    )
    calendar = _trading_calendar_rows(
        [
            ("2020-01-01", 64, True),
            ("2020-01-02", 1, True),
        ]
    )

    panel = panels.build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=pd.DataFrame([_market_cap_row(share_change_date="2020-01-01")]),
        trading_calendar=calendar,
        min_listing_trading_days=1,
    )

    assert panel["listing_age_trading_days"].tolist() == [0, 1]
    assert panel["eligible_after_listing_age"].tolist() == [False, True]


def test_a_share_panel_from_interim_requires_trading_calendar(tmp_path) -> None:
    interim_root = tmp_path / "data" / "interim" / "csmar"
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_minimal_panel_interim_fixture(interim_root)

    with pytest.raises(ValueError, match="missing_trading_calendar_interim"):
        panels.build_panel_asset_batches_from_interim(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2024,
            end_year=2024,
            end_date="2024-06-30",
            min_listing_trading_days=1,
        )


def test_a_share_panel_from_interim_preserves_invalid_calendar_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="calendar_missing_normalized_columns"):
        panels.build_research_panel(
            pd.DataFrame([_daily_market_row("2024-06-28", market_type=1)]),
            stock_master=pd.DataFrame(
                [
                    _stock_master_row(
                        asset_id="000001",
                        listing_date="2020-01-01",
                        status_change_date="2020-01-01",
                        company_status="active",
                        industry_code="I",
                    )
                ]
            ),
            market_cap=pd.DataFrame(
                [_market_cap_row(asset_id="000001", share_change_date="2020-01-01")]
            ),
            trading_calendar=pd.DataFrame(
                {
                    "calendar_date": [pd.Timestamp("2024-06-28")],
                    "is_trading_day": [True],
                }
            ),
            min_listing_trading_days=1,
        )


def test_a_share_panel_builder_does_not_depend_on_raw_calendar() -> None:
    source = Path(panels.__file__).read_text(encoding="utf-8")

    forbidden_terms = (
        "data/raw/csmar/trading_calendar",
        'a_share_csmar_normalize_private_export_frame(raw_frame, "trading_calendar")',
        "Markettype",
        "Clddt",
        "Daywk",
        "State",
    )
    assert not any(term in source for term in forbidden_terms)


def test_a_share_panel_from_interim_applies_exact_end_date(tmp_path) -> None:
    interim_root = tmp_path / "data" / "interim" / "csmar"
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_interim_fixture(
        interim_root,
        "daily_market",
        pd.DataFrame(
            [
                _daily_market_row("2024-06-28"),
                _daily_market_row("2024-07-01"),
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "stock_master",
        pd.DataFrame(
            [
                _stock_master_row(
                    asset_id="000001",
                    listing_date="2020-01-01",
                    status_change_date="2020-01-01",
                    company_status="active",
                    industry_code="I",
                )
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [_market_cap_row(asset_id="000001", share_change_date="2020-01-01")]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(
            pd.date_range("2020-01-01", "2024-07-01", freq="D"),
            market_type=1,
        ),
    )

    output_root, rows = panels.build_panel_asset_batches_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2024,
        end_year=2024,
        end_date="2024-06-30",
        min_listing_trading_days=1,
    )
    written = _read_artifact(output_root, years=(2024,))

    assert rows == 1
    assert written["timestamp"].tolist() == [pd.Timestamp("2024-06-28")]


def test_a_share_panel_asset_batches_preserve_next_session_boundary(
    tmp_path,
) -> None:
    interim_root = tmp_path / "data" / "interim" / "csmar"
    processed_root = tmp_path / "data" / "processed" / "a_share"
    dates = pd.to_datetime(["2014-12-30", "2014-12-31", "2015-01-02"])
    daily = pd.DataFrame(
        [
            _daily_market_row(str(date.date()), asset_id=asset_id, market_type=1)
            for asset_id in ("000001", "000002")
            for date in dates
        ]
    )
    _write_interim_fixture(interim_root, "daily_market", daily)
    _write_interim_fixture(
        interim_root,
        "stock_master",
        pd.DataFrame(
            [
                _stock_master_row(
                    asset_id=asset_id,
                    listing_date="2010-01-01",
                    status_change_date="2010-01-01",
                    company_status="active",
                    industry_code="I",
                )
                for asset_id in ("000001", "000002")
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [
                _market_cap_row(asset_id=asset_id, share_change_date="2010-01-01")
                for asset_id in ("000001", "000002")
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(
            pd.date_range("2010-01-01", "2015-01-02"),
            market_type=1,
        ),
    )

    output_root, rows = panels.build_panel_asset_batches_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
        end_date="2014-12-31",
        calculation_end_year=2015,
        min_listing_trading_days=1,
        asset_chunk_size=1,
    )
    actual = _read_artifact(output_root, years=(2014,))

    assert rows == len(actual)
    assert rows == 4
    assert tuple(actual.columns) == tuple(
        panels.a_share_processed_catalog().get("daily_panel").required_columns
    )
    assert list(zip(actual["asset_id"], actual["timestamp"], strict=True)) == [
        ("000001", pd.Timestamp("2014-12-30")),
        ("000001", pd.Timestamp("2014-12-31")),
        ("000002", pd.Timestamp("2014-12-30")),
        ("000002", pd.Timestamp("2014-12-31")),
    ]
    assert actual["timestamp"].max() == pd.Timestamp("2014-12-31")
    assert actual.loc[
        actual["timestamp"].eq(pd.Timestamp("2014-12-31")),
        "can_enter",
    ].all()


def _daily_market_row(
    timestamp_value: str,
    *,
    asset_id: str = "000001",
    market_type: object = "A",
) -> dict[str, object]:
    timestamp = pd.Timestamp(timestamp_value)
    return {
        "market_id": "a_share",
        "asset_id": asset_id,
        "timestamp": timestamp,
        "session_date": timestamp,
        "available_at": timestamp,
        "market_type": market_type,
        "trading_status_code": 1,
        "filling_flag": "0",
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "adj_open": 10.0,
        "adj_close": 10.2,
        "prev_close": 9.9,
        "volume": 1000.0,
        "amount": 10200.0,
        "return_1d": 0.03,
        "limit_status": 0,
        "is_limit_up": False,
        "is_limit_down": False,
        "is_trading": True,
        "is_st": False,
        "is_pt": False,
        "is_a_share_scope": True,
        "eligible_base": True,
        "total_market_cap": 200000.0,
        "float_market_cap": 100000.0,
    }


def _stock_master_row(
    *,
    asset_id: str = "000001",
    listing_date: str = "2010-01-01",
    status_change_date: str,
    company_status: str,
    industry_code: str,
) -> dict[str, object]:
    return {
        "market_id": "a_share",
        "asset_id": asset_id,
        "source_id": "csmar",
        "source_asset_id": asset_id,
        "stock_name": "demo",
        "listing_date": pd.Timestamp(listing_date),
        "delisting_date": pd.NaT,
        "company_status": company_status,
        "status_change_date": pd.Timestamp(status_change_date),
        "csrc_2012_industry_code": industry_code,
        "csrc_2012_industry_name": f"industry {industry_code}",
    }


def _market_cap_row(
    *,
    asset_id: str = "000001",
    share_change_date: str = "2010-01-01",
) -> dict[str, object]:
    timestamp = pd.Timestamp(share_change_date)
    return {
        "market_id": "a_share",
        "asset_id": asset_id,
        "timestamp": timestamp,
        "share_change_date": timestamp,
        "total_shares": 1000.0,
        "a_float_shares": 500.0,
    }


def _basic_trading_calendar() -> pd.DataFrame:
    return _trading_calendar_frame(
        pd.date_range("2010-01-01", "2024-12-31", freq="D"),
        market_type=1,
    )


def _factor_panel_for_dates(dates) -> pd.DataFrame:
    rows = []
    for offset, date in enumerate(pd.to_datetime(dates), start=1):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": pd.Timestamp(date),
                "session_date": pd.Timestamp(date),
                "available_at": pd.Timestamp(date),
                "adj_open": float(100 + offset),
                "adj_close": float(100 + offset),
                "return_1d": 0.001 * offset,
                "amount": 10000.0 + offset,
                "turnover": 0.01 + offset / 100000.0,
                "market_type": 1,
                "high": float(101 + offset),
                "low": float(99 + offset),
                "prev_close": float(99 + offset),
                "total_market_cap": float(1_000_000 + offset),
                "float_market_cap": float(500_000 + offset),
                "is_st": False,
                "is_pt": False,
                "limit_status": 0,
                "listing_age_trading_days": 130 + offset,
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
            }
        )
    return pd.DataFrame(rows)


def _trading_calendar_frame(
    calendar_dates,
    *,
    market_type: int,
    is_trading_day: bool = True,
) -> pd.DataFrame:
    return _trading_calendar_rows(
        [(date, market_type, is_trading_day) for date in calendar_dates]
    )


def _trading_calendar_rows(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "source_id": "csmar",
                "timestamp": pd.Timestamp(date),
                "calendar_date": pd.Timestamp(date),
                "source_market_type": market_type,
                "is_trading_day": is_trading_day,
                "source_day_of_week": pd.Timestamp(date).dayofweek + 1,
                "source_state": "O" if is_trading_day else "C",
            }
            for date, market_type, is_trading_day in rows
        ]
    )


def _write_processed_panel_fixture(processed_root, frame: pd.DataFrame) -> None:
    spec = a_share_processed_catalog().get("daily_panel")
    write_dataset(
        _processed_artifact_root(processed_root, spec.category, spec.processed_data_id),
        _select_fixture_columns(frame, spec),
        spec=spec,
    )


def _write_minimal_panel_interim_fixture(interim_root) -> None:
    _write_interim_fixture(
        interim_root,
        "daily_market",
        pd.DataFrame([_daily_market_row("2024-06-28", market_type=1)]),
    )
    _write_interim_fixture(
        interim_root,
        "stock_master",
        pd.DataFrame(
            [
                _stock_master_row(
                    asset_id="000001",
                    listing_date="2020-01-01",
                    status_change_date="2020-01-01",
                    company_status="active",
                    industry_code="I",
                )
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [_market_cap_row(asset_id="000001", share_change_date="2020-01-01")]
        ),
    )


def _write_interim_fixture(
    interim_root,
    interim_data_id: str,
    frame: pd.DataFrame,
) -> None:
    spec = csmar_binding.a_share_csmar_interim_catalog().get(interim_data_id)
    write_dataset(
        Path(interim_root) / interim_data_id,
        _select_fixture_columns(frame, spec),
        spec=spec,
    )


def _select_fixture_columns(frame: pd.DataFrame, spec) -> pd.DataFrame:
    result = frame.copy()
    for column in spec.required_columns:
        if column not in result:
            result[column] = _fixture_default_value(column, result)
    return result.loc[:, list(spec.required_columns)].copy()


def _fixture_default_value(column: str, frame: pd.DataFrame):
    if column == "market_id":
        return "a_share"
    if column == "source_id":
        return "csmar"
    if column == "source_asset_id":
        if "asset_id" in frame:
            return frame["asset_id"].astype(str)
        return "source_asset"
    if column in {"session_date", "available_at", "timestamp"}:
        if "timestamp" in frame:
            return pd.to_datetime(frame["timestamp"])
        return pd.Timestamp("2014-01-02")
    if column in {
        "listing_date",
        "status_change_date",
        "share_change_date",
        "calendar_date",
    }:
        if "timestamp" in frame:
            return pd.to_datetime(frame["timestamp"])
        return pd.Timestamp("2010-01-01")
    if column == "delisting_date":
        return pd.NaT
    if column in {"stock_name", "company_status", "trading_status_code"}:
        return "normal"
    if column in {"filling_flag", "csrc_2012_industry_code"}:
        return "I"
    if column == "csrc_2012_industry_name":
        return "industry"
    if column in {
        "is_limit_up",
        "is_limit_down",
        "is_st",
        "is_pt",
        "is_trading",
        "is_a_share_scope",
        "eligible_base",
        "eligible",
        "can_enter",
        "can_exit",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "is_trading_day",
    }:
        return True
    if column in {"market_type", "source_market_type", "source_day_of_week"}:
        return 1
    if column == "source_state":
        return "O"
    if column == "limit_status":
        return 0
    if column in {"listing_age_days", "listing_age_trading_days"}:
        return 130
    if column in {
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_close",
        "prev_close",
        "volume",
        "amount",
        "turnover",
        "return_1d",
        "total_market_cap",
        "float_market_cap",
        "total_shares",
        "a_float_shares",
    }:
        return 1.0
    return pd.NA


def _looks_like_price_volume_factor_column(column: str) -> bool:
    prefixes = (
        "mom_",
        "rev_",
        "vol_",
        "downside_vol_",
        "upside_vol_",
        "max_return_",
        "min_return_",
        "drawdown_",
        "skew_",
        "kurt_",
        "avg_amount_",
        "turnover_",
        "turnover_chg_",
        "turnover_vol_",
        "amihud_",
        "zero_volume_ratio_",
        "amplitude_",
        "intraday_range_",
        "log_float_market_cap",
        "log_total_market_cap",
    )
    return column.startswith(prefixes)
