from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.factors import (
    build_factors,
    price_volume,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    calculate_literature_price_volume_factors,
    exact_literature_price_volume_factor_input_columns,
    expected_literature_price_volume_factor_columns,
    largest_literature_price_volume_lookback_days,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    build_labels,
    forward_returns,
)
from quant_research_platform.markets.implementations.a_share.masks import (
    build as build_masks,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    contexts,
    panels,
)
from quant_research_platform.markets.implementations.a_share.universes import (
    build as build_universes,
)
from quant_research_platform.storage.layout import (
    processed_data_manifest_path,
    processed_data_parquet_root,
)
from quant_research_platform.storage.parquet import read_partitioned_parquet_dataset


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


def test_literature_factor_lookback_infers_requested_factor_window() -> None:
    assert largest_literature_price_volume_lookback_days(("mom_3d",)) == 3
    assert largest_literature_price_volume_lookback_days(()) == 0


def _panel() -> pd.DataFrame:
    rows = []
    for day in range(1, 6):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "stock_code": "000001",
                "timestamp": pd.Timestamp(f"2014-01-0{day}"),
                "session_date": pd.Timestamp(f"2014-01-0{day}"),
                "available_at": pd.Timestamp(f"2014-01-0{day}"),
                "trade_date": pd.Timestamp(f"2014-01-0{day}"),
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
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
                "high": float(day + 11),
                "low": float(day + 9),
                "prev_close": float(day + 9),
            }
        )
    return pd.DataFrame(rows)


def test_a_share_price_volume_factor_contract_is_56_columns() -> None:
    columns = expected_literature_price_volume_factor_columns()

    assert len(columns) == 56
    assert "mom_20d" in columns
    assert "forward_return_20d" not in columns
    assert expected_literature_price_volume_factor_columns(factor_set="core")
    for factor_id in columns:
        assert exact_literature_price_volume_factor_input_columns(factor_id)
    with pytest.raises(ValueError, match="invalid_factor_set"):
        expected_literature_price_volume_factor_columns(
            factor_set="core" + "_long_period"
        )


def test_a_share_factor_and_label_helpers_use_past_and_forward_domains() -> None:
    panel = _panel()
    factored = price_volume.calculate_price_volume_factors(
        panel,
        windows=(2,),
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
    panel_path = (
        processed_data_parquet_root(processed_root, "panels", "daily_panel")
        / "year=2014"
        / "part-000.parquet"
    )
    panel_path.parent.mkdir(parents=True)
    panel.to_parquet(panel_path, index=False)
    _write_interim_fixture(
        interim_root,
        "trading_calendar",
        _trading_calendar_frame(panel["trade_date"], market_type=1),
    )
    factored = build_factors.calculate_panel_factors(panel, windows=(2,))
    labels = build_labels.calculate_panel_labels(panel)
    factor_path, factor_rows = build_factors.calculate_price_volume_factor_full(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    label_path, label_rows = build_labels.write_forward_labels(
        labels,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    factor_manifest = processed_data_manifest_path(
        processed_root, "factors", "price_volume_56"
    )
    label_manifest = processed_data_manifest_path(
        processed_root, "labels", "forward_returns"
    )

    assert "mom_2d" in factored.columns
    assert not any(column.startswith("forward_return_") for column in factored)
    assert "forward_return_1d" in labels.columns
    assert factor_rows == len(panel)
    assert label_rows == len(panel)
    assert factor_path.as_posix().endswith(
        "processed/a_share/factors/price_volume_56/data"
    )
    assert label_path.as_posix().endswith(
        "processed/a_share/labels/forward_returns/data"
    )
    factor_payload = json.loads(factor_manifest.read_text(encoding="utf-8"))
    label_payload = json.loads(label_manifest.read_text(encoding="utf-8"))
    assert factor_payload["schema_version"] == "0.1"
    assert label_payload["schema_version"] == "0.1"


def test_a_share_factor_full_passes_spec_columns_to_literature_helper(
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
        _trading_calendar_frame(panel["trade_date"], market_type=1),
    )
    captured: dict[str, object] = {}

    def fake_literature_helper(frame, **kwargs):
        captured["expected_columns"] = kwargs.get("expected_columns")
        result = frame.copy()
        for column in expected_literature_price_volume_factor_columns():
            result[column] = 0.0
        return result

    monkeypatch.setattr(
        build_factors,
        "calculate_literature_price_volume_factors",
        fake_literature_helper,
    )

    output_root, rows = build_factors.calculate_price_volume_factor_full(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = read_partitioned_parquet_dataset(output_root, years=(2014,))

    assert captured["expected_columns"] == build_factors._factor_output_columns()
    assert set(captured["expected_columns"]) == set(
        expected_literature_price_volume_factor_columns()
    )
    assert rows == len(output)
    assert set(expected_literature_price_volume_factor_columns()).issubset(
        output.columns
    )


def test_literature_helper_with_contract_columns_avoids_extra_factor_outputs() -> None:
    contract_columns = expected_literature_price_volume_factor_columns()
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=260))

    result = calculate_literature_price_volume_factors(
        panel,
        expected_columns=contract_columns,
    )
    slim = build_factors.select_slim_factor_columns(result)
    extra_factor_columns = {
        column
        for column in result.columns
        if _looks_like_literature_factor_column(column)
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
    assert tuple(slim.columns) == tuple(
        build_factors.a_share_processed_catalog()
        .get("price_volume_56")
        .required_columns
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

    output_root, rows = build_factors.calculate_price_volume_factor_full(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2020,
        end_year=2020,
    )
    output = read_partitioned_parquet_dataset(output_root, years=(2020,))

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

    output_root, rows = build_factors.calculate_price_volume_factor_full(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = read_partitioned_parquet_dataset(output_root, years=(2014,))

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
    no_warmup_output_root, _ = build_factors.calculate_price_volume_factor_full(
        interim_root=no_warmup_interim,
        processed_root=no_warmup_root,
        start_year=2014,
        end_year=2014,
    )
    no_warmup_output = read_partitioned_parquet_dataset(
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
    warmup_output_root, _ = build_factors.calculate_price_volume_factor_full(
        interim_root=warmup_interim,
        processed_root=warmup_root,
        start_year=2014,
        end_year=2014,
    )
    warmup_output = read_partitioned_parquet_dataset(
        warmup_output_root,
        years=(2014,),
    )

    assert warmup_output["timestamp"].min() >= pd.Timestamp("2014-01-01")
    assert warmup_output["timestamp"].max() <= pd.Timestamp("2014-12-31")
    assert (
        warmup_output["mom_252d"].isna().sum()
        < no_warmup_output["mom_252d"].isna().sum()
    )


def test_a_share_factor_builder_requires_trading_calendar(tmp_path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    interim_root = tmp_path / "data" / "interim" / "csmar"
    panel = _factor_panel_for_dates(pd.bdate_range("2020-01-01", periods=3))
    _write_processed_panel_fixture(processed_root, panel)

    with pytest.raises(ValueError, match="missing_trading_calendar_interim"):
        build_factors.calculate_price_volume_factor_full(
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
        build_factors.calculate_price_volume_factor_full(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2020,
            end_year=2020,
        )


def test_a_share_label_builder_allows_missing_future_lookahead_year(tmp_path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    panel = _factor_panel_for_dates(pd.bdate_range("2014-12-29", "2014-12-31"))
    labels = build_labels.calculate_panel_labels(panel)

    label_path, label_rows = build_labels.write_forward_labels(
        labels,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    output = read_partitioned_parquet_dataset(label_path, years=(2014,))

    assert label_rows == len(output)
    assert output["timestamp"].dt.year.tolist() == [2014, 2014, 2014]
    assert output["forward_return_1d"].isna().sum() == 2


def test_a_share_label_builder_uses_future_lookahead_when_available(
    tmp_path,
) -> None:
    panel_without_future = _factor_panel_for_dates(
        pd.bdate_range("2014-12-29", "2014-12-31")
    )
    labels_without_future = build_labels.calculate_panel_labels(panel_without_future)

    panel_with_future = _factor_panel_for_dates(
        pd.bdate_range("2014-12-29", "2015-01-02")
    )
    labels_with_future = build_labels.calculate_panel_labels(panel_with_future)

    tail_without_future = labels_without_future.loc[
        labels_without_future["trade_date"].dt.year == 2014,
        "forward_return_1d",
    ]
    tail_with_future = labels_with_future.loc[
        labels_with_future["trade_date"].dt.year == 2014,
        "forward_return_1d",
    ]
    assert tail_with_future.isna().sum() < tail_without_future.isna().sum()


def test_a_share_factor_builder_does_not_depend_on_raw_calendar() -> None:
    source = Path(build_factors.__file__).read_text(encoding="utf-8")

    forbidden_terms = (
        "data/raw/csmar/trading_calendar",
        'normalize_private_export_frame(raw_frame, "trading_calendar")',
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

    context_path, context_rows = contexts.write_contexts(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    universe_path, universe_rows = build_universes.write_universe(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    mask_outputs = build_masks.write_masks(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )

    assert context_rows == len(panel)
    assert universe_rows == len(panel)
    assert context_path.as_posix().endswith(
        "processed/a_share/contexts/daily_context/data"
    )
    assert universe_path.as_posix().endswith(
        "processed/a_share/universes/a_share_default/data"
    )
    assert sorted(mask_outputs) == ["as_of", "evaluation", "execution"]
    assert mask_outputs["execution"][0].as_posix().endswith(
        "processed/a_share/masks/execution/a_share_execution/data"
    )
    assert json.loads(
        processed_data_manifest_path(
            processed_root, "masks/execution", "a_share_execution"
        ).read_text(encoding="utf-8")
    )["schema_version"] == "0.1"


def test_a_share_panel_joins_stock_master_as_of_trade_date() -> None:
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
    assert not panel.duplicated(["stock_code", "trade_date"]).any()
    by_date = panel.set_index("trade_date").sort_index()
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


def test_a_share_panel_stock_master_duplicate_effective_date_keeps_last() -> None:
    daily = pd.DataFrame([_daily_market_row("2020-07-02")])
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="older_duplicate",
                industry_code="I1",
            ),
            _stock_master_row(
                status_change_date="2020-07-01",
                company_status="newer_duplicate",
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
    assert not panel.duplicated(["stock_code", "trade_date"]).any()
    assert panel.loc[0, "company_status"] == "newer_duplicate"
    assert panel.loc[0, "csrc_2012_industry_code"] == "I2"


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
    assert not panel.duplicated(["stock_code", "trade_date"]).any()
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
    trade_dates = pd.date_range("2020-01-01", periods=3, freq="D")
    daily = pd.DataFrame(
        [_daily_market_row(str(date.date()), market_type=1) for date in trade_dates]
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
        trading_calendar=_trading_calendar_frame(trade_dates, market_type=1),
        min_listing_trading_days=3,
    )

    assert panel["listing_age_trading_days"].tolist() == [1, 2, 3]
    assert panel["eligible_after_listing_age"].tolist() == [False, False, True]


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
            _daily_market_row("2020-01-01", stock_code="000001", market_type=1),
            _daily_market_row("2020-01-01", stock_code="000002", market_type=32),
        ]
    )
    stock_master = pd.DataFrame(
        [
            _stock_master_row(
                stock_code="000001",
                listing_date="2020-01-01",
                status_change_date="2020-01-01",
                company_status="normal",
                industry_code="I1",
            ),
            _stock_master_row(
                stock_code="000002",
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
                _market_cap_row(stock_code="000001", share_change_date="2020-01-01"),
                _market_cap_row(stock_code="000002", share_change_date="2020-01-01"),
            ]
        ),
        trading_calendar=calendar,
        min_listing_trading_days=1,
    )

    ages = panel.set_index("stock_code")["listing_age_trading_days"].to_dict()
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
        panels.build_panel_from_interim(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2024,
            end_year=2024,
            end_date="2024-06-30",
            min_listing_trading_days=1,
        )


def test_a_share_panel_from_interim_preserves_invalid_calendar_error(tmp_path) -> None:
    interim_root = tmp_path / "data" / "interim" / "csmar"
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_minimal_panel_interim_fixture(interim_root)
    data_root = interim_root / "trading_calendar" / "data" / "year=2024"
    data_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "calendar_date": [pd.Timestamp("2024-06-28")],
            "is_trading_day": [True],
        }
    ).to_parquet(data_root / "part-000.parquet", index=False)

    with pytest.raises(ValueError, match="calendar_missing_normalized_columns"):
        panels.build_panel_from_interim(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=2024,
            end_year=2024,
            end_date="2024-06-30",
            min_listing_trading_days=1,
        )


def test_a_share_panel_builder_does_not_depend_on_raw_calendar() -> None:
    source = Path(panels.__file__).read_text(encoding="utf-8")

    forbidden_terms = (
        "data/raw/csmar/trading_calendar",
        'normalize_private_export_frame(raw_frame, "trading_calendar")',
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
                {
                    "stock_code": "000001",
                    "stock_name": "demo",
                    "listing_date": pd.Timestamp("2020-01-01"),
                    "delisting_date": pd.NaT,
                    "company_status": "active",
                    "status_change_date": pd.Timestamp("2020-01-01"),
                    "csrc_2012_industry_code": "I",
                    "csrc_2012_industry_name": "industry",
                }
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [
                {
                    "stock_code": "000001",
                    "share_change_date": pd.Timestamp("2020-01-01"),
                    "total_shares": 1000.0,
                    "a_float_shares": 500.0,
                }
            ]
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

    output_root, rows = panels.build_panel_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2024,
        end_year=2024,
        end_date="2024-06-30",
        min_listing_trading_days=1,
    )
    written = read_partitioned_parquet_dataset(output_root, years=(2024,))

    assert rows == 1
    assert written["trade_date"].tolist() == [pd.Timestamp("2024-06-28")]


def _daily_market_row(
    trade_date: str,
    *,
    stock_code: str = "000001",
    market_type: object = "A",
) -> dict[str, object]:
    timestamp = pd.Timestamp(trade_date)
    return {
        "market_id": "a_share",
        "asset_id": stock_code,
        "timestamp": timestamp,
        "session_date": timestamp,
        "available_at": timestamp,
        "stock_code": stock_code,
        "trade_date": timestamp,
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
    stock_code: str = "000001",
    listing_date: str = "2010-01-01",
    status_change_date: str,
    company_status: str,
    industry_code: str,
) -> dict[str, object]:
    return {
        "stock_code": stock_code,
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
    stock_code: str = "000001",
    share_change_date: str = "2010-01-01",
) -> dict[str, object]:
    return {
        "stock_code": stock_code,
        "share_change_date": pd.Timestamp(share_change_date),
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
                "stock_code": "000001",
                "timestamp": pd.Timestamp(date),
                "session_date": pd.Timestamp(date),
                "available_at": pd.Timestamp(date),
                "trade_date": pd.Timestamp(date),
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
    data_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    result = frame.copy()
    result["_fixture_year"] = pd.to_datetime(result["timestamp"]).dt.year
    for year, year_frame in result.groupby("_fixture_year"):
        partition_root = data_root / f"year={int(year)}"
        partition_root.mkdir(parents=True)
        year_frame.drop(columns=["_fixture_year"]).to_parquet(
            partition_root / "part-000.parquet",
            index=False,
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
                {
                    "stock_code": "000001",
                    "stock_name": "demo",
                    "listing_date": pd.Timestamp("2020-01-01"),
                    "delisting_date": pd.NaT,
                    "company_status": "active",
                    "status_change_date": pd.Timestamp("2020-01-01"),
                    "csrc_2012_industry_code": "I",
                    "csrc_2012_industry_name": "industry",
                }
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [
                {
                    "stock_code": "000001",
                    "share_change_date": pd.Timestamp("2020-01-01"),
                    "total_shares": 1000.0,
                    "a_float_shares": 500.0,
                }
            ]
        ),
    )


def _write_interim_fixture(
    interim_root,
    interim_data_id: str,
    frame: pd.DataFrame,
) -> None:
    result = frame.copy()
    date_column = next(
        column
        for column in (
            "calendar_date",
            "timestamp",
            "trade_date",
            "listing_date",
            "share_change_date",
        )
        if column in result.columns
    )
    result["_fixture_year"] = pd.to_datetime(result[date_column]).dt.year
    for year, year_frame in result.groupby("_fixture_year"):
        data_root = interim_root / interim_data_id / "data" / f"year={int(year)}"
        data_root.mkdir(parents=True)
        year_frame.drop(columns=["_fixture_year"]).to_parquet(
            data_root / "part-000.parquet",
            index=False,
        )


def _looks_like_literature_factor_column(column: str) -> bool:
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
