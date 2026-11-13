from __future__ import annotations

from pathlib import Path

from tests.helpers.import_graph import import_edges_under

SOURCE_ROOT = Path("src/quant_research_platform")


def test_query_owner_has_no_workflow_or_source_traversal(import_edges):
    for edge in import_edges_under(import_edges, SOURCE_ROOT / "interfaces/queries.py"):
        assert not edge.module.startswith(
            ("quant_research_platform.workflows", "quant_research_platform.sources")
        )


def test_a_share_pit_owner_and_lifecycle_inputs_have_one_direction(
    import_edges,
) -> None:
    factor_path = (
        SOURCE_ROOT / "research" / "factors" / "fundamental" / "calculations.py"
    )
    factor_modules = (
        edge.module for edge in import_edges_under(import_edges, factor_path)
    )
    assert all(
        not module.startswith("quant_research_platform.markets")
        for module in factor_modules
    )

    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.sources.choice.interim import first_disclosure_spec

    panel_sources = (
        catalog.processed_catalog().get("market_panel_1d").source_input_data_ids
    )
    # Industry announcements do not replace stock_master as listing truth.
    assert {"listing_status_changes", "stock_info_annual"}.isdisjoint(panel_sources)
    assert {"stock_master", "stock_info_changes"}.issubset(panel_sources)
    assert first_disclosure_spec().primary_key == (
        "asset_id",
        "report_period",
        "first_disclosure_date",
    )


def test_financial_pit_owner_converts_quarterly_flows_to_ttm():
    import pandas as pd

    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from tests.helpers.a_share_sources import _synthetic_financial_frames

    source = _synthetic_financial_frames()
    source["trading_calendar"] = pd.DataFrame(
        {
            "calendar_date": pd.bdate_range("2019-01-01", "2021-02-01"),
            "source_market_type": 4,
            "is_trading_day": True,
        }
    )
    events = producer._financial_pit_events(
        source,
        required_sources=(
            "financial_income",
            "financial_first_disclosure",
            "trading_calendar",
        ),
    )
    quarter = events.loc[
        events.asset_id.eq("000001")
        & events.metric_id.eq("operating_revenue")
        & events.report_period.eq(pd.Timestamp("2020-06-30"))
    ]
    # Current half-year + prior annual - prior half-year = 6 + 4 - 2.
    assert quarter.value.tolist() == [8.0]
    assert quarter.available_at.tolist() == [pd.Timestamp("2020-07-21")]


def test_factor_dispatcher_does_not_own_market_eligibility(import_edges):
    factor_path = SOURCE_ROOT / "research" / "factors" / "engine.py"
    assert all(
        not edge.module.startswith("quant_research_platform.markets")
        for edge in import_edges_under(import_edges, factor_path)
    )
