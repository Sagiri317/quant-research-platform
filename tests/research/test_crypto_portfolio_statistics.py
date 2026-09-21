import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    _record_execution_result,
)
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    research_profile,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_RESEARCH_DATASET_ID,
    protocol,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.software_identity import (
    production_software_identity,
)
from quant_research_platform.workflows.research import _target_envelope_for_spec
from tests.helpers.research_configuration import research_configuration


@pytest.mark.parametrize("quantity_scale", (1.0, 2.0**-50))
@pytest.mark.parametrize("missing_boundary", (False, True))
def test_sparse_published_ledgers_project_utc_days_gap_and_endpoint_inventory(
    tmp_path, missing_boundary, quantity_scale
):
    plan = {
        "folds": [
            {
                "fold_id": "first",
                "train": {"start": "2023-12-01T00:00:00", "end": "2023-12-10T00:00:00"},
                "validation": {
                    "start": "2023-12-11T00:00:00",
                    "end": "2023-12-31T00:00:00",
                },
                "development_test": {
                    "start": "2024-01-01T00:00:00",
                    "end": "2024-01-02T00:00:00",
                },
            },
            {
                "fold_id": "second",
                "train": {"start": "2023-12-02T00:00:00", "end": "2024-01-01T00:00:00"},
                "validation": {
                    "start": "2024-01-01T00:00:00.100000",
                    "end": "2024-01-02T12:00:00",
                },
                "development_test": {
                    "start": "2024-01-03T00:00:00",
                    "end": "2024-01-06T00:00:00",
                },
            },
        ],
        "final_holdout": {"start": "2024-01-07T00:00:00", "end": "2024-01-08T00:00:00"},
        "initial_capital": 1000.0,
        "valuation_currency": "USDT",
        "exposure_limits": {
            asset: {
                "max_order_quantity": quantity_scale,
                "max_position_quantity": 2.0 * quantity_scale,
            }
            for asset in ("btc_usdt", "eth_usdt")
        },
        "execution": {
            "taker_fee_bps": 0.0,
            "latency_ms": 0,
            "markout_horizon_ms": 500,
            "latency_scenarios_ms": [0, 100, 200],
            "fee_scenarios_bps": [0.0, 1.0],
            "additional_spread_scenarios_bps": [0.0, 1.0],
            "maker": {
                "maker_fee_bps": 0.0,
                "latency_ms": 0,
                "min_calibration_samples": 1,
                "inventory_target": 0.0,
                "inventory_skew": 0.0,
            },
        },
    }
    configuration = research_configuration("crypto", "baseline")
    configuration["research_plan"] = plan
    bound = replace(
        protocol(research_plan=plan),
        configuration_json=json.dumps(configuration),
    )
    profile = research_profile(CRYPTO_PROFILE_ID)
    layout = StorageLayout(tmp_path)
    start = pd.Timestamp("2024-01-01T00:00:00").value
    day = pd.Timedelta(days=1).value
    window = TimeWindow("2024-01-01T00:00:00", "2024-01-06T00:00:00")
    research_window = {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-06T00:00:00",
        "access": "development",
    }
    roots, identities = {}, {}
    with runtime_resource_policy_scope() as policy:
        for candidate in bound.portfolio_candidates():
            capital = PortfolioCapital(
                candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
            )
            envelope = build_business_manifest_envelope(
                logical_artifact_id=candidate.target_id,
                schema_identity="a" * 64,
                protocol_or_spec_identity="b" * 64,
                software_identity="c" * 64,
                research_window=research_window,
                upstream_identities={},
                semantic_metadata={"portfolio_capital": asdict(capital)},
            )

            def produce(emit, capital=capital, candidate=candidate):
                economic = PortfolioEconomics(capital, candidate.asset_id, emit)

                def mark(offset, price, name=None, number=None):
                    diagnostics = (
                        ()
                        if name is None
                        else (
                            PortfolioDiagnostic(
                                candidate.asset_id, None, name, number=number
                            ),
                        )
                    )
                    economic.mark(
                        start + offset, price / quantity_scale, diagnostics=diagnostics
                    )

                mark(0, 100.0, "oos_window_start", 0.0)
                economic.fill(
                    timestamp_ns=start,
                    order_id="first",
                    quantity=quantity_scale,
                    price=100.0 / quantity_scale,
                    mark_price=100.0 / quantity_scale,
                    fee=0.0,
                )
                mark(day, 200.0, "oos_window_end", 0.0)
                economic.funding(
                    FundingEvent(start + day + day // 2, -0.5, 200.0 / quantity_scale)
                )
                mark(2 * day, 500.0, "oos_window_start", 1.0)
                mark(
                    3 * day,
                    500.0,
                    None if missing_boundary else "utc_day_boundary",
                    None if missing_boundary else float((start + 3 * day) // day),
                )
                mark(
                    4 * day, 500.0, "utc_day_boundary", float((start + 4 * day) // day)
                )
                mark(5 * day, 350.0)
                economic.fill(
                    timestamp_ns=start + 5 * day,
                    order_id="terminal",
                    quantity=quantity_scale,
                    price=350.0 / quantity_scale,
                    mark_price=350.0 / quantity_scale,
                    fee=0.0,
                )
                mark(5 * day, 350.0, "oos_window_end", 1.0)
                _record_execution_result(
                    economic,
                    {
                        "simulation": "sparse_ledger_fixture",
                        "net_pnl": 350.0,
                        "created_orders": 2,
                        "filled_orders": 2,
                        "selection_status": "COMPLETE",
                    },
                    start + 5 * day,
                    350.0 / quantity_scale,
                )

            roots[candidate.target_id] = write_portfolio_ledger_artifact(
                layout.semantic_artifact_root("crypto", candidate.target_id),
                envelope=envelope,
                produce=produce,
            )
            identities[candidate.target_id] = envelope.instance_identity
        target = f"{HFT_RESEARCH_DATASET_ID}_portfolio_statistics"
        spec = next(
            spec
            for spec in bound.target_specs("research run")
            if spec.target_id == target
        )
        envelope = _target_envelope_for_spec(
            profile=profile,
            protocol=bound,
            target_spec=spec,
            data_spec=None,
            research_window=research_window,
            upstream_identities=identities,
            software_identity=production_software_identity(),
            raw_content_identity=None,
        )
        owned = profile.producer_bindings[target].producer(
            layout=layout.with_target_roots(roots),
            protocol_manifest=bound.manifest(),
            time_window=window,
            upstream_identities=identities,
            resource_policy=policy,
        )
        try:
            root = write_statistics_artifact(
                layout.semantic_artifact_root("crypto", target),
                envelope=envelope,
                tables=owned.value,
                table_group="crypto_portfolio",
            )
        finally:
            owned.release(owner="research.statistics.output")
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with (
            artifact.table("crypto_portfolio_statistics.parquet") as statistics,
            artifact.table("crypto_portfolio_daily.parquet") as daily,
            artifact.table("crypto_portfolio_intervals.parquet") as intervals,
        ):
            for candidate in bound.portfolio_candidates():
                metrics = statistics.loc[
                    statistics.candidate_id.eq(candidate.candidate_id)
                ].set_index("metric_id")
                assert metrics.loc["terminal_nav", "value"] == 1350.0
                assert metrics.loc["continuous_max_drawdown", "value"] == pytest.approx(
                    -0.1
                )
                assert metrics.loc["gap_net_pnl", "value"] == 400.0
                assert metrics.loc["oos_net_pnl", "value"] == -50.0
                assert (
                    metrics.loc["oos_inventory_max_abs", "value"]
                    == 2.0 * quantity_scale
                )
                assert metrics.loc["oos_inventory_mean", "value"] == pytest.approx(
                    quantity_scale, abs=0.0
                )
                assert (
                    metrics.loc["oos_inventory_time_at_limit_fraction", "value"] == 0.0
                )
                rows = daily.loc[daily.candidate_id.eq(candidate.candidate_id)]
                assert len(rows) == 4 and rows.benchmark_return.eq(0).all()
                assert rows.iloc[-1].start_nav == 1500.0
                assert rows.iloc[-1].net_return == -0.1
                if missing_boundary:
                    assert rows.status.eq("DEFINED").sum() == 2
                    assert metrics.loc["sharpe", "status"] == "UNDEFINED"
                    assert metrics.loc["annualized_volatility", "status"] == "UNDEFINED"
                    assert (
                        metrics.loc["sharpe", "reason"]
                        == "planned_utc_day_valuation_undefined"
                    )
                else:
                    assert rows.net_return.to_numpy() == pytest.approx(
                        [0.1, 0.0, 0.0, -0.1]
                    )
                    assert metrics.loc["sharpe", "value"] == pytest.approx(0.0)
                    assert metrics.loc[
                        "annualized_volatility", "value"
                    ] == pytest.approx(np.sqrt(0.02 / 3 * 365))
                gap = intervals.loc[
                    intervals.candidate_id.eq(candidate.candidate_id)
                    & intervals.kind.eq("gap")
                ].iloc[0]
                assert gap.funding == 100.0 and gap.net_pnl == 400.0
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("scale", (1.0, 2.0**-50))
@pytest.mark.parametrize("direction", (-1.0, 1.0))
@pytest.mark.parametrize(
    "fraction,expected_fraction",
    ((0.0, 0.0), (0.5, 0.0), (float.fromhex("0x1.fffffffffffffp-1"), 0.0), (1.0, 1.0)),
)
def test_inventory_limit_duration_has_exact_execution_boundary(
    tmp_path, scale, direction, fraction, expected_fraction
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _crypto_ledger_observations,
    )

    capital = PortfolioCapital("limit", 1000.0, "USDT", "linear_perpetual")
    window = TimeWindow("1970-01-01T00:00:00", "1970-01-01T00:00:10")
    duration = 10_000_000_000
    envelope = build_business_manifest_envelope(
        logical_artifact_id="limit",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def produce(emit):
        economics = PortfolioEconomics(capital, "btc_usdt", emit)
        economics.mark(0, 1.0 / scale)
        if fraction:
            economics.fill(
                timestamp_ns=0,
                order_id="open",
                quantity=direction * fraction * scale,
                price=1.0 / scale,
                mark_price=1.0 / scale,
                fee=0.0,
            )
        economics.mark(duration, 1.0 / scale)

    with runtime_resource_policy_scope() as policy:
        root = write_portfolio_ledger_artifact(
            tmp_path / "limit", envelope=envelope, produce=produce
        )
        artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
        _daily, inventory, _drawdown = _crypto_ledger_observations(
            artifact, (("test", window),), scale
        )
        assert inventory["oos"]["duration"] == duration
        assert inventory["oos"]["at_limit"] / duration == expected_fraction
        assert inventory["gap"]["duration"] == 0
        assert policy.used_memory_bytes == 0
