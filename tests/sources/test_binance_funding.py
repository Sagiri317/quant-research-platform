from copy import deepcopy

import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
    funding_source_identity,
    read_funding_observations,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
    write_csv_snapshot,
)


def _write(raw, *, rows=(), metadata_changes=None):
    metadata = {
        "schema_id": FUNDING_SCHEMA_ID,
        "symbol": "BTCUSDT",
        "market_type": FUTURES_MARKET_TYPE,
        "valuation_currency": "USDT",
        "coverage": {"start_ns": 0, "end_ns": 100, "status": "COMPLETE"},
        "provenance": "explicit_synthetic_funding_calendar",
        "settlement_source": None,
    }
    metadata.update(metadata_changes or {})

    def validate(root):
        manifest = read_manifest(root / "response.metadata.json")
        assert manifest["metadata"] == metadata
        for _row in iter_csv_snapshot(
            root,
            expected_manifest=manifest,
            max_bytes=1024 * 1024,
        ):
            pass

    with runtime_resource_policy_scope():
        write_csv_snapshot(
            raw.parent / "funding" / "BTCUSDT",
            columns=FUNDING_COLUMNS,
            rows=rows,
            metadata=metadata,
            max_bytes=1024 * 1024,
            validate=validate,
        )


def test_funding_missing_and_explicit_zero_have_distinct_evidence(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(PortfolioTargetBlocked, match="binance_funding_input_missing"):
        funding_source_identity(
            raw.parent / "funding" / "BTCUSDT", "ledger", symbol="BTCUSDT"
        )
    _write(raw, rows=(("forecast", 0, 100, None, None, []),))
    with runtime_resource_policy_scope() as policy:
        with read_funding_observations(
            raw.parent / "funding" / "BTCUSDT", symbol="BTCUSDT", resource_policy=policy
        ) as source:
            assert source.events == () and source.coverage == (0, 100)
            forecasts = tuple(source.forecasts)
            assert len(forecasts) == 1 and forecasts[0].cashflows == ()
            original = source.identity
        assert (
            funding_source_identity(
                raw.parent / "funding" / "BTCUSDT", "ledger", symbol="BTCUSDT"
            )
            == original
        )
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_input_missing"
        ):
            funding_source_identity(
                raw.parent / "funding" / "ETHUSDT", "ledger", symbol="ETHUSDT"
            )
        assert policy.used_memory_bytes == 0
    changed = tmp_path / "changed" / "raw"
    _write(changed, rows=(("forecast", 0, 100, None, None, [[50, -1.0]]),))
    assert (
        funding_source_identity(
            changed.parent / "funding" / "BTCUSDT", "ledger", symbol="BTCUSDT"
        )
        != original
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"coverage": {"start_ns": 0, "end_ns": 100, "status": "UNPROVEN"}},
        {"provenance": ""},
    ),
)
def test_funding_unproven_coverage_never_becomes_zero(tmp_path, changes):
    raw = tmp_path / "raw"
    _write(raw, metadata_changes=deepcopy(changes))
    with pytest.raises(
        PortfolioTargetBlocked, match="binance_funding_coverage_unproven"
    ):
        funding_source_identity(
            raw.parent / "funding" / "BTCUSDT", "ledger", symbol="BTCUSDT"
        )


@pytest.mark.parametrize(
    "rows",
    (
        (("settlement", 101, None, 0.001, 100.0, None),),
        (
            ("settlement", 50, None, 0.001, 100.0, None),
            ("settlement", 50, None, 0.001, 100.0, None),
        ),
        (("forecast", 0, 100, None, None, [[101, 1.0]]),),
    ),
)
def test_funding_malformed_calendar_rejected_with_resources_released(tmp_path, rows):
    raw = tmp_path / "raw"
    _write(raw, rows=rows)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="binance_funding_input_invalid"):
            funding_source_identity(
                raw.parent / "funding" / "BTCUSDT", "ledger", symbol="BTCUSDT"
            )
        assert policy.used_memory_bytes == 0


def test_funding_changed_during_consumption_is_rejected_and_primary_preserved(tmp_path):
    raw = tmp_path / "raw"
    row = ("forecast", 0, 100, None, None, [[50, 0.0]])
    _write(raw, rows=(row,))
    changed = tmp_path / "changed" / "raw"
    _write(changed, rows=(("forecast", 0, 100, None, None, [[50, 1.0]]),))
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="raw_csv_metadata_invalid"):
            with read_funding_observations(
                raw.parent / "funding" / "BTCUSDT", symbol="BTCUSDT"
            ) as source:
                assert tuple(source.forecasts)
                for name in ("response.csv", "response.metadata.json"):
                    (raw.parent / "funding" / "BTCUSDT" / name).write_bytes(
                        (changed.parent / "funding" / "BTCUSDT" / name).read_bytes()
                    )
        primary = KeyboardInterrupt("primary")
        with pytest.raises(KeyboardInterrupt) as raised:
            with read_funding_observations(
                raw.parent / "funding" / "BTCUSDT", symbol="BTCUSDT"
            ):
                raise primary
        assert raised.value is primary
        assert policy.used_memory_bytes == 0
