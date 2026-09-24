"""Native settlement identity and vendor forecast availability are distinct."""

import json

import pandas as pd
import pytest

from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.helpers.runtime_resources import capture_test_policy
from tests.sources.tardis.test_raw import _slices
from tests.sources.tardis.test_replay import _line


@pytest.mark.parametrize("source_id", ["binance", "xoomar"])
@pytest.mark.parametrize("slice_count", [1, 144, 289])
def test_forecast_does_not_use_future_settlement_and_original_is_verified(
    tmp_path, slice_count, source_id
):
    start_ms = pd.Timestamp("2026-09-01T00:00:00Z").value // 1_000_000
    native = _line(
        1,
        "markPrice@1s",
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": start_ms + 900,
            "p": "100",
            "r": "0.05",
            "T": start_ms + 10000,
        },
    )
    slices = _slices(tmp_path / "native", native, count=slice_count)
    response = tmp_path / "official.json"
    original = json.dumps(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": start_ms + 10000,
                "fundingRate": "0.1",
                "markPrice": "200",
            }
        ]
    ).encode()
    if source_id == "xoomar":
        original = json.dumps(
            {
                "schema_id": "qrp-xoomar-settlement-extract",
                "upstream_response_sha256": "a" * 64,
                "records": [
                    {
                        "exchange": "binance",
                        "baseAsset": "BTC",
                        "slug": "btc",
                        "fundingTime": "2026-09-01T00:00:10.000Z",
                        "fundingRate": "0.1",
                        "markPrice": "200",
                    }
                ],
            }
        ).encode()
    response.write_bytes(original)
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_slices(
            tmp_path / "published",
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with open_tardis_archive(tmp_path / "published", policy) as day:
            import_funding_history(
                day,
                symbol="BTCUSDT",
                source_id=source_id,
                response_path=response,
                http_status=200,
                start_ms=start_ms,
                end_ms=start_ms + slice_count * 600_000 - 1,
                byte_budget=1024**2,
            )
            with read_funding_observations(
                day.funding_root("BTCUSDT"), symbol="BTCUSDT", resource_policy=policy
            ) as source:
                assert source.events[0].rate == 0.1
                assert source.events[0].mark_price == 200.0
                forecasts = tuple(source.forecasts)
                assert len(forecasts) == 1
                assert forecasts[0].available_at_ns == (start_ms + 1000) * 1_000_000
                assert forecasts[0].cashflows == (
                    ((start_ms + 10000) * 1_000_000, -5.0),
                )
            metadata = json.loads(
                (day.funding_root("BTCUSDT") / "response.metadata.json").read_text(
                    encoding="utf-8"
                )
            )["metadata"]["settlement_source"]
            assert metadata["source_id"] == source_id
            assert ("xoomar.com" in metadata["endpoint"]) == (source_id == "xoomar")
            preserved = day.funding_root("BTCUSDT").parent / "settlements.json"
            assert preserved.read_bytes() == original == response.read_bytes()
            preserved.write_bytes(original.replace(b"200", b"201"))
            with pytest.raises(
                ValueError, match="binance_funding_settlement_source_changed"
            ):
                with read_funding_observations(
                    day.funding_root("BTCUSDT"),
                    symbol="BTCUSDT",
                    resource_policy=policy,
                ):
                    pass
        assert policy.used_memory_bytes == 0
