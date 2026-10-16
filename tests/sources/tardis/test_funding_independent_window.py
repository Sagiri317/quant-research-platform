"""Settlement response coverage does not inherit market archive boundaries."""

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.sources.tardis.test_funding import _publish_funding_archive


@pytest.mark.parametrize("offset,length", [(-1000, 700000), (5000, 10000)])
def test_settlement_window_and_output_are_independent(tmp_path, offset, length):
    destination = _publish_funding_archive(
        tmp_path, "2026-09-01", include_funding=False
    )
    origin = pd.Timestamp("2026-09-01").value // 1000000
    start, end = origin + offset, origin + offset + length
    response = tmp_path / "settlements.json"
    response.write_text(
        json.dumps(
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": origin + 10000,
                    "fundingRate": "0.1",
                    "markPrice": "200",
                }
            ]
        )
    )
    output = tmp_path / "independent"
    with open_tardis_archive(destination) as archive:
        kwargs = dict(
            symbol="BTCUSDT",
            source_id="binance",
            response_path=response,
            http_status=200,
            start_ms=start,
            end_ms=end,
            output_root=output,
        )
        with pytest.raises(PortfolioTargetBlocked, match="coverage_unproven"):
            import_funding_history(archive, **kwargs, request_limit=1)
        assert not output.exists()
        import_funding_history(archive, **kwargs, request_limit=1000)
    assert not (destination / "funding").exists()
    assert (output / "settlements.json").read_bytes() == response.read_bytes()
    with read_funding_observations(
        (output / "observations",), symbol="BTCUSDT"
    ) as source:
        assert source.coverage == ((start * 1000000, (end + 1) * 1000000 - 1),)
        assert source.events[0].timestamp_ns == (origin + 10000) * 1000000
        forecasts = tuple(source.forecasts)
        assert forecasts[0].available_at_ns == (origin + 1000) * 1000000
        assert forecasts[0].cashflows == (((origin + 10000) * 1000000, -5.0),)


def test_multiple_forecast_archives_can_share_one_settlement_response(tmp_path):
    archives = [
        _publish_funding_archive(tmp_path, date, include_funding=False)
        for date in ("2026-09-01", "2026-09-02")
    ]
    origin = pd.Timestamp("2026-09-01").value // 1000000
    response = tmp_path / "response.json"
    response.write_bytes(b"[]")
    output = tmp_path / "funding"
    with open_tardis_archive(
        tmp_path, archive_paths=tuple(path.name for path in archives)
    ) as source:
        import_funding_history(
            source,
            symbol="BTCUSDT",
            source_id="binance",
            response_path=response,
            http_status=200,
            start_ms=origin,
            end_ms=origin + 2 * 86400000 - 1,
            request_limit=1000,
            output_root=output,
        )
    with read_funding_observations(
        (output / "observations",), symbol="BTCUSDT"
    ) as source:
        assert source.events == ()
        assert len(tuple(source.forecasts)) == 2
