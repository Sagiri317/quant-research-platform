"""Independent imported funding reaches both public native ledger consumers."""

import json
from dataclasses import asdict

import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.helpers.binance_hft import write_binance_hft_run, write_hft_funding_fixture
from tests.helpers.tardis import _line, _slices
from tests.markets.crypto import test_direct_portfolio

explicit_models = test_direct_portfolio.explicit_models


@pytest.fixture
def settlement_rate():
    return "0.1"


@pytest.fixture
def independent_inputs(tmp_path, settlement_rate):
    native = _line(
        1,
        "markPrice",
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": 900,
            "p": "100",
            "r": "0.05",
            "T": 10000,
        },
    ).replace(b"2026-09-01T", b"1970-01-01T")
    archive = tmp_path / "archive"
    import_slices(
        archive, date="1970-01-01", slices=_slices(tmp_path / "native", native)
    )
    response = tmp_path / "official.json"
    response.write_text(
        json.dumps(
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": 8750,
                    "fundingRate": settlement_rate,
                    "markPrice": "200",
                }
            ]
        )
    )
    funding = tmp_path / "independent-funding"
    with open_tardis_archive(archive) as source:
        import_funding_history(
            source,
            symbol="BTCUSDT",
            source_id="binance",
            response_path=response,
            http_status=200,
            start_ms=0,
            end_ms=10000,
            request_limit=1000,
            output_root=funding,
        )
    raw = tmp_path / "market" / "raw"
    write_binance_hft_run(raw, count=90, maker_liquidity=True)
    # A valid but different archive-owned source must never override selection.
    write_hft_funding_fixture(raw)
    window = {"start": "1970-01-01T00:00:08.600", "end": "1970-01-01T00:00:09"}
    panel = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "panel"),
        }
    )
    return {
        "source": str(raw),
        "panel": str(panel),
        "inputs": str(panel),
        "prediction_window": window,
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "execution_mode": "continuous",
        "funding_roots": [str(funding / "observations")],
        "capital": asdict(
            PortfolioCapital("independent", 1000.0, "USDT", "linear_perpetual")
        ),
        "output": str(tmp_path / "ledger"),
    }


def _config(kind, independent_inputs, explicit_models):
    config = dict(independent_inputs)
    if kind == "maker":
        config.update(
            models=[],
            calibration=None,
            execution=asdict(portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0)),
        )
    else:
        inputs, _chronology, _validation, final = explicit_models
        config.update(
            inputs=str(inputs),
            predictions=final[0]["predictions"],
            execution=asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
        )
    return config


@pytest.mark.parametrize("settlement_rate", ["0.1", None])
@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_imported_funding_is_consumed_without_relocation(
    kind, independent_inputs, request, settlement_rate
):
    models = request.getfixturevalue("explicit_models") if kind == "taker" else None
    config = _config(kind, independent_inputs, models)
    output = getattr(portfolio, "backtest_" + kind)(config)
    artifact = read_portfolio_ledger_artifact(output)
    summary = summarize_crypto_ledger(artifact, artifact.iter_chunks())
    economics = summary["economics"]
    if kind == "maker" and settlement_rate is None:
        assert economics["funding"] is None
        assert economics["nav"] is None
        assert economics["account_path_status"] == "PARTIAL"
        assert summary["execution"]["account_stop_ns"] == 8_750_000_000
        assert economics["funding_reason"] == "portfolio_required_funding_missing"
    elif kind == "maker":
        # One long unit at settlement: -1 asset * 0.1 * 200 USDT/asset.
        assert economics["funding"] == -20.0
        assert economics["fill_events"] == 2
    else:
        # Missing settlement amounts do not contaminate a known flat account.
        assert economics["funding"] == 0.0
        assert economics["nav"] == 1000.0
    assert artifact.metadata["funding_roots"] == config["funding_roots"]
    qualification = summary["funding_qualification"]["sources"]
    assert len(qualification) == 1
    assert qualification[0]["verification"] == "official_response_validated"
    assert qualification[0]["settlement_source"]["settlement_count"] == 1
    assert qualification[0]["coverage"]["end_ns"] == 10_001_000_000 - 1


@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_selected_corrupt_funding_never_falls_back(
    tmp_path, kind, independent_inputs, request
):
    models = request.getfixturevalue("explicit_models") if kind == "taker" else None
    config = _config(kind, independent_inputs, models)
    # The original response is part of the native source contract, not a copied CSV.
    (tmp_path / "independent-funding" / "settlements.json").write_text("[]")
    with pytest.raises(ValueError, match="binance_funding_settlement_source_invalid"):
        getattr(portfolio, "backtest_" + kind)(config)
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("roots", [None, [], "one-root", [1], [""], ["a", "a"]])
@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_explicit_funding_roots_are_strict_json_lists(tmp_path, roots, kind):
    config = dict.fromkeys(portfolio._BACKTEST_KEYS)
    config.update(asset_id="btc_usdt", symbol="BTCUSDT", funding_roots=roots)
    config.update(
        {"models": [], "calibration": None}
        if kind == "maker"
        else {"predictions": "unused"}
    )
    with pytest.raises(ValueError, match="crypto_funding_roots_invalid"):
        getattr(portfolio, "backtest_" + kind)(config)
