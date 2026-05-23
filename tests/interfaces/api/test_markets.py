from __future__ import annotations

from fastapi.testclient import TestClient

from quant_research_platform.interfaces.api.app import create_app


def test_api_lists_static_market_capabilities() -> None:
    client = TestClient(create_app(run_roots=[("platform", "reports/runs")]))

    response = client.get("/markets")

    assert response.status_code == 200
    markets = response.json()["markets"]
    assert [market["market"] for market in markets] == ["a_share", "crypto"]
    assert markets[0]["enabled"] is True
    assert markets[0]["status"] == "enabled"
    assert markets[1]["enabled"] is False
    assert markets[1]["status"] == "planned"


def test_api_returns_market_capability_detail() -> None:
    client = TestClient(create_app(run_roots=[("platform", "reports/runs")]))

    a_share = client.get("/markets/a_share/capabilities")
    crypto = client.get("/markets/crypto/capabilities")
    missing = client.get("/markets/unknown/capabilities")

    assert a_share.status_code == 200
    assert a_share.json()["market"] == "a_share"
    assert a_share.json()["enabled"] is True
    assert "local_csmar_data" in a_share.json()["capabilities"]
    assert crypto.status_code == 200
    assert crypto.json()["market"] == "crypto"
    assert crypto.json()["enabled"] is False
    assert crypto.json()["status"] == "planned"
    assert missing.status_code == 404
