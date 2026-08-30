from __future__ import annotations

import json
from pathlib import Path

from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MULTI_ASSET_SCOPES,
)
from quant_research_platform.workflows.unified_research import (
    run_unified_research_smoke,
)
from tests.helpers.binance_hft import write_binance_hft_run


def test_unified_smoke_consumes_persisted_market_statistics_and_ledger_owners(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_binance_hft_run(raw_root)

    payload = run_unified_research_smoke(
        raw_crypto_root=raw_root,
        output_root=output_root,
    )

    assert payload["status"] == {
        "implementation": "IMPLEMENTATION_COMPLETE",
        "data": "REAL_DATA_PENDING",
        "holdout": "HOLDOUT_LOCKED",
        "readiness": "READY_FOR_REAL_DATA",
    }
    assert payload["holdout_accessed"] is False
    assert payload["protocols"]["a_share"]["max_label_lookahead_steps"] == 20
    assert payload["protocols"]["crypto"]["max_label_lookahead_steps"] == 5
    assert payload["crypto"]["data_quality"]["panel_rows"] == 122
    assert set(payload["crypto"]["mode_evaluations"]) == set(HFT_MULTI_ASSET_SCOPES)
    assert set(payload["crypto"]["execution"]["multi_asset_execution"]) == set(
        HFT_MULTI_ASSET_SCOPES
    )
    assert payload["crypto"]["execution"]["cost_sensitivity"]
    assert payload["ledger"] == {
        "a_share": ["abandoned", "abandoned", "abandoned"],
        "crypto": ["completed"] * 5,
    }
    assert set(payload["statistics"]["families"]) == {"btc_usdt", "eth_usdt"}
    for family in payload["statistics"]["families"].values():
        assert {
            "hac",
            "stationary_bootstrap",
            "fdr_by",
            "spa",
            "stepm",
            "mcs",
            "psr",
            "dsr",
            "pbo",
        }.issubset(family)
    assert payload["statistics"]["fama_macbeth"]
    assert payload["a_share"]["factor_rows"]
    assert set(payload["a_share"]["portfolios"]) == {"1d", "1w", "1m"}
    assert payload["a_share"]["model_status"] == "REAL_DATA_PENDING"
    assert len(list((output_root / "experiment-ledger" / "crypto").glob("*.json"))) == 5
    persisted = json.loads(
        (output_root / "unified-research-smoke.json").read_text("utf-8")
    )
    assert persisted["holdout_accessed"] is False
    markdown = (output_root / "unified-research-interview-demo.md").read_text("utf-8")
    assert "SMOKE_ONLY / SYNTHETIC_DATA / REAL_DATA_PENDING" in markdown
    tracked_markdown = (
        Path(__file__).resolve().parents[2]
        / "docs/research/unified-research-interview-demo.md"
    ).read_text("utf-8")
    for heading in (
        "## 摘要",
        "## 背景与方法",
        "### 关键假设",
        "## 数据",
        "## A股结果",
        "## Crypto 结果",
        "## 稳健性与过拟合诊断",
        "## 局限",
        "## 结论",
    ):
        assert heading in markdown
        assert heading in tracked_markdown
    for token in (
        "REAL_DATA_PENDING",
        "HOLDOUT_LOCKED",
        "stationary bootstrap",
        "SPA/Reality Check",
        "StepM",
        "MCS",
        "Fama-MacBeth",
    ):
        assert token in markdown
        assert token in tracked_markdown
