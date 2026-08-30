from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MULTI_ASSET_SCOPES,
)
from quant_research_platform.sources.binance.fixtures import (
    write_synthetic_hft_recorder_run,
)
from quant_research_platform.workflows import unified_research
from quant_research_platform.workflows.unified_research import (
    run_unified_research_smoke,
)


def test_unified_smoke_consumes_persisted_market_statistics_and_ledger_owners(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_synthetic_hft_recorder_run(raw_root)

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
    for protocol in payload["protocols"].values():
        fold = protocol["folds"][0]
        assert (
            fold["final_holdout_start_step"] - fold["development_test_end_step"] - 1
            >= protocol["max_label_lookahead_steps"]
        )
    assert payload["crypto"]["data_quality"]["panel_rows"] == 122
    assert all(
        set(metrics) == set(HFT_MULTI_ASSET_SCOPES)
        for metrics in payload["crypto"]["role_evaluations"].values()
    )
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
        assert set(family) == {"validation", "development_test"}
        for role in family.values():
            assert {
                "hac",
                "stationary_bootstrap",
                "fdr_bh",
                "fdr_by",
                "spa",
                "stepm",
                "mcs",
                "predictive_score_overfit",
            }.issubset(role)
            assert role["common_observations"] == 8
    assert set(payload["statistics"]["role_metrics"]) == {
        "validation",
        "development_test",
    }
    assert all(
        set(metrics) == set(HFT_MULTI_ASSET_SCOPES)
        for metrics in payload["statistics"]["role_metrics"].values()
    )
    assert payload["statistics"]["global"]["a_share"] == {
        "status": "UNDEFINED",
        "reason": "real_data_pending",
        "candidate_count": 3,
    }
    assert len(payload["statistics"]["global"]["candidate_ids"]) == 8
    for role in payload["statistics"]["global"]["crypto"].values():
        assert len(role["comparison_ids"]) == len(role["fdr_bh"]) == len(role["fdr_by"])
    assert payload["statistics"]["fama_macbeth"] == {
        "status": "UNDEFINED",
        "reason": "real_data_pending",
    }
    assert payload["a_share"]["factor_rows"]
    assert payload["a_share"]["factor_artifacts"]["persisted_data_ids"] == [
        "fundamental_pit",
        "fundamental_raw",
    ]
    assert {
        row["fundamental_raw_asset_growth"] for row in payload["a_share"]["factor_rows"]
    } == {0.25}
    assert set(payload["a_share"]["portfolios"]) == {"1d", "1w", "1m"}
    summaries = [
        portfolio["summary"] for portfolio in payload["a_share"]["portfolios"].values()
    ]
    assert len({summary["total_turnover"] for summary in summaries}) == 3
    assert all(
        {
            "tracking_error",
            "information_ratio",
            "constant_notional_additive_pnl",
            "benchmark_nav",
            "capacity_notional",
        }.issubset(summary)
        for summary in summaries
    )
    assert all(
        portfolio["active_exposures"] and portfolio["path"]
        for portfolio in payload["a_share"]["portfolios"].values()
    )
    assert payload["a_share"]["model_status"] == "REAL_DATA_PENDING"
    ledger_paths = sorted((output_root / "experiment-ledger" / "crypto").glob("*.json"))
    assert len(ledger_paths) == 5
    for path in ledger_paths:
        record = json.loads(path.read_text("utf-8"))
        assert record["validation_metrics"]["observation_count"] in {8, 16}
        assert record["development_test_metrics"]["observation_count"] in {8, 16}
        assert record["validation_metrics"] != record["development_test_metrics"]
    persisted = json.loads(
        (output_root / "unified-research-smoke.json").read_text("utf-8")
    )
    assert persisted == payload
    assert (output_root / "crypto" / "outputs").is_dir()
    markdown = (output_root / "unified-research-interview-demo.md").read_text("utf-8")
    assert "SMOKE_ONLY / SYNTHETIC_DATA / REAL_DATA_PENDING" in markdown
    tracked_markdown = (
        Path(__file__).resolve().parents[2]
        / "docs/research/unified-research-interview-demo.md"
    ).read_text("utf-8")
    assert tracked_markdown == markdown
    notebook = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/research/unified-research-interview-demo.ipynb"
        ).read_text("utf-8")
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "tests.helpers" not in source
    assert "sys.path" not in source
    assert "repo_root" not in source
    assert "write_synthetic_hft_recorder_run" in source


def test_unified_parent_publish_is_atomic_on_late_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"

    def write_crypto_child(*, output_root: Path, **_kwargs: object) -> dict:
        output_root.mkdir(parents=True)
        (output_root / "child-complete").write_text("complete", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        unified_research, "run_crypto_hft_experiment", write_crypto_child
    )
    monkeypatch.setattr(
        unified_research, "_a_share_smoke", lambda _root: ({}, object())
    )
    monkeypatch.setattr(unified_research, "_crypto_plan", object)
    monkeypatch.setattr(
        unified_research,
        "_statistics_from_crypto_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late-failure")),
    )

    with pytest.raises(RuntimeError, match="^late-failure$"):
        run_unified_research_smoke(
            raw_crypto_root=tmp_path / "raw",
            output_root=output_root,
        )

    assert not output_root.exists()
