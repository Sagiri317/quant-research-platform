"""One study expands into independent accounts with native prediction windows."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.research.policies import etf_execution


def test_study_shares_rules_and_selects_predictions_for_each_stage(
    tmp_path, monkeypatch
):
    config = json.loads(
        Path("configs/high-frequency/run-etf-execution.json").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        prediction_root=str(tmp_path / "predictions"), output=str(tmp_path / "results")
    )
    windows = {
        "before": {"start": "2024-01-01", "end": "2024-12-31"},
        "training": {"start": "2025-01-01", "end": "2025-09-30"},
        "validation": {"start": "2025-10-01", "end": "2025-12-31"},
        "test": {"start": "2026-10-30T01:30:00", "end": "2026-10-30T07:00:00"},
    }
    for asset in config["asset_ids"]:
        for period, window in windows.items():
            directory = (
                tmp_path
                / "predictions"
                / asset
                / "models"
                / period
                / "xgboost"
                / "predictions"
            )
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(
                json.dumps({"metadata": {"prediction_window": window}})
            )
    original = deepcopy(config)
    calls = []

    def capture(request):
        calls.append(request)
        return Path(request["output"])

    monkeypatch.setattr(etf_execution, "backtest_policy", capture)
    monkeypatch.setattr(etf_execution, "train_policy", capture)
    results = etf_execution.run_etf_execution(config, stage="direct")
    assert list(results) == ["510300", "510500", "159915", "518880"]
    for asset, request in zip(config["asset_ids"], calls, strict=True):
        episode = request["episode"]
        assert episode["asset_ids"] == [asset]
        assert episode["account"]["parameters"]["initial_capital"] == 1_000_000
        assert episode["account"]["parameters"]["initial_weights"] == {asset: 0.5}
        assert episode["account"]["parameters"]["maximum_weights"] == {asset: 1.0}
        assert [Path(p).parents[1].name for p in episode["prediction"]["paths"]] == [
            "test"
        ]
        assert request["strategy"] == "immediate"
    direct_account = calls[0]["episode"]["account"]
    calls.clear()
    etf_execution.run_etf_execution(config, stage="train", asset_id="510300")
    train = calls.pop()
    assert train["fit_cutoff"] == "2025-12-31T23:59:59.999999999"
    assert len(train["episodes"]) == 3
    assert all(
        Path(e["prediction"]["paths"][0]).parents[1].name == "training"
        for e in train["episodes"]
    )
    assert (
        Path(train["validation_episodes"][0]["prediction"]["paths"][0]).parents[1].name
        == "validation"
    )
    config_file = tmp_path / "study.json"
    config_file.write_text(json.dumps(config))
    result = CliRunner().invoke(
        app,
        [
            "run-etf-execution",
            "--config",
            str(config_file),
            "--stage",
            "ppo",
            "--asset-id",
            "510300",
        ],
    )
    assert result.exit_code == 0, result.output
    ppo = calls.pop()
    assert ppo["policy"] == train["output"]
    assert json.loads(result.output) == {"510300": ppo["output"]}
    assert ppo["episode"]["account"]["parameters"] == direct_account["parameters"]
    assert config == original
    with pytest.raises(ValueError, match="asset_not_selected"):
        etf_execution.run_etf_execution(config, stage="direct", asset_id="unknown")
    with pytest.raises(ValueError, match="stage_invalid"):
        etf_execution.run_etf_execution(config, stage="unknown")
    config["backtest_window"] = {"start": "2027-01-01", "end": "2027-01-02"}
    with pytest.raises(ValueError, match="predictions_missing"):
        etf_execution.run_etf_execution(config, stage="direct", asset_id="510300")
