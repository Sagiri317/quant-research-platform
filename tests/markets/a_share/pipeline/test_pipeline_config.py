import json

import pytest

from quant_research_platform.markets.a_share.pipeline.config import (
    PipelineConfig,
    validate_pipeline_config,
)
from quant_research_platform.markets.a_share.pipeline.snapshots import (
    PipelineSnapshotError,
    write_run_snapshot,
)


def _valid_config(tmp_path):
    return PipelineConfig(
        market="a_share",
        run_id="sample_run",
        date_start="2023-01-01",
        date_end="2025-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        factor_template_source="config/a_share/factor_templates/reversal_20d.yaml",
        portfolio_mode="long_only",
        cost_bps=10.0,
        pnl_return_column="forward_return_20d",
        label_columns=("forward_return_5d", "forward_return_20d"),
        output_root=tmp_path,
    )


def test_valid_config_writes_snapshot(tmp_path):
    config = _valid_config(tmp_path)

    snapshot = write_run_snapshot(config)

    assert snapshot.run_dir == tmp_path / "sample_run"
    assert (snapshot.run_dir / "run_config.yaml").is_file()
    assert (snapshot.run_dir / "warnings.json").is_file()
    assert (snapshot.run_dir / "summary.md").is_file()
    assert "factor_template_id: reversal_20d" in (
        snapshot.run_dir / "run_config.yaml"
    ).read_text(encoding="utf-8")


def test_snapshot_records_expression_metadata_and_template_source(tmp_path):
    config = _valid_config(tmp_path)

    snapshot = write_run_snapshot(config)
    run_config = (snapshot.run_dir / "run_config.yaml").read_text(encoding="utf-8")

    assert (
        "factor_template_source: config/a_share/factor_templates/reversal_20d.yaml"
        in run_config
    )
    assert "expression_dependencies:" in run_config
    assert "  - close" in run_config
    assert "parsed_expression:" in run_config
    assert "name: return" in run_config


def test_same_run_id_creates_unique_directory_by_default(tmp_path):
    config = _valid_config(tmp_path)

    first = write_run_snapshot(config)
    second = write_run_snapshot(config)

    assert first.run_dir == tmp_path / "sample_run"
    assert second.run_dir != first.run_dir
    assert second.run_dir.name.startswith("sample_run_")
    assert (second.run_dir / "run_config.yaml").is_file()


def test_blocking_issue_prevents_snapshot(tmp_path):
    config = _valid_config(tmp_path)
    config.factor_expression = ""

    validation = validate_pipeline_config(config)

    assert validation.has_blocking
    assert any(
        issue.code == "missing_factor_expression" for issue in validation.blocking
    )
    with pytest.raises(PipelineSnapshotError):
        write_run_snapshot(config)


def test_validation_classifies_market_and_portfolio_issues(tmp_path):
    missing_market = _valid_config(tmp_path)
    missing_market.market = ""
    unsupported_market = _valid_config(tmp_path)
    unsupported_market.market = "crypto"
    unsupported_mode = _valid_config(tmp_path)
    unsupported_mode.portfolio_mode = "market_neutral"

    missing_market_validation = validate_pipeline_config(missing_market)
    unsupported_market_validation = validate_pipeline_config(unsupported_market)
    unsupported_mode_validation = validate_pipeline_config(unsupported_mode)

    assert any(
        issue.code == "missing_market"
        for issue in missing_market_validation.blocking
    )
    assert any(
        issue.code == "unsupported_market"
        for issue in unsupported_market_validation.blocking
    )
    assert any(
        issue.code == "unsupported_portfolio_mode"
        for issue in unsupported_mode_validation.blocking
    )


def test_warning_issue_is_recorded(tmp_path):
    config = _valid_config(tmp_path)
    config.diagnostics = ("industry_neutrality",)

    validation = validate_pipeline_config(config)
    snapshot = write_run_snapshot(config)
    warnings = json.loads((snapshot.run_dir / "warnings.json").read_text("utf-8"))

    assert not validation.has_blocking
    assert any(issue["code"] == "missing_optional_industry" for issue in warnings)
