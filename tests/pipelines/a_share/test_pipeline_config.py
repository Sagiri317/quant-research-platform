import json

import pytest

from quant_research_platform.pipelines.a_share.config import (
    PipelineConfig,
    validate_pipeline_config,
)
from quant_research_platform.pipelines.a_share.snapshots import (
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
        return_source_column="forward_return_20d",
        label_columns=("forward_return_5d", "forward_return_20d"),
        output_root=tmp_path,
    )


def test_valid_config_writes_snapshot(tmp_path):
    config = _valid_config(tmp_path)
    config.raw_root = tmp_path / "raw_private" / "csmar"
    config.interim_root = tmp_path / "interim_private" / "csmar"
    config.processed_root = tmp_path / "processed_private" / "csmar"

    snapshot = write_run_snapshot(config)
    run_config = (snapshot.run_dir / "run_config.yaml").read_text(encoding="utf-8")

    assert snapshot.run_dir == tmp_path / "sample_run"
    assert (snapshot.run_dir / "run_config.yaml").is_file()
    assert (snapshot.run_dir / "warnings.json").is_file()
    assert (snapshot.run_dir / "summary.md").is_file()
    assert "factor_template_id: reversal_20d" in run_config
    assert "raw_private" not in run_config
    assert "interim_private" not in run_config
    assert "processed_private" not in run_config
    assert str(tmp_path) not in run_config


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


def test_snapshot_rejects_bare_processed_column_expression(tmp_path):
    config = _valid_config(tmp_path)
    config.factor_expression = "rev_20d"
    config.factor_template_id = "custom"
    config.factor_template_source = None

    validation = validate_pipeline_config(config)

    assert validation.has_blocking
    assert any(
        issue.code == "invalid_factor_expression" for issue in validation.blocking
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


def test_validation_rejects_invalid_dates_cost_and_labels(tmp_path):
    reverse_dates = _valid_config(tmp_path)
    reverse_dates.date_start = "2025-01-01"
    reverse_dates.date_end = "2024-12-31"
    malformed_date = _valid_config(tmp_path)
    malformed_date.date_start = "not-a-date"
    compact_date = _valid_config(tmp_path)
    compact_date.date_start = "20240101"
    week_date = _valid_config(tmp_path)
    week_date.date_start = "2024-W01-1"
    negative_cost = _valid_config(tmp_path)
    negative_cost.cost_bps = -0.1
    missing_labels = _valid_config(tmp_path)
    missing_labels.label_columns = ()
    unsafe_label = _valid_config(tmp_path)
    unsafe_label.label_columns = ("forward-return-1d",)
    missing_return_source = _valid_config(tmp_path)
    missing_return_source.return_source_column = ""
    unsafe_return_source = _valid_config(tmp_path)
    unsafe_return_source.return_source_column = "forward-return-1d"

    cases = [
        (reverse_dates, "invalid_date_range"),
        (malformed_date, "invalid_date"),
        (compact_date, "invalid_date"),
        (week_date, "invalid_date"),
        (negative_cost, "negative_cost_bps"),
        (missing_labels, "missing_label_columns"),
        (unsafe_label, "unsafe_label_column"),
        (missing_return_source, "missing_return_source_column"),
        (unsafe_return_source, "unsafe_return_source_column"),
    ]
    for config, code in cases:
        validation = validate_pipeline_config(config)

        assert any(issue.code == code for issue in validation.blocking)
        with pytest.raises(PipelineSnapshotError):
            write_run_snapshot(config)


def test_validation_rejects_invalid_build_numeric_fields(tmp_path):
    negative_listing_days = _valid_config(tmp_path)
    negative_listing_days.min_listing_trading_days = -1
    string_factor_window = _valid_config(tmp_path)
    string_factor_window.factor_windows = ("20",)
    zero_factor_window = _valid_config(tmp_path)
    zero_factor_window.factor_windows = (0,)
    string_label_horizon = _valid_config(tmp_path)
    string_label_horizon.label_horizons = ("1",)
    empty_label_horizons = _valid_config(tmp_path)
    empty_label_horizons.label_horizons = ()

    cases = [
        (negative_listing_days, "negative_min_listing_trading_days"),
        (string_factor_window, "invalid_factor_windows"),
        (zero_factor_window, "invalid_factor_windows"),
        (string_label_horizon, "invalid_label_horizons"),
        (empty_label_horizons, "invalid_label_horizons"),
    ]
    for config, code in cases:
        validation = validate_pipeline_config(config)

        assert any(issue.code == code for issue in validation.blocking)


def test_validation_rejects_unknown_start_layer(tmp_path):
    config = _valid_config(tmp_path)
    config.start_from = "middle_layer"

    validation = validate_pipeline_config(config)

    assert any(issue.code == "unsupported_start_from" for issue in validation.blocking)


def test_validation_allows_separate_label_and_return_source_columns(tmp_path):
    config = _valid_config(tmp_path)
    config.label_columns = ("forward_return_20d",)
    config.return_source_column = "forward_return_1d"

    validation = validate_pipeline_config(config)

    assert not validation.has_blocking


def test_validation_rejects_unsafe_processed_factor_column(tmp_path):
    config = _valid_config(tmp_path)
    config.processed_factor_column = "forward-return-1d"

    validation = validate_pipeline_config(config)

    assert any(
        issue.code == "unsafe_processed_factor_column"
        for issue in validation.blocking
    )


def test_warning_issue_is_recorded(tmp_path):
    config = _valid_config(tmp_path)
    config.diagnostics = ("industry_neutrality",)

    validation = validate_pipeline_config(config)
    snapshot = write_run_snapshot(config)
    warnings = json.loads((snapshot.run_dir / "warnings.json").read_text("utf-8"))

    assert not validation.has_blocking
    assert any(issue["code"] == "missing_optional_industry" for issue in warnings)


def test_start_from_and_build_config_fields_are_serialized(tmp_path):
    config = _valid_config(tmp_path)
    config.start_from = "rebuild_panel"
    config.raw_root = tmp_path / "raw"
    config.interim_root = tmp_path / "interim"
    config.processed_root = tmp_path / "processed"
    config.min_listing_trading_days = 90
    config.factor_windows = (20, 60)
    config.label_horizons = (1, 20)

    payload = config.to_snapshot_dict()

    assert payload["start_from"] == "rebuild_panel"
    assert "raw_root" not in payload
    assert "interim_root" not in payload
    assert "processed_root" not in payload
    assert "output_root" not in payload
    assert payload["min_listing_trading_days"] == 90
    assert payload["factor_windows"] == [20, 60]
    assert payload["label_horizons"] == [1, 20]
