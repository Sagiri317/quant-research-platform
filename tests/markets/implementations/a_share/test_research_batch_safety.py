from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.research_batch_safety import (  # noqa: E501
    build_schema_inventory,
    run_memory_preflight,
    write_json_sanitized,
)


def _panel_path(tmp_path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000001"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "market_type": ["A", "A", "A"],
            "is_a_share_scope": [True, True, True],
            "eligible": [True, True, False],
            "can_enter": [True, True, False],
            "can_exit": [True, True, True],
            "amount": [100.0, 200.0, 150.0],
            "turnover": [0.1, 0.2, 0.15],
            "mom_20d": [0.01, 0.02, 0.03],
            "forward_return_20d": [0.02, -0.01, 0.04],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def test_schema_inventory_is_aggregate_only_and_role_tagged(tmp_path) -> None:
    path = _panel_path(tmp_path)

    inventory = build_schema_inventory(
        path,
        expected_columns=("stock_code", "trade_date"),
    )

    assert inventory["row_count"] == 3
    assert inventory["date_range"] == {"min": "2024-01-02", "max": "2024-01-03"}
    assert inventory["duplicate_key_count"] == 0
    roles = {column["name"]: column["role"] for column in inventory["columns"]}
    assert roles["stock_code"] == "key"
    assert roles["amount"] == "raw_price_volume"
    assert roles["mom_20d"] == "factor_candidate"
    assert roles["forward_return_20d"] == "label_candidate"
    assert roles["eligible"] == "tradability"
    assert inventory["market_scope"] == {
        "field": "is_a_share_scope",
        "true_count": 3,
        "false_count": 0,
        "null_count": 0,
    }

    serialized = json.dumps(inventory, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "000001" not in serialized
    assert "row_sample" not in serialized
    assert "stock_sample" not in serialized


def test_schema_inventory_redacts_unsafe_column_names(tmp_path) -> None:
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "amount": [100.0, 200.0],
            "mom_20d": [0.01, 0.02],
            "forward_return_20d": [0.02, -0.01],
            "vendor_internal_flag": [1, 0],
            "source_download_code": ["x", "y"],
            "processed_private": ["x", "y"],
            "credential_token": ["x", "y"],
            "C:/vendor/path": ["x", "y"],
            "local_path": ["x", "y"],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    inventory = build_schema_inventory(
        path,
        expected_columns=(
            "stock_code",
            "trade_date",
            "amount",
            "vendor_expected",
            "credential_token",
            "local_expected_path",
        ),
    )
    serialized = json.dumps(inventory, ensure_ascii=False)

    assert {column["name"] for column in inventory["columns"]} == {
        "stock_code",
        "trade_date",
        "amount",
        "mom_20d",
        "forward_return_20d",
    }
    assert inventory["redacted_column_count"] == 6
    assert inventory["redacted_missing_expected_column_count"] == 3
    assert inventory["redacted_unexpected_column_count"] == 5
    for forbidden in (
        "vendor_internal_flag",
        "source_download_code",
        "processed_private",
        "credential_token",
        "C:/vendor/path",
        "local_path",
        "vendor_expected",
        "local_expected_path",
    ):
        assert forbidden not in serialized


def test_memory_preflight_uses_parquet_metadata_and_budget_gate(tmp_path) -> None:
    path = _panel_path(tmp_path)

    passed = run_memory_preflight(
        path,
        required_columns=("stock_code",),
        memory_budget_gb=1.0,
    )
    failed = run_memory_preflight(
        path,
        required_columns=("missing_column",),
        memory_budget_gb=0.000000001,
    )

    assert passed["status"] == "passed"
    assert passed["row_count"] == 3
    assert passed["column_count"] == 11
    assert passed["estimated_memory_bytes"] > 0
    assert passed["missing_required_columns"] == []
    assert failed["status"] == "failed"
    assert failed["missing_required_columns"] == ["missing_column"]
    assert failed["estimated_memory_bytes"] > failed["memory_budget_bytes"]


def test_write_json_sanitized_rejects_private_or_absolute_path_leaks(tmp_path) -> None:
    output = tmp_path / "safe.json"

    write_json_sanitized(output, {"status": "passed", "run_id": "safe_run"})
    assert output.is_file()

    with pytest.raises(ValueError, match="unsafe private or local path"):
        write_json_sanitized(output, {"path": str(tmp_path / "private_real" / "x")})


def test_write_json_sanitized_rejects_sensitive_values_and_keys(tmp_path) -> None:
    output = tmp_path / "safe.json"

    write_json_sanitized(
        output,
        {"status": "passed", "factor_name": "reversal_20d", "model_id": "ridge"},
    )
    assert output.is_file()

    with pytest.raises(ValueError, match="sensitive_metadata_policy_violation") as exc:
        write_json_sanitized(output, {"status": "token=abc123"})
    assert "abc123" not in str(exc.value)
    assert "token=abc123" not in str(exc.value)

    with pytest.raises(ValueError, match="sensitive_metadata_policy_violation") as exc:
        write_json_sanitized(output, {"api_key": "redacted"})
    assert "api_key" not in str(exc.value)
    assert "redacted" not in str(exc.value)
