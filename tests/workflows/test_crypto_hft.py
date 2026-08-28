from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.crypto_hft import run_crypto_hft_experiment
from tests.helpers.binance_hft import write_binance_hft_run


def _taker_config() -> TakerConfig:
    return TakerConfig(1.0, 1.0, 10, 1.0, 0.0, 500)


def _maker_config() -> MakerConfig:
    return MakerConfig(1.0, 2.0, -1.0, 0.0, 0.0, 500)


def test_crypto_hft_workflow_closes_standard_chain_and_execution(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_binance_hft_run(raw_root)

    report = run_crypto_hft_experiment(
        output_root=output_root,
        raw_root=raw_root,
        start_ms=0,
        end_ms=6_000,
        taker_config=_taker_config(),
        maker_config=_maker_config(),
    )

    assert report["status"] == "PASS"
    assert report["lineage"]["source_artifact_identity"]["artifact_id"] == (
        "workflow-run"
    )
    assert report["lineage"]["panel"] == HFT_PANEL_DATA_ID
    assert report["lineage"]["factor"] == HFT_FACTOR_DATA_ID
    assert report["lineage"]["label"] == HFT_LABEL_DATA_ID
    assert report["lineage"]["research_dataset"] == HFT_RESEARCH_DATASET_ID
    assert set(report["models"]["prediction_outputs"]) == {
        "crypto_hft_lightgbm_regressor",
        "crypto_hft_linear_regression",
        "crypto_hft_xgboost_regressor",
    }
    assert report["execution"]["taker_model_id"] == "crypto_hft_linear_regression"
    assert report["execution"]["maker_policy_id"] == (
        "public_l2_inventory_maker_baseline"
    )
    assert report["execution"]["maker_prediction_used"] is False
    assert report["execution"]["taker"]["latency_fidelity"] == (
        "full_l2_observed_event"
    )
    assert report["execution"]["maker"]["queue_model"] == ("estimated_queue_ahead")
    assert report["resource"]["available0_bytes"] > 0
    assert report["resource"]["memory_budget_bytes"] > 0
    assert report["resource"]["wall_seconds"] > 0
    assert report["resource"]["threads"] >= 1
    assert (output_root / "report.json").is_file()
    assert (output_root / report["execution"]["path"]).is_file()
    layout = StorageLayout(output_root)
    assert layout.research_dataset_root(
        "crypto",
        HFT_RESEARCH_DATASET_ID,
    ).is_dir()
    assert not list(tmp_path.glob(".a-*"))


def test_crypto_hft_workflow_identity_failure_publishes_nothing(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_binance_hft_run(raw_root)
    raw_file = next(raw_root.rglob("*.jsonl.gz"))
    raw_file.write_bytes(raw_file.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="file_identity_mismatch"):
        run_crypto_hft_experiment(
            output_root=output_root,
            raw_root=raw_root,
            start_ms=0,
            end_ms=6_000,
            taker_config=_taker_config(),
            maker_config=_maker_config(),
        )

    assert not output_root.exists()


def test_crypto_hft_workflow_completes_under_explicit_memory_cap(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_binance_hft_run(raw_root)

    report = run_crypto_hft_experiment(
        output_root=output_root,
        raw_root=raw_root,
        start_ms=0,
        end_ms=6_000,
        taker_config=_taker_config(),
        maker_config=_maker_config(),
        explicit_cap_bytes=256 * 1024**2,
    )

    assert report["status"] == "PASS"
    assert report["data_quality"]["panel_rows"] == 61
    assert report["data_quality"]["factor_rows"] == 61
    assert report["data_quality"]["label_rows"] == 61
    assert report["resource"]["memory_budget_bytes"] <= 256 * 1024**2
    assert report["resource"]["sampled_peak_reserved_bytes"] <= report["resource"][
        "memory_budget_bytes"
    ]
