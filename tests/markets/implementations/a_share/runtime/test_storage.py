from __future__ import annotations

from quant_research_platform.markets.implementations.a_share.runtime import (
    storage as space_gate,
)


def test_evaluate_space_gate_passes_when_estimate_is_smaller_and_fits_disk() -> None:
    result = space_gate.evaluate_space_gate(
        observed_bytes=2_000,
        observed_rows=1_000,
        target_rows=5_000,
        free_bytes=100_000,
        legacy_processed_bytes=20_000,
        max_legacy_ratio=0.8,
        max_free_ratio=0.8,
    )

    assert result.estimated_bytes == 10_000
    assert result.allowed
    assert result.reasons == []


def test_evaluate_space_gate_blocks_when_estimate_is_too_close_to_legacy() -> None:
    result = space_gate.evaluate_space_gate(
        observed_bytes=2_000,
        observed_rows=1_000,
        target_rows=9_000,
        free_bytes=100_000,
        legacy_processed_bytes=20_000,
        max_legacy_ratio=0.8,
        max_free_ratio=0.8,
    )

    assert result.estimated_bytes == 18_000
    assert not result.allowed
    assert any("legacy" in reason for reason in result.reasons)


def test_evaluate_space_gate_blocks_when_estimate_exceeds_disk_budget() -> None:
    result = space_gate.evaluate_space_gate(
        observed_bytes=2_000,
        observed_rows=1_000,
        target_rows=9_000,
        free_bytes=20_000,
        legacy_processed_bytes=100_000,
        max_legacy_ratio=0.8,
        max_free_ratio=0.8,
    )

    assert result.estimated_bytes == 18_000
    assert not result.allowed
    assert any("free" in reason for reason in result.reasons)


def test_collect_observed_processed_bytes_counts_full_processed_files(tmp_path) -> None:
    processed_root = tmp_path / "csmar"
    panel_dir = processed_root / "panel_full"
    factor_dir = processed_root / "price_volume_factors_full"
    panel_dir.mkdir(parents=True)
    factor_dir.mkdir(parents=True)
    (panel_dir / "research_panel_2014_2025.parquet").write_bytes(b"panel-full")
    (factor_dir / "price_volume_factors_2014_2025.parquet").write_bytes(
        b"factor-full"
    )

    observed = space_gate.collect_observed_processed_bytes(
        processed_root,
        start_year=2014,
        end_year=2025,
    )

    assert observed == len(b"panel-fullfactor-full")
