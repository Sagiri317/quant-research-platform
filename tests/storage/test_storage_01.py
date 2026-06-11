from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.manifests import (
    write_internal_artifact_manifest,
    write_layer_manifest,
    write_manifest,
    write_run_manifest,
)


def _processed_spec(data_id: str = "x") -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )


def test_dataset_writer_rejects_public_output_before_iterating_chunks(
    tmp_path: Path,
) -> None:
    iterated = False

    def chunks():
        nonlocal iterated
        iterated = True
        yield _frame()

    with pytest.raises(ValueError, match="outside_private_boundary"):
        write_dataset(tmp_path / "public" / "x", chunks(), spec=_processed_spec())

    assert not iterated
    assert not (tmp_path / "public" / "x").exists()


def test_dataset_writer_accepts_private_processed_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / "x"

    write_dataset(root, _frame(), spec=_processed_spec())

    assert (root / "spec.json").is_file()
    assert (root / "_manifest" / "manifest.json").is_file()
    assert (root / "data" / "year=2014" / "part-000.parquet").is_file()


def test_manifest_writer_sanitizes_public_summary_payload(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "reports" / "private_real" / "a_share" / "run" / "manifest.json",
        {
            "path": r"C:\Users\DELL\data\raw\csmar\000001.SZ",
            "traceback": "Traceback (most recent call last): /mnt/data/raw",
            "token": "token=abcdef123456",
        },
    )
    text = manifest.read_text(encoding="utf-8")

    assert "C:" not in text
    assert "/mnt/" not in text
    assert "000001" not in text
    assert "abcdef123456" not in text


def test_manifest_writer_rejects_existing_target_file(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "run"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_manifest(manifest, {"schema_version": "0.1"})

    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_output_already_exists_error_public_summary() -> None:
    assert OutputAlreadyExistsError().to_public_summary() == {
        "status": "failed",
        "failure_code": "output_already_exists",
    }


def test_layer_manifest_detects_missing_dataset_manifests(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    present = layer_root / "a_share" / "factor" / "present"
    missing = layer_root / "a_share" / "label" / "missing"
    write_internal_artifact_manifest(
        present / "_manifest" / "manifest.json",
        _artifact_manifest_payload(
            data_id="present",
            row_count=3,
            date_min="2014-01-02",
            date_max="2014-01-03",
        ),
    )

    manifest = write_layer_manifest(
        layer_root=layer_root,
        logical_layer="processed",
        run_id="run",
        dataset_roots=(present, missing),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["dataset_count"] == 1
    assert payload["total_rows"] == 3
    assert payload["missing_dataset_manifests"] == [
        "data/processed/a_share/label/missing/_manifest/manifest.json"
    ]
    assert payload["layer_validation_status"] == "fail"


def test_run_manifest_aggregates_layers(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    evidence = tmp_path / "data" / "evidence"
    processed.mkdir(parents=True)
    evidence.mkdir(parents=True)
    write_layer_manifest(
        layer_root=processed,
        logical_layer="processed",
        run_id="run",
        dataset_roots=(),
    )
    write_layer_manifest(
        layer_root=evidence,
        logical_layer="evidence",
        run_id="run",
        dataset_roots=(),
    )

    manifest = write_run_manifest(
        manifests_root=tmp_path / "manifests",
        run_id="run",
        output_root=Path(".run") / "s1",
        layer_manifest_paths=(
            processed / "_manifests" / "layer_manifest.json",
            evidence / "_manifests" / "layer_manifest.json",
        ),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["run_id"] == "run"
    assert payload["output_root"] == ".run/s1"
    assert payload["layer_count"] == 2
    assert payload["run_validation_status"] == "pass"


def _artifact_manifest_payload(
    *,
    data_id: str,
    row_count: int,
    date_min: str,
    date_max: str,
) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "data_id": data_id,
        "row_count": row_count,
        "date_min": date_min,
        "date_max": date_max,
        "duplicate_primary_key_count": 0,
        "validation_status": "pass",
    }
