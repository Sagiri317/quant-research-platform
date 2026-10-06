from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.interfaces import private_cli
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.schema import data_spec_payload
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_read_text,
    filesystem_rename,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.diagnostics import schema_inventory


def _collect_inventory(
    root: Path,
    *,
    important_null_count_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    with schema_inventory.iter_schema_inventory_items(
        root,
        important_null_count_columns=important_null_count_columns,
    ) as items:
        return {"schema_id": "qrp-schema-inventory", "datasets": list(items)}


@pytest.mark.parametrize("deep_root", (False, True))
def test_schema_inventory_keeps_all_current_roots_and_instances(
    tmp_path: Path, deep_root: bool
) -> None:
    if deep_root:
        while len(str(tmp_path)) <= 270:
            tmp_path /= "segment_" + "a" * 32
        assert len(str(tmp_path)) > 260
    layout = StorageLayout(tmp_path)
    common = {
        "data_id": "inventory_current",
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
            "label": ColumnRole.LABEL,
        },
        "primary_key": ("market_id", "asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    specs = (
        (
            InterimDataSpec(source_id="fixture", **common),
            layout.interim_data_root("fixture", common["data_id"]),
            "interim",
            "interim",
        ),
        (
            ProcessedDataSpec(
                market_id="a_share", kind=ProcessedDataKind.FACTOR, **common
            ),
            layout.processed_data_root("a_share", "factor", common["data_id"]),
            "processed",
            "factor",
        ),
        (
            ResearchDatasetSpec(
                market_id="a_share",
                input_data_ids=("panel",),
                base_input_data_id="panel",
                factor_columns=("value",),
                label_columns=("label",),
                **common,
            ),
            layout.research_dataset_root("a_share", common["data_id"]),
            "research",
            "research_dataset",
        ),
    )
    expected = {}
    for spec, logical_root, layer, kind in specs:
        for rows in (1, 2, 3):
            envelope = build_business_manifest_envelope(
                logical_artifact_id=spec.data_id,
                schema_identity=canonical_json_hash(data_spec_payload(spec)),
                protocol_or_spec_identity="a" * 64,
                software_identity="b" * 64,
                research_window={
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                    "access": "development",
                },
                upstream_identities={},
                semantic_metadata={"root_seed": rows},
            )
            root = (
                logical_root
                if rows == 1
                else layout.instance_root(logical_root, envelope.instance_identity)
            )
            frame = pd.DataFrame(
                {
                    "market_id": ["a_share"] * rows,
                    "asset_id": [f"asset_{index}" for index in range(rows)],
                    "timestamp": pd.date_range("2024-01-01", periods=rows),
                    "value": [1.0] + [float("nan")] * (rows - 1),
                    "label": [0.5] * rows,
                }
            )
            write_dataset(
                root,
                frame,
                spec=spec,
                instance_metadata=None if rows == 1 else {"common_manifest": envelope},
            )
            expected[root.relative_to(tmp_path).as_posix()] = (layer, kind, rows)
    output = tmp_path / "inventory.json"
    schema_inventory.write_schema_inventory(
        tmp_path, output, important_null_count_columns=("value",)
    )
    items = json.loads(filesystem_read_text(output, encoding="utf-8"))["datasets"]
    assert len(items) == 9
    assert sum(item["row_count"] for item in items) == 18
    assert [item["logical_relative_root"] for item in items] == sorted(expected)
    for item in items:
        layer, kind, rows = expected[item["logical_relative_root"]]
        assert (
            item["artifact_id"],
            item["layer"],
            item["kind"],
            item["row_count"],
        ) == (common["data_id"], layer, kind, rows)
        assert item["duplicate_primary_key_count"] == 0
        assert item["selected_important_column_null_counts"] == {"value": rows - 1}
        invalid = deepcopy(item)
        invalid["logical_relative_root"] = (
            str(item["logical_relative_root"]) + "/instances/not-an-identity"
        )
        with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
            schema_inventory._validate_schema_inventory_item(invalid)
    cli_output = tmp_path / "cli-inventory.json"
    result = CliRunner().invoke(
        private_cli.app, ["--project-root", str(tmp_path), "--output", str(cli_output)]
    )
    assert result.exit_code == 0, result.output
    cli_items = json.loads(filesystem_read_text(cli_output, encoding="utf-8"))[
        "datasets"
    ]
    assert [item["logical_relative_root"] for item in cli_items] == sorted(expected)


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_schema_inventory_does_not_descend_nested_junction(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    filesystem_mkdir(layer_root, parents=True)
    external = tmp_path / "external"
    filesystem_mkdir(external)
    filesystem_write_text(external / "spec.json", "{}", encoding="utf-8")
    junction = layer_root / "nested"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert tuple(schema_inventory._iter_dataset_artifact_roots(tmp_path)) == ()


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_schema_inventory_rejects_junction_project_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    filesystem_mkdir(external)
    root = tmp_path / "project"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(external)],
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(root))


def test_schema_inventory_rejects_aliased_data_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "project"
    filesystem_mkdir(root)
    external = tmp_path / "external"
    filesystem_mkdir(external / "processed", parents=True)
    try:
        Path(native_filesystem_path(root / "data")).symlink_to(
            native_filesystem_path(external), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(root))


def test_schema_inventory_rejects_alias_above_supplied_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    filesystem_mkdir(target / "project" / "data" / "processed", parents=True)
    alias = tmp_path / "alias"
    try:
        Path(native_filesystem_path(alias)).symlink_to(
            native_filesystem_path(target), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(alias / "project"))


def test_schema_inventory_rejects_spec_symlink(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    artifact_root = layer_root / "artifact"
    filesystem_mkdir(artifact_root, parents=True)
    external = tmp_path / "spec.json"
    filesystem_write_text(external, "{}", encoding="utf-8")
    try:
        Path(native_filesystem_path(artifact_root / "spec.json")).symlink_to(
            native_filesystem_path(external)
        )
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(tmp_path))


def test_schema_inventory_uses_storage_exact_duplicate_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_factor",
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
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share"] * 3,
                "asset_id": ["asset_a", "asset_b", "asset_c"],
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
                "value": [1.0, 2.0, 3.0],
            }
        ),
        spec=spec,
    )

    monkeypatch.setattr(
        pd.util,
        "hash_pandas_object",
        lambda values, **_kwargs: pd.Series(
            [0] * len(values),
            dtype="uint64",
        ),
    )

    payload = _collect_inventory(tmp_path)

    assert payload["datasets"][0]["duplicate_primary_key_count"] == 0


def test_schema_inventory_rejects_replacement_between_open_and_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_replacement",
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
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    current_root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    replacement_root = (
        tmp_path
        / "replacement"
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / spec.data_id
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")

    def frame(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "market_id": ["a_share"] * rows,
                "asset_id": [f"asset_{index}" for index in range(rows)],
                "timestamp": pd.date_range("2026-01-01", periods=rows, freq="D"),
                "value": [float(index) for index in range(rows)],
            }
        )

    write_dataset(current_root, frame(2), spec=spec)
    write_dataset(replacement_root, frame(3), spec=spec)
    real_open = schema_inventory.open_dataset_session
    real_aggregate = schema_inventory._aggregate_dataset_stats
    open_count = 0

    def record_open(root: Path, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(root, **kwargs)

    def replace_before_stats(session, *args, **kwargs):  # noqa: ANN001
        filesystem_rename(current_root, backup_root)
        filesystem_rename(replacement_root, current_root)
        return real_aggregate(session, *args, **kwargs)

    monkeypatch.setattr(schema_inventory, "open_dataset_session", record_open)
    monkeypatch.setattr(
        schema_inventory,
        "_aggregate_dataset_stats",
        replace_before_stats,
    )

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
            tuple(items)

    assert open_count == 1


@pytest.mark.parametrize(
    ("forged_summary", "error"),
    (
        (
            {"row_count": 3, "partition_row_counts": {"year=2026": 3}},
            "manifest_row_count_mismatch",
        ),
        (
            {
                "partitions": ["year=2099"],
                "partition_row_counts": {"year=2099": 2},
            },
            "manifest_partition_date_mismatch",
        ),
        (
            {
                "null_count_summary": {
                    "market_id": 1,
                    "asset_id": 0,
                    "timestamp": 0,
                }
            },
            "dataset_manifest_summary_mismatch",
        ),
    ),
    ids=("row-count", "partition-summary", "observer-summary"),
)
def test_schema_inventory_rejects_forged_storage_summary(
    tmp_path: Path,
    forged_summary: dict[str, object],
    error: str,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_forged_summary",
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
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["asset_a", "asset_b"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "value": [1.0, 2.0],
            }
        ),
        spec=spec,
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest.update(forged_summary)
    filesystem_write_text(manifest_path, json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
            tuple(items)


@pytest.mark.parametrize("value", (True, 1, ""))
def test_schema_inventory_rejects_important_column_coercion(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_inventory_important_columns_invalid",
    ):
        with schema_inventory.iter_schema_inventory_items(
            tmp_path,
            important_null_count_columns=(value,),  # type: ignore[arg-type]
        ):
            pass


def _valid_inventory_payload() -> dict[str, object]:
    columns = ("market_id", "asset_id", "timestamp", "value")
    dtypes = {
        "market_id": "object",
        "asset_id": "object",
        "timestamp": "datetime64[ns]",
        "value": "float64",
    }
    primary_key = ("market_id", "asset_id", "timestamp")
    partition_columns = ("year",)
    return {
        "schema_id": "qrp-schema-inventory",
        "datasets": [
            {
                "artifact_id": "inventory_factor",
                "layer": "processed",
                "kind": "factor",
                "logical_relative_root": (
                    "data/processed/a_share/factor/inventory_factor"
                ),
                "columns": list(columns),
                "dtypes": dtypes,
                "primary_key": list(primary_key),
                "partition_columns": list(partition_columns),
                "file_count": 1,
                "row_count": 3,
                "row_count_by_year": {"2026": 3},
                "min_timestamp": "2026-01-01",
                "max_timestamp": "2026-01-03",
                "duplicate_primary_key_count": 0,
                "null_key_counts": {column: 0 for column in primary_key},
                "selected_important_column_null_counts": {"value": 0},
                "schema_hash": schema_inventory._schema_hash(
                    columns=columns,
                    dtypes=dtypes,
                    primary_key=primary_key,
                    partition_columns=partition_columns,
                ),
                "warnings": [],
            }
        ],
    }


@pytest.mark.parametrize(
    "time_column", ("timestamp", "report_period", "effective_date")
)
@pytest.mark.parametrize("cancel", (False, True))
def test_inventory_uses_declared_time_and_closes_early_context(
    tmp_path, time_column, cancel
):
    spec = InterimDataSpec(
        data_id="declared_time",
        source_id="fixture",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "asset_id": ColumnRole.KEY,
            time_column: ColumnRole.TIMESTAMP,
            "value": ColumnRole.CONTEXT,
        },
        primary_key=("asset_id", time_column),
        partition_columns=("year",),
    )
    layout = StorageLayout(tmp_path)
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            time_column: pd.to_datetime(["2022-12-31", "2023-01-03", "2023-01-04"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
    write_dataset(layout.interim_data_root("fixture", spec.data_id), frame, spec=spec)
    primary = KeyboardInterrupt("inventory_consumer_primary")
    items = None
    try:
        try:
            with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
                item = next(items)
                assert item["row_count_by_year"] == {"2022": 1, "2023": 2}
                assert schema_inventory._validate_schema_inventory_item(item) is item
                if cancel:
                    raise primary
        except KeyboardInterrupt as error:
            assert cancel and error is primary
        assert next(items, None) is None
    finally:
        if items is not None:
            items.close()
