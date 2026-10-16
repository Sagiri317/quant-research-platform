from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

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
from quant_research_platform.interfaces import private_cli
from quant_research_platform.storage import schema_inventory
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_write_text,
    native_filesystem_path,
)
from tests.helpers.paths import StorageLayout


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


@pytest.mark.parametrize("layout_kind", ("year", "reference"))
@pytest.mark.parametrize("context", (False, True))
def test_inventory_reuses_verified_null_profiles_and_reads_remaining_facts(
    tmp_path, monkeypatch, layout_kind, context
):
    from dataclasses import replace

    from tests.storage.low_level_fixtures import make_processed_dataset_spec

    partitions = ("year",) if layout_kind == "year" else ()
    spec = make_processed_dataset_spec("statistics")
    spec = replace(spec, column_roles={**spec.column_roles, "note": ColumnRole.CONTEXT})
    root = StorageLayout(tmp_path).processed_data_root(
        "a_share", "factor", "statistics"
    )
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-04"]),
            "value": [1.0, float("nan"), float("nan")],
            "note": pd.Series([None, "known", None], dtype="string"),
        }
    )
    if layout_kind == "reference":
        spec = replace(
            spec,
            grain=DataGrain.ASSET_REFERENCE,
            partition_columns=partitions,
            primary_key=("market_id", "asset_id"),
            column_roles={
                key: value
                for key, value in spec.column_roles.items()
                if key != "timestamp"
            },
        )
        frame = frame.drop(columns="timestamp")
    write_dataset(root, frame, spec=spec)
    projections = []
    from quant_research_platform.storage.datasets import inspection

    original = inspection.iter_dataset_file_batches

    def record(session, **kwargs):
        projections.append(kwargs["columns"])
        yield from original(session, **kwargs)

    monkeypatch.setattr(inspection, "iter_dataset_file_batches", record)
    (item,) = _collect_inventory(
        tmp_path,
        important_null_count_columns=("asset_id", "value", "note")
        if context
        else ("asset_id", "value"),
    )["datasets"]
    assert item["row_count_by_year"] == (
        {"2020": 1, "2021": 2} if layout_kind != "reference" else {}
    )
    assert item["null_key_counts"] == {key: 0 for key in spec.primary_key}
    expected_nulls = {
        "asset_id": 0,
        "value": 2,
    }
    if context:
        expected_nulls["note"] = 2
    assert item["selected_important_column_null_counts"] == expected_nulls
    assert len(projections) == 1
    assert set(projections[0]) == set(frame.columns)


def test_inventory_ignores_only_unpublished_atomic_directory(tmp_path, monkeypatch):
    from quant_research_platform.storage.datasets import writing as dataset_writing
    from quant_research_platform.storage.identity import filesystem_exists
    from tests.storage.low_level_fixtures import (
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    layout = StorageLayout(tmp_path)
    ordinary_file = tmp_path / "data" / "processed" / ".a-0123456789ab"
    filesystem_mkdir(ordinary_file.parent, parents=True)
    filesystem_write_text(ordinary_file, "ordinary file", encoding="utf-8")
    observed = []
    original_atomic = dataset_writing.atomic_write_directory
    for index, name in enumerate(("published", "new_build")):
        spec = make_processed_dataset_spec(name)
        root = layout.processed_data_root("a_share", "factor", name)

        def inspect_staged(staging, root=root, index=index):
            assert staging.name.startswith(".a-")
            assert not filesystem_exists(root)
            items = _collect_inventory(tmp_path)["datasets"]
            observed.append(items)
            assert len(items) == index
            assert all(".a-" not in item["logical_relative_root"] for item in items)

        def observed_atomic(final_root, writer):
            def observe(staging):
                result = writer(staging)
                inspect_staged(staging)
                return result

            return original_atomic(final_root, observe)

        monkeypatch.setattr(dataset_writing, "atomic_write_directory", observed_atomic)
        write_dataset(root, make_default_dataset_frame(), spec=spec)
    assert len(observed) == 2
    assert len(_collect_inventory(tmp_path)["datasets"]) == 2
    assert (
        Path(native_filesystem_path(ordinary_file)).read_text(encoding="utf-8")
        == "ordinary file"
    )


@pytest.mark.parametrize(
    "names", [("foo", "foo-bar"), ("foo", "foo.bar"), ("foo", "foo_bar")]
)
def test_schema_inventory_uses_full_posix_path_order_for_prefix_names(tmp_path, names):
    from tests.storage.low_level_fixtures import (
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    layout = StorageLayout(tmp_path)
    roots = []
    for name in names:
        spec = make_processed_dataset_spec(name)
        roots.append(
            write_dataset(
                layout.processed_data_root("a_share", "factor", name),
                make_default_dataset_frame(),
                spec=spec,
            )
        )
    output = tmp_path / "inventory.json"
    schema_inventory.write_schema_inventory(tmp_path, output)
    items = json.loads(
        Path(native_filesystem_path(output)).read_text(encoding="utf-8")
    )["datasets"]
    assert [item["logical_relative_root"] for item in items] == sorted(
        root.root.relative_to(tmp_path).as_posix() for root in roots
    )
    assert all(
        item["row_count"] == 3 and item["duplicate_primary_key_count"] == 0
        for item in items
    )


@pytest.mark.parametrize("deep_root", (False, True))
def test_schema_inventory_keeps_all_explicit_dataset_paths(
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
            root = logical_root / f"sample-{rows}"
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
            )
            expected[root.relative_to(tmp_path).as_posix()] = (layer, kind, rows)
    output = tmp_path / "inventory.json"
    schema_inventory.write_schema_inventory(
        tmp_path, output, important_null_count_columns=("value",)
    )
    items = json.loads(
        Path(native_filesystem_path(output)).read_text(encoding="utf-8")
    )["datasets"]
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
    cli_output = tmp_path / "cli-inventory.json"
    result = CliRunner().invoke(
        private_cli.app, ["--project-root", str(tmp_path), "--output", str(cli_output)]
    )
    assert result.exit_code == 0, result.output
    cli_items = json.loads(
        Path(native_filesystem_path(cli_output)).read_text(encoding="utf-8")
    )["datasets"]
    assert [item["logical_relative_root"] for item in cli_items] == sorted(expected)


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
@pytest.mark.parametrize("name", ("nested", ".a-0123456789ab"))
def test_schema_inventory_does_not_descend_nested_junction(
    tmp_path: Path, name: str
) -> None:
    layer_root = tmp_path / "data" / "processed"
    filesystem_mkdir(layer_root, parents=True)
    external = tmp_path / "external"
    filesystem_mkdir(external)
    filesystem_write_text(external / "spec.json", "{}", encoding="utf-8")
    junction = layer_root / name
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


@pytest.mark.parametrize(
    ("forged_summary", "error"),
    (
        (
            {"row_count": 3, "partition_row_counts": {"year=2026": 3}},
            "dataset_manifest_summary_mismatch",
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
    manifest = json.loads(
        Path(native_filesystem_path(manifest_path)).read_text(encoding="utf-8")
    )
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
    caught = False
    try:
        try:
            with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
                item = next(items)
                assert item["row_count_by_year"] == {"2022": 1, "2023": 2}
                if cancel:
                    raise primary
        except KeyboardInterrupt as error:
            assert cancel and error is primary
            caught = True
        assert caught == cancel
        assert next(items, None) is None
    finally:
        if items is not None:
            items.close()


@pytest.mark.parametrize("forged", (False, True))
def test_inventory_nonkey_time_does_not_define_year_axis(tmp_path, forged):
    spec = InterimDataSpec(
        data_id="nullable_time",
        source_id="fixture",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE,
        column_roles={
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.CONTEXT,
        },
        primary_key=("asset_id",),
        partition_columns=(),
    )
    root = StorageLayout(tmp_path).interim_data_root("fixture", spec.data_id)
    write_dataset(
        root,
        pd.DataFrame(
            {
                "asset_id": ["a", "b"],
                "timestamp": pd.to_datetime([None, "2021-01-04"]),
                "value": [1.0, 2.0],
            }
        ),
        spec=spec,
    )
    if forged:
        path = root / "_manifest" / "manifest.json"
        manifest = json.loads(
            Path(native_filesystem_path(path)).read_text(encoding="utf-8")
        )
        manifest["null_count_summary"]["asset_id"] = 1
        filesystem_write_text(path, json.dumps(manifest) + "\n", encoding="utf-8")
    if forged:
        with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
            _collect_inventory(tmp_path)
    else:
        (item,) = _collect_inventory(
            tmp_path, important_null_count_columns=("timestamp",)
        )["datasets"]
        assert item["row_count_by_year"] == {}
        assert item["selected_important_column_null_counts"] == {"timestamp": 1}
