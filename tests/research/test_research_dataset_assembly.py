from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.datasets.assembler as assembler_module
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.paths import StorageLayout
from tests.helpers.research_processed_inputs import (
    _demo_research_rows,
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    assemble_research_dataset,
    make_research_dataset_catalog,
)


def test_research_dataset_assembles_from_processed_inputs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(layout, processed_catalog)
    root = assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    inspection = inspect_dataset(root)
    assert root == tmp_path / "data" / "research" / "a_share" / "demo"
    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "momentum_20d",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert len(result) == 4
    assert set(inspection.manifest["inputs"]) == {
        "demo_factors",
        "demo_labels",
        "demo_annotations",
    }


@pytest.mark.parametrize(
    "scenario", ("pv_all", "pv_expanded", "fusion_nf_core", "fusion_nf_expectations")
)
def test_non_a_share_named_scenario_keeps_input_semantics_and_requires_real_columns(
    tmp_path, scenario
):
    from dataclasses import replace

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import (
        make_dummy_processed_catalog,
        write_dummy_processed_inputs,
    )
    from tests.helpers.research_specs import make_dummy_research_dataset_spec

    layout = StorageLayout(tmp_path)
    processed = make_dummy_processed_catalog()
    spec = make_dummy_research_dataset_spec(data_id=scenario)
    write_dummy_processed_inputs(layout, processed)
    root = assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed.list()
            if item.data_id
            in ResearchDatasetCatalog((spec,)).get(scenario).input_data_ids
        },
        output_root=layout.research_dataset_root(
            ResearchDatasetCatalog((spec,)).get(scenario).market_id, scenario
        ),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        processed_catalog=processed,
        dataset_id=scenario,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    frame = read_dataset(root)
    assert frame.eligible.tolist() == [True, False]
    assert assembler_module._generated_semantic_columns(spec) == ()
    assert assembler_module._add_a_share_semantic_columns(frame, spec) is frame
    a_share = replace(
        spec, market_id=MarketId("a_share"), eligibility_recipe="a_share_all"
    )
    assert assembler_module._generated_semantic_columns(a_share) == ("eligible",)
    with pytest.raises(ValueError, match="^a_share_eligibility_facts_missing:"):
        assembler_module._add_a_share_semantic_columns(
            frame.assign(market_id="a_share"), a_share
        )
    missing = ProcessedDataCatalog(
        tuple(
            replace(
                item,
                column_roles={
                    key: role
                    for key, role in item.column_roles.items()
                    if key != "eligible"
                },
            )
            if str(item.data_id) == "filters"
            else item
            for item in processed.list()
        )
    )
    with pytest.raises(
        ValueError, match="^missing_research_required_columns: eligible$"
    ):
        assembler_module._validate_research_input_specs(
            spec, {item.data_id: item for item in missing.list()}
        )


def test_research_dataset_assembles_intraday_timestamp_window(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    primary_key = PLATFORM_ASSET_TIME_KEY
    factor_spec = ProcessedDataSpec(
        data_id="minute_factors",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_factor": ColumnRole.FACTOR,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    label_spec = ProcessedDataSpec(
        data_id="minute_labels",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.LABEL,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_label": ColumnRole.LABEL,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    processed_catalog = ProcessedDataCatalog((factor_spec, label_spec))
    dataset_catalog = make_research_dataset_catalog(
        data_id="minute_research",
        market_id="crypto",
        input_data_ids=("minute_factors", "minute_labels"),
        base_input_data_id="minute_factors",
        factor_columns=("minute_factor",),
        label_columns=("minute_label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
    )
    timestamps = pd.to_datetime(
        ["2026-07-06T00:00:00", "2026-07-06T00:01:00", "2026-07-06T00:02:00"]
    )
    keys = pd.DataFrame(
        [
            {"market_id": "crypto", "asset_id": asset_id, "timestamp": timestamp}
            for timestamp in timestamps
            for asset_id in ("asset_a", "asset_b")
        ]
    )
    for spec, value_column in (
        (factor_spec, "minute_factor"),
        (label_spec, "minute_label"),
    ):
        frame = keys.assign(**{value_column: range(len(keys))})
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame,
            spec=spec,
        )
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("minute_research").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset_catalog.get("minute_research").market_id, "minute_research"
        ),
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="minute_research",
        time_window=TimeWindow("2026-07-06T00:01:00", "2026-07-06T00:02:00"),
    )
    result = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("crypto"), "minute_research"),
            time_window=TimeWindow("2026-07-06T00:01:00", "2026-07-06T00:02:00"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    assert len(result) == 4
    assert sorted(result["timestamp"].unique()) == list(timestamps[1:])


def test_assembly_keeps_nanosecond_windows_through_persisted_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path)
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
    }
    source = ProcessedDataSpec(
        data_id="nanosecond_input",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.TICK,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    dataset = make_research_dataset_catalog(
        data_id="nanosecond_research",
        market_id="crypto",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.TICK,
    )
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "asset",
            "timestamp": times,
            "factor": [1.0, 2.0],
            "label": [3.0, 4.0],
        }
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        frame,
        spec=source,
    )
    window = TimeWindow(times[0], times[-1])
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in ProcessedDataCatalog((source,)).list()
            if item.data_id in dataset.get("nanosecond_research").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset.get("nanosecond_research").market_id, "nanosecond_research"
        ),
        dataset_catalog=dataset,
        processed_catalog=ProcessedDataCatalog((source,)),
        dataset_id="nanosecond_research",
        time_window=window,
    )
    result = pd.concat(
        tuple(
            read_dataset(
                layout.research_dataset_root(MarketId("crypto"), "nanosecond_research"),
                time_window=window,
                mode=DatasetReadMode.CHUNKED,
            )
        ),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(result, frame, check_exact=True)


def test_research_dataset_session_complete_chunks_preserve_output(
    tmp_path: Path,
) -> None:
    frame = _research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(default_layout, processed_catalog, frame=frame)
    for spec in processed_catalog.list():
        source = frame.loc[:, list(spec.column_roles)]
        root = tiny_layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(
            root,
            (source.iloc[position : position + 1] for position in range(len(source))),
            spec=spec,
        )
        assert inspect_dataset(root).file_count == len(source)
    base_spec = processed_catalog.get(dataset_catalog.get("demo").base_input_data_id)
    base_root = tiny_layout.processed_data_root(
        base_spec.market_id, base_spec.kind, base_spec.data_id
    )
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )

    batches = list(
        iter_dataset_chunks(
            open_dataset_session(base_root, columns=None, time_window=None)
        )
    )
    assert len(batches) == len(frame)
    assert all(len(batch) == 1 for batch in batches)
    assert batches[0].timestamp.iloc[0] == batches[1].timestamp.iloc[0]
    default_root = assemble_research_dataset(
        inputs={
            item.data_id: default_layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=default_layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    tiny_root = assemble_research_dataset(
        inputs={
            item.data_id: tiny_layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=tiny_layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    default = pd.concat(
        read_dataset(
            default_layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = pd.concat(
        read_dataset(
            tiny_layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    pd.testing.assert_frame_equal(
        default.reset_index(drop=True), tiny.reset_index(drop=True)
    )
    assert (
        inspect_dataset(tiny_root).file_count > inspect_dataset(default_root).file_count
    )


def test_research_chunk_view_prunes_top_level_multi_year_physical_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    frame = _research_rows_for_dates(["2014-12-31", "2015-01-02"])
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    processed_spec = processed_catalog.get("demo_factors")
    dataset_spec = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    ).get("demo")
    columns = assembler_module._processed_input_columns(
        dataset_spec, processed_spec, existing_columns=()
    )
    session = assembler_module.open_dataset_session(
        layout.processed_data_root(
            processed_spec.market_id, processed_spec.kind, processed_spec.data_id
        ),
        columns=columns,
        time_window=TimeWindow("2014-12-31", "2015-01-02"),
    )
    view = session.filtered_view(TimeWindow("2015-01-02", "2015-01-02"))
    assert len(view.read_paths) == 1
    assert view.read_paths[0].parent.name == "year=2015"
    assert len(session.read_paths) == 2


def test_sparse_research_dataset_left_joins_missing_non_base_input_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        key_coverage={"demo_labels": "sparse"},
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    frame = _demo_research_rows()
    labels = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    write_demo_processed_inputs(layout, processed_catalog, label_frame=labels)
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    missing_row = result.loc[
        result["asset_id"].eq("000002")
        & result["timestamp"].eq(pd.Timestamp("2014-01-03"))
    ].iloc[0]
    assert len(result) == 4
    assert pd.isna(missing_row["forward_return_20d"])


def test_research_dataset_role_validation_rejects_wrong_factor_source(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id="bad_demo",
                market_id=MarketId("a_share"),
                input_data_ids=("demo_labels", "demo_annotations"),
                base_input_data_id="demo_annotations",
                factor_columns=("market_type",),
                label_columns=("forward_return_20d",),
                frequency=DataFrequency.DAILY,
                grain=DataGrain.RESEARCH_ROW,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    "market_type": ColumnRole.FACTOR,
                    "forward_return_20d": ColumnRole.LABEL,
                },
                primary_key=PLATFORM_ASSET_TIME_KEY,
                partition_columns=("year",),
            ),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)
    with pytest.raises(ValueError, match="assemble_factor_not_available"):
        assemble_research_dataset(
            inputs={
                item.data_id: layout.processed_data_root(
                    item.market_id, item.kind, item.data_id
                )
                for item in processed_catalog.list()
                if item.data_id in dataset_catalog.get("bad_demo").input_data_ids
            },
            output_root=layout.research_dataset_root(
                dataset_catalog.get("bad_demo").market_id, "bad_demo"
            ),
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="bad_demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


@pytest.mark.parametrize("coverage", ("strict", "sparse"))
@pytest.mark.parametrize("physical_rows", (1, 2))
@pytest.mark.parametrize(
    "extra_date", (None, "2023-12-31", "2024-01-02", "2024-01-04", "2024-01-06")
)
def test_full_request_key_domain_is_independent_of_base_file_boundaries(
    tmp_path, coverage, physical_rows, extra_date
):
    import hashlib

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_specs import (
        DEMO_FACTOR_COLUMN,
        DEMO_LABEL_COLUMN,
        make_demo_research_dataset_spec,
    )

    layout = StorageLayout(tmp_path)
    processed = make_demo_processed_catalog()
    spec = make_demo_research_dataset_spec(key_coverage={"demo_factors": coverage})
    rows = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset_a",
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            DEMO_FACTOR_COLUMN: 1.0,
            DEMO_LABEL_COLUMN: 0.01,
            "eligible": True,
            "can_enter": True,
            "can_exit": True,
            "market_type": 1,
        }
    )
    factors = (
        rows
        if extra_date is None
        else pd.concat(
            (rows, rows.iloc[:1].assign(timestamp=pd.Timestamp(extra_date))),
            ignore_index=True,
        ).sort_values("timestamp")
    )
    roots = []
    for data_id, frame in (
        ("demo_annotations", rows),
        ("demo_labels", rows),
        ("demo_factors", factors),
    ):
        definition = processed.get(data_id)
        root = layout.processed_data_root(
            definition.market_id, definition.kind, data_id
        )
        selected = frame.loc[:, list(definition.required_columns)]
        chunks = tuple(
            selected.iloc[start : start + physical_rows]
            for start in range(0, len(selected), physical_rows)
        )
        write_dataset(root, chunks, spec=definition)
        roots.append(root)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }

    def assemble():
        return assemble_research_dataset(
            inputs={
                item.data_id: layout.processed_data_root(
                    item.market_id, item.kind, item.data_id
                )
                for item in processed.list()
                if item.data_id
                in ResearchDatasetCatalog((spec,)).get(spec.data_id).input_data_ids
            },
            output_root=layout.research_dataset_root(
                ResearchDatasetCatalog((spec,)).get(spec.data_id).market_id,
                spec.data_id,
            ),
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            processed_catalog=processed,
            dataset_id=spec.data_id,
            time_window=TimeWindow("2023-12-31", "2024-01-04"),
        )

    if extra_date in {"2023-12-31", "2024-01-02", "2024-01-04"}:
        with pytest.raises(
            ValueError, match="extra_research_input_keys: demo_factors: 1"
        ):
            assemble()
        assert not layout.research_dataset_root(spec.market_id, spec.data_id).exists()
    else:
        frame = read_dataset(assemble())
        assert frame.timestamp.tolist() == rows.timestamp.tolist()
    assert all(
        (
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path, digest in before.items()
        )
    )


def test_model_target_uses_last_participating_label_maturity_and_keeps_raw_clock(
    tmp_path,
):
    from dataclasses import replace

    from quant_research_platform.contracts.experiments import (
        legal_training_rows,
    )
    from quant_research_platform.research.protocols import a_share
    from quant_research_platform.storage.datasets import read_dataset

    base = a_share.research_dataset_catalog().get("pv_all")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "ineligible"],
            "timestamp": pd.Timestamp("2020-01-02"),
            "forward_return_20d": [0.1, 0.3, 0.5],
            "forward_return_20d_available_at": pd.to_datetime(
                ["2020-02-01", "2020-02-04", "2020-02-10"]
            ),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": [False, False, True],
            "is_pt": False,
            "listing_age_trading_days": 400,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 10.0,
            "amount": 100.0,
            "turnover": 0.1,
        }
    )
    output = assembler_module._add_a_share_semantic_columns(frame, base)
    output["momentum_20d"] = [1.0, 2.0, 3.0]
    pd.testing.assert_series_equal(
        output.forward_return_20d_available_at, frame.forward_return_20d_available_at
    )
    assert (
        output.model_target_available_at.iloc[:2].eq(pd.Timestamp("2020-02-04")).all()
    )
    assert pd.isna(output.model_target_available_at.iloc[2])
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        "forward_return_20d",
        "forward_return_20d_available_at",
        "model_target",
        "model_target_available_at",
        "eligible",
        "momentum_20d",
    )
    spec = replace(
        base,
        factor_columns=("momentum_20d",),
        feature_columns=(),
        feature_transforms=(),
        context_columns=("model_target_available_at",),
        label_columns=(
            "forward_return_20d",
            "forward_return_20d_available_at",
            "model_target",
        ),
        column_roles={key: base.column_roles[key] for key in columns},
        label_availability_columns={
            label: maturity
            for label, maturity in base.label_availability_columns.items()
            if label in columns and maturity in columns
        },
    )
    root = StorageLayout(tmp_path).research_dataset_root(spec.market_id, spec.data_id)
    write_dataset(root, output.loc[:, columns], spec=spec)
    saved = read_dataset(root)
    maturity = spec.label_availability_columns["model_target"]
    assert maturity == "model_target_available_at"
    assert legal_training_rows(
        saved,
        fitting_cutoff=pd.Timestamp("2020-02-02"),
        availability_column=maturity,
        target_column="model_target",
        feature_columns=(),
    ).empty
    assert (
        len(
            legal_training_rows(
                saved,
                fitting_cutoff=pd.Timestamp("2020-02-04"),
                availability_column=maturity,
                target_column="model_target",
                feature_columns=(),
            )
        )
        == 2
    )


@pytest.mark.parametrize("recipe", ("a_share_canonical", "a_share_quantile_1_99"))
def test_named_feature_recipe_assembles_complete_sessions_and_persists_identity(
    tmp_path, recipe
):
    import numpy as np

    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )
    from quant_research_platform.storage.datasets.schema import data_spec_payload

    layout = StorageLayout(project_root=tmp_path)
    transform = FeatureTransformSpec(recipe, ("factor",))
    context = ("industry_standard_id", "industry_level1_code", "log_total_market_cap")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in range(120)],
            "timestamp": pd.Timestamp("2020-01-03"),
            "factor": np.sin(np.arange(120)),
            "label": 0.0,
            "eligible": True,
            "industry_standard_id": "fixture",
            "industry_level1_code": "A",
            "log_total_market_cap": np.arange(120) * 0.01,
        }
    )
    frame.loc[119, "factor"] = 1000.0
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
        "eligible": ColumnRole.UNIVERSE,
        **dict.fromkeys(context, ColumnRole.CONTEXT),
    }
    source = ProcessedDataSpec(
        data_id="recipe_inputs",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    datasets = make_research_dataset_catalog(
        data_id="recipe_case",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        feature_columns=transform.output_columns,
        feature_transforms=(transform,),
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=context,
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        (frame.iloc[i : i + 25] for i in range(0, len(frame), 25)),
        spec=source,
    )
    window = TimeWindow("2020-01-03", "2020-01-03")
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in ProcessedDataCatalog((source,)).list()
            if item.data_id in datasets.get("recipe_case").input_data_ids
        },
        output_root=layout.research_dataset_root(
            datasets.get("recipe_case").market_id, "recipe_case"
        ),
        dataset_catalog=datasets,
        processed_catalog=ProcessedDataCatalog((source,)),
        dataset_id="recipe_case",
        time_window=window,
    )
    saved = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("a_share"), "recipe_case"),
            time_window=window,
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    expected = a_share_feature_pipeline(
        frame, factor_columns=("factor",), recipe_id=recipe
    )
    for column in ("factor", *transform.output_columns):
        pd.testing.assert_series_equal(
            saved[column], expected[column], check_exact=True
        )
    assert saved.feature_factor.notna().all()
    assert data_spec_payload(datasets.get("recipe_case"))["feature_transforms"] == [
        {
            "recipe_id": recipe,
            "source_columns": ["factor"],
            "clip": transform.clip,
            "industry_neutralize": True,
            "size_neutralize": True,
            "size_column": "log_total_market_cap",
            "center": True,
            "scale": True,
            "impute_missing": True,
        },
    ]


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "momentum_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)
