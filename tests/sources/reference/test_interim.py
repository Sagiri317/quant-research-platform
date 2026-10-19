from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _canonical_stock_master,
)
from quant_research_platform.sources.reference import interim
from quant_research_platform.sources.reference.a_share_fees import (
    FEE_CSV_COLUMNS,
    load_a_share_fee_source,
)
from quant_research_platform.sources.reference.interim import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SCHEDULE_DATA_ID,
    A_SHARE_FEE_SIDECAR_NAME,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_mkdir,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from tests.helpers.paths import StorageLayout
from tests.helpers.source_preparation import prepared_source_evidence


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): Path(
            native_filesystem_path(parent / name)
        ).read_bytes()
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


def _write_reference_root(root: Path, *, commission: float = 0.0003) -> Path:
    schedule_root = root / "reference" / "a_share_fee_schedule"
    filesystem_mkdir(schedule_root, parents=True, exist_ok=True)
    csv_path = schedule_root / A_SHARE_FEE_CSV_NAME
    rows = [
        _row(
            "BROKER_COMMISSION",
            "2019-01-01",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            commission,
        ),
        _row("TRANSFER_FEE", "2019-01-01", "2025-12-31", "沪深", "买入+卖出", 1e-05),
        _row("STAMP_DUTY", "2019-01-01", "2025-12-31", "沪深", "卖出", 0.001),
    ]
    with open_filesystem_file(
        csv_path, "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FEE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = {
        "file_sha256": hashlib.sha256(
            Path(native_filesystem_path(csv_path)).read_bytes()
        ).hexdigest(),
        "file_bytes": filesystem_stat(csv_path).st_size,
        "row_count": len(rows),
        "fields_raw": list(FEE_CSV_COLUMNS),
    }
    filesystem_write_text(
        schedule_root / A_SHARE_FEE_SIDECAR_NAME,
        json.dumps(sidecar, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _row(
    fee_code: str, start: str, end: str, market: str, direction: str, rate: float
) -> dict[str, str]:
    row = dict.fromkeys(FEE_CSV_COLUMNS, "")
    row.update(
        {
            "费用代码": fee_code,
            "生效日": start,
            "截止日": end,
            "市场": market,
            "方向": direction,
            "计费基数": "成交金额",
            "十进制费率": str(rate),
            "费率(‰)": str(rate * 1000.0),
            "费率(bps)": str(rate * 10000.0),
            "进入正式账本": "是",
        }
    )
    return row


def test_fee_source_publishes_explicit_path_and_records_source_evidence(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    output = tmp_path / "fees"
    result = interim.materialize_interim_data(
        raw_root=raw, output_roots={A_SHARE_FEE_SCHEDULE_DATA_ID: output}
    )
    assert result.artifact_roots[A_SHARE_FEE_SCHEDULE_DATA_ID] == output
    frame = read_dataset(output)
    assert tuple(frame.columns) == interim.FEE_SCHEDULE_COLUMNS
    manifest = read_dataset_manifest_evidence(output)
    assert (
        manifest["source_evidence"]["content_hash"]
        == load_a_share_fee_source(reference_root=raw).source_identity
    )
    before = _file_bytes(output)
    with pytest.raises((ValueError, FileExistsError)):
        interim.materialize_interim_data(
            raw_root=raw, output_roots={A_SHARE_FEE_SCHEDULE_DATA_ID: output}
        )
    assert _file_bytes(output) == before


def test_fee_reader_cleanup_preserves_primary(tmp_path, monkeypatch):
    import io

    from quant_research_platform.sources.reference import a_share_fees

    root = _write_reference_root(tmp_path / "raw") / "reference"
    path = next(filesystem_glob(root, "**/*.csv"))
    primary = MemoryError("primary")

    class Handle(io.BufferedReader):
        def read1(self, *args):
            raise primary

        def close(self):
            super().close()
            raise OSError("secondary")

    handle = Handle(io.FileIO(native_filesystem_path(path)))
    original_open = a_share_fees.open_filesystem_file
    monkeypatch.setattr(
        a_share_fees,
        "open_filesystem_file",
        lambda candidate: handle if candidate == path else original_open(candidate),
    )
    with pytest.raises(MemoryError) as caught:
        load_a_share_fee_source(reference_root=root)
    assert caught.value is primary
    assert handle.closed


def _industry_observations():
    return (
        (
            "snapshot",
            "CSRC2012",
            "2019-12-31",
            "2020-01-03",
            [
                {
                    "code": "000001",
                    "name": "synthetic issuer",
                    "category": "A",
                    "major": "",
                    "industry": "01",
                    "source_locator": "page:1",
                }
            ],
        ),
    )


def test_public_industry_source_reaches_interim_and_real_panel(tmp_path):
    import pandas as pd

    from quant_research_platform.markets.implementations.a_share.processed import panels
    from tests.helpers.a_share_raw import write_industry_reference_fixture
    from tests.helpers.a_share_sources import _synthetic_interim_frames

    raw_root = tmp_path / "raw"
    write_industry_reference_fixture(raw_root, observations=_industry_observations())
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    root = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=raw_root / "reference",
        output_roots={data_id: root},
        data_ids=(data_id,),
        end_date="2020-12-31",
    )
    snapshots = read_dataset(root)
    assert str(snapshots["timestamp"].dtype) == "datetime64[ns]"
    spec = json.loads(
        Path(native_filesystem_path(root / "spec.json")).read_text(encoding="utf-8")
    )
    assert spec["primary_key"] == ["source_asset_id", "timestamp", "snapshot_id"]
    assert snapshots["observation_date"].tolist() == [pd.Timestamp("2019-12-31")]
    assert snapshots["timestamp"].tolist() == [pd.Timestamp("2020-01-04")]
    manifest = read_dataset_manifest_evidence(root)
    assert manifest["source_evidence"]["content_hash"] == prepared_source_evidence(
        interim, raw_root / "reference", data_id
    )
    dates = pd.bdate_range("2020-01-02", periods=5)
    inputs = _synthetic_interim_frames(dates)
    stocks = inputs["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    dictionary = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "CSRC2012"],
            "industry_code": ["A", "A01"],
            "industry_name": ["agriculture", "crops"],
            "parent_industry_code": ["", "A"],
            "industry_level": ["1", "2"],
        }
    )
    versions = inputs["industry_classification_versions"].assign(
        implement_date=pd.Timestamp("2012-10-26")
    )
    panel = panels.compute_market_panel(
        inputs["market_bars_1d"],
        **panels._prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stocks),
            market_cap=inputs["market_cap"],
            trading_calendar=inputs["trading_calendar"],
            industry_sets=dictionary,
            industry_versions=versions,
            industry_snapshots=snapshots,
        ),
    )
    asset = panel.loc[panel["asset_id"].eq("000001")]
    assert asset["timestamp"].tolist() == list(dates)
    assert asset["industry_level1_code"].fillna("unknown").tolist() == [
        "unknown",
        "unknown",
        "A",
        "A",
        "A",
    ]
    assert (
        panel.loc[~panel["asset_id"].eq("000001"), "industry_level1_code"].isna().all()
    )


def test_industry_snapshot_range_uses_publication_not_observation(tmp_path):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    root = tmp_path / "raw"
    observations = _industry_observations() + (
        (
            "capco",
            "CAPCO",
            "2023-06-30",
            "2024-02-08",
            [
                {
                    "code": "000001",
                    "name": "synthetic issuer",
                    "category": "C",
                    "major": "CE",
                    "industry": "26",
                    "source_locator": "page:1",
                }
            ],
        ),
    )
    write_industry_reference_fixture(root, observations=observations)
    layout = StorageLayout(tmp_path)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key) for key in (data_id,)
        },
        raw_root=root / "reference",
        data_ids=(data_id,),
        end_date="2023-12-31",
    )
    frame = read_dataset(layout.interim_data_root("reference", data_id))
    assert frame["snapshot_id"].tolist() == ["snapshot"]
    assert frame["industry_standard_id"].tolist() == ["CSRC2012"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_member",
        "bad_hash",
        "traversal",
        "non_mapping_record",
        "publication_before_observation",
    ],
)
def test_industry_reference_boundary_rejects_unproved_source(tmp_path, failure):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=_industry_observations()
    )
    metadata = json.loads(
        Path(native_filesystem_path(root / "metadata.json")).read_text(encoding="utf-8")
    )
    if failure == "missing_member":
        metadata["snapshots"][0]["document_file"] = "missing.pdf"
    elif failure == "bad_hash":
        metadata["files"]["snapshot.pdf"] = "0" * 64
    elif failure == "traversal":
        metadata["files"]["../snapshot.pdf"] = metadata["files"].pop("snapshot.pdf")
    elif failure == "publication_before_observation":
        metadata["snapshots"][0]["publication_date"] = "2019-12-01"
    elif failure == "non_mapping_record":
        candidate = json.loads(
            Path(native_filesystem_path(root / "snapshot.json")).read_text(
                encoding="utf-8"
            )
        )
        candidate["records"].append(42)
        filesystem_write_text(
            root / "snapshot.json", json.dumps(candidate), encoding="utf-8"
        )
        metadata["files"]["snapshot.json"] = hashlib.sha256(
            Path(native_filesystem_path(root / "snapshot.json")).read_bytes()
        ).hexdigest()
    filesystem_write_text(
        root / "metadata.json", json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="industry_snapshot_"):
        interim.materialize_interim_data(
            output_roots={
                key: StorageLayout(tmp_path).interim_data_root("reference", key)
                for key in (interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,)
            },
            raw_root=root.parent,
            data_ids=(interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,),
        )


def test_industry_reference_preserves_table_row_locator(tmp_path):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    observations[0][4][0]["source_locator"] = "table:2;row:7"
    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=observations, document_suffix=".docx"
    )
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    layout = StorageLayout(tmp_path)
    interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key) for key in (data_id,)
        },
        raw_root=root.parent,
        data_ids=(data_id,),
    )
    frame = read_dataset(layout.interim_data_root("reference", data_id))
    assert frame["source_locator"].tolist() == ["table:2;row:7"]
    assert "source_page" not in frame


def test_invalid_fee_csv_header_preserves_previous_artifact(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    layout = StorageLayout(project_root=tmp_path / "project")
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key)
            for key in ("a_share_fee_schedule",)
        },
        raw_root=raw,
    )
    root = result.artifact_roots[A_SHARE_FEE_SCHEDULE_DATA_ID]
    before = _file_bytes(root)
    source = raw / "a_share_fee_schedule"
    csv_path = source / A_SHARE_FEE_CSV_NAME
    original = csv_path.read_bytes()
    hostile = b"unexpected_column," + original
    csv_path.write_bytes(hostile)
    sidecar = source / A_SHARE_FEE_SIDECAR_NAME
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["file_sha256"] = hashlib.sha256(hostile).hexdigest()
    metadata["file_bytes"] = len(hostile)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="a_share_fee_source_schema_invalid"):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("reference", key)
                for key in ("a_share_fee_schedule",)
            },
            raw_root=raw,
        )
    assert _file_bytes(root) == before


@pytest.mark.parametrize("boundary", ["header", "record"])
def test_malformed_fee_csv_is_source_local_and_preserves_old_good(
    tmp_path, monkeypatch, boundary
):
    from quant_research_platform.sources.reference import a_share_fees
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    raw = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(project_root=tmp_path / "project")
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key)
            for key in ("a_share_fee_schedule",)
        },
        raw_root=raw / "reference",
    )
    old_root = result.artifact_roots[A_SHARE_FEE_SCHEDULE_DATA_ID]
    before = _file_bytes(old_root)
    source = raw / "reference" / "a_share_fee_schedule"
    path = source / A_SHARE_FEE_CSV_NAME
    header = (
        path.read_bytes().splitlines(keepends=True)[0] if boundary == "record" else b""
    )
    payload = header + b'"unterminated\n'
    path.write_bytes(payload)
    sidecar = source / A_SHARE_FEE_SIDECAR_NAME
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.update(
        file_bytes=len(payload), file_sha256=hashlib.sha256(payload).hexdigest()
    )
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    opened = []
    original_open = a_share_fees.open_filesystem_file

    def tracked_open(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(a_share_fees, "open_filesystem_file", tracked_open)
    with pytest.raises(
        ValueError, match="^a_share_fee_source_schema_invalid$"
    ) as caught:
        load_a_share_fee_source(reference_root=raw / "reference")
    assert isinstance(caught.value.__cause__, csv.Error)
    write_industry_reference_fixture(raw, observations=_industry_observations())
    target = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    root = tmp_path / "unaffected-industry"
    interim.materialize_interim_data(
        raw_root=raw / "reference", output_roots={target: root}, data_ids=(target,)
    )
    assert len(read_dataset(root)) == 1
    assert _file_bytes(old_root) == before
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.parametrize("publication", ("2025-01-01", "9999-12-31"))
def test_industry_future_bad_row_does_not_change_selected_history(
    tmp_path, publication
):
    import pandas as pd

    from tests.helpers.a_share_raw import write_industry_reference_fixture

    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    expected_root = tmp_path / "expected"
    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=_industry_observations()
    )
    interim.materialize_interim_data(
        output_roots={data_id: expected_root},
        raw_root=root.parent,
        data_ids=(data_id,),
        end_date="2020-12-31",
    )
    write_industry_reference_fixture(
        tmp_path / "raw",
        observations=_industry_observations()
        + (("future", "CSRC2012", "2020-01-01", publication, [{"code": "invalid"}]),),
    )
    raw_before = _file_bytes(root)
    actual_root = tmp_path / "actual"
    interim.materialize_interim_data(
        output_roots={data_id: actual_root},
        raw_root=root.parent,
        data_ids=(data_id,),
        end_date="2020-12-31",
    )
    pd.testing.assert_frame_equal(
        read_dataset(actual_root), read_dataset(expected_root)
    )
    assert _file_bytes(root) == raw_before


@pytest.mark.parametrize("optional", ["name", "source_locator"])
@pytest.mark.parametrize("value", [None, "", "omitted", 42])
def test_industry_optional_metadata_does_not_gate_classification(
    tmp_path, optional, value
):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    row = observations[0][4][0]
    if value == "omitted":
        del row[optional]
    else:
        row[optional] = value
    root = write_industry_reference_fixture(tmp_path / "raw", observations=observations)
    before = _file_bytes(root)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=root.parent, output_roots={data_id: output}, data_ids=(data_id,)
    )
    frame = read_dataset(output)
    assert frame.source_asset_id.tolist() == ["000001"]
    assert frame.industry_category_code.tolist() == ["A"]
    assert frame.industry_code.tolist() == ["01"]
    if value is None or not isinstance(value, str) or value == "omitted":
        column = "company_name" if optional == "name" else "source_locator"
        assert frame[column].isna().all()
    assert _file_bytes(root) == before


def test_industry_invalid_asset_and_industry_cells_are_local(tmp_path):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    good = observations[0][4][0]
    observations[0][4].extend(
        [
            {**good, "code": "not-an-asset"},
            {key: value for key, value in good.items() if key != "code"},
            {**good, "code": "000002", "industry": "invalid"},
            {**good, "code": "000003", "category": "J", "industry": "66"},
        ]
    )
    root = write_industry_reference_fixture(tmp_path / "raw", observations=observations)
    before = _file_bytes(root)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=root.parent, output_roots={data_id: output}, data_ids=(data_id,)
    )
    frame = read_dataset(output).set_index("source_asset_id")
    assert set(frame.index) == {"000001", "000002", "000003"}
    assert frame.loc["000001", "industry_code"] == "01"
    assert frame.loc["000003", "industry_code"] == "66"
    assert (
        frame.loc[
            "000002", ["industry_category_code", "industry_major_code", "industry_code"]
        ]
        .isna()
        .all()
    )
    assert frame.loc["000002", "company_name"] == "synthetic issuer"
    diagnostics = read_dataset_manifest_evidence(output)["producer_diagnostics"]
    assert diagnostics["source_row_counts"] == {
        "excluded_missing_required_rows": 2,
        "invalid_industry_rows": 1,
    }
    assert _file_bytes(root) == before


@pytest.mark.parametrize("conflict", [None, "name", "industry"])
def test_industry_duplicate_reference_facts_coalesce_by_field(tmp_path, conflict):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    row = observations[0][4][0]
    duplicate = dict(row)
    if conflict is not None:
        duplicate[conflict] = "02" if conflict == "industry" else "other issuer label"
    observations[0][4].extend([duplicate, {**row, "code": "000002"}])
    root = write_industry_reference_fixture(tmp_path / "raw", observations=observations)
    before = _file_bytes(root)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=root.parent, output_roots={data_id: output}, data_ids=(data_id,)
    )
    frame = read_dataset(output).set_index("source_asset_id")
    assert len(frame) == 2
    assert frame.loc["000002", "industry_code"] == "01"
    assert frame.loc["000001", "industry_category_code"] == "A"
    if conflict == "industry":
        assert frame.loc[["000001"], "industry_code"].isna().all()
    else:
        assert frame.loc["000001", "industry_code"] == "01"
    if conflict == "name":
        assert frame.loc[["000001"], "company_name"].isna().all()
    assert _file_bytes(root) == before


@pytest.mark.parametrize("reason", ["outside_window", "invalid_asset"])
def test_industry_no_selected_records_publishes_typed_empty_reference(tmp_path, reason):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    if reason == "invalid_asset":
        observations[0][4][0]["code"] = "invalid"
    root = write_industry_reference_fixture(tmp_path / "raw", observations=observations)
    before = _file_bytes(root)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=root.parent,
        output_roots={data_id: output},
        data_ids=(data_id,),
        end_date="2020-01-03" if reason == "outside_window" else "2020-12-31",
    )
    frame = read_dataset(output)
    assert frame.empty
    assert tuple(frame.columns) == interim._INDUSTRY_COLUMNS
    for column in frame:
        expected = (
            "datetime64[ns]"
            if column in {"timestamp", "observation_date", "publication_date"}
            else "string"
        )
        assert str(frame[column].dtype) == expected
    assert _file_bytes(root) == before


def test_industry_unreadable_future_json_remains_a_source_boundary(tmp_path):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations() + (
        ("future", "CSRC2012", "2025-01-01", "2025-01-02", [{"code": "invalid"}]),
    )
    root = write_industry_reference_fixture(tmp_path / "raw", observations=observations)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=root.parent,
        output_roots={data_id: output},
        data_ids=(data_id,),
        end_date="2020-12-31",
    )
    before = _file_bytes(output)
    malformed = b'{"records": ['
    filesystem_write_bytes(root / "future.json", malformed)
    metadata = json.loads(
        Path(native_filesystem_path(root / "metadata.json")).read_text(encoding="utf-8")
    )
    metadata["files"]["future.json"] = hashlib.sha256(malformed).hexdigest()
    filesystem_write_text(
        root / "metadata.json", json.dumps(metadata), encoding="utf-8"
    )
    raw_before = _file_bytes(root)
    with pytest.raises(ValueError):
        interim.materialize_interim_data(
            raw_root=root.parent,
            output_roots={data_id: output},
            data_ids=(data_id,),
            end_date="2020-12-31",
        )
    assert _file_bytes(output) == before
    assert _file_bytes(root) == raw_before


@pytest.mark.parametrize("terminator", ("\n", "\r\n", "\r", "mixed"))
def test_fee_csv_native_newlines_and_unrestricted_metadata(tmp_path, terminator):
    import io
    from datetime import date, timedelta

    raw = _write_reference_root(tmp_path / "raw") / "reference"
    root = raw / "a_share_fee_schedule"
    path = root / A_SHARE_FEE_CSV_NAME
    rows = []
    for day in range(129):
        stamp = (date(2020, 1, 1) + timedelta(days=day)).isoformat()
        rows.append(_row("BROKER_COMMISSION", stamp, stamp, "沪深", "买入", 0.0003))
    rows[0]["说明"] = "x" * 16800 + '\rquoted, "text"\nnext\r\nline'
    output = io.StringIO(newline="")
    for index, record in enumerate(
        [dict(zip(FEE_CSV_COLUMNS, FEE_CSV_COLUMNS, strict=True)), *rows]
    ):
        ending = (
            ("\n", "\r", "\r\n")[index % 3] if terminator == "mixed" else terminator
        )
        csv.DictWriter(
            output, fieldnames=FEE_CSV_COLUMNS, lineterminator=ending
        ).writerow(record)
    payload = output.getvalue().encode("utf-8-sig")
    path.write_bytes(payload)
    metadata = dict(
        file_bytes=len(payload),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        row_count=len(rows),
        fields_raw=list(FEE_CSV_COLUMNS),
        notes="x" * 140000,
    )
    (root / A_SHARE_FEE_SIDECAR_NAME).write_text(json.dumps(metadata), encoding="utf-8")
    result = load_a_share_fee_source(reference_root=raw)
    assert len(result.rows) == 129
    assert result.csv_sha256 == metadata["file_sha256"]
    assert result.rows[0].rate == 0.0003
    assert path.read_bytes() == payload


def test_fee_source_touch_preserves_consumed_hash_and_byte_count(tmp_path, monkeypatch):
    import os

    from quant_research_platform.sources.reference import a_share_fees

    root = _write_reference_root(tmp_path / "raw") / "reference"
    expected = load_a_share_fee_source(reference_root=root)
    original = a_share_fees._read_rows

    def read_then_touch(path):
        result = original(path)
        for member in path.parent.iterdir():
            before = member.stat()
            os.utime(
                member, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000)
            )
        return result

    monkeypatch.setattr(a_share_fees, "_read_rows", read_then_touch)
    assert load_a_share_fee_source(reference_root=root) == expected
