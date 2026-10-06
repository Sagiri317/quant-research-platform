from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.profiles.research import research_profile
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
    filesystem_exists,
    filesystem_glob,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.workflows import research


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
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
        "file_sha256": hashlib.sha256(filesystem_read_bytes(csv_path)).hexdigest(),
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


def test_fee_interim_target_routes_build_reuse_and_identity_invalidation(
    tmp_path: Path,
) -> None:
    raw_root = _write_reference_root(tmp_path / "raw")
    raw_identity = load_a_share_fee_source(
        reference_root=raw_root / "reference"
    ).source_identity
    layout = StorageLayout(project_root=tmp_path)
    profile = research_profile()
    plan = research.plan_targets(
        profile=profile,
        action="data materialize",
        terminal_targets=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
        layout=layout,
        raw_root=raw_root,
    )
    planned = plan["targets"][0]
    assert planned["decision"] == "BUILD"
    assert planned["requested_identity"]
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    first_target = first["target_results"][0]
    assert first_target["status"] == "BUILT"
    first_frame = read_dataset(
        layout.instance_root(
            layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID),
            first_target["instance_identity"],
        )
    )
    manifest = read_dataset_manifest_evidence(
        layout.instance_root(
            layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID),
            first_target["instance_identity"],
        )
    )
    assert (
        manifest["common_manifest"]["semantic_metadata"]["raw_content_identity"]
        == raw_identity
    )
    assert tuple(first_frame.columns) == (
        "fee_code",
        "effective_date",
        "end_date",
        "market",
        "side",
        "basis",
        "rate",
        "included_in_ledger",
        "included_in",
    )
    second = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    assert second["target_results"][0]["status"] == "REUSED"
    sidecar_path = (
        raw_root / "reference" / "a_share_fee_schedule" / A_SHARE_FEE_SIDECAR_NAME
    )
    sidecar = json.loads(filesystem_read_text(sidecar_path, encoding="utf-8"))
    sidecar["notes"] = "identity mutation"
    filesystem_write_text(
        sidecar_path, json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )
    third = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    assert third["target_results"][0]["status"] == "BUILT"
    assert (
        third["target_results"][0]["instance_identity"]
        != first_target["instance_identity"]
    )


@pytest.mark.parametrize("mutation", ("planned", "loaded", "staged"))
@pytest.mark.parametrize("existing_target", (False, True))
def test_fee_source_identity_race_preserves_old_good(
    tmp_path, monkeypatch, mutation, existing_target
):
    raw_root = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(project_root=tmp_path)
    built = research.data_materialize(
        profile=research_profile(),
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    identity = built["target_results"][0]["instance_identity"]
    root = layout.instance_root(
        layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID), identity
    )
    metadata = {
        "common_manifest": read_dataset_manifest_evidence(root)["common_manifest"]
    }
    old_good = _file_bytes(root)
    target = (
        root
        if existing_target
        else StorageLayout(tmp_path / "candidate").interim_data_root(
            "reference", A_SHARE_FEE_SCHEDULE_DATA_ID
        )
    )
    target_layout = layout.with_target_roots({A_SHARE_FEE_SCHEDULE_DATA_ID: target})
    filesystem_mkdir(target.parent, parents=True, exist_ok=True)
    unrelated = target.parent / "unrelated.txt"
    filesystem_write_text(unrelated, "keep", encoding="utf-8")
    previous_members = set(filesystem_iterdir(target.parent))
    if mutation == "planned":
        _write_reference_root(raw_root, commission=0.0004)
    elif mutation == "loaded":
        original = interim.load_a_share_fee_source
        calls = 0

        def load(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                _write_reference_root(raw_root, commission=0.0004)
            return original(**kwargs)

        monkeypatch.setattr(interim, "load_a_share_fee_source", load)
    else:
        original_frame = interim._frame_from_rows

        def frame(rows):
            value = original_frame(rows)
            _write_reference_root(raw_root, commission=0.0004)
            return value

        monkeypatch.setattr(interim, "_frame_from_rows", frame)
    expected_error = (
        "dataset_exists"
        if existing_target and mutation == "staged"
        else "reference_raw_identity_changed"
    )
    with pytest.raises(ValueError, match=expected_error):
        interim.materialize_interim_data(
            layout=target_layout,
            raw_root=raw_root / "reference",
            instance_metadata=metadata,
        )
    assert _file_bytes(root) == old_good
    assert set(filesystem_iterdir(target.parent)) == previous_members
    assert filesystem_read_text(unrelated, encoding="utf-8") == "keep"
    if not existing_target:
        assert not filesystem_exists(target)


def test_fee_identity_rechecks_sidecar_after_rows(tmp_path, monkeypatch):
    from quant_research_platform.sources.reference import a_share_fees

    root = _write_reference_root(tmp_path / "raw") / "reference"
    sidecar = next(filesystem_glob(root, "**/*.metadata.json"))
    original = a_share_fees._read_rows

    def read_then_change(*args):
        rows = original(*args)
        filesystem_write_bytes(sidecar, filesystem_read_bytes(sidecar) + b" ")
        return rows

    monkeypatch.setattr(a_share_fees, "_read_rows", read_then_change)
    with pytest.raises(ValueError, match="identity"):
        load_a_share_fee_source(reference_root=root)


def test_fee_reader_cleanup_preserves_primary(tmp_path, monkeypatch):
    import io

    from quant_research_platform.sources.reference import a_share_fees

    root = _write_reference_root(tmp_path / "raw") / "reference"
    path = next(filesystem_glob(root, "**/*.csv"))
    primary = MemoryError("primary")

    class Handle(io.BufferedReader):
        def read(self, *args):
            raise primary

        def close(self):
            super().close()
            raise OSError("secondary")

    handle = Handle(io.FileIO(native_filesystem_path(path)))
    monkeypatch.setattr(a_share_fees, "open_filesystem_file", lambda _: handle)
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
    from tests.markets.a_share.test_target_dag_chain import _synthetic_interim_frames

    raw_root = tmp_path / "raw"
    write_industry_reference_fixture(raw_root, observations=_industry_observations())
    layout = StorageLayout(tmp_path)
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    profile = research_profile()
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2019-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    target = result["target_results"][0]
    assert target["status"] == "BUILT"
    root = layout.instance_root(
        layout.interim_data_root("reference", data_id), target["instance_identity"]
    )
    snapshots = read_dataset(root)
    assert str(snapshots["timestamp"].dtype) == "datetime64[ns]"
    spec = json.loads(filesystem_read_text(root / "spec.json", encoding="utf-8"))
    assert spec["primary_key"] == ["source_asset_id", "timestamp", "snapshot_id"]
    assert snapshots["observation_date"].tolist() == [pd.Timestamp("2019-12-31")]
    assert snapshots["timestamp"].tolist() == [pd.Timestamp("2020-01-04")]
    manifest = read_dataset_manifest_evidence(root)
    assert manifest["common_manifest"]["semantic_metadata"][
        "raw_content_identity"
    ] == interim.source_content_identity(raw_root / "reference", data_id)
    dates = pd.bdate_range("2020-01-02", periods=5)
    inputs = _synthetic_interim_frames(dates)
    stocks = inputs["stock_master"].assign(market_type=1)
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
        stock_master=stocks,
        market_cap=inputs["market_cap"],
        trading_calendar=inputs["trading_calendar"],
        industry_sets=dictionary,
        industry_versions=versions,
        industry_snapshots=snapshots,
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
    again = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2019-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    assert again["target_results"][0]["status"] == "REUSED"


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
        layout=layout,
        raw_root=root / "reference",
        data_ids=(data_id,),
        start_date="2011-01-01",
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
        "duplicate_asset",
        "publication_before_observation",
    ],
)
def test_industry_reference_boundary_rejects_unproved_source(tmp_path, failure):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=_industry_observations()
    )
    metadata = json.loads(
        filesystem_read_text(root / "metadata.json", encoding="utf-8")
    )
    if failure == "missing_member":
        metadata["snapshots"][0]["document_file"] = "missing.pdf"
    elif failure == "bad_hash":
        metadata["files"]["snapshot.pdf"] = "0" * 64
    elif failure == "traversal":
        metadata["files"]["../snapshot.pdf"] = metadata["files"].pop("snapshot.pdf")
    elif failure == "publication_before_observation":
        metadata["snapshots"][0]["publication_date"] = "2019-12-01"
    else:
        candidate = json.loads(
            filesystem_read_text(root / "snapshot.json", encoding="utf-8")
        )
        candidate["records"].append(candidate["records"][0])
        filesystem_write_text(
            root / "snapshot.json", json.dumps(candidate), encoding="utf-8"
        )
        metadata["files"]["snapshot.json"] = hashlib.sha256(
            filesystem_read_bytes(root / "snapshot.json")
        ).hexdigest()
    filesystem_write_text(
        root / "metadata.json", json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="industry_snapshot_"):
        interim.materialize_interim_data(
            layout=StorageLayout(tmp_path),
            raw_root=root.parent,
            data_ids=(interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,),
        )


def test_industry_source_change_before_publish_preserves_old_good(
    tmp_path, monkeypatch
):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=_industry_observations()
    )
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    layout = StorageLayout(tmp_path)
    interim.materialize_interim_data(
        layout=layout, raw_root=root.parent, data_ids=(data_id,)
    )
    output = layout.interim_data_root("reference", data_id)
    before = _file_bytes(output)
    candidate_layout = StorageLayout(tmp_path / "candidate")
    candidate_root = candidate_layout.interim_data_root("reference", data_id)
    writer = interim.write_dataset

    def mutate_source(*args, **kwargs):
        filesystem_write_text(
            root / "snapshot.html", "<html>changed</html>", encoding="utf-8"
        )
        return writer(*args, **kwargs)

    monkeypatch.setattr(interim, "write_dataset", mutate_source)
    with pytest.raises(ValueError, match="industry_snapshot_member_hash_mismatch"):
        interim.materialize_interim_data(
            layout=candidate_layout, raw_root=root.parent, data_ids=(data_id,)
        )
    assert not filesystem_exists(candidate_root)
    assert _file_bytes(output) == before
    assert read_dataset(output)["snapshot_id"].tolist() == ["snapshot"]


@pytest.mark.parametrize("document_suffix", [".docx", ".xls"])
def test_industry_reference_preserves_table_row_locator(tmp_path, document_suffix):
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    observations = _industry_observations()
    observations[0][4][0]["source_locator"] = "table:2;row:7"
    root = write_industry_reference_fixture(
        tmp_path / "raw", observations=observations, document_suffix=document_suffix
    )
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    layout = StorageLayout(tmp_path)
    interim.materialize_interim_data(
        layout=layout, raw_root=root.parent, data_ids=(data_id,)
    )
    frame = read_dataset(layout.interim_data_root("reference", data_id))
    assert frame["source_locator"].tolist() == ["table:2;row:7"]
    assert "source_page" not in frame


def test_invalid_fee_csv_header_preserves_previous_artifact(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    layout = StorageLayout(project_root=tmp_path / "project")
    result = interim.materialize_interim_data(layout=layout, raw_root=raw)
    root = result.artifact_roots[A_SHARE_FEE_SCHEDULE_DATA_ID]
    before = _file_bytes(root)
    source = raw / "a_share_fee_schedule"
    csv_path = source / A_SHARE_FEE_CSV_NAME
    original = csv_path.read_bytes()
    hostile = b"unexpected_column," + original
    csv_path.write_bytes(hostile)
    sidecar = source / A_SHARE_FEE_SIDECAR_NAME
    metadata = json.loads(sidecar.read_text())
    metadata["file_sha256"] = hashlib.sha256(hostile).hexdigest()
    metadata["file_bytes"] = len(hostile)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="a_share_fee_source_schema_invalid"):
        interim.materialize_interim_data(layout=layout, raw_root=raw)
    assert _file_bytes(root) == before


def test_fee_csv_bounded_decoder_preserves_quoted_multiline_fields(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    source = raw / "a_share_fee_schedule"
    path = source / A_SHARE_FEE_CSV_NAME
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["说明"] = 'synthetic, quoted "field"\nsecond line'
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = source / A_SHARE_FEE_SIDECAR_NAME
    metadata = json.loads(sidecar.read_text())
    payload = path.read_bytes()
    metadata.update(
        file_bytes=len(payload), file_sha256=hashlib.sha256(payload).hexdigest()
    )
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    result = load_a_share_fee_source(reference_root=raw)
    assert len(result.rows) == 5
