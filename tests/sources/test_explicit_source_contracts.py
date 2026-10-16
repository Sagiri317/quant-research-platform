"""Source selection stays independent of research dates and storage names."""

import hashlib
import json

import pandas as pd
import pytest

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.reference import interim
from quant_research_platform.storage.datasets import read_dataset
from tests.sources.reference.test_equity_events import write_equity_source
from tests.sources.reference.test_interim import _write_reference_root


def test_default_csmar_extent_and_explicit_state_prefix(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dates = ("2010-01-04", "2020-01-06", "2026-01-05")
    (raw / "adjustment.csv").write_text(
        "Symbol,TradingDate,FwardFactor,BwardFactor,CumulateFwardFactor,CumulateBwardFactor\n"
        + "".join(f"000001,{day},1,1,1,1\n" for day in dates)
    )
    (raw / "market_cap.csv").write_text(
        "Stkcd,Shrchgdt,Nshrttl,Nshra\n"
        + "".join(f"000001,{day},120,100\n" for day in dates)
    )
    for suffix, window in (
        ("all", {}),
        ("window", {"start_date": "2020-01-01", "end_date": "2020-12-31"}),
    ):
        outputs = {
            name: str(tmp_path / suffix / name) for name in ("adjustment", "market_cap")
        }
        prepare_source(
            {"source_id": "csmar", "raw_root": str(raw), "outputs": outputs, **window}
        )
        adjustment = read_dataset(outputs["adjustment"])
        shares = read_dataset(outputs["market_cap"])
        expected_adjustment = dates if not window else dates[1:2]
        expected_shares = dates if not window else dates[:2]
        assert adjustment.trade_date.tolist() == list(
            pd.to_datetime(expected_adjustment)
        )
        assert shares.share_change_date.tolist() == list(
            pd.to_datetime(expected_shares)
        )


def test_reference_multiple_outputs_preserve_prefix_and_skip_independently(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    write_equity_source(raw)
    names = (
        interim.A_SHARE_FEE_SCHEDULE_DATA_ID,
        interim.A_SHARE_EQUITY_EVENTS_DATA_ID,
        interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
    )
    outputs = {name: str(tmp_path / name) for name in names}
    result = prepare_source(
        {
            "source_id": "reference",
            "raw_root": str(raw),
            "outputs": outputs,
            "end_date": "2024-12-31",
        }
    )
    assert set(result.artifact_roots) == set(names[:2])
    assert result.skipped == names[2:]
    assert read_dataset(outputs[names[1]]).timestamp.min() < pd.Timestamp("2024-01-01")
    fees = read_dataset(outputs[names[0]])
    assert fees.effective_date.min() == pd.Timestamp("2019-01-01")
    assert fees.end_date.max() == pd.Timestamp("2025-12-31")


def test_reference_rejects_fake_start_filter_before_publication(tmp_path):
    with pytest.raises(ValueError, match="reference_start_date_not_consumed"):
        prepare_source(
            {
                "source_id": "reference",
                "raw_root": str(tmp_path),
                "outputs": {
                    interim.A_SHARE_FEE_SCHEDULE_DATA_ID: str(tmp_path / "out")
                },
                "start_date": "2024-01-01",
            }
        )
    assert not (tmp_path / "out").exists()


def test_explicit_fee_pair_ignores_historical_name_and_checks_content(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    root = raw / interim.A_SHARE_FEE_SCHEDULE_DATA_ID
    csv = tmp_path / "actual-fees.csv"
    sidecar = tmp_path / "actual-evidence.json"
    (root / interim.A_SHARE_FEE_CSV_NAME).rename(csv)
    (root / interim.A_SHARE_FEE_SIDECAR_NAME).rename(sidecar)
    root.rmdir()
    csv.write_bytes(
        csv.read_bytes()
        .replace(b"2019-01-01", b"2026-01-01")
        .replace(b"2025-12-31", b"2027-12-31")
    )
    metadata = json.loads(sidecar.read_text())
    metadata["file_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    metadata["file_bytes"] = csv.stat().st_size
    sidecar.write_text(json.dumps(metadata))
    config = {
        "source_id": "reference",
        "raw_root": str(raw),
        "outputs": {interim.A_SHARE_FEE_SCHEDULE_DATA_ID: str(tmp_path / "data")},
        "fee_csv": str(csv),
        "fee_sidecar": str(sidecar),
    }
    prepare_source(config)
    saved = read_dataset(tmp_path / "data")
    assert len(saved) == 5
    assert saved.effective_date.min() == pd.Timestamp("2026-01-01")
    assert saved.end_date.max() == pd.Timestamp("2027-12-31")
    with pytest.raises(ValueError):
        read_dataset(tmp_path / "data" / "data")
    metadata = json.loads(sidecar.read_text())
    metadata["file_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata))
    config["outputs"][interim.A_SHARE_FEE_SCHEDULE_DATA_ID] = str(tmp_path / "bad")
    with pytest.raises(ValueError, match="a_share_fee_source_metadata_invalid"):
        prepare_source(config)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("field", ["fee_csv", "fee_sidecar"])
def test_explicit_fee_files_require_pair(tmp_path, field):
    with pytest.raises(ValueError, match="a_share_fee_source_pair_required"):
        prepare_source(
            {
                "source_id": "reference",
                "raw_root": str(tmp_path),
                "outputs": {
                    interim.A_SHARE_FEE_SCHEDULE_DATA_ID: str(tmp_path / "out")
                },
                field: str(tmp_path / "member"),
            }
        )


def test_named_data_root_still_rejects_symlink(tmp_path):
    raw = _write_reference_root(tmp_path / "raw") / "reference"
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "data").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_source(
            {
                "source_id": "reference",
                "raw_root": str(raw),
                "outputs": {
                    interim.A_SHARE_FEE_SCHEDULE_DATA_ID: str(tmp_path / "data")
                },
            }
        )
    assert not list(target.iterdir())
