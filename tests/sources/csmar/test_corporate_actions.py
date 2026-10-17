"""Corporate-action sources preserve event identity and incomplete local facts."""

import csv

import pandas as pd
import pytest

from quant_research_platform.sources.csmar.field_mappings import RAW_TABLE_CONTRACTS
from quant_research_platform.sources.csmar.interim import (
    interim_catalog,
    normalize_raw_frame,
)
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

HEADER = RAW_TABLE_CONTRACTS["corporate_actions"].required_columns


def row(kind="CA", **changes):
    values = dict(
        zip(
            HEADER,
            ("1", kind, "2024-06-03", "4", "2024-05-20", "2024-06-05", "0.25", ""),
            strict=True,
        )
    )
    return values | changes


def prepare(tmp_path, rows, chunksize=1):
    raw = tmp_path / "raw_only" / "TRDNEW_Cptl"
    raw.mkdir(parents=True)
    path = raw / "raw__TRDNEW_Cptl__part-00__observed-20260902.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    original = path.read_bytes()
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "csmar",
            "raw_root": str(raw),
            "outputs": {"corporate_actions": str(output)},
            "chunksize": chunksize,
        }
    )
    assert path.read_bytes() == original
    return output


def test_same_day_distributions_and_exact_duplicates_survive_persistence(tmp_path):
    output = prepare(tmp_path, [row(), row("SD", Amount="0E-7"), row()])
    frame = read_dataset(output).sort_values("kind")
    assert frame.source_asset_id.tolist() == ["000001", "000001"]
    assert frame.kind.tolist() == ["CA", "SD"]
    assert frame.amount.tolist() == [0.25, 0.0]
    assert interim_catalog().get("corporate_actions").primary_key == (
        "source_asset_id",
        "timestamp",
        "kind",
    )
    metadata = read_dataset_manifest_evidence(output)
    assert metadata["producer_diagnostics"]["exact_duplicate_count"] == 1


def test_optional_bad_values_are_unknown_and_counted_not_dropped(tmp_path):
    output = prepare(
        tmp_path,
        [
            row(
                Annodt="bad",
                Paydt="bad",
                Amount="invalid",
                Roprc="inf",
                Markettype="bad",
            ),
            row("SD", Amount="-0.25", Paydt="1990-01-01"),
        ],
    )
    frame = read_dataset(output).sort_values("kind").reset_index(drop=True)
    assert len(frame) == 2
    assert (
        frame.loc[
            0,
            [
                "announcement_date",
                "settlement_date",
                "amount",
                "subscription_price",
                "market_type",
            ],
        ]
        .isna()
        .all()
    )
    assert frame.settlement_date_unresolved.tolist() == [True, False]
    assert frame.loc[1, "amount"] == -0.25
    assert frame.loc[1, "settlement_date"] == pd.Timestamp("1990-01-01")
    metadata = read_dataset_manifest_evidence(output)
    counts = metadata["producer_diagnostics"]["source_row_counts"]
    assert counts["raw_rows"] == 2
    assert counts["invalid_settlement_date_rows"] == 1
    assert counts["invalid_amount_rows"] == 1


def test_pre_ex_dates_and_source_amount_units_remain_facts():
    frame = normalize_raw_frame(
        pd.DataFrame([row(Paydt="2024-05-30", Amount="12.5")]), "corporate_actions"
    )
    assert frame.loc[0, "amount"] == 12.5
    assert frame.loc[0, "settlement_date"] == pd.Timestamp("2024-05-30")
    assert "record_date" not in frame
    assert RAW_TABLE_CONTRACTS["equity_changes"].table_code == "FN_FN046"


@pytest.mark.parametrize(
    "changes",
    [{"Exdistdt": "bad"}, {"Stkcd": "bad"}, {"Disttyp": ""}],
)
def test_invalid_event_key_excludes_only_unlocatable_row(tmp_path, changes):
    output = prepare(tmp_path, [row(**changes), row(Stkcd="2", Amount="0.75")])
    frame = read_dataset(output)
    assert frame.source_asset_id.tolist() == ["000002"]
    assert frame.amount.tolist() == [0.75]
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == 2
    assert counts["excluded_missing_required_rows"] == 1
    assert counts["retained_after_date_filter_rows"] == 1


@pytest.mark.parametrize(
    "changes", [{"Exdistdt": "bad"}, {"Stkcd": "bad"}, {"Disttyp": ""}]
)
def test_all_unlocatable_events_publish_typed_empty(tmp_path, changes):
    output = prepare(tmp_path, [row(**changes)])
    frame = read_dataset(output)
    assert frame.empty
    assert str(frame.timestamp.dtype) == "datetime64[ns]"
    assert str(frame.market_type.dtype) == "Int64"
    assert str(frame.amount.dtype) == "float64"
    assert str(frame.settlement_date_unresolved.dtype) == "bool"
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["excluded_missing_required_rows"] == 1
    assert counts["retained_after_date_filter_rows"] == 0


@pytest.mark.parametrize("chunksize", (1, 100))
def test_conflicting_event_amount_keeps_key_and_independent_facts(tmp_path, chunksize):
    output = prepare(
        tmp_path, [row(), row(Amount="0.5"), row(Stkcd="2", Amount="0.75")], chunksize
    )
    frame = read_dataset(output).set_index("source_asset_id")
    assert len(frame) == 2
    assert pd.isna(frame.loc["000001", "amount"])
    assert frame.loc["000001", "timestamp"] == pd.Timestamp("2024-06-03")
    assert frame.loc["000001", "kind"] == "CA"
    assert frame.loc["000001", "settlement_date"] == pd.Timestamp("2024-06-05")
    assert not frame.loc["000001", "settlement_date_unresolved"]
    assert frame.loc["000002", "amount"] == 0.75
    assert "version_identity" not in frame
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["conflicting_amount_rows"] == 2
    assert counts["raw_rows"] == 3


@pytest.mark.parametrize("payments", [("bad", ""), ("2024-06-05", "2024-06-06")])
@pytest.mark.parametrize("chunksize", (1, 100))
def test_unresolved_provided_payment_date_never_becomes_genuine_blank(
    tmp_path, payments, chunksize
):
    output = prepare(
        tmp_path,
        [row(Paydt=value) for value in payments] + [row(Stkcd="2", Paydt="")],
        chunksize,
    )
    frame = read_dataset(output).set_index("source_asset_id")
    assert len(frame) == 2
    assert frame.settlement_date.isna().all()
    assert bool(frame.loc["000001", "settlement_date_unresolved"])
    assert not bool(frame.loc["000002", "settlement_date_unresolved"])
    assert frame.amount.tolist() == [0.25, 0.25]
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    if "bad" in payments:
        assert counts["invalid_settlement_date_rows"] == 1
    else:
        assert counts["conflicting_settlement_date_rows"] == 2


def test_empty_source_has_stable_nullable_schema(tmp_path):
    output = prepare(tmp_path, [])
    frame = read_dataset(output)
    assert frame.empty
    assert str(frame.timestamp.dtype) == "datetime64[ns]"
    assert str(frame.market_type.dtype) == "Int64"
    assert str(frame.amount.dtype) == "float64"


def test_raw_only_table_identity_rejects_missing_fields(tmp_path):
    raw = tmp_path / "raw_only"
    raw.mkdir()
    (raw / "raw__TRDNEW_Cptl__part-00__observed-20260902.csv").write_text(
        "Stkcd,Disttyp,Exdistdt\n000001,CA,2024-06-03\n"
    )
    with pytest.raises(ValueError, match="missing CSMAR columns"):
        prepare_source(
            {
                "source_id": "csmar",
                "raw_root": str(raw),
                "outputs": {"corporate_actions": str(tmp_path / "interim")},
            }
        )


def test_all_vendor_kinds_remain_distinct_source_events():
    kinds = ["CA", "SD", "RO", "DS", "GQ", "SN"]
    frame = normalize_raw_frame(
        pd.DataFrame([row(kind) for kind in kinds]), "corporate_actions"
    )
    assert set(frame.kind) == set(kinds)
    assert len(frame) == 6


def test_unknown_nonempty_kind_is_retained_with_diagnostics(tmp_path):
    output = prepare(tmp_path, [row("UNKNOWN")])
    assert read_dataset(output).kind.tolist() == ["UNKNOWN"]
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["unsupported_kind_rows"] == 1
