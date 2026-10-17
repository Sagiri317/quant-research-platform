"""Industry uncertainty stays on its causal asset frontier, preserving prices."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import panels
from tests.helpers.a_share_sources import _synthetic_interim_frames

INDUSTRY_COLUMNS = [
    "industry_standard_id",
    "industry_level1_code",
    "industry_level1_name",
    "is_nonfinancial_industry",
]


def _dictionary():
    return pd.DataFrame(
        [
            (standard, code, name, parent, level)
            for standard in ("CSRC2012", "CAPCO")
            for code, name, parent, level in (
                ("A", "agriculture", "", "1"),
                ("A01", "crops", "A", "2"),
                ("J", "finance", "", "1"),
                ("J66", "banking", "J", "2"),
            )
        ],
        columns=[
            "industry_standard_id",
            "industry_code",
            "industry_name",
            "parent_industry_code",
            "industry_level",
        ],
    )


def _snapshot(
    asset, available, *, observed="2020-01-01", standard="CSRC2012", code="A01"
):
    available = pd.Timestamp(available)
    return {
        "source_asset_id": asset,
        "timestamp": available,
        "observation_date": pd.Timestamp(observed),
        "publication_date": available - pd.Timedelta(days=1),
        "industry_standard_id": standard,
        "industry_category_code": code[0],
        "industry_major_code": "",
        "industry_code": code[1:],
        "company_name": None,
        "snapshot_id": f"{asset}-{standard}-{code}-{available.date()}",
        "source_locator": None,
    }


def _panel(snapshots, *, dictionary=None, history=None, changes=None, versions=None):
    dates = pd.bdate_range("2020-01-02", periods=6).as_unit("ns")
    inputs = _synthetic_interim_frames(dates)
    stocks = inputs["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    if versions is None:
        versions = pd.DataFrame(
            {
                "industry_standard_id": ["CSRC2012", "CAPCO"],
                "implement_date": pd.to_datetime(["2012-10-26", "2012-10-26"]),
                "abandon_date": pd.Series(
                    pd.NaT, index=range(2), dtype="datetime64[ns]"
                ),
            }
        )
    industry = None if snapshots is None else pd.DataFrame(snapshots)
    return panels.compute_market_panel(
        inputs["market_bars_1d"],
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(stocks),
            market_cap=inputs["market_cap"],
            trading_calendar=inputs["trading_calendar"],
            industry_history=history,
            industry_changes=changes,
            industry_sets=_dictionary() if dictionary is None else dictionary,
            industry_versions=versions,
            industry_snapshots=industry,
        ),
    )


def _assert_prices_unchanged(actual):
    baseline = _panel(None)
    pd.testing.assert_frame_equal(
        actual.drop(columns=INDUSTRY_COLUMNS), baseline.drop(columns=INDUSTRY_COLUMNS)
    )
    assert len(actual) == 18
    assert actual.close.notna().all()


@pytest.mark.parametrize("conflict", ["value", "standard"])
@pytest.mark.parametrize("reverse", [False, True])
def test_public_conflict_starts_when_both_facts_are_known_then_recovers(
    conflict, reverse
):
    records = [
        _snapshot("000001", "2020-01-02"),
        _snapshot("000002", "2020-01-02", code="J66"),
        _snapshot("000003", "2020-01-02"),
        _snapshot(
            "000001",
            "2020-01-06",
            standard="CAPCO" if conflict == "standard" else "CSRC2012",
            code="J66",
        ),
        _snapshot("000001", "2020-01-08", observed="2020-01-07", code="J66"),
    ]
    if reverse:
        records.reverse()
    actual = _panel(records)
    affected = actual.query("asset_id == '000001'").reset_index(drop=True)
    assert affected.industry_level1_code.fillna("unknown").tolist() == [
        "A",
        "A",
        "unknown",
        "unknown",
        "J",
        "J",
    ]
    assert affected.iloc[2:4][INDUSTRY_COLUMNS].isna().all().all()
    assert affected.iloc[[0, 1]].is_nonfinancial_industry.tolist() == [True, True]
    assert affected.iloc[4:].is_nonfinancial_industry.tolist() == [False, False]
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert (
        actual.query("asset_id == '000003'").industry_level1_code.tolist() == ["A"] * 6
    )
    _assert_prices_unchanged(actual)


@pytest.mark.parametrize("dictionary_conflict", [False, True])
def test_dictionary_duplicates_affect_only_the_consumed_path(dictionary_conflict):
    dictionary = _dictionary()
    duplicate = dictionary.query(
        "industry_standard_id == 'CSRC2012' and industry_code == 'A01'"
    ).copy()
    if dictionary_conflict:
        duplicate["parent_industry_code"] = "J"
    dictionary = pd.concat([dictionary, duplicate], ignore_index=True)
    actual = _panel(
        [
            _snapshot("000001", "2020-01-02"),
            _snapshot("000001", "2020-01-08", observed="2020-01-07", code="J66"),
            _snapshot("000002", "2020-01-02", code="J66"),
            _snapshot("000003", "2020-01-02", standard="CAPCO"),
        ],
        dictionary=dictionary,
    )
    affected = actual.query("asset_id == '000001'")
    if dictionary_conflict:
        assert affected.iloc[:4][INDUSTRY_COLUMNS].isna().all().all()
    else:
        assert affected.iloc[:4].industry_level1_code.tolist() == ["A"] * 4
    assert affected.iloc[4:].industry_level1_code.tolist() == ["J", "J"]
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert (
        actual.query("asset_id == '000003'").industry_level1_code.tolist() == ["A"] * 6
    )
    _assert_prices_unchanged(actual)


def test_unknown_industry_is_a_barrier_until_newer_known_observation():
    unknown = _snapshot("000001", "2020-01-06", observed="2020-01-05")
    unknown["industry_category_code"] = None
    unknown["industry_code"] = None
    actual = _panel(
        [
            _snapshot("000001", "2020-01-02"),
            unknown,
            # A late copy of the older fact cannot fill a newer unknown state.
            _snapshot("000001", "2020-01-07"),
            _snapshot("000001", "2020-01-08", observed="2020-01-07", code="J66"),
            _snapshot("000002", "2020-01-02", code="J66"),
        ]
    )
    affected = actual.query("asset_id == '000001'")
    assert affected.industry_level1_code.fillna("unknown").tolist() == [
        "A",
        "A",
        "unknown",
        "unknown",
        "J",
        "J",
    ]
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert actual.query("asset_id == '000003'")[INDUSTRY_COLUMNS].isna().all().all()
    _assert_prices_unchanged(actual)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_standard_history_conflict_is_local_and_recovers(reverse):
    history = pd.DataFrame(
        [
            ("000001", "2019-12-30", "A01", "crops"),
            ("000001", "2020-01-03", "A01", "shared reported name"),
            ("000001", "2020-01-03", "J66", "shared reported name"),
            ("000001", "2020-01-07", "J66", "banking"),
            ("000002", "2020-01-01", "J66", "banking"),
            ("000003", "2020-01-01", "A01", "crops"),
        ],
        columns=[
            "source_asset_id",
            "timestamp",
            "industry_level1_code",
            "industry_level1_name",
        ],
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["industry_standard_id"] = "CSRC2012"
    changes = history.drop_duplicates(["source_asset_id", "timestamp"]).rename(
        columns={"timestamp": "effective_date", "industry_level1_name": "value_after"}
    )
    changes["changed_item"] = "P0207行业代码"
    changes["announcement_date"] = pd.to_datetime(
        ["2020-01-01", "2020-01-05", "2020-01-07", "2020-01-01", "2020-01-01"]
    )
    if reverse:
        history = history.iloc[::-1].copy()
        changes = changes.iloc[::-1].copy()
    history_before = history.copy(deep=True)
    changes_before = changes.copy(deep=True)
    actual = _panel(None, history=history, changes=changes)
    affected = actual.query("asset_id == '000001'")
    assert affected.industry_level1_code.fillna("unknown").tolist() == [
        "A",
        "A",
        "unknown",
        "unknown",
        "J",
        "J",
    ]
    assert affected.iloc[2:4][INDUSTRY_COLUMNS].isna().all().all()
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert (
        actual.query("asset_id == '000003'").industry_level1_code.tolist() == ["A"] * 6
    )
    pd.testing.assert_frame_equal(history, history_before)
    pd.testing.assert_frame_equal(changes, changes_before)
    _assert_prices_unchanged(actual)


@pytest.mark.parametrize("broken_path", ["cycle", "missing_parent", "missing_node"])
@pytest.mark.parametrize("source", ["snapshot", "history"])
def test_unresolvable_dictionary_path_is_unknown_without_harming_other_assets(
    broken_path, source
):
    dictionary = _dictionary()
    selected = dictionary.industry_standard_id.eq(
        "CSRC2012"
    ) & dictionary.industry_code.eq("A")
    if broken_path == "cycle":
        dictionary.loc[selected, ["industry_level", "parent_industry_code"]] = [
            "2",
            "A01",
        ]
    elif broken_path == "missing_parent":
        dictionary.loc[selected, ["industry_level", "parent_industry_code"]] = [
            "2",
            "unavailable",
        ]
    else:
        dictionary = dictionary.loc[~selected].copy()
    snapshots = [
        _snapshot("000002", "2020-01-02", code="J66"),
        _snapshot("000003", "2020-01-02", standard="CAPCO"),
    ]
    history = changes = None
    if source == "snapshot":
        snapshots.append(_snapshot("000001", "2020-01-02"))
    else:
        history = pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": pd.to_datetime(["2020-01-01"]),
                "industry_standard_id": ["CSRC2012"],
                "industry_level1_code": ["A01"],
                "industry_level1_name": ["crops"],
            }
        )
        changes = history.rename(
            columns={
                "timestamp": "effective_date",
                "industry_level1_name": "value_after",
            }
        ).assign(
            changed_item="P0207行业代码", announcement_date=pd.Timestamp("2020-01-01")
        )
    actual = _panel(snapshots, dictionary=dictionary, history=history, changes=changes)
    assert actual.query("asset_id == '000001'")[INDUSTRY_COLUMNS].isna().all().all()
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert (
        actual.query("asset_id == '000003'").industry_level1_code.tolist() == ["A"] * 6
    )
    _assert_prices_unchanged(actual)


def test_coalesced_reference_conflict_remains_unknown_until_newer_evidence(tmp_path):
    from quant_research_platform.sources.reference import interim
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    agriculture = {"code": "000001", "category": "A", "major": "", "industry": "01"}
    finance = {"code": "000001", "category": "J", "major": "", "industry": "66"}
    raw = write_industry_reference_fixture(
        tmp_path / "raw",
        observations=(
            (
                "earlier",
                "CSRC2012",
                "2020-01-01",
                "2020-01-01",
                [agriculture, {**finance, "code": "000002"}],
            ),
            (
                "conflicted",
                "CSRC2012",
                "2020-01-05",
                "2020-01-05",
                [agriculture, finance],
            ),
            ("recovered", "CSRC2012", "2020-01-07", "2020-01-07", [finance]),
        ),
    )
    before = {path: path.read_bytes() for path in raw.iterdir()}
    data_id = interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    output = tmp_path / "industry"
    interim.materialize_interim_data(
        raw_root=raw.parent, output_roots={data_id: output}, data_ids=(data_id,)
    )
    snapshots = read_dataset(output)
    conflicted = snapshots.query("snapshot_id == 'conflicted'")
    assert len(conflicted) == 1
    assert conflicted[["industry_category_code", "industry_code"]].isna().all().all()
    actual = _panel(snapshots.to_dict("records"))
    affected = actual.query("asset_id == '000001'")
    assert affected.industry_level1_code.fillna("unknown").tolist() == [
        "A",
        "A",
        "unknown",
        "unknown",
        "J",
        "J",
    ]
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert actual.query("asset_id == '000003'")[INDUSTRY_COLUMNS].isna().all().all()
    _assert_prices_unchanged(actual)
    assert {path: path.read_bytes() for path in raw.iterdir()} == before


def _raw_version(standard, abandon, *, implement="2012-10-26"):
    return {
        "IndClassifySystemCode": standard,
        "IndClassifySystem": standard,
        "ImplementDate": implement,
        "AbandonDate": abandon,
        "InstitutionName": "synthetic classification publisher",
    }


def _version_inputs(rows, route, tmp_path):
    from quant_research_platform.sources.csmar import interim
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_raw import _write_csmar

    if route == "direct":
        return pd.DataFrame(rows).rename(
            columns={
                "IndClassifySystemCode": "industry_standard_id",
                "ImplementDate": "implement_date",
                "AbandonDate": "abandon_date",
            }
        )
    if route == "normalize":
        return interim.normalize_raw_frame(
            pd.DataFrame(rows), "industry_classification_versions"
        )
    raw = tmp_path / "raw"
    raw.mkdir()
    data_id = "industry_classification_versions"
    _write_csmar(raw, data_id, rows)
    before = {path: path.read_bytes() for path in raw.iterdir()}
    output = tmp_path / "versions"
    interim.materialize_interim_data(
        raw_root=raw,
        output_roots={data_id: output},
        data_ids=(data_id,),
        chunksize=1,
    )
    assert {path: path.read_bytes() for path in raw.iterdir()} == before
    return read_dataset(output)


def _version_panel(versions):
    return _panel(
        [
            _snapshot("000001", "2020-01-02"),
            _snapshot("000002", "2020-01-02", standard="CAPCO", code="J66"),
            _snapshot("000003", "2020-01-02", standard="CAPCO"),
        ],
        versions=versions,
    )


def _assert_other_standard_unchanged(actual):
    assert (
        actual.query("asset_id == '000002'").industry_level1_code.tolist() == ["J"] * 6
    )
    assert (
        actual.query("asset_id == '000003'").industry_level1_code.tolist() == ["A"] * 6
    )
    _assert_prices_unchanged(actual)


@pytest.mark.parametrize(
    "end,unresolved",
    [
        (None, False),
        ("", False),
        (" \t", False),
        ("invalid", True),
        ("2020-02-30", True),
    ],
)
def test_version_source_distinguishes_open_end_from_invalid_nonempty_end(
    end, unresolved
):
    from quant_research_platform.sources.csmar.interim import normalize_raw_frame

    raw = pd.DataFrame([_raw_version("P0207", end)])
    before = raw.copy(deep=True)
    actual = normalize_raw_frame(raw, "industry_classification_versions")
    assert len(actual) == 1
    assert actual.abandon_date.isna().all()
    assert actual.abandon_date_unresolved.tolist() == [unresolved]
    assert actual.implement_date.tolist() == [pd.Timestamp("2012-10-26")]
    assert actual.timestamp.tolist() == [pd.Timestamp("2012-10-26")]
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("route", ["direct", "normalize"])
@pytest.mark.parametrize(
    "end,expected",
    [
        (None, ["A"] * 6),
        ("", ["A"] * 6),
        (" \t", ["A"] * 6),
        ("invalid", ["unknown"] * 6),
        ("2020-02-30", ["unknown"] * 6),
        ("2020-01-06", ["A", "A", "unknown", "unknown", "unknown", "unknown"]),
    ],
)
def test_version_end_uncertainty_is_local_to_its_standard(
    tmp_path, route, end, expected
):
    versions = _version_inputs(
        [_raw_version("CSRC2012", end), _raw_version("CAPCO", None)], route, tmp_path
    )
    before = versions.copy(deep=True)
    actual = _version_panel(versions)
    assert (
        actual.query("asset_id == '000001'")
        .industry_level1_code.fillna("unknown")
        .tolist()
        == expected
    )
    affected_unknown = actual.asset_id.eq("000001") & actual.industry_level1_code.isna()
    assert actual.loc[affected_unknown, INDUSTRY_COLUMNS].isna().all().all()
    _assert_other_standard_unchanged(actual)
    pd.testing.assert_frame_equal(versions, before)


@pytest.mark.parametrize("route", ["direct", "materialize"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("other_end", ["2020-01-08", "", "invalid"])
def test_conflicting_version_ends_preserve_only_common_valid_interval(
    tmp_path, route, reverse, other_end
):
    rows = [
        _raw_version("CSRC2012", "2020-01-06"),
        _raw_version("CSRC2012", other_end),
        _raw_version("CAPCO", None),
    ]
    if reverse:
        rows.reverse()
    versions = _version_inputs(rows, route, tmp_path)
    canonical = panels._canonical_industry_versions(versions, {"CSRC2012", "CAPCO"})
    affected_version = canonical.query("industry_standard_id == 'CSRC2012'")
    assert len(affected_version) == 1
    assert affected_version.abandon_date_unresolved.tolist() == [True]
    if other_end == "invalid":
        assert affected_version.abandon_date.isna().all()
        expected = ["unknown"] * 6
    else:
        assert affected_version.abandon_date.tolist() == [pd.Timestamp("2020-01-06")]
        expected = ["A", "A", "unknown", "unknown", "unknown", "unknown"]
    actual = _version_panel(versions)
    assert (
        actual.query("asset_id == '000001'")
        .industry_level1_code.fillna("unknown")
        .tolist()
        == expected
    )
    _assert_other_standard_unchanged(actual)


@pytest.mark.parametrize("route", ["direct", "materialize"])
@pytest.mark.parametrize(
    "future_case", ["malformed_later_version", "conflicting_later_ends"]
)
def test_future_version_uncertainty_does_not_rewrite_earlier_classification(
    tmp_path, route, future_case
):
    baseline = _version_panel(None)
    rows = [_raw_version("CAPCO", None)]
    if future_case == "malformed_later_version":
        rows += [
            _raw_version("CSRC2012", "2025-01-01"),
            _raw_version("CSRC2012", "invalid", implement="2025-01-01"),
        ]
    else:
        rows += [
            _raw_version("CSRC2012", "2025-01-01"),
            _raw_version("CSRC2012", "2026-01-01"),
        ]
    versions = _version_inputs(rows, route, tmp_path)
    actual = _version_panel(versions)
    pd.testing.assert_frame_equal(actual, baseline)
    _assert_other_standard_unchanged(actual)
