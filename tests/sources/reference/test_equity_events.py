import hashlib
import json

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import EQUITY_EVENT_COLUMNS
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import decode_exact_numeric
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.research.portfolios import production
from quant_research_platform.sources.reference import interim
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.paths import StorageLayout
from tests.helpers.portfolio_inputs import _interim_specs, _processed_specs
from tests.helpers.source_preparation import prepared_source_evidence


def write_equity_source(raw_root, *, events=None):
    root = raw_root / "a_share_equity_events"
    root.mkdir(parents=True, exist_ok=True)
    document = (
        b"Synthetic official evidence. CNY per share; ordinary holder share quantities."
    )
    (root / "evidence.txt").write_bytes(document)
    event = {
        "event_id": "quote-1",
        "asset_id": "000001",
        "listing_date": "2022-12-01",
        "kind": "quote",
        "effective_date": "2023-01-03",
        "publication_date": "2023-01-03",
        "observed_at": "2026-10-13T00:00:00Z",
        "currency": "CNY",
        "amount_per_share": 2.5,
        "remaining_old_units_per_old_unit": 1.0,
        "successor_asset_id": None,
        "successor_listing_date": None,
        "successor_shares_per_old_share": None,
        "source_url": "https://www.sse.com.cn/synthetic-equity-evidence",
        "document_file": "evidence.txt",
        "source_locator": "page:1",
    }
    metadata = {
        "schema_id": "a_share_equity_events",
        "files": {"evidence.txt": hashlib.sha256(document).hexdigest()},
        "events": [event] if events is None else events(event),
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def test_equity_event_raw_to_persisted_normalization_preserves_units_and_dates(
    tmp_path,
):
    raw = tmp_path / "raw"

    def events(base):
        return [
            base,
            dict(base, event_id="cash", kind="cash_recovery", amount_per_share=3.0),
            dict(
                base,
                event_id="cancel",
                kind="cancel",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
            ),
            dict(
                base,
                event_id="convert",
                kind="conversion",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
                successor_asset_id="000002",
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=0.5,
            ),
        ]

    write_equity_source(raw, events=events)
    identity = prepared_source_evidence(interim, raw, "a_share_equity_events")
    result = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path).interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    frame = read_dataset(result.artifact_roots["a_share_equity_events"]).set_index(
        "event_id"
    )
    assert decode_exact_numeric(frame.loc["quote-1", "value_per_old_unit"]) == 2.5
    assert decode_exact_numeric(frame.loc["cash", "value_per_old_unit"]) == 3.0
    assert (
        decode_exact_numeric(frame.loc["convert", "successor_units_per_old_unit"])
        == 0.5
    )
    assert frame.timestamp.eq(pd.Timestamp("2023-01-04")).all()
    assert frame.effective_at.eq(pd.Timestamp("2023-01-03")).all()
    assert frame.observed_at.eq(pd.Timestamp("2026-10-13")).all()
    assert frame.evidence_identity.str.len().eq(64).all()
    assert prepared_source_evidence(interim, raw, "a_share_equity_events") == identity


@pytest.mark.parametrize(
    "damage",
    [
        "amount",
        "document",
        "hash",
        "future_publication",
        "currency",
        "cancel_quantity",
        "unproved_successor",
        "unofficial_source",
    ],
)
def test_equity_evidence_invalid_never_publishes_over_old_good(tmp_path, damage):
    raw = tmp_path / "raw"
    root = write_equity_source(raw)
    layout = StorageLayout(tmp_path)
    original = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    artifact = original.artifact_roots["a_share_equity_events"]
    before = read_dataset(artifact)
    path = root / "metadata.json"
    metadata = json.loads(path.read_text())
    event = metadata["events"][0]
    if damage == "amount":
        event["amount_per_share"] = None
    elif damage == "document":
        event["document_file"] = "missing.txt"
    elif damage == "hash":
        (root / "evidence.txt").write_bytes(b"replaced")
    elif damage == "future_publication":
        event["publication_date"] = "2027-01-01"
    elif damage == "currency":
        event["currency"] = "USD"
    elif damage == "cancel_quantity":
        event.update(kind="cancel", amount_per_share=None)
    elif damage == "unofficial_source":
        event["source_url"] = "https://example.com/evidence"
    else:
        event["successor_asset_id"] = "000002"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="equity_event"):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("reference", key)
                for key in ("a_share_equity_events",)
            },
            raw_root=raw,
            data_ids=("a_share_equity_events",),
        )
    pd.testing.assert_frame_equal(before, read_dataset(artifact))


def native_equity_inputs(
    tmp_path, *, events=None, panel_transform=None, fault="delisted"
):
    from tests.helpers.portfolio_inputs import closed_market_inputs as _native_inputs

    inputs, roots, calendar, _panel = _native_inputs(tmp_path, fault=fault)
    for target, input_key in (
        ("market_panel_1d", "panel"),
        ("market_bars_1d", "market_bars"),
        ("stock_master", "stock_master"),
    ):
        frame = read_dataset(roots[target])
        column = "asset_id" if target == "market_panel_1d" else "source_asset_id"
        frame[column] = frame[column].replace({"a0": "000001"})
        if target == "market_panel_1d":
            if fault == "delisted":
                frame = frame.loc[
                    ~(frame.asset_id.eq("000001") & frame.timestamp.gt("2023-01-02"))
                ].copy()
            selected = frame.asset_id.eq("000001")
            frame.loc[selected, "close"] = frame.loc[selected, "adj_close"] / 2
            if panel_transform is not None:
                frame = panel_transform(frame.copy())
        spec = (
            _processed_specs().get(target)
            if target == "market_panel_1d"
            else _interim_specs().get(target)
        )
        root = tmp_path / "renamed" / target
        write_dataset(root, frame, spec=spec)
        roots[target] = inputs[input_key] = root
    raw = tmp_path / "raw"

    write_equity_source(raw, events=events)
    imported = interim.materialize_interim_data(
        output_roots={"a_share_equity_events": tmp_path / "rights"},
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    inputs["equity_events"] = imported.artifact_roots["a_share_equity_events"]
    return (inputs, roots, calendar)


@pytest.mark.parametrize("adjusted_scale", [1, 3])
def test_native_source_to_session_carries_recovery_without_forged_panel_row(
    tmp_path, adjusted_scale
):
    layout, roots, calendar = native_equity_inputs(
        tmp_path,
        panel_transform=lambda frame: frame.assign(
            adj_close=frame.adj_close * adjusted_scale
        ),
    )
    source = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        equity_events_root=layout["equity_events"],
        panel_root=roots["market_panel_1d"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
    )
    views = list(source)
    recovered = [view for view in views if not view.rights_events.empty]
    assert len(recovered) == 1 and recovered[0].timestamp == pd.Timestamp("2023-01-04")
    assert "000001" not in set(recovered[0].panel.asset_id)
    event = recovered[0].rights_events.iloc[0]
    assert event.asset_id == "000001"
    assert event.value_per_old_unit == 2.5
    assert bool(event.entitlement_proved)
    assert ("000001", pd.Timestamp("2022-12-01")) in recovered[
        0
    ].confirmed_delisted_lifecycles


def test_calendar_coalesced_rights_events_do_not_use_event_id_as_order(tmp_path):

    def events(base):
        return [
            dict(base, publication_date="2022-12-31", effective_date="2023-01-01"),
            dict(
                base,
                event_id="cash",
                kind="cash_recovery",
                amount_per_share=3.0,
                publication_date="2023-01-01",
                effective_date="2023-01-02",
            ),
        ]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    source = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        equity_events_root=layout["equity_events"],
        panel_root=roots["market_panel_1d"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
    )
    with pytest.raises(ValueError, match="equity_event_session_order_unproved"):
        list(source)


def test_source_exact_event_values_preserve_cash_and_conversion_units(tmp_path):
    from fractions import Fraction

    raw = tmp_path / "raw"

    def events(base):
        return [
            dict(base, amount_per_share=30.01),
            dict(
                base,
                event_id="conversion",
                kind="conversion",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
                successor_asset_id="000002",
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=1.25,
            ),
        ]

    write_equity_source(raw, events=events)
    imported = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path).interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    frame = read_dataset(imported.artifact_roots["a_share_equity_events"]).set_index(
        "event_id"
    )
    quote = decode_exact_numeric(frame.loc["quote-1", "value_per_old_unit"])
    multiplier = decode_exact_numeric(
        frame.loc["conversion", "successor_units_per_old_unit"]
    )
    assert quote == Fraction(30.01)
    assert multiplier == Fraction(1.25)
    assert float(Fraction(1) / quote * quote) == 1.0
    assert Fraction(1) / multiplier * multiplier == 1


def test_per_share_integer_does_not_round_before_exact_normalization(tmp_path):
    raw = tmp_path / "raw"
    shares = 2**53 + 1
    write_equity_source(
        raw,
        events=lambda base: [dict(base, amount_per_share=shares)],
    )
    imported = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path).interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    value = read_dataset(
        imported.artifact_roots["a_share_equity_events"]
    ).value_per_old_unit.iloc[0]
    assert decode_exact_numeric(value) == shares


def test_observation_timezone_does_not_confuse_chinese_publication_date(tmp_path):
    raw = tmp_path / "raw"
    write_equity_source(
        raw, events=lambda base: [dict(base, observed_at="2023-01-03T00:30:00+08:00")]
    )
    imported = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path).interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    row = read_dataset(imported.artifact_roots["a_share_equity_events"]).iloc[0]
    assert row.observed_at == pd.Timestamp("2023-01-02T16:30:00")
    assert row.available_at == pd.Timestamp("2023-01-04")


def test_present_future_only_source_materializes_empty_window_without_fake_history(
    tmp_path,
):
    raw = tmp_path / "raw"
    write_equity_source(raw)
    imported = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path).interim_data_root("reference", key)
            for key in ("a_share_equity_events",)
        },
        raw_root=raw,
        data_ids=("a_share_equity_events",),
        end_date="2023-01-02",
    )
    assert prepared_source_evidence(interim, raw, "a_share_equity_events") is not None
    frame = read_dataset(imported.artifact_roots["a_share_equity_events"])
    assert frame.empty and tuple(frame.columns) == EQUITY_EVENT_COLUMNS


def test_absent_source_is_distinct_from_partial_present_source(tmp_path):
    assert prepared_source_evidence(interim, tmp_path, "a_share_equity_events") is None
    (tmp_path / "a_share_equity_events").mkdir()
    with pytest.raises(ValueError, match="equity_event_metadata_missing"):
        prepared_source_evidence(interim, tmp_path, "a_share_equity_events")


@pytest.mark.parametrize(
    "host,allowed",
    [
        ("www.neeq.com.cn", True),
        ("neeq.cc", True),
        ("disclosure.neeq.cc", True),
        ("neeq.cc.example.com", False),
        ("notneeq.cc", False),
        ("neeq.com.cn.evil", False),
    ],
)
def test_equity_official_transfer_domains_reject_lookalikes(tmp_path, host, allowed):
    write_equity_source(
        tmp_path,
        events=lambda base: [dict(base, source_url=f"https://{host}/evidence.pdf")],
    )
    if allowed:
        assert (
            prepared_source_evidence(interim, tmp_path, "a_share_equity_events")
            is not None
        )
    else:
        with pytest.raises(ValueError, match="equity_event_source_invalid"):
            prepared_source_evidence(interim, tmp_path, "a_share_equity_events")


def test_narrow_session_does_not_coalesce_different_historical_event_sessions(tmp_path):

    def events(base):
        return [
            base,
            dict(
                base,
                event_id="cash-later",
                kind="cash_recovery",
                amount_per_share=3,
                publication_date="2023-01-04",
                effective_date="2023-01-05",
            ),
        ]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=layout["market_bars"],
            stock_master_root=layout["stock_master"],
            equity_events_root=layout["equity_events"],
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2023-01-06", "2023-01-06"),
        )
    )
    assert len(views) == 1
    assert views[0].rights_events.event_id.tolist() == ["quote-1", "cash-later"]
    assert views[0].rights_events.timestamp.tolist() == list(
        pd.to_datetime(["2023-01-04", "2023-01-05"])
    )


def test_future_event_attachment_does_not_change_prior_session_event_proof(tmp_path):
    layout, roots, calendar = native_equity_inputs(tmp_path)

    def views(current_layout):
        return list(
            portfolio_sessions.iter_portfolio_sessions(
                market_bars_root=current_layout["market_bars"],
                stock_master_root=current_layout["stock_master"],
                equity_events_root=current_layout["equity_events"],
                panel_root=roots["market_panel_1d"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=TimeWindow("2022-12-30", "2023-01-06"),
            )
        )

    before = views(layout)
    raw = tmp_path / "raw"
    old_identity = prepared_source_evidence(interim, raw, "a_share_equity_events")
    source = raw / "a_share_equity_events"
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    future = dict(
        metadata["events"][0],
        event_id="future",
        effective_date="2023-01-10",
        publication_date="2023-01-09",
        document_file="future.txt",
    )
    attachment = b"Synthetic future fact and unit evidence."
    (source / "future.txt").write_bytes(attachment)
    metadata["files"]["future.txt"] = hashlib.sha256(attachment).hexdigest()
    metadata["events"].append(future)
    path.write_text(json.dumps(metadata))
    root = tmp_path / "future_rights"
    changed_layout = {**layout, "equity_events": root}
    interim.materialize_interim_data(
        output_roots={"a_share_equity_events": root},
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    after = views(changed_layout)
    assert old_identity != prepared_source_evidence(
        interim, raw, "a_share_equity_events"
    )
    for old, new in zip(before, after, strict=True):
        pd.testing.assert_frame_equal(old.rights_events, new.rights_events)


@pytest.mark.parametrize("fault", ["active", "late_status", "before_termination"])
def test_equity_events_require_terminated_origin_at_actual_recognition(tmp_path, fault):
    events = (
        (lambda base: [dict(base, effective_date="2023-01-01")])
        if fault == "before_termination"
        else None
    )
    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    if fault != "before_termination":
        stocks = read_dataset(roots["stock_master"])
        selected = stocks.source_asset_id.eq("000001")
        if fault == "active":
            stocks.loc[selected, "delisting_date"] = pd.NaT
        else:
            stocks.loc[selected, "timestamp"] = pd.Timestamp("2023-01-05")
        root = roots["stock_master"].parent / ("d" * 64)
        write_dataset(root, stocks, spec=_interim_specs().get("stock_master"))
        layout = {**layout, "stock_master": root}
    source = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        equity_events_root=layout["equity_events"],
        panel_root=roots["market_panel_1d"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2023-01-06", "2023-01-06"),
    )
    events = pd.concat(view.rights_events for view in source)
    assert not events.entitlement_proved.all()


@pytest.mark.parametrize("active_successor", [False, True, "unresolved", "null_flag"])
def test_proved_conversion_qualifies_external_successor_but_not_active_distribution(
    tmp_path, active_successor
):

    def events(base):
        converted = dict(
            base,
            event_id="z-conversion",
            kind="conversion",
            amount_per_share=None,
            remaining_old_units_per_old_unit=0,
            successor_asset_id="000001",
            successor_listing_date="2023-01-03",
            successor_shares_per_old_share=0.5,
        )
        successor = dict(
            base,
            event_id="a-successor",
            listing_date="2023-01-03",
            kind="cash_recovery",
        )
        return [successor, converted]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    if active_successor:
        stocks = read_dataset(roots["stock_master"])
        successor = stocks.loc[stocks.source_asset_id.eq("000001")].assign(
            timestamp=pd.Timestamp("2023-01-03"),
            listing_date=pd.Timestamp("2023-01-03"),
            delisting_date=pd.NaT,
            delisting_date_unresolved=(
                pd.NA
                if active_successor == "null_flag"
                else active_successor == "unresolved"
            ),
        )
        root = roots["stock_master"].parent / ("e" * 64)
        write_dataset(
            root,
            pd.concat((stocks, successor)),
            spec=_interim_specs().get("stock_master"),
        )
        layout = {**layout, "stock_master": root}
    source = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        equity_events_root=layout["equity_events"],
        panel_root=roots["market_panel_1d"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
    )
    if active_successor:
        events = pd.concat(view.rights_events for view in source)
        assert events.set_index("event_id").loc["z-conversion", "entitlement_proved"]
        assert not events.set_index("event_id").loc["a-successor", "entitlement_proved"]
    else:
        events = pd.concat(view.rights_events for view in source)
        assert events.event_id.tolist() == ["z-conversion", "a-successor"]


def test_circular_conversion_cannot_revive_the_extinguished_original_life(tmp_path):

    def events(base):
        forward = dict(
            base,
            event_id="forward",
            kind="conversion",
            amount_per_share=None,
            remaining_old_units_per_old_unit=0,
            successor_asset_id="000001",
            successor_listing_date="2023-01-03",
            successor_shares_per_old_share=1,
        )
        backward = dict(
            forward,
            event_id="backward",
            listing_date="2023-01-03",
            successor_listing_date="2022-12-01",
        )
        return [backward, forward]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    source = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        equity_events_root=layout["equity_events"],
        panel_root=roots["market_panel_1d"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
    )
    with pytest.raises(ValueError, match="equity_event_successor_lineage_cycle"):
        list(source)


def test_explicit_ordinary_holder_share_change_uses_same_lifecycle_without_price_basis(
    tmp_path,
):
    raw = tmp_path / "raw"
    write_equity_source(
        raw,
        events=lambda base: [
            dict(
                base,
                kind="share_change",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.16,
            )
        ],
    )
    imported = interim.materialize_interim_data(
        output_roots={"a_share_equity_events": tmp_path / "rights"},
        raw_root=raw,
        data_ids=("a_share_equity_events",),
    )
    events = read_dataset(imported.artifact_roots["a_share_equity_events"])
    stocks = pd.DataFrame(
        {
            "asset_id": ["000001"],
            "listing_date": pd.to_datetime(["2022-12-01"]),
            "stock_effective_date": pd.to_datetime(["2022-12-01"]),
            "delisting_date": [pd.Timestamp.max.normalize()],
        }
    )
    qualified = portfolio_sessions._qualify_equity_events(
        events, pd.Series(pd.to_datetime(["2023-01-04"])), stocks, {}
    )
    assert qualified.entitlement_proved.all()
    assert (
        decode_exact_numeric(qualified.remaining_old_units_per_old_unit.iloc[0]) == 0.16
    )
    assert "shares_per_old_unit" not in qualified
