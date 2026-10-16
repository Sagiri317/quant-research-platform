"""Persisted local validity never introduces peer or unrelated-channel gates."""

import copy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.markets.implementations.crypto.hft import (
    iter_factor_frames,
    iter_label_frames,
    iter_panel_frames,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_CROSS_ASSET_DATA_ID,
    HFT_DERIVATIVE_DATA_ID,
    HFT_FACTOR_DATA_ID,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_FACTOR_COLUMNS,
    factor_catalog,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.binance_hft import (
    _coverage,
    _write_file,
    coverage_lookup,
    write_binance_hft_run,
)
from tests.markets.crypto.test_hft import _state, _trade
from tests.research.direct_model_test_support import fit_frame


def _persist(path, frame):
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(market_id="key", asset_id="key", timestamp="timestamp")
    write_dataset(
        path,
        frame,
        spec=DataSpec(
            data_id="local_fixture",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
        ),
    )
    return path


def _window(start=0, end=1000):
    return {
        "start": pd.Timestamp(start, unit="ms").isoformat(),
        "end": pd.Timestamp(end, unit="ms").isoformat(),
    }


def _panel(count=60, batch=7, coverage=None):
    return list(
        iter_panel_frames(
            [_state(0), _state(count * 100, bid_index=996, ask_index=1006)],
            [
                _trade(index * 100, 1.0, maker_buyer=False)
                for index in range(1, count + 1)
            ],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=0,
            end_ms=count * 100,
            book_coverage={"s1": _coverage(0, count * 100 + 1)},
            trade_coverage=coverage_lookup(
                coverage or {"trade-connection-1": _coverage(0, count * 100 + 1)}
            ),
            batch_rows=batch,
        )
    )


@pytest.mark.parametrize("factor", HFT_FACTOR_COLUMNS)
def test_every_selected_factor_prepares_from_its_minimal_persisted_inputs(
    tmp_path, factor
):
    panel = pd.concat(_panel(), ignore_index=True)
    expected = pd.concat(
        iter_factor_frames([panel], columns=(factor,)), ignore_index=True
    )
    names = (
        "market_id",
        "asset_id",
        "timestamp",
        "segment_id",
        *factor_catalog().get(factor).input_columns,
    )
    minimal = panel.loc[:, list(dict.fromkeys(names))]
    root = _persist(tmp_path / "input", minimal)
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_FACTOR_DATA_ID,
            "factor_columns": [factor],
            "parameters": {"asset_ids": ["btc_usdt"]},
            "inputs": {
                HFT_PANEL_DATA_ID: {
                    "path": str(root),
                    "columns": list(factor_catalog().get(factor).input_columns),
                }
            },
            "time_window": _window(0, 6000),
            "output_root": str(tmp_path / "factor"),
        }
    )
    actual = read_dataset(output)
    pd.testing.assert_series_equal(actual[factor], expected[factor], check_exact=True)
    assert len(actual) == 61
    assert actual[factor].notna().any()


@pytest.mark.parametrize("mode", ["gap", "empty", "recover", "initially_empty"])
def test_real_source_peer_gap_preserves_own_panel_factors_labels_and_model(
    tmp_path, mode
):
    raw = tmp_path / "raw"
    manifest = write_binance_hft_run(raw, count=20)
    snapshots = []
    for index, member in enumerate(manifest["files"]):
        if member["symbol"] != "ETHUSDT" or member["data_kind"] == "metadata":
            continue
        with gzip.open(raw / member["path"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            row["capture_seq"] *= 10
            row["receive_monotonic_ns"] *= 10
        if member["data_kind"] == "depth_snapshot":
            if mode in {"empty", "initially_empty"}:
                rows[0]["payload"]["bids"][0][1] = "NaN"
            if mode in {"recover", "initially_empty"}:
                later = copy.deepcopy(rows[0])
                later.update(
                    capture_seq=100099,
                    receive_monotonic_ns=100099,
                    observed_time_ns=400_000_000,
                    receive_time="1970-01-01T00:00:00.400000Z",
                )
                later["payload"].update(
                    lastUpdateId=104,
                    E=400,
                    T=400,
                    bids=[[str(199.6 - i), "2"] for i in range(10)],
                    asks=[[str(200.6 + i), "2"] for i in range(10)],
                )
                rows.append(later)
                snapshots.append(later)
        if member["data_kind"] == "depth" and mode in {"gap", "recover"}:
            next(row for row in rows if row["observed_time_ns"] == 200_000_000)[
                "payload"
            ]["b"][0][1] = "NaN"
        manifest["files"][index] = _write_file(
            raw,
            symbol=member["symbol"],
            data_kind=member["data_kind"],
            stream=member["stream"],
            rows=rows,
        )
    (raw / "run.json").write_text(json.dumps(manifest))
    window = _window(0, 2000)
    panel_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "panel"),
        }
    )
    panel = read_dataset(panel_root)
    btc = panel.loc[panel.asset_id.eq("btc_usdt")].reset_index(drop=True)
    assert (
        btc.timestamp.tolist()
        == pd.date_range("1970-01-01", periods=21, freq="100ms").tolist()
    )
    assert btc.segment_id.nunique() == 1
    eth = panel.loc[panel.asset_id.eq("eth_usdt")]
    assert len(eth) < len(btc)
    if mode == "empty":
        assert eth.empty
    if mode in {"recover", "initially_empty"}:
        assert eth.timestamp.max() == btc.timestamp.max()
    roots = {}
    for data_id, selection in (
        (HFT_FACTOR_DATA_ID, {"factor_columns": ["return_100ms"]}),
        (HFT_LABEL_DATA_ID, {"label_columns": ["future_mid_return_100ms"]}),
    ):
        roots[data_id] = prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                **selection,
                "parameters": {"asset_ids": ["btc_usdt"]},
                "inputs": {HFT_PANEL_DATA_ID: str(panel_root)},
                "time_window": window,
                "output_root": str(tmp_path / data_id),
            }
        )
    factors, labels = (
        read_dataset(roots[key]) for key in (HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID)
    )
    assert factors.return_100ms.notna().sum() == 20
    assert labels.future_mid_return_100ms.notna().sum() == 20
    training = factors.merge(labels, on=["market_id", "asset_id", "timestamp"]).assign(
        eligible=True
    )
    case = fit_frame(
        tmp_path / "model",
        training,
        ModelSpec(
            model_id="own_return",
            model_algorithm="linear_regression",
            input_factor_columns=("return_100ms",),
            target_column="future_mid_return_100ms",
            random_seed=1,
        ),
    )
    assert case.root.exists()
    assert case.result.metadata["rows_after_input_label_dropna"] == 19
    if mode == "empty":
        information = prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_INFORMATION_STATE_DATA_ID,
                "inputs": {HFT_PANEL_DATA_ID: str(panel_root)},
                "time_window": window,
                "output_root": str(tmp_path / "information"),
            }
        )
        cross = prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_CROSS_ASSET_DATA_ID,
                "factor_columns": ["eth_return_100ms"],
                "parameters": {
                    "max_age_ms": 100,
                    "information_assets": ["eth_usdt"],
                    "target_assets": ["btc_usdt"],
                },
                "inputs": {
                    HFT_PANEL_DATA_ID: str(panel_root),
                    HFT_INFORMATION_STATE_DATA_ID: str(information),
                },
                "time_window": window,
                "output_root": str(tmp_path / "cross"),
            }
        )
        dependent = (
            read_dataset(cross)
            .merge(labels, on=["market_id", "asset_id", "timestamp"])
            .assign(eligible=True)
        )
        assert len(dependent) == 21 and dependent.eth_return_100ms.isna().all()
        with pytest.raises(ValueError, match="model_training_sample_empty"):
            fit_frame(
                tmp_path / "cross_model",
                dependent,
                ModelSpec(
                    model_id="peer_return",
                    model_algorithm="linear_regression",
                    input_factor_columns=("eth_return_100ms",),
                    target_column="future_mid_return_100ms",
                    random_seed=1,
                ),
            )


@pytest.mark.parametrize("batch", [1, 7, 60])
def test_trade_gap_persists_book_rows_and_only_invalidates_dependent_windows(
    tmp_path, batch
):
    # The same observed event list has different completeness evidence.
    coverage = {"before": _coverage(0, 200), "trade-connection-1": _coverage(300, 2001)}
    frames = _panel(count=20, batch=batch, coverage=coverage)
    panel = read_dataset(
        _persist(tmp_path / "panel", pd.concat(frames, ignore_index=True))
    )
    assert len(panel) == 21
    assert panel.segment_id.nunique() == 1
    assert str(panel.trade_count_100ms.dtype) == "Int64"
    assert str(panel.trade_observation_id.dtype).startswith("string")
    assert panel.trade_count_100ms.iloc[2:4].isna().all()
    assert panel.trade_volume_100ms.iloc[0] == 0
    assert panel.trade_observation_id.iloc[2:4].isna().all()
    factors = pd.concat(
        iter_factor_frames(
            [panel.iloc[i : i + batch] for i in range(0, len(panel), batch)]
        ),
        ignore_index=True,
    )
    assert factors.spread.notna().all()
    assert factors.return_100ms.notna().sum() == 20
    assert factors.ofi_500ms.iloc[4:].notna().all()
    assert factors.trade_flow_imbalance_500ms.iloc[:8].isna().all()
    assert factors.trade_flow_imbalance_500ms.iloc[8:].notna().all()
    labels = pd.concat(
        iter_label_frames([panel], horizons_ms=(100, 500)), ignore_index=True
    )
    assert labels.future_mid_return_100ms.notna().sum() == 20
    assert labels.maker_bid_sell_pressure_proxy_500ms.iloc[:3].isna().all()
    assert labels.maker_bid_sell_pressure_proxy_500ms.iloc[3:16].notna().all()
    training = factors.merge(
        labels, on=["market_id", "asset_id", "timestamp", "timestamp_ms"]
    ).assign(eligible=True)
    book_model = fit_frame(
        tmp_path / "book_model",
        training,
        ModelSpec(
            model_id="book",
            model_algorithm="linear_regression",
            input_factor_columns=("spread",),
            target_column="future_mid_return_100ms",
            random_seed=1,
        ),
    )
    flow_model = fit_frame(
        tmp_path / "flow_model",
        training,
        ModelSpec(
            model_id="flow",
            model_algorithm="linear_regression",
            input_factor_columns=("trade_flow_imbalance_500ms",),
            target_column="future_mid_return_100ms",
            random_seed=1,
        ),
    )
    assert book_model.root.exists() and flow_model.root.exists()
    assert book_model.result.metadata["rows_after_input_label_dropna"] == 20
    assert flow_model.result.metadata["rows_after_input_label_dropna"] == 12


def test_missing_and_bad_required_values_are_local_and_windows_keep_elapsed_time():
    panel = pd.concat(_panel(count=20), ignore_index=True)
    panel.loc[5, "mid"] = np.inf
    panel.loc[7, "observed_taker_buy_volume_100ms"] = np.nan
    panel.loc[9, "bid_depth_1"] = -1.0
    result = pd.concat(
        iter_factor_frames([panel.iloc[:6], panel.iloc[6:]]), ignore_index=True
    )
    assert result.spread.notna().all()
    assert result.return_100ms.iloc[5:7].isna().all()
    assert result.return_500ms.iloc[5:11].isna().all()
    assert result.return_500ms.iloc[11:].notna().all()
    assert result.ofi_500ms.iloc[4:].notna().all()
    assert result.trade_flow_imbalance_500ms.iloc[7:12].isna().all()
    assert result.taker_buy_pressure_500ms.iloc[12:].notna().all()
    assert result.imbalance_1.iloc[9:10].isna().all()
    missing = panel.drop(columns=["book_spread"])
    out = pd.concat(iter_factor_frames([missing], columns=("spread", "return_100ms")))
    assert out.spread.isna().all()
    assert out.return_100ms.notna().any()


@pytest.mark.parametrize("has_open_interest", [False, True])
@pytest.mark.parametrize("bad_update", ["NaN", None, "missing"])
def test_selected_open_interest_uses_only_its_channel_and_age(
    tmp_path, has_open_interest, bad_update
):
    from quant_research_platform.sources.tardis.raw import import_slices
    from tests.helpers.tardis import _slices

    origin = pd.Timestamp("2026-09-01")
    channel = "openInterest" if has_open_interest else "topLongShortPositionRatio"
    records = []
    for stamp, value in ((0, "10"), (100, bad_update), (200, "20")):
        for symbol in ("BTCUSDT", "ETHUSDT"):
            payload = (
                {
                    "symbol": symbol,
                    "time": origin.value // 1000000 + stamp,
                    "openInterest": value,
                }
                if has_open_interest
                else {
                    "symbol": symbol,
                    "timestamp": origin.value // 1000000 + stamp,
                    "longShortRatio": "1",
                    "longAccount": "0.5",
                    "shortAccount": "0.5",
                }
            )
            if has_open_interest and value == "missing":
                del payload["openInterest"]
            records.append(
                (origin + pd.Timedelta(milliseconds=stamp)).isoformat()
                + "Z "
                + json.dumps(
                    {"stream": symbol.lower() + "@" + channel, "data": payload}
                )
                + "\n"
            )
    archive = tmp_path / "archive"
    import_slices(
        archive,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", "".join(records).encode()),
        channels=(channel,),
    )
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": [asset for asset in ("btc_usdt", "eth_usdt") for _ in range(4)],
            "timestamp": list(pd.date_range(origin, periods=4, freq="100ms")) * 2,
            "segment_id": "s1",
        }
    )
    root = _persist(tmp_path / "input", frame)
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_DERIVATIVE_DATA_ID,
            "factor_columns": ["log_open_interest"],
            "parameters": {"max_age_ms": {"openInterest": 1000}},
            "inputs": {HFT_PANEL_DATA_ID: str(root)},
            "source": {"root": str(archive)},
            "time_window": {
                "start": origin.isoformat(),
                "end": (origin + pd.Timedelta(milliseconds=300)).isoformat(),
            },
            "output_root": str(tmp_path / "oi"),
        }
    )
    actual = read_dataset(output)
    assert not any(name.startswith("premiumIndex") for name in actual)
    for _, rows in actual.groupby("asset_id"):
        if has_open_interest:
            np.testing.assert_allclose(
                rows.log_open_interest,
                [np.log(10), np.nan, np.log(20), np.log(20)],
                equal_nan=True,
            )
            np.testing.assert_array_equal(rows.openInterest_age_ms, [0, 0, 0, 100])
            assert rows.openInterest_observed_at.iloc[1] == origin + pd.Timedelta(
                milliseconds=100
            )
        else:
            assert rows.log_open_interest.isna().all()


def _native_book_archive(
    tmp_path, *, empty=False, date="2026-09-01", symbols=("btcusdt", "ethusdt")
):
    from quant_research_platform.sources.tardis.raw import import_slices
    from tests.helpers.tardis import _slices

    origin = pd.Timestamp(date)
    rows = []
    for index in range(21 if not empty else 0):
        stamp = origin + pd.Timedelta(milliseconds=index * 100)
        clock = stamp.value // 1_000_000
        payload = (
            {
                "lastUpdateId": 100,
                "bids": [[str(100 - level), "2"] for level in range(12)],
                "asks": [[str(101 + level), "2"] for level in range(12)],
            }
            if index == 0
            else {
                "e": "depthUpdate",
                "E": clock,
                "T": clock,
                "s": "BTCUSDT",
                "st": 1,
                "ps": "BTCUSDT",
                "U": 100 if index == 1 else 100 + index,
                "u": 100 + index,
                "pu": 99 + index,
                "b": [["100", str(index + 2)]],
                "a": [],
            }
        )
        native = {
            "stream": "btcusdt@" + ("depthSnapshot" if index == 0 else "depth@0ms"),
            "data": payload,
        }
        if index == 0:
            native["generated"] = True
        rows.append(stamp.isoformat() + "Z " + json.dumps(native) + "\n")
    archive = tmp_path / "archive"
    import_slices(
        archive,
        date=date,
        symbols=symbols,
        slices=_slices(tmp_path / "native", "".join(rows).encode()),
    )
    return archive, origin


def test_native_entirely_absent_peer_keeps_own_real_research_and_reports_universe(
    tmp_path,
):
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

    raw, origin = _native_book_archive(tmp_path)
    window = {
        "start": origin.isoformat(),
        "end": (origin + pd.Timedelta(seconds=2)).isoformat(),
    }
    panel_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "panel"),
        }
    )
    panel = read_dataset(panel_root)
    assert set(panel.asset_id) == {"btc_usdt"}
    assert len(panel) == 19
    assert panel.timestamp.diff().dropna().eq(pd.Timedelta(milliseconds=100)).all()
    assert panel.trade_count_100ms.isna().all()
    manifest = read_dataset_manifest_evidence(panel_root)
    diagnostics = manifest["producer_diagnostics"]
    assert diagnostics["requested_assets"] == ["BTCUSDT", "ETHUSDT"]
    assert diagnostics["available_assets"] == ["BTCUSDT"]
    assert diagnostics["unavailable_assets"] == {
        "ETHUSDT": "tardis_price_observations_missing"
    }
    roots = {}
    for data_id, selection in (
        (HFT_FACTOR_DATA_ID, {"factor_columns": ["return_100ms", "spread"]}),
        (HFT_LABEL_DATA_ID, {"label_columns": ["future_mid_return_100ms"]}),
    ):
        roots[data_id] = prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                **selection,
                "parameters": {"asset_ids": ["btc_usdt"]},
                "inputs": {HFT_PANEL_DATA_ID: str(panel_root)},
                "time_window": window,
                "output_root": str(tmp_path / data_id),
            }
        )
    training = (
        read_dataset(roots[HFT_FACTOR_DATA_ID])
        .merge(
            read_dataset(roots[HFT_LABEL_DATA_ID]),
            on=["market_id", "asset_id", "timestamp"],
        )
        .assign(eligible=True)
    )
    result = fit_frame(
        tmp_path / "model",
        training,
        ModelSpec(
            model_id="native_own_return",
            model_algorithm="linear_regression",
            input_factor_columns=("return_100ms", "spread"),
            target_column="future_mid_return_100ms",
            random_seed=1,
        ),
    )
    assert result.result.metadata["rows_after_input_label_dropna"] == 17
    btc_only = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "parameters": {"asset_ids": ["btc_usdt"]},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "btc_only"),
        }
    )
    pd.testing.assert_frame_equal(read_dataset(btc_only), panel)
    own_diagnostics = read_dataset_manifest_evidence(btc_only)["producer_diagnostics"]
    assert own_diagnostics["requested_assets"] == ["BTCUSDT"]
    assert own_diagnostics["unavailable_assets"] == {}
    reversed_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "parameters": {"asset_ids": ["eth_usdt", "btc_usdt"]},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "reversed"),
        }
    )
    assert read_dataset_manifest_evidence(reversed_root)["producer_diagnostics"][
        "requested_assets"
    ] == ["ETHUSDT", "BTCUSDT"]
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    with pytest.raises(PortfolioTargetBlocked, match="hft_source_assets_unavailable"):
        prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_PANEL_DATA_ID,
                "inputs": {},
                "parameters": {"asset_ids": ["eth_usdt"]},
                "source": {"root": str(raw)},
                "time_window": window,
                "output_root": str(tmp_path / "eth_only"),
            }
        )
    assert not (tmp_path / "eth_only").exists()


def test_native_all_requested_assets_absent_does_not_publish_research(tmp_path):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    raw, origin = _native_book_archive(tmp_path, empty=True)
    with pytest.raises(PortfolioTargetBlocked, match="hft_source_assets_unavailable"):
        prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_PANEL_DATA_ID,
                "inputs": {},
                "source": {"root": str(raw)},
                "time_window": {
                    "start": origin.isoformat(),
                    "end": (origin + pd.Timedelta(seconds=2)).isoformat(),
                },
                "output_root": str(tmp_path / "panel"),
            }
        )
    assert not (tmp_path / "panel").exists()


def test_selected_panel_asset_does_not_require_an_undeclared_peer(tmp_path):
    raw, origin = _native_book_archive(tmp_path, symbols=("btcusdt",))
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "parameters": {"asset_ids": ["btc_usdt"]},
            "source": {"root": str(raw)},
            "time_window": {
                "start": origin.isoformat(),
                "end": (origin + pd.Timedelta(seconds=2)).isoformat(),
            },
            "output_root": str(tmp_path / "own"),
        }
    )
    assert len(read_dataset(output)) == 19


@pytest.mark.parametrize(
    "data_id,assets",
    [
        (HFT_PANEL_DATA_ID, ["sol_usdt"]),
        ("binance_spot_l2_panel_100ms", ["sol_usdt"]),
        ("binance_sol_futures_l2_panel_100ms", ["btc_usdt"]),
        (HFT_PANEL_DATA_ID, ["btc_usdt", "btc_usdt"]),
    ],
)
def test_panel_selection_rejects_unsupported_or_duplicate_assets(
    tmp_path, data_id, assets
):
    from quant_research_platform.markets.implementations.crypto.processed.catalog import (  # noqa: E501
        hft_processed_catalog,
    )
    from quant_research_platform.markets.prepare import _configured_spec

    with pytest.raises(ValueError, match="data_operation_asset_ids_invalid"):
        _configured_spec(
            hft_processed_catalog().get(data_id), {"parameters": {"asset_ids": assets}}
        )


def test_native_empty_archive_day_does_not_gate_later_observed_day(tmp_path):
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

    for name in ("first", "second"):
        (tmp_path / name).mkdir()
    first, _ = _native_book_archive(tmp_path / "first", empty=True)
    second, origin = _native_book_archive(tmp_path / "second", date="2026-09-02")
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "parameters": {"asset_ids": ["btc_usdt"]},
            "source": {
                "root": str(tmp_path),
                "archive_paths": [
                    str(first.relative_to(tmp_path)),
                    str(second.relative_to(tmp_path)),
                ],
            },
            "time_window": {
                "start": "2026-09-01T00:00:00",
                "end": (origin + pd.Timedelta(seconds=2)).isoformat(),
            },
            "output_root": str(tmp_path / "later"),
        }
    )
    frame = read_dataset(output)
    assert len(frame) == 19
    assert frame.timestamp.min() == origin + pd.Timedelta(milliseconds=200)
    assert (
        read_dataset_manifest_evidence(output)["producer_diagnostics"][
            "source_file_count"
        ]
        == 2
    )


def test_panel_unrepresentable_totals_do_not_block_healthy_book_research(tmp_path):
    from dataclasses import replace

    state = replace(_state(0), bids=((995, 1e308), (994, 1e308)))
    panel = pd.concat(
        iter_panel_frames(
            [state],
            [
                _trade(50, 1e308, maker_buyer=False),
                _trade(80, 1e308, maker_buyer=False),
            ],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=0,
            end_ms=100,
            book_coverage={"s1": _coverage(0, 200)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 200)}),
            batch_rows=1,
        ),
        ignore_index=True,
    )
    root = _persist(tmp_path / "panel", panel)
    persisted = read_dataset(root)
    assert persisted.trade_count_100ms.tolist() == [0, 2]
    assert pd.isna(persisted.trade_volume_100ms.iloc[1])
    assert persisted.bid_depth_5.isna().all()
    assert persisted.bid_depth_1.eq(1e308).all()
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_FACTOR_DATA_ID,
            "factor_columns": [
                "spread",
                "signed_trade_flow_100ms",
                "imbalance_1",
                "imbalance_5",
            ],
            "parameters": {"asset_ids": ["btc_usdt"]},
            "inputs": {HFT_PANEL_DATA_ID: str(root)},
            "time_window": _window(0, 100),
            "output_root": str(tmp_path / "factor"),
        }
    )
    result = read_dataset(output)
    assert result.spread.notna().all() and result.imbalance_1.notna().all()
    assert result.imbalance_5.isna().all()
    assert result.signed_trade_flow_100ms.iloc[0] == 0
    assert pd.isna(result.signed_trade_flow_100ms.iloc[1])
