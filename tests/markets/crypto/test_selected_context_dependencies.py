"""Selected context recipes consume only their real mathematical dependencies."""

import gzip
import io

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_factor_frames,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_CROSS_ASSET_DATA_ID,
    HFT_EXTERNAL_DATA_IDS,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_SPOT_CONTEXT_DATA_ID,
    HFT_SPOT_PANEL_DATA_ID,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_CROSS_ASSET_FACTOR_COLUMNS,
    HFT_EXTERNAL_FIELDS,
    HFT_SPOT_FACTOR_COLUMNS,
    cross_asset_inputs,
    spot_dependencies,
)
from quant_research_platform.sources.tardis.normalized import _publish_dataset
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.markets.crypto.test_local_data_dependencies import _persist
from tests.markets.crypto.test_spot_context import panel

KEYS = ("market_id", "asset_id", "timestamp")


def _run(tmp_path, name, data_id, factor, inputs, frame, **options):
    return read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                "factor_columns": [factor],
                "inputs": inputs,
                "parameters": {"max_age_ms": 200, **options.pop("parameters", {})},
                "time_window": {
                    "start": frame.timestamp.min().isoformat(),
                    "end": frame.timestamp.max().isoformat(),
                },
                "output_root": str(tmp_path / name),
                **options,
            }
        )
    )


@pytest.mark.parametrize("factor", HFT_SPOT_FACTOR_COLUMNS)
def test_every_spot_factor_minimal_inputs_match_full_public_prepare(tmp_path, factor):
    frames = {HFT_PANEL_DATA_ID: panel(), HFT_SPOT_PANEL_DATA_ID: panel(spot=True)}
    results = []
    for minimal in (False, True):
        inputs = {}
        for data_id, frame in frames.items():
            dependencies, factors = spot_dependencies(
                (factor,), perpetual=data_id == HFT_PANEL_DATA_ID
            )
            names = list(
                dict.fromkeys(
                    (*KEYS, *dependencies, *(("segment_id",) if factors else ()))
                )
            )
            root = _persist(
                tmp_path / f"source_{data_id}_{minimal}",
                frame.loc[:, names] if minimal else frame,
            )
            inputs[data_id] = {
                "path": str(root),
                "columns": [name for name in names if name not in KEYS]
                or ["timestamp"],
            }
        results.append(
            _run(
                tmp_path,
                f"output{minimal}",
                HFT_SPOT_CONTEXT_DATA_ID,
                factor,
                inputs,
                frames[HFT_PANEL_DATA_ID],
            )
        )
    pd.testing.assert_frame_equal(results[0].isna(), results[1].isna())
    pd.testing.assert_frame_equal(
        results[0].drop(columns=[factor]),
        results[1].drop(columns=[factor]),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        results[0][factor], results[1][factor], check_exact=False, rtol=2e-12, atol=0
    )
    assert results[0][factor].notna().any()


@pytest.mark.parametrize("factor", HFT_CROSS_ASSET_FACTOR_COLUMNS)
def test_every_cross_asset_factor_minimal_information_matches_full(tmp_path, factor):
    target = panel()
    information = next(iter_factor_frames([panel(spot=True)]))
    asset = factor.split("_", 1)[0] + "_usdt"
    information["asset_id"] = asset
    target_root = _persist(tmp_path / "target", target.loc[:, list(KEYS)])
    results = []
    for minimal in (False, True):
        names = (*KEYS, *cross_asset_inputs((factor,)))
        selected = information.loc[:, list(names)] if minimal else information
        root = _persist(tmp_path / f"information{minimal}", selected)
        results.append(
            _run(
                tmp_path,
                f"output{minimal}",
                HFT_CROSS_ASSET_DATA_ID,
                factor,
                {
                    HFT_PANEL_DATA_ID: str(target_root),
                    HFT_INFORMATION_STATE_DATA_ID: str(root),
                },
                target,
                parameters={
                    "information_assets": [asset],
                    "target_assets": ["btc_usdt"],
                },
            )
        )
    # Cross-asset alignment copies the already computed information factor.
    pd.testing.assert_frame_equal(*results, check_exact=True)
    assert results[0][factor].notna().any()


@pytest.mark.parametrize("field", HFT_EXTERNAL_FIELDS)
def test_every_external_factor_ignores_unrelated_verified_archive_and_target_fields(
    tmp_path, field
):
    raw = tmp_path / "raw"
    raw.mkdir()
    epoch = pd.Timestamp("2026-08-01").value // 1000
    for kind in ("quotes", "trades"):
        header = (
            "exchange,symbol,timestamp,local_timestamp,"
            + (
                "ask_amount,ask_price,bid_price,bid_amount"
                if kind == "quotes"
                else "id,side,price,amount"
            )
            + "\n"
        )
        rows = []
        for index in range(30):
            stamp = epoch + index * 100000
            values = "2,201,199,2" if kind == "quotes" else f"{index},buy,200,1"
            rows.append(f"bybit,BTCUSDT,{stamp},{stamp},{values}\n")
        data = gzip.compress((header + "".join(rows)).encode(), mtime=0)
        _publish_dataset(
            raw / kind,
            lambda data=data: io.BytesIO(data),
            exchange="bybit",
            dataset_type=kind,
            symbol="BTCUSDT",
            date="2026-08-01",
        )
    frame = panel()
    results = []
    for minimal in (False, True):
        names = (
            *KEYS,
            *(("segment_id",) if field.startswith(("return_", "observed_")) else ()),
            *(
                ("mid",)
                if field in {"price_difference", "relative_mid_dispersion"}
                else ()
            ),
        )
        target = _persist(
            tmp_path / f"target{minimal}",
            frame.loc[:, list(names)] if minimal else frame,
        )
        kinds = ("trades",) if field.startswith("observed_") else ("quotes",)
        archives = {
            kind: [kind] for kind in (kinds if minimal else ("quotes", "trades"))
        }
        results.append(
            _run(
                tmp_path,
                f"output{minimal}",
                HFT_EXTERNAL_DATA_IDS["bybit"],
                "bybit_" + field,
                {HFT_PANEL_DATA_ID: str(target)},
                frame,
                source={"root": str(raw), "archives": {"BTCUSDT": archives}},
            )
        )
    factor = "bybit_" + field
    pd.testing.assert_frame_equal(results[0].isna(), results[1].isna())
    pd.testing.assert_frame_equal(
        results[0].drop(columns=[factor]),
        results[1].drop(columns=[factor]),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        results[0][factor], results[1][factor], check_exact=False, rtol=2e-12, atol=0
    )
    assert results[0]["bybit_" + field].notna().any()


@pytest.mark.parametrize("dtype", ["bool", "boolean", "object"])
def test_persisted_boolean_numeric_cells_are_unknown_not_one(tmp_path, dtype):
    frame = panel().iloc[:3].copy()
    frame["book_spread"] = pd.Series([True, False, True], dtype=dtype)
    source = _persist(tmp_path / "source", frame)
    output = prepare(
        {
            "market_id": "crypto",
            "data_id": "binance_futures_microstructure_factors_100ms",
            "factor_columns": ["spread"],
            "inputs": {HFT_PANEL_DATA_ID: str(source)},
            "parameters": {"asset_ids": ["btc_usdt"]},
            "time_window": {
                "start": frame.timestamp.min().isoformat(),
                "end": frame.timestamp.max().isoformat(),
            },
            "output_root": str(tmp_path / "output"),
        }
    )
    assert read_dataset(output).spread.isna().all()


@pytest.mark.parametrize(
    "columns, expected_error",
    [
        (["book_spraed_typo"], "dataset_column_not_found"),
        (["mid"], "data_operation_input_columns_missing"),
    ],
)
def test_explicit_bad_projection_is_configuration_error(
    tmp_path, columns, expected_error
):
    frame = panel()
    root = _persist(tmp_path / "source", frame)
    with pytest.raises(ValueError, match=f"^{expected_error}$"):
        prepare(
            {
                "market_id": "crypto",
                "data_id": "binance_futures_microstructure_factors_100ms",
                "factor_columns": ["spread"],
                "parameters": {"asset_ids": ["btc_usdt"]},
                "inputs": {HFT_PANEL_DATA_ID: {"path": str(root), "columns": columns}},
                "time_window": {
                    "start": frame.timestamp.min().isoformat(),
                    "end": frame.timestamp.max().isoformat(),
                },
                "output_root": str(tmp_path / "output"),
            }
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "field,output",
    [
        ("depth_normalized_ofi_500ms", "eth_normalized_ofi_500ms"),
        ("trade_flow_imbalance_500ms", "eth_lagged_tfi_500ms"),
        ("return_100ms", "eth_lagged_return_100ms"),
    ],
)
def test_signed_information_and_known_lag_do_not_depend_on_current_value(
    tmp_path, field, output
):
    target = panel().iloc[:4]
    info = target.loc[:, list(KEYS)].assign(
        asset_id="eth_usdt", segment_id="eth-source"
    )
    info[field] = [-0.5, float("nan"), 0.7, 0.8]
    target_root = _persist(tmp_path / "target", target.loc[:, list(KEYS)])
    info_root = _persist(tmp_path / "info", info)
    result = _run(
        tmp_path,
        "output",
        HFT_CROSS_ASSET_DATA_ID,
        output,
        {
            HFT_PANEL_DATA_ID: str(target_root),
            HFT_INFORMATION_STATE_DATA_ID: str(info_root),
        },
        target,
        parameters={"information_assets": ["eth_usdt"], "target_assets": ["btc_usdt"]},
    )
    expected = (
        [float("nan"), -0.5, float("nan"), 0.7]
        if "lagged" in output
        else info[field].tolist()
    )
    pd.testing.assert_series_equal(result[output], pd.Series(expected, name=output))


@pytest.mark.parametrize(
    "archives",
    [{"BTCUSDT": {"quoes": ["quotes"]}}, {"BTCTYPO": {"quotes": ["quotes"]}}],
)
def test_unknown_external_archive_selector_is_configuration_error(tmp_path, archives):
    frame = panel().loc[:, list(KEYS)]
    target = _persist(tmp_path / "target", frame)
    with pytest.raises(ValueError, match="normalized_source_archives_invalid"):
        _run(
            tmp_path,
            "output",
            HFT_EXTERNAL_DATA_IDS["bybit"],
            "bybit_spread_usdt",
            {HFT_PANEL_DATA_ID: str(target)},
            frame,
            source={"root": str(tmp_path), "archives": archives},
        )


def test_cross_asset_lag_requires_persisted_source_segment_only_when_selected(tmp_path):
    target = panel().iloc[:4]
    info = target.loc[:, list(KEYS)].assign(asset_id="eth_usdt", return_100ms=0.25)
    inputs = {
        HFT_PANEL_DATA_ID: str(
            _persist(tmp_path / "target", target.loc[:, list(KEYS)])
        ),
        HFT_INFORMATION_STATE_DATA_ID: str(
            _persist(tmp_path / "legacy-information", info)
        ),
    }
    current = _run(
        tmp_path,
        "current",
        HFT_CROSS_ASSET_DATA_ID,
        "eth_return_100ms",
        inputs,
        target,
        parameters={"information_assets": ["eth_usdt"], "target_assets": ["btc_usdt"]},
    )
    assert current.eth_return_100ms.eq(0.25).all()
    with pytest.raises(ValueError, match="crypto_information_segment_required"):
        _run(
            tmp_path,
            "lag",
            HFT_CROSS_ASSET_DATA_ID,
            "eth_lagged_return_100ms",
            inputs,
            target,
            parameters={
                "information_assets": ["eth_usdt"],
                "target_assets": ["btc_usdt"],
            },
        )


@pytest.mark.parametrize("route", ["spot", "union"])
@pytest.mark.parametrize("declared_market", [None, "crypto", "a_share"])
def test_native_context_respects_explicit_market(tmp_path, route, declared_market):
    from quant_research_platform.contracts.data import DataSpec, ProcessedDataSpec
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_INFORMATION_PANEL_DATA_ID,
        HFT_SOL_PANEL_DATA_ID,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.markets.crypto.test_local_data_dependencies import _panel

    if route == "spot":
        frames = {HFT_PANEL_DATA_ID: panel(), HFT_SPOT_PANEL_DATA_ID: panel(spot=True)}
        output_id = HFT_SPOT_CONTEXT_DATA_ID
    else:
        base = pd.concat(_panel(count=5), ignore_index=True)
        frames = {
            HFT_PANEL_DATA_ID: base,
            HFT_SOL_PANEL_DATA_ID: base.assign(asset_id="sol_usdt"),
        }
        output_id = HFT_INFORMATION_PANEL_DATA_ID
    inputs = {}
    for data_id, frame in frames.items():
        frame = frame.assign(market_id=declared_market or "crypto")
        roles = dict.fromkeys(frame.columns, "context")
        roles.update(market_id="key", asset_id="key", timestamp="timestamp")
        fields = dict(
            data_id=data_id,
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=KEYS,
            partition_columns=("year",),
        )
        source_spec = (
            DataSpec(**fields)
            if declared_market is None
            else ProcessedDataSpec(**fields, market_id=declared_market, kind="panel")
        )
        root = tmp_path / data_id
        write_dataset(root, frame, spec=source_spec)
        frames[data_id] = read_dataset(root)
        inputs[data_id] = str(root)
    output = tmp_path / "output"
    config = {
        "market_id": "crypto",
        "data_id": output_id,
        "inputs": inputs,
        "time_window": {
            "start": frame.timestamp.min().isoformat(),
            "end": frame.timestamp.max().isoformat(),
        },
        "output_root": str(output),
    }
    if route == "spot":
        config.update(
            factor_columns=["spot_perp_basis"], parameters={"max_age_ms": 200}
        )
    if declared_market == "a_share":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            prepare(config)
        assert not output.exists()
        return
    result = read_dataset(prepare(config))
    assert set(result.market_id) == {"crypto"}
    if route == "spot":
        assert result.spot_perp_basis.tolist() == pytest.approx(
            frames[HFT_SPOT_PANEL_DATA_ID].mid / frames[HFT_PANEL_DATA_ID].mid - 1
        )
    else:
        expected = pd.concat(frames.values(), ignore_index=True).sort_values(
            ["timestamp", "asset_id"], ignore_index=True
        )
        pd.testing.assert_frame_equal(
            result[expected.columns], expected, check_exact=True
        )
