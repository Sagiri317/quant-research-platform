from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


def test_interim_identity_receives_common_root_joined_with_source_id(
    tmp_path: Path,
) -> None:
    common_root = tmp_path / "raw"
    source_root = common_root / "csmar"
    source_root.mkdir(parents=True)
    marker = source_root / "market_bars.raw"
    marker.write_bytes(b"synthetic")
    observed: list[Path] = []
    data_id = "market_bars_1d"

    def source_identity(raw_root: Path, _data_id: str) -> str:
        observed.append(raw_root)
        return content_hash_for_files(
            ((data_id, raw_root / marker.name),),
            error="synthetic_source_identity_invalid",
        )

    profile = replace(
        research_profile(),
        producer_bindings={
            **research_profile().producer_bindings,
            data_id: ProducerBinding(
                ProducerKind.INTERIM,
                lambda **_: None,
                source_identity=source_identity,
            ),
        },
    )
    result = research.plan_targets(
        profile=profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=StorageLayout(project_root=tmp_path),
        raw_root=common_root,
    )

    assert result["status"] == "COMPLETE"
    assert result["targets"][0]["decision"] == "BUILD"
    assert observed == [source_root]


def test_a_share_target_windows_are_shared_by_plan_build_and_reuse() -> None:
    profile = research_profile()
    protocol = profile.protocol("pv_all")
    requested = {
        "start": "2019-01-01",
        "end": "2019-12-31",
        "access": "development",
    }
    assert (
        research._target_time_window(
            "research run",
            protocol,
            profile.interim.get("market_bars_1d"),
            requested,
        )
        == protocol.data_window
    )
    assert (
        research._target_time_window(
            "research run",
            protocol,
            profile.processed.get("market_panel_1d"),
            requested,
        )
        == protocol.data_window
    )
    assert (
        research._target_time_window(
            "research run",
            protocol,
            profile.research_datasets.get("pv_all"),
            requested,
        )
        == protocol.formal_dataset_window
    )
    requested_window = TimeWindow(requested["start"], requested["end"])
    assert (
        research._target_time_window(
            "research run",
            protocol,
            None,
            requested,
        )
        == requested_window
    )
    assert (
        research._target_time_window(
            "dataset build",
            protocol,
            profile.research_datasets.get("pv_all"),
            requested,
        )
        == requested_window
    )


def test_crypto_plan_preserves_source_system_io_failure(tmp_path, monkeypatch):
    import errno

    import pytest

    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_PANEL_DATA_ID,
    )
    from quant_research_platform.profiles.research import CRYPTO_PROFILE_ID
    from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
    from tests.sources.binance.test_replay import _agg_trade, _write_run

    root = tmp_path / "native"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    primary = OSError(errno.EIO, "controlled_io_failure")
    original = Path.lstat

    def fail(path, *args, **kwargs):
        if path == root / "run.json":
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fail)
    with pytest.raises(OSError) as caught:
        research.plan_targets(
            profile=research_profile(CRYPTO_PROFILE_ID),
            action="data materialize",
            terminal_targets=(HFT_PANEL_DATA_ID,),
            layout=StorageLayout(tmp_path / "project"),
            raw_root=root,
            research_window={
                "start": "2026-08-27T00:00:00",
                "end": "2026-08-27T00:00:01",
                "access": "development",
            },
        )
    assert caught.value is primary


def test_plan_preserves_persisted_instance_system_io_failure(tmp_path, monkeypatch):
    import pytest

    from tests.sources.reference.test_interim import _write_reference_root

    raw = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(tmp_path)
    profile = research_profile()
    data_id = "a_share_fee_schedule"
    built = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw,
        start_date=None,
        end_date=None,
        data_ids=(data_id,),
    )
    assert built["target_results"][0]["status"] == "BUILT"
    arguments = dict(
        profile=profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=layout,
        raw_root=raw,
    )
    planned = research.plan_targets(**arguments)
    assert planned["targets"][0]["decision"] == "REUSE"
    target = layout.instance_root(
        layout.interim_data_root("reference", data_id),
        planned["targets"][0]["requested_identity"],
    )
    primary = OSError(5, "synthetic_storage_io")
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path == target:
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(OSError) as caught:
        research.plan_targets(**arguments)
    assert caught.value is primary
