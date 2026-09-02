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
