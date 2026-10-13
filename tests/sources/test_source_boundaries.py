"""Present malformed sources remain distinct from absent datasets."""

import pytest

from quant_research_platform.profiles.research import research_profile
from quant_research_platform.sources.market import market_source_identity
from quant_research_platform.sources.reference import interim
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import operations
from tests.helpers.a_share_raw import write_industry_reference_fixture
from tests.sources.reference.test_equity_events import write_equity_source
from tests.sources.reference.test_interim import (
    _file_bytes,
    _industry_observations,
    _write_reference_root,
)


@pytest.mark.parametrize("damage", ["root_file", "manifest_directory", "run_directory"])
def test_present_crypto_source_is_invalid(tmp_path, damage):
    root = tmp_path / "source"
    if damage == "root_file":
        root.write_text("not a source directory")
    else:
        root.mkdir()
        (root / ("run.json" if damage == "run_directory" else "manifest.json")).mkdir()
    with pytest.raises(ValueError):
        market_source_identity(root, "market_bars")


def test_absent_crypto_source_does_not_consume_unrelated_data(tmp_path):
    assert market_source_identity(tmp_path / "absent", "market_bars") is None
    assert (
        market_source_identity(
            tmp_path / "absent", "market_bars", archive_paths=("declared",)
        )
        is None
    )
    (tmp_path / "unrelated.json").write_text("{}")
    assert market_source_identity(tmp_path, "market_bars") is None


@pytest.mark.parametrize(
    "data_id",
    [
        interim.A_SHARE_EQUITY_EVENTS_DATA_ID,
        interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
        interim.A_SHARE_FEE_SCHEDULE_DATA_ID,
    ],
)
@pytest.mark.parametrize(
    "damage", ["root_file", "dataset_file", "empty", "member_directory", "partial"]
)
def test_reference_present_invalid_is_not_absent(tmp_path, data_id, damage):
    raw = tmp_path / "raw"
    if damage == "root_file":
        raw.write_text("invalid root")
    elif damage == "dataset_file":
        raw.mkdir()
        (raw / data_id).write_text("invalid dataset")
    else:
        root = raw / data_id
        root.mkdir(parents=True)
        member = root / (
            interim.A_SHARE_FEE_CSV_NAME
            if data_id == interim.A_SHARE_FEE_SCHEDULE_DATA_ID
            else "metadata.json"
        )
        if damage == "member_directory":
            member.mkdir()
        elif damage == "partial":
            member.write_text("{}")
    with pytest.raises(ValueError):
        interim.source_content_identity(raw, data_id)


@pytest.mark.parametrize(
    "data_id",
    [
        interim.A_SHARE_EQUITY_EVENTS_DATA_ID,
        interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
        interim.A_SHARE_FEE_SCHEDULE_DATA_ID,
    ],
)
def test_reference_absent_requested_dataset_with_other_data(tmp_path, data_id):
    (tmp_path / "unrelated").mkdir()
    assert interim.source_content_identity(tmp_path, data_id) is None


@pytest.mark.parametrize("source", ["industry", "equity"])
@pytest.mark.parametrize("failure", [None, KeyboardInterrupt, GeneratorExit])
def test_reference_source_exit_checks_preserve_primary(tmp_path, source, failure):
    if source == "equity":
        root = write_equity_source(tmp_path / "reference")
        context = interim._equity_source(tmp_path / "reference")
    else:
        write_industry_reference_fixture(
            tmp_path, observations=_industry_observations()
        )
        root = tmp_path / "reference" / interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
        context = interim._industry_source(tmp_path / "reference")
    error = failure("consumer cancellation") if failure else None
    with pytest.raises(failure or ValueError) as caught:
        with context:
            member = next(
                path for path in root.iterdir() if path.name != "metadata.json"
            )
            member.write_bytes(b"changed evidence")
            if error:
                raise error
    if error:
        assert caught.value is error
        assert caught.value.__notes__ == [
            "reference source identity verification also failed: ValueError"
        ]
    else:
        assert str(caught.value) == "reference_raw_identity_changed"


def test_partial_source_workflow_blocks_and_preserves_old_good(tmp_path):
    raw = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(project_root=tmp_path / "project")
    result = interim.materialize_interim_data(layout=layout, raw_root=raw / "reference")
    target = interim.A_SHARE_FEE_SCHEDULE_DATA_ID
    old_root = result.artifact_roots[target]
    before = _file_bytes(old_root)
    (raw / "reference" / target / interim.A_SHARE_FEE_SIDECAR_NAME).unlink()
    result = operations.data_materialize(
        profile=research_profile(),
        layout=layout,
        raw_root=raw,
        start_date=None,
        end_date=None,
        data_ids=(target,),
    )
    assert result["target_results"][0]["status"] == "BLOCKED"
    assert result["target_results"][0]["reason"] == "source_identity_invalid"
    assert _file_bytes(old_root) == before


def test_present_crypto_source_disappearing_during_open_is_not_absence(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from quant_research_platform.sources import market

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    @contextmanager
    def vanished(*args, **kwargs):
        manifest.unlink()
        raise FileNotFoundError("source disappeared during admission")
        yield

    monkeypatch.setattr(market, "open_market_source", vanished)
    with pytest.raises(ValueError, match="crypto_source_member_missing"):
        market.market_source_identity(tmp_path, "market_bars")


@pytest.mark.parametrize("location", ["root", "dataset"])
@pytest.mark.parametrize(
    "data_id",
    [
        interim.A_SHARE_EQUITY_EVENTS_DATA_ID,
        interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
        interim.A_SHARE_FEE_SCHEDULE_DATA_ID,
    ],
)
def test_dangling_reference_source_alias_is_not_absent(tmp_path, location, data_id):
    root = tmp_path / "reference"
    if location == "root":
        alias = root
    else:
        root.mkdir()
        alias = root / data_id
    try:
        alias.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(ValueError, match="reference_source_root_invalid"):
        interim.source_content_identity(root, data_id)
