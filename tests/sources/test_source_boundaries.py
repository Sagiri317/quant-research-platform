"""Present malformed sources remain distinct from absent datasets."""

import pytest

from quant_research_platform.sources.market import open_market_source
from quant_research_platform.sources.reference import interim
from tests.helpers.paths import StorageLayout
from tests.helpers.source_preparation import prepared_source_evidence
from tests.sources.reference.test_interim import (
    _file_bytes,
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
        with open_market_source(root):
            pass


@pytest.mark.parametrize(
    "location,archive_paths",
    [("absent", None), ("absent", ("declared",)), (".", None)],
)
def test_absent_crypto_source_does_not_consume_unrelated_data(
    tmp_path, location, archive_paths
):
    (tmp_path / "unrelated.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        with open_market_source(tmp_path / location, archive_paths=archive_paths):
            pass


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
        prepared_source_evidence(interim, raw, data_id)


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
    assert prepared_source_evidence(interim, tmp_path, data_id) is None


def test_partial_source_workflow_blocks_and_preserves_old_good(tmp_path):
    raw = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(project_root=tmp_path / "project")
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("reference", key)
            for key in ("a_share_fee_schedule",)
        },
        raw_root=raw / "reference",
    )
    target = interim.A_SHARE_FEE_SCHEDULE_DATA_ID
    old_root = result.artifact_roots[target]
    before = _file_bytes(old_root)
    (raw / "reference" / target / interim.A_SHARE_FEE_SIDECAR_NAME).unlink()
    with pytest.raises(ValueError):
        interim.materialize_interim_data(
            raw_root=raw / "reference",
            output_roots={target: tmp_path / "new-fees"},
            data_ids=(target,),
        )
    assert _file_bytes(old_root) == before


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
        prepared_source_evidence(interim, root, data_id)
