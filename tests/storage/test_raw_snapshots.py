"""Consumed CSV member identity, including cancellation and primary failures."""

import pytest

from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
    write_csv_snapshot,
)


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("termination", ["exhaust", "close", "primary"])
def test_csv_consumed_digest_is_verified_on_every_exit(tmp_path, changed, termination):
    root = tmp_path / "snapshot"
    write_csv_snapshot(
        root,
        columns=("value",),
        rows=((1,), (2,)),
        metadata={"source": "synthetic"},
        max_bytes=1024,
        validate=lambda path: list(
            iter_csv_snapshot(
                path,
                expected_manifest=read_manifest(path / "response.metadata.json"),
                max_bytes=1024,
            )
        ),
    )
    manifest = read_manifest(root / "response.metadata.json")
    if changed:
        path = root / "response.csv"
        original = path.read_bytes()
        replacement = original.replace(b"2\n", b"3\n")
        assert replacement != original and len(replacement) == len(original)
        path.write_bytes(replacement)
    rows = iter_csv_snapshot(root, expected_manifest=manifest, max_bytes=1024)
    assert next(rows) == (1,)
    if termination == "primary":
        primary = KeyboardInterrupt("body_primary")
        with pytest.raises(KeyboardInterrupt) as caught:
            rows.throw(primary)
        assert caught.value is primary
        assert bool(getattr(primary, "__notes__", [])) == changed
    elif changed:
        with pytest.raises(ValueError, match="raw_csv_content_invalid"):
            list(rows) if termination == "exhaust" else rows.close()
    elif termination == "exhaust":
        assert list(rows) == [(2,)]
    else:
        rows.close()
    assert read_manifest(root / "response.metadata.json") == manifest
