"""Consumed CSV member identity, including cancellation and primary failures."""

import pytest

from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
    write_csv_snapshot,
)


@pytest.mark.parametrize("outer_handled", [False, True])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("termination", ["exhaust", "close", "primary"])
def test_csv_consumed_digest_is_verified_on_every_exit(
    tmp_path, changed, termination, outer_handled
):
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

    def finish():
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

    if outer_handled:
        try:
            raise RuntimeError("handled_outer")
        except RuntimeError as outer:
            finish()
            assert not getattr(outer, "__notes__", [])
    else:
        finish()
    assert read_manifest(root / "response.metadata.json") == manifest


def test_raw_cell_codec_preserves_types_canonical_bytes_and_bounds():
    from quant_research_platform.storage.raw_snapshots import (
        decode_raw_cell,
        encode_raw_cell,
    )

    pairs = [
        (None, "null"),
        (True, "true"),
        (1234, "1234"),
        (-0.0, "-0.0"),
        (1.2345, "1.2345"),
        ("中文", '"中文"'),
        ([2020, 1, None], "[2020,1,null]"),
        (b"abc", '{"bytes_hex":"616263"}'),
    ]
    for value, encoded in pairs:
        assert encode_raw_cell(value) == encoded
        decoded = decode_raw_cell(encoded)
        assert type(decoded) is type(value)
        assert decoded == value
    for invalid in (" 1", "1 ", "1e0", '"\\u4e2d"', '{"other":1}'):
        with pytest.raises(ValueError):
            decode_raw_cell(invalid)
    with pytest.raises(ValueError, match="raw_csv_cell_bound_exceeded"):
        decode_raw_cell('"' + "x" * (64 * 1024 + 1) + '"')


@pytest.mark.parametrize("field", ["byte_count", "metadata"])
def test_raw_manifest_expected_identity_preserves_json_numeric_types(tmp_path, field):
    from copy import deepcopy

    from quant_research_platform.contracts.json_codec import canonical_identity_bytes

    root = tmp_path / "typed_snapshot"
    write_csv_snapshot(
        root,
        columns=("value",),
        rows=((1,), (2,)),
        metadata={"revision": 1},
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
    assert list(
        iter_csv_snapshot(root, expected_manifest=manifest, max_bytes=1024)
    ) == [(1,), (2,)]
    changed = deepcopy(manifest)
    if field == "byte_count":
        changed["byte_count"] = float(changed["byte_count"])
    else:
        changed["metadata"]["revision"] = True
    assert changed == manifest
    assert canonical_identity_bytes(changed) != canonical_identity_bytes(manifest)
    with pytest.raises(ValueError, match="raw_csv_metadata_invalid"):
        list(iter_csv_snapshot(root, expected_manifest=changed, max_bytes=1024))
