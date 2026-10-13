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
def test_csv_acceptance_verifies_consumed_digest_without_draining_on_close(
    tmp_path, changed, termination, outer_handled
):
    root = tmp_path / "snapshot"
    write_csv_snapshot(
        root,
        columns=("value",),
        rows=((1,), (2,)),
        metadata={"source": "synthetic"},
        validate=lambda path: list(
            iter_csv_snapshot(
                path,
                expected_manifest=read_manifest(path / "response.metadata.json"),
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
    rows = iter_csv_snapshot(root, expected_manifest=manifest)
    assert next(rows) == (1,)

    def finish():
        if termination == "primary":
            primary = KeyboardInterrupt("body_primary")
            with pytest.raises(KeyboardInterrupt) as caught:
                rows.throw(primary)
            assert caught.value is primary
            assert not getattr(primary, "__notes__", [])
        elif changed and termination == "exhaust":
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


def test_raw_cell_codec_preserves_types_and_canonical_bytes():
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
    value = "x" * (1024 * 1024 + 1)
    assert decode_raw_cell(encode_raw_cell(value)) == value
    nested = b"bytes"
    for _ in range(70):
        nested = [nested]
    assert decode_raw_cell(encode_raw_cell(nested)) == nested
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="raw_csv_cycle"):
        encode_raw_cell(cyclic)


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
        validate=lambda path: list(
            iter_csv_snapshot(
                path,
                expected_manifest=read_manifest(path / "response.metadata.json"),
            )
        ),
    )
    manifest = read_manifest(root / "response.metadata.json")
    assert list(iter_csv_snapshot(root, expected_manifest=manifest)) == [(1,), (2,)]
    changed = deepcopy(manifest)
    if field == "byte_count":
        changed["byte_count"] = float(changed["byte_count"])
    else:
        changed["metadata"]["revision"] = True
    assert changed == manifest
    assert canonical_identity_bytes(changed) != canonical_identity_bytes(manifest)
    with pytest.raises(ValueError, match="raw_csv_metadata_invalid"):
        list(iter_csv_snapshot(root, expected_manifest=changed))


def test_raw_snapshot_long_cell_and_record_roundtrip(tmp_path):
    root = tmp_path / "large"
    value = "x" * (1024 * 1024 + 1) + '\n"quoted"'

    def read(path):
        return list(
            iter_csv_snapshot(
                path, expected_manifest=read_manifest(path / "response.metadata.json")
            )
        )

    write_csv_snapshot(
        root, columns=("value",), rows=((value,),), metadata={}, validate=read
    )
    assert read(root) == [(value,)]
