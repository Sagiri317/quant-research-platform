from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".parquet",
    ".csv",
    ".pickle",
    ".pkl",
    ".feather",
    ".arrow",
}
TEXT_LIKE_SUFFIXES = {
    ".log",
    ".txt",
    ".md",
    ".json",
    ".patch",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".js",
    ".xml",
}


def test_handoff_safety_accepts_utf8_patch_and_matching_hashes(
    tmp_path: Path,
) -> None:
    review_root = tmp_path / "review"
    review_root.mkdir()
    patch = review_root / "changes.patch"
    summary = review_root / "implementation_summary.md"
    patch.write_text(
        "diff --git a/src/a.py b/src/a.py\n+print('ok')\n",
        encoding="utf-8",
    )
    summary.write_text("# Summary\n\nok\n", encoding="utf-8")
    log = review_root / "test.log"
    log.write_text("command ok\n", encoding="utf-8")
    bundle = review_root / "final-branch.bundle"
    bundle.write_bytes(b"bundle bytes")
    manifest = review_root / "handoff-manifest.json"
    zip_path = review_root / "handoff.zip"
    files = (patch, summary, log, bundle)
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    path.name: {"sha256": _sha256(path)}
                    for path in files
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)

    _assert_safe_handoff(review_root)


def test_handoff_safety_rejects_external_manifest_inside_zip(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    review_root.mkdir()
    (review_root / "changes.patch").write_text("patch\n", encoding="utf-8")
    (review_root / "handoff-manifest.json").write_text(
        '{"artifacts": {"changes.patch": {"sha256": "bad"}}}\n',
        encoding="utf-8",
    )
    with zipfile.ZipFile(review_root / "handoff.zip", "w") as archive:
        archive.write(review_root / "handoff-manifest.json", "handoff-manifest.json")

    try:
        _assert_safe_handoff(review_root)
    except AssertionError as exc:
        assert "external_manifest_inside_zip" in str(exc)
    else:
        raise AssertionError("manifest in zip should fail")


def test_handoff_safety_rejects_private_or_binary_data_entries(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    review_root.mkdir()
    (review_root / "handoff-manifest.json").write_text(
        '{"artifacts": {}}\n',
        encoding="utf-8",
    )
    with zipfile.ZipFile(review_root / "handoff.zip", "w") as archive:
        archive.writestr("data/raw/vendor/file.parquet", b"private")

    try:
        _assert_safe_handoff(review_root)
    except AssertionError as exc:
        assert "forbidden_handoff_data_entry" in str(exc)
    else:
        raise AssertionError("private data entry should fail")


def test_handoff_safety_rejects_utf16_text_like_zip_entry(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    review_root.mkdir()
    patch = review_root / "changes.patch"
    patch.write_text("patch\n", encoding="utf-8")
    (review_root / "handoff-manifest.json").write_text(
        json.dumps(
            {"artifacts": {"changes.patch": {"sha256": _sha256(patch)}}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(review_root / "handoff.zip", "w") as archive:
        archive.write(patch, "changes.patch")
        archive.writestr("test_commands_and_logs/bad.log", "bad\n".encode("utf-16"))

    try:
        _assert_safe_handoff(review_root)
    except AssertionError as exc:
        assert "non_utf8_text_like_entry" in str(exc)
    else:
        raise AssertionError("UTF-16 log should fail")


def _assert_safe_handoff(review_root: Path) -> None:
    manifest_path = review_root / "handoff-manifest.json"
    zip_path = review_root / "handoff.zip"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in review_root.iterdir():
        if path.suffix in {".md", ".json", ".patch", ".txt", ".log"}:
            path.read_text(encoding="utf-8")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        text_decode_failures = []
        for name in names:
            if Path(name).suffix.lower() in TEXT_LIKE_SUFFIXES:
                try:
                    archive.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    text_decode_failures.append(name)
    assert "handoff-manifest.json" not in names, "external_manifest_inside_zip"
    assert text_decode_failures == [], (
        "non_utf8_text_like_entry: " + ", ".join(text_decode_failures)
    )
    bad_paths = [
        name
        for name in names
        if "\\" in name
        or Path(name).suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES
        or name.startswith("data/raw/")
        or "/raw/" in name
    ]
    assert bad_paths == [], "forbidden_handoff_data_entry: " + ", ".join(bad_paths)
    for relative_path, metadata in manifest.get("artifacts", {}).items():
        path = review_root / relative_path
        assert path.is_file(), f"manifest_artifact_missing: {relative_path}"
        assert _sha256(path) == metadata["sha256"], (
            f"manifest_hash_mismatch: {relative_path}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
