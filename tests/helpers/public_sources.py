"""Construct public-source collection fixtures."""

import json

import pandas as pd


def write_source_manifest(root, manifest, *, name="manifest.json"):
    path = root / "manifest.json"
    source = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"source_root": ".", "collections": {}}
    )
    source["collections"][name] = manifest
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")


def write_source_csv(path, rows):
    frame = pd.DataFrame(rows)
    frame["source_sha256"] = "original-response"
    frame["source_url"] = "https://example.org/original"
    frame.to_csv(path, index=False)
    return path.read_bytes()
