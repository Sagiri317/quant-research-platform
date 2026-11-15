"""Construct public-source collection fixtures."""

import csv
import gzip
import hashlib
import io
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_source_tables(path, tables):
    path.mkdir()
    metadata = {}
    identity = hashlib.sha256()
    for name, payload in tables.items():
        text = payload.decode() if isinstance(payload, bytes) else payload
        identity.update(text.encode())
        rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
        fields, values = rows[0], rows[1:]
        table = pa.Table.from_arrays(
            [
                pa.array([row[i] for row in values], type=pa.string())
                for i in range(len(fields))
            ],
            names=fields,
        )
        name = name.removesuffix(".tsv")
        pq.write_table(table, path / (name + ".parquet"))
        metadata[name] = {"row_groups": [0], "columns": fields}
    return {"path": path.name, "sha256": identity.hexdigest(), "tables": metadata}


def write_fund_records(path, rows):
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    compressed = gzip.compress(payload)
    with path.open("ab") as stream:
        offset = stream.tell()
        stream.write(compressed)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "gzip_member": {"offset": offset, "length": len(compressed)},
        "columns": list(dict.fromkeys(key for row in rows for key in row)),
    }


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
