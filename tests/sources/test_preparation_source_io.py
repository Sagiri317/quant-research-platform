"""Actual source reads, codec roundtrips and pre-publication failures."""

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.sources.choice import interim as choice
from quant_research_platform.sources.choice.raw_exports import choice_session
from quant_research_platform.sources.csmar import interim as csmar
from quant_research_platform.sources.csmar import raw_schemas
from quant_research_platform.sources.reference import a_share_fees
from quant_research_platform.sources.reference import interim as reference
from quant_research_platform.storage import identity, manifests, raw_snapshots
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from tests.helpers.a_share_raw import write_industry_reference_fixture
from tests.helpers.choice_raw import SDK, response
from tests.helpers.paths import StorageLayout
from tests.sources.reference.test_equity_events import write_equity_source
from tests.sources.reference.test_interim import (
    _industry_observations,
    _write_reference_root,
)
from tests.sources.test_choice_interim import FIELDS, _disclosure, _snapshot


class CountedFile:
    def __init__(self, handle, path, counts):
        self.handle, self.path, self.counts = handle, path, counts

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def read(self, *args):
        data = self.handle.read(*args)
        self.counts[self.path] += len(data)
        return data

    def read1(self, *args):
        data = self.handle.read1(*args)
        self.counts[self.path] += len(data)
        return data

    def readline(self, *args):
        data = self.handle.readline(*args)
        self.counts[self.path] += len(data)
        return data


def count_source_reads(monkeypatch, raw):
    opened, counts, handles = Counter(), Counter(), []
    original = identity.open_filesystem_file

    def tracked(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        path = Path(path)
        if path.is_relative_to(raw):
            opened[path] += 1
            handles.append(handle)
            return CountedFile(handle, path, counts)
        return handle

    for owner in (identity, manifests, raw_snapshots, raw_schemas, a_share_fees):
        monkeypatch.setattr(owner, "open_filesystem_file", tracked)
    return opened, counts, handles


def member_digest(entries):
    digest = hashlib.sha256()
    for name, member_hash in sorted(entries):
        for value in (name, member_hash):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def test_csmar_known_members_parse_and_hash_once_before_publication(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = []
    for shard in range(2):
        path = raw / f"csmar__trading_calendar__p{shard}__full__export-20260101.csv"
        rows = [
            f"4,{day.date()},4,O"
            for day in pd.date_range("2019-01-01", periods=500)[
                shard * 250 : (shard + 1) * 250
            ]
        ]
        path.write_bytes(
            ("Markettype,Clddt,Daywk,State\r\n" + "\r\n".join(rows) + "\r\n").encode()
        )
        selected.append(path)
    unrelated = raw / "csmar__stock_master__all__full__export-20260101.csv"
    unrelated.write_bytes(b"invalid unrelated schema")
    expected = member_digest(
        [
            ("trading_calendar", hashlib.sha256(path.read_bytes()).hexdigest())
            for path in selected
        ]
    )
    opened, counts, handles = count_source_reads(monkeypatch, raw)
    output = tmp_path / "calendar"
    csmar.materialize_interim_data(
        raw_root=raw,
        output_roots={"trading_calendar": output},
        data_ids=("trading_calendar",),
        chunksize=37,
    )
    assert opened == Counter({path: 1 for path in selected})
    assert counts == Counter({path: path.stat().st_size for path in selected})
    assert handles and all(handle.closed for handle in handles)
    assert len(read_dataset(output)) == 500
    assert (
        read_dataset_manifest_evidence(output)["source_evidence"]["content_hash"]
        == expected
    )


def test_choice_selected_response_bytes_and_sidecars_are_read_once(
    tmp_path, monkeypatch
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
    raw = layout.raw_source_root("choice")
    selected = list(raw.rglob("*.csv")) + list(raw.rglob("*.json"))
    expected = member_digest(
        [
            (
                path.relative_to(raw).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in selected
        ]
    )
    opened, counts, handles = count_source_reads(monkeypatch, raw)
    output = tmp_path / "balance"
    choice.materialize_interim_data(
        raw_root=raw,
        output_roots={"financial_balance": output},
        data_ids=("financial_balance",),
    )
    assert opened == Counter({path: 1 for path in selected})
    assert counts == Counter({path: path.stat().st_size for path in selected})
    assert handles and all(handle.closed for handle in handles)
    assert read_dataset(output).total_assets.tolist() == [100.0]
    assert (
        read_dataset_manifest_evidence(output)["source_evidence"]["content_hash"]
        == expected
    )


@pytest.mark.parametrize("kind", ["fees", "industry", "equity"])
def test_reference_parsed_members_and_proof_documents_each_read_once(
    tmp_path, monkeypatch, kind
):
    raw = tmp_path / "raw"
    if kind == "fees":
        _write_reference_root(raw)
        data_id = reference.A_SHARE_FEE_SCHEDULE_DATA_ID
    elif kind == "industry":
        write_industry_reference_fixture(raw, observations=_industry_observations())
        data_id = reference.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    else:
        write_equity_source(raw / "reference")
        data_id = "a_share_equity_events"
    raw = raw / "reference"
    selected = list((raw / data_id).iterdir())
    opened, counts, handles = count_source_reads(monkeypatch, raw)
    output = tmp_path / "prepared"
    reference.materialize_interim_data(
        raw_root=raw,
        output_roots={data_id: output},
        data_ids=(data_id,),
    )
    assert opened == Counter({path: 1 for path in selected})
    assert counts == Counter({path: path.stat().st_size for path in selected})
    assert handles and all(handle.closed for handle in handles)
    assert not read_dataset(output).empty


@pytest.mark.parametrize("source", ["csmar", "choice", "reference"])
def test_source_close_failure_prevents_publication_and_preserves_old_good(
    tmp_path, monkeypatch, source
):
    raw = tmp_path / "raw"
    if source == "csmar":
        raw.mkdir()
        path = raw / "csmar__trading_calendar__all__full__export-20260101.csv"
        path.write_bytes(b"Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n")
        owner, data_id, io_owner = csmar, "trading_calendar", raw_schemas
    elif source == "choice":
        layout = StorageLayout(tmp_path)
        with choice_session(SDK(response())) as session:
            _disclosure(session, layout)
        raw = layout.raw_source_root("choice")
        path = next(raw.rglob("response.csv"))
        owner, data_id, io_owner = choice, "financial_first_disclosure", raw_snapshots
    else:
        _write_reference_root(raw)
        raw /= "reference"
        path = next(raw.rglob("*.csv"))
        owner, data_id, io_owner = (
            reference,
            reference.A_SHARE_FEE_SCHEDULE_DATA_ID,
            a_share_fees,
        )
    old = tmp_path / "old"
    owner.materialize_interim_data(
        raw_root=raw, output_roots={data_id: old}, data_ids=(data_id,)
    )
    saved = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    original = io_owner.open_filesystem_file
    failure = OSError("source close failed")
    handles = []

    class FailingClose:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            self.handle.close()
            raise failure

    def open_source(candidate, *args, **kwargs):
        handle = original(candidate, *args, **kwargs)
        if Path(candidate) == path:
            handles.append(handle)
            return FailingClose(handle)
        return handle

    monkeypatch.setattr(io_owner, "open_filesystem_file", open_source)
    output = tmp_path / "new"
    with pytest.raises(OSError) as caught:
        owner.materialize_interim_data(
            raw_root=raw, output_roots={data_id: output}, data_ids=(data_id,)
        )
    assert caught.value is failure
    assert handles and all(handle.closed for handle in handles)
    assert not output.exists()
    assert {
        p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()
    } == saved


@pytest.mark.parametrize("field", ["RequestID", "after_cutoff_count"])
def test_choice_materialization_rejects_boolean_numeric_source_facts(tmp_path, field):
    import json

    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        source, _ = _disclosure(session, layout)
    sidecar = source / "response.metadata.json"
    manifest = json.loads(sidecar.read_bytes())
    facts = manifest["metadata"]["response" if field == "RequestID" else "scope"]
    assert facts[field] == 0
    facts[field] = False
    sidecar.write_text(json.dumps(manifest))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="choice_(response_type|saved_scope)_invalid"):
        choice.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={"financial_first_disclosure": output},
            data_ids=("financial_first_disclosure",),
        )
    assert not output.exists()


def test_funding_consumption_rejects_boolean_coverage_time(tmp_path):
    from quant_research_platform.sources.binance.funding import (
        read_funding_observations,
    )
    from tests.sources.test_binance_funding import _write

    _write(
        tmp_path / "raw",
        metadata_changes={
            "coverage": {"start_ns": False, "end_ns": 100, "status": "COMPLETE"},
        },
    )
    with pytest.raises(ValueError, match="binance_funding_coverage_invalid"):
        with read_funding_observations(
            (tmp_path / "funding/BTCUSDT",), symbol="BTCUSDT"
        ):
            pass


@pytest.mark.parametrize("method", ["read", "read1", "readinto", "readinto1"])
def test_digest_reader_counts_every_consumed_byte_through_binary_methods(
    tmp_path, method
):
    payload = ('\ufeff"中文,\r\n引号""",2\r\n' * 1000).encode()
    path = tmp_path / "bytes.csv"
    path.write_bytes(payload)
    restored = bytearray()
    with path.open("rb") as handle:
        reader = identity._DigestingReader(handle)
        while True:
            if method in {"readinto", "readinto1"}:
                buffer = bytearray(13)
                size = getattr(reader, method)(buffer)
                data = buffer[:size]
            else:
                data = getattr(reader, method)(13)
            if not data:
                break
            restored.extend(data)
        assert restored == payload
        assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()


def test_csmar_early_close_cannot_claim_a_completed_source_digest(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(b"first,second\n1,2\n")
    hashes = []
    rows = raw_schemas.iter_export_csv_records(path, content_hashes=hashes)
    assert next(rows) == ["first", "second"]
    rows.close()
    assert hashes == []
    assert list(raw_schemas.iter_export_csv_records(path, content_hashes=hashes)) == [
        ["first", "second"],
        ["1", "2"],
    ]
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest()]
