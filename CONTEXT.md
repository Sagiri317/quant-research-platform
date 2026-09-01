# QRP context

## Active raw root

CSMAR raw exports live under one recursively scanned active root: `data/raw/csmar`.
The source owner discovers files below that root; a sibling raw root is not an
input or fallback.

## Raw family

A raw family is a provider/table identity with one source-owner contract. Families
that cannot be proven to satisfy an active adapter remain raw-only.

## Exact duplicate

An exact duplicate has identical bytes. Canonical staging keeps one copy and
records the discarded source names and content hash in its sidecar provenance.

## Same-table replacement

When a newer export is selected for the same provider/table identity, it is the
active raw candidate. Coverage and structural differences remain sidecar evidence;
the bytes are not rewritten to repair the export.

## KEEP_RAW_ONLY and unclassified

`KEEP_RAW_ONLY` means the bytes are retained but excluded from active producer
materialization until an owner contract can prove their semantics. `unclassified`
is reserved for files whose table identity cannot be established without guessing.

## Canonical sidecar

Each retained CSV has one deterministic UTF-8 `.metadata.json` sidecar containing
schema, counts, dates, hash, provenance, and readiness facts. It is operational
metadata, not a second dataset truth.

## Formal readiness

Formal readiness requires an owner, a real production consumer, persisted output,
and the applicable semantic and evidence checks. A readable standalone fee
candidate is provisional and remains unverified until its execution semantic batch
provides the formal consumer and evidence.
