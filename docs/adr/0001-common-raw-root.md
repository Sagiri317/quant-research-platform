# ADR 0001: common raw root with source-owned namespaces

## Context

Provider exports were previously addressed as separate caller roots. That
layout could not express reference data alongside CSMAR and Binance data.

## Decision

Use `data/raw` as the common raw root. Each interim source receives only its
source-owned namespace, `data/raw/<source_id>`, and recursively validates the
requested provider files from their observed filename and header. No alternate
root detection or second raw truth is part of the interface.

Retain each exact byte sequence once during canonical staging and record duplicate
source provenance in the retained file's sidecar. Keep unknown or unproven files
under a named raw-only family when their table identity is proven; reserve
`unclassified` for files whose identity cannot be established without guessing.
Do not infer a contract from a folder name.

The raw cutover uses same-parent owner-qualified staging and old-good names. A
readback failure restores the old names. Residual old-good data is evidence for
manual recovery, not an input selected by production code.

The standalone A-share fee CSV remains the one fee reference source. Its sidecar
retains exact checksum, size, row-count, and field facts; the designated input is
accepted for the current QRP research purpose and no normalized raw duplicate is
created here.

## Status

Accepted for the current local raw layout.

## Consequences

- Source identity and materialization receive one common root and use the same
  source-owned namespace.
- Table families retain separate owners, and unproven families cannot silently
  enter research materialization.
- Sidecars preserve checksum and quality facts without becoming a second raw
  manifest or registry.
- New raw acquisition must be staged and read back before replacing the active
  root.
