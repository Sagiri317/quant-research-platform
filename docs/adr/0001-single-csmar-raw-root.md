# ADR 0001: single CSMAR raw root

## Context

CSMAR exports were previously split across more than one local raw root. That
layout allowed source discovery to choose a root outside the caller's requested
root and made duplicate table exports difficult to reason about.

## Decision

Use `data/raw/csmar` as the only active CSMAR raw root. The source owner recursively
discovers provider files within that root and validates each requested table from
its observed filename and header. No sibling-root fallback or second raw truth is
part of the interface.

Retain each exact byte sequence once during canonical staging and record duplicate
source provenance in the retained file's sidecar. Keep unknown or unproven files
under a named raw-only family when their table identity is proven; reserve
`unclassified` for files whose identity cannot be established without guessing.
Do not infer a contract from a folder name.

The raw cutover uses same-parent owner-qualified staging and old-good names. A
readback failure restores the old names. Residual old-good data is evidence for
manual recovery, not an input selected by production code.

The standalone A-share fee CSV remains the one fee reference source. Its sidecar
marks the evidence provisional and unverified; the execution semantic batch is the
future formal consumer and no normalized duplicate is created here.

## Status

Accepted for the current local raw layout.

## Consequences

- Source identity and materialization receive one root and use the same recursive
  source-owner discovery.
- Table families retain separate owners, and unproven families cannot silently
  enter research materialization.
- Sidecars preserve provenance and quality facts without becoming a second raw
  manifest or registry.
- New raw acquisition must be staged and read back before replacing the active
  root.
