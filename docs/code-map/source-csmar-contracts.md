# CSMAR Source Contract Boundary

The CSMAR package owns source-local raw export inspection and raw-to-interim
normalization. It is an internal provider implementation boundary, not a public
API/CLI surface.

## Current Boundary

```text
sources/csmar/raw_exports.py     -> raw export discovery and reads
sources/csmar/field_mappings.py  -> raw-to-interim field mapping metadata
sources/csmar/interim.py         -> CSMAR raw -> CSMAR interim
data/interim/csmar/<data_id>     -> source-normalized interim artifacts

markets/implementations/a_share/ -> A-share market semantics
data/processed/a_share/<kind>/<data_id>
```

`sources/csmar` must not import `markets.implementations.a_share`. It does not
produce `market_id`, platform `asset_id`, eligibility masks, execution masks, or
A-share processed artifacts.

## Source-Owned Responsibilities

- CSMAR source identity and source-local ids.
- Raw export schema descriptors and CSV read helpers.
- Source-to-interim field mappings.
- Source-normalized interim specs with `data_id` and `source_id`.
- Interim row fields such as `source_asset_id` where the source table is
  asset-level.

## Forbidden Responsibilities

- A-share market semantics.
- Platform `asset_id` mapping.
- `market_id` output.
- `eligible`, `can_enter`, or `can_exit` output.
- Processed data materialization.
- Factor, label, annotation, research dataset, evaluation, prediction, model,
  or portfolio logic.
- Public API/CLI routes or commands.
- Compatibility modules for retired clean-rewrite entrypoints.

## Notes

- The current supported raw export table format is CSV. Spreadsheet exports are
  not accepted as raw tables unless a real read path and tests are added.
- TXT field descriptions are semantic references for source fields. They are
  not raw table inputs.
- JSON metadata sidecars are check material for private intake evidence. They
  are not required pipeline inputs and are not written by the source reader.
- `source_id` is artifact metadata, not an interim row column.
- `source_table` is raw schema or manifest metadata when needed, not an interim
  row column.
- `available_at` is not invented for source data that does not provide it.
- `source_exchange` is emitted only if the raw schema already has a reliable
  exchange or venue field and the code already reliably uses it.
