# CSMAR Source Contract Boundary

The internal provider implementation is split by owner module. Retired
entrypoint facades are absent. The source-level dataset ids are owned by this
package, while the public data product id remains owned by the platform
catalog.

Registry boundary:

```text
catalog/data_products.py
  a_share_daily_bar = public data product id
  a_share_daily_bar_interim = product-level interim output id

catalog/provider_bindings.py
  a_share_daily_bar -> csmar source
  required_source_dataset_ids = daily_market / stock_master / market_cap /
    adjustment / index_info / benchmark_index_market / trading_calendar

catalog/sources.py
  csmar = internal source identity and implementation package

sources/csmar/catalog.py
  csmar source identity

markets/implementations/a_share/source_bindings/csmar.py
  A-share CSMAR persisted interim catalog, provider binding, market semantics,
    interim pruning, and private raw-to-interim import orchestration
```

The split source modules are:

- `catalog.py`
- `fields.py`
- `ids.py`
- `raw_schemas.py`
- `raw_exports.py`
- `field_mappings.py`
- `checksums.py`

Allowed responsibilities:

- `catalog.py`: CSMAR source definition only
- `ids.py`: CSMAR source identifier only
- `fields.py`: CSMAR source-local field roles only
- `raw_schemas.py`: raw export schema descriptors and raw table inspection only
- `raw_exports.py`: raw export file discovery and read helpers only
- `field_mappings.py`: source-to-interim field mapping metadata, approved
  aliases, field descriptions, and mapping helpers
- `checksums.py`: source-level checksum and metadata checksum helpers

Forbidden responsibilities:

- A-share market semantics
- A-share CSMAR interim catalog, column pruning, or importer orchestration
- processed data materialization
- factor, label, mask, universe, dataset, evaluation, prediction, model, or
  portfolio logic
- output path selection or persisted writer calls
- convenience imports from `__init__.py`
- compatibility modules for retired clean-rewrite entrypoints

This is an internal provider implementation boundary, not a public API/CLI
surface.

## Follow-up TODOs

These are deferred cleanup considerations, not roadmap commitments:

- Possible future rename `interim_spec_ids` -> `interim_output_ids` to reduce
  naming ambiguity.
- Possible future removal of duplicated `implementation_package` from
  `ProviderBinding` after multi-source semantics stabilize.
- Raw `index_market` support remains in the source schema, mapping, raw loader,
  and benchmark extraction path. The current persisted A-share CSMAR interim
  catalog exposes `benchmark_index_market`, derived from raw `index_market`,
  but does not expose direct materialized `index_market`.
- `benchmark_index_market` filters to the benchmark subset before duplicate
  validation. Exact duplicate benchmark keys are deterministically deduped,
  conflicting benchmark keys fail closed, and unrelated non-benchmark raw
  conflicts do not block the benchmark artifact.
- Dev/test tooling is declared in the optional `dev` dependency group; use the
  README conda-first install command rather than adding pytest/ruff to runtime
  dependencies.
- Future broader coverage improvements for tensor, raw_exports, checksums, and
  selected error paths.
- Optional private verification B rerun as verification-only, not code cleanup.
- Future calendar reference storage/catalog: 0.1.0 persists CSMAR
  `trading_calendar` as a private interim source artifact at
  `data/interim/csmar/trading_calendar/data`, so processed/factor builders never
  read raw calendar directly. If calendars become shared across multiple
  sources, markets, or databases, promote this source-specific interim artifact
  into a dedicated reference-data layer or database-backed calendar catalog.
