# CSMAR Source Contract Boundary

The internal provider implementation is split by owner module. There is no
compatibility facade for retired compatibility entrypoints. The source-level
dataset ids are owned by this package, while the public data product id remains
owned by the platform catalog.

Registry boundary:

```text
catalog/data_products.py
  a_share_daily_bar = public data product id
  a_share_daily_bar_interim = product-level interim output id

catalog/provider_bindings.py
  a_share_daily_bar -> csmar source
  required_source_dataset_ids = daily_market / stock_master / market_cap /
    adjustment / index_market / index_info / benchmark_index_market

catalog/sources.py
  csmar = internal source identity and implementation package

sources/csmar/catalog.py
  daily_market / stock_master / market_cap / adjustment / index_market /
    index_info / benchmark_index_market = CSMAR source dataset ids
```

The split source modules are:

- `catalog.py`
- `columns.py`
- `fields.py`
- `raw_schemas.py`
- `raw_exports.py`
- `dataset_contracts.py`
- `field_mappings.py`
- `checksums.py`
- `mappers.py`
- `interim.py`
- `importer.py`

Allowed responsibilities:

- `raw_schemas.py`: raw export schema descriptors and raw table inspection
  helpers only
- `dataset_contracts.py`: canonical dataset required columns, primary keys,
  duplicate-key checks, OHLC checks, and liquidity sign checks
- `field_mappings.py`: source-to-interim field mapping metadata, approved
  aliases, field descriptions, and mapping helpers
- `checksums.py`: source-level checksum and metadata checksum helpers
- `mappers.py`: raw-to-canonical transform logic
- `interim.py`: platform identity and time-field normalization
- `importer.py`: the only CSMAR raw-to-interim orchestration entrypoint

Forbidden responsibilities:

- A-share market semantics
- processed data materialization
- factor, label, mask, universe, dataset, evaluation, prediction, model, or
  portfolio logic
- output path selection or persisted writer calls
- convenience re-exports from `__init__.py`
- compatibility modules for retired clean-rewrite entrypoints

This is an internal provider implementation boundary, not a public API/CLI
surface.
