# CSMAR Source Contract Boundary

The internal provider implementation is split by owner module. There is no
compatibility facade for old clean-rewrite entrypoints. The split modules are:

- `raw_schemas.py`
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
