# CSMAR Source Contract Boundary

`src/quant_research_platform/sources/csmar/contracts.py` is retained as a thin
compatibility facade for the internal provider implementation. The split
modules are:

- `raw_schemas.py`
- `field_mappings.py`
- `checksums.py`

Allowed responsibilities:

- `raw_schemas.py`: raw export schema descriptors and raw inspection helpers
- `field_mappings.py`: source-to-interim field mapping helpers
- `checksums.py`: source-level checksum helpers
- `contracts.py`: compatibility exports for existing internal imports
- portable validation errors for the source adapter

Forbidden responsibilities:

- A-share market semantics
- processed data materialization
- factor, label, mask, universe, dataset, evaluation, prediction, model, or
  portfolio logic
- output path selection or persisted writer calls

This is an internal provider implementation boundary, not a public API/CLI
surface.
