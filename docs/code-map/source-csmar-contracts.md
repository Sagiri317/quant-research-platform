# CSMAR Source Contract Boundary

`src/quant_research_platform/sources/csmar/contracts.py` is retained as the
source-adapter contract boundary for QRP `0.1`.

Allowed responsibilities:

- raw export schema descriptors
- source-to-interim field mapping descriptors
- source-level checksum helpers
- portable validation errors for the source adapter

Forbidden responsibilities:

- A-share market semantics
- processed data materialization
- factor, label, mask, universe, dataset, evaluation, prediction, model, or
  portfolio logic
- output path selection or persisted writer calls

If this file grows again, split it into `raw_schemas.py`, `field_mappings.py`,
and `checksums.py`, leaving `contracts.py` as public exports only.
