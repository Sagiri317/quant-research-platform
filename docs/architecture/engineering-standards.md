# Engineering Standards

These standards apply to QRP Clean Rewrite `0.1.0`.

## Versioning

- Project package version is `0.1.0`.
- API/spec/contract version is `0.1`.
- API paths use `/api/0.1/...`.
- Durable docs must not record real commit SHAs.

## Module Boundaries

- `contracts/` defines stable definitions and specs.
- `storage/` owns file layout, parquet helpers, manifests, partitions,
  checksums, and storage privacy helpers.
- `catalog/` owns queryable registries and must be ready for a future database
  adapter.
- `sources/` owns vendor/source ingestion into canonical interim data.
- `markets/` owns market-specific semantics.
- `research/` owns research datasets, evaluations, models, predictions, and
  portfolios.
- `workflows/` owns local DAG planning/execution and checkpoints.
- `compute/` owns dataframe/tensor math helpers only.
- `agents/` owns proposal/spec boundaries only.
- `interfaces/` owns local API/CLI adapters.
- `reports/` owns report manifests, summaries, and bundles.
- `runtime/` owns logging, profiling, privacy, sanitization, and errors.

## Data Safety

- Private-derived row-level output must stay under approved private roots.
- Metadata, stdout, stderr, API errors, reports, and manifests must be
  sanitized.
- Do not expose local absolute paths, row samples, stock-code samples,
  vendor/source markers, secrets, tokens, or raw tracebacks with local/private
  context.

## Testing

- Put enforceable contract rules in pytest or scripts.
- Run targeted tests before broad tests.
- Before commit, run `ruff`, `git diff --check`, targeted pytest, and the clean
  rewrite stale-term gate.
