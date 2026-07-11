# Platform Flow

本文描述 QRP 当前代码流程。QRP 是 local research platform，只覆盖本地研究、验证和 REVIEW_SAFE evidence；边界是 no trading execution 和 no production trading boundary。

## 总体链路

```text
sources -> markets implementation -> storage -> research -> workflows diagnostics -> reports -> artifacts
```

研究链路：

```text
dataset -> factor -> feature -> label -> model -> prediction -> signal -> evaluation -> portfolio -> result -> diagnostics payload -> evidence/report payload -> files
```

长期合同锚点：

```text
data -> feature -> signal/model -> evaluate -> evidence
```

该链路受 current contract 和 current-contract-only 约束：文档、脚本、tests 和 handoff evidence 只保护当前合同。

## 关键目录

```text
src/quant_research_platform/policy
src/quant_research_platform/contracts
src/quant_research_platform/catalog
src/quant_research_platform/profiles
src/quant_research_platform/sources
src/quant_research_platform/markets
src/quant_research_platform/markets/implementations/a_share
src/quant_research_platform/markets/implementations/crypto
src/quant_research_platform/storage
src/quant_research_platform/research
src/quant_research_platform/compute
src/quant_research_platform/reports
src/quant_research_platform/artifacts
src/quant_research_platform/workflows
src/quant_research_platform/runtime
src/quant_research_platform/interfaces
src/quant_research_platform/agents
```

## Layer Contracts

`policy/` owns share safety, identifier safety, and check status aggregation. `contracts/` owns structure and generic strict validation primitives only. It may validate types, nulls, finite values, integer-ness, strings, membership, timestamps, booleans, and hashable keys; market-specific asset semantics stay in market owner layers. `catalog/` owns generic registry classes. `profiles/` is the composition root used by API, CLI, agents, and workflows.

`research/` builds results only and does not write files. `workflows/diagnostics/` reads storage datasets when a report needs schema or validation payloads. `reports/` builds and renders prepared evidence/report content only; it does not read storage datasets. `artifacts/` writes LOCAL_ONLY outputs and REVIEW_SAFE evidence files. `workflows/` only orchestrates research -> diagnostics payload -> reports -> artifacts. `storage/` only owns dataset IO and data layout.

`storage/` also owns the generic dataset instance-metadata schema. `input_dataset_identities` is the only supported dependency-lineage field: it is an exact JSON-safe mapping from input dataset id to that dataset's `spec.json` and `_manifest/manifest.json` hashes. Content fingerprints remain a dataset comparison capability and are not a parallel lineage representation. Storage validates this generic shape but does not decide market lineage, dependency coverage, or materialization ordering; those lifecycle semantics remain in the relevant market implementation.

`storage/_dataset_reading.py` owns the single physical Parquet-file batch primitive used by materialized reads, chunked reads, time-window inspection, fingerprinting, and bounded internal consumers of standalone Parquet output bundles. Logical `DatasetReadMode.CHUNKED` boundaries do not imply whole-file reads: numeric part indexes define deterministic file order, Parquet statistics prune non-matching row groups, and Arrow predicates apply time and owner-approved equality filters before Pandas conversion. Fixed-width row-group metadata derives a conservative batch size. Variable-width and Pandas object-expanding Arrow types first use a conservative conversion probe; Parquet uncompressed chunk bounds plus measured expansion then size later batches before allocation, while Arrow buffers and Python batch state remain inside their allocations. A single row above the smallest physical or conversion budget fails closed. Projection, time-window filtering, path order, fingerprint, and final row order remain dataset contracts; physical chunk boundaries do not. Dataset-wide Parquet null statistics stabilize nullable numeric and boolean Pandas dtypes across CHUNKED, MATERIALIZED, filtered, and empty results; nullable integers retain Arrow signedness and bit width, and nullable booleans retain `BooleanDtype`, rather than passing through lossy or generic object representations. Dictionary domains are collected through bounded Arrow metadata batches, charged to physical state, applied identically to every logical chunk, and included in content fingerprints. Nested Arrow values accepted by the writer use a deterministic typed fallback only when Pandas' native row hash cannot hash the restored object. `MATERIALIZED` may retain the final complete DataFrame, but the one storage-owned materialization boundary consumes the same bounded physical conversion path.

`storage/_dataset_key_tracking.py` owns the exact spillable primary-key index used by dataset writes and owner-approved exact set comparison. Keys are cast through the same final Arrow schema used by the staged Parquet writer before their typed, length-framed payload enters SQLite; dictionary keys use their stored dictionary type plus canonical value type, decimal payloads avoid context-dependent rounding, timestamps use naive nanosecond identity, null and NaN fail closed, negative zero follows stored float equality, and strings are not Unicode-normalized. Namespaces use indexed anti-existence queries for exact missing/extra counts without a row-proportional Python set or a temporary sort. The SQLite database is a temporary uniqueness/comparison index, not a business database: writer scratch lives under the staged root, post-publication comparison scratch is a hidden sibling outside the final artifact on the same filesystem, batched transactions use a bounded page cache, and SQLite temporary query state stays in memory rather than creating an unowned system-temp file. Ordinary completion removes the database plus journal, WAL, and SHM sidecars. Scratch names carry an explicit QRP owner tag and PID; recovery deletes only recognized stale ownership, compares directory time with the current process start time to reject PID reuse, normalizes Windows case aliases, and preserves unclaimed sibling directories.

`storage/atomic.py` owns the package-internal fail-if-exists publication primitive shared by staged datasets and artifact file writers. Windows rename and Linux `renameat2` claim the final filesystem name without replacing a concurrently completed target; temporary ownership, retry, cleanup, and recovery remain with the lifecycle owner, and the primitive is not a second dataset layout or public execution mode.

Materialization owners that accept `RuntimeResourcePolicy` derive internal physical read targets from that policy and execute the minimum-free-disk gate before staged output, between output chunks, and once more before the writer can publish the staged artifact. A session planner rejects a single estimated session that already exceeds `effective_chunk_bytes` instead of returning an impossible plan. Each disk check reserves additional same-volume headroom for the next staged chunk, SQLite key index and journal, and safety margin. `storage/` remains independent of `RuntimeResourcePolicy`.

`artifacts/` owns artifact membership validation for generated outputs. Generic
artifact validation checks DataFrame membership values directly and must not use
hash-set conversion when untrusted artifact columns may contain unhashable
values. Membership checks compare the exact stored identifiers and do not strip or
coerce generated artifact membership values.

Research assembly uses session planning for date-session datasets and exact
timestamp chunking for intraday datasets. Intraday timestamps are not normalized
to dates or routed through the date-session planner. Its distinct time-key state
and its physical scan receive disjoint sub-budgets; an unrepresentable namespace
fails closed rather than growing an unbounded list/set. Processed input reads use
the caller policy's physical target. Its output generator applies the same
chunk-boundary and pre-publication disk checks as market materializers.
Local verification validates
prediction windows only for profiles that build prediction specs.

## Interfaces

Current public API:

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

Current public CLI:

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

`.agents tooling` only carries governance, handoff, and prompt tooling. `.run artifacts` are generated evidence, not an architecture or tool source.

`private data safety` remains a hard boundary: REVIEW_SAFE evidence must use summaries, counts, hashes, schemas, command status, and sanitized text only.

## Crypto Binance Flow

当前 crypto Binance 链路由 `docs/code-map/crypto-data-flow.md` 细化。核心 owner-flow 是 `sources/binance` raw envelope/interim normalization -> `markets/implementations/crypto/processed` processed datasets -> `research` factor/label calculations -> crypto research dataset/evaluation specs。

`sources/binance` owns Binance source/raw/interim semantics: local JSONL discovery, canonical envelope validation, raw payload semantic validation, and source-normalized interim datasets. It does not own crypto market binding or research factor semantics.

`markets/implementations/crypto` owns crypto market binding and processed semantics: canonical crypto asset ids, processed source-input selection, pure market-panel calculation, observation diagnostics, factor/label dataset materialization, research dataset specs, and evaluation scenarios.

`research/factors/microstructure` owns generic microstructure factor definitions and calculations. Market implementations bind those generic definitions to market ids and required input data ids; they do not duplicate generic factor construction.

Crypto coverage remains minute-grid based and owner-local to crypto processed
flows. It is not mixed with A-share trading-session coverage.
