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

`policy/` owns share safety, identifier safety, sanitize-key collision rejection, and exact check status aggregation. `contracts/` owns structure and generic strict validation primitives only. Its internal, non-package-root-exported `_json.py` is the sole QRP-owned structural JSON codec and typed canonical identity hash owner: it performs strict UTF-8/duplicate-key/nonfinite/top-level-object validation, deterministic JSON-native writes without coercion, and type-tagged length-framed identity. Its bounded writer admits finite Decimal values only within the existing 4096 coefficient-digit ceiling and the current runtime `MIN_EMIN..MAX_EMAX` adjusted-exponent domain before materializing the digit tuple; unbounded identity and admitted golden bytes are unchanged. Provider/raw JSON readers remain source-owned input parsers and cannot emit QRP artifact bytes. Dataset fingerprinting reuses that typed framing for runtime nested-value normalization; it does not render a second JSON identity or artifact codec. Market-specific asset semantics stay in market owner layers. `catalog/` owns generic registry classes. `profiles/` is the composition root used by API, CLI, agents, and workflows.

`research/` builds results only and does not write files. `workflows/diagnostics/` reads storage datasets when a report needs schema or validation payloads. `reports/` builds and renders prepared evidence/report content only; it does not read storage datasets. `artifacts/` writes LOCAL_ONLY outputs and REVIEW_SAFE evidence files. `workflows/` only orchestrates research -> diagnostics payload -> reports -> artifacts. `storage/` only owns dataset IO and data layout.

`storage/` also owns the generic dataset instance-metadata schema. `input_dataset_identities` is the only supported dependency-lineage field: it is an exact JSON-safe mapping from input dataset id to that dataset's `spec.json` and `_manifest/manifest.json` hashes. Content fingerprints remain a dataset comparison capability and are not a parallel lineage representation. Storage validates this generic shape but does not decide market lineage, dependency coverage, or materialization ordering; those lifecycle semantics remain in the relevant market implementation.

`storage/_dataset_schema.py` owns the exact base/interim/processed/research spec grammars and matching manifest grammar. Unknown or wrong-kind fields fail closed; write-side dataclass fields are validated before path or Parquet IO, and copied spec/manifest fields are compared through the sole typed canonical identity owner rather than Python numeric equality. Manifest structural validation, diagnostic readability, normal consumability, and full persisted inspection are distinct truth tiers in the same owner pipeline. Counts are exact nonnegative integers bounded by `row_count`; canonical primary-key timestamp date bounds and canonical `year=YYYY` partitions are cross-validated. Columns and partitions are unique typed lists, hashes are lowercase 64-hex, and validation status uses the Policy vocabulary without normalization. The same owner derives one recursive physical Arrow schema descriptor from an already-open `ParquetFile` schema: field and nested-child nullability, parameterized nested types, decimal representation bounds, map-key non-nullability, and dictionary index/value/ordered identity all participate in the canonical schema hash. Normal read and fingerprint reject manifest-declared failed status, duplicate keys, primary-key nulls, or nonfinite values, while diagnostic inspection retains structural access.

`storage/_dataset_reading.py` owns the stable `DatasetSession` and sole physical traversal used by materialized reads, chunked reads, time-window inspection, fingerprinting, persisted writer readback, and approved internal consumers. Its Storage-private SQLite inventory is the sole physical-discovery, numeric-order, path-identity and part-metadata owner: filesystem scans and later verification stream one path at a time, while numeric year/part keys, physical identities, and borrowed Crypto and Research path-view membership are exact indexed constraints rather than Python path or identity containers. Borrowed views share the same inventory ownership and full-identity verification. The session freezes bounded strict `spec.json` and manifest bytes, exact part inventory, file identity, schema and logical dtype plan; each file traversal performs pre/post identity checks on the same open Parquet handle, while session final verification reopens and re-scans filesystem paths and identities. The owner-controlled cache is fixed to one 1024-byte SQLite page; other small SQLite native bookkeeping is bounded external-library overhead and is not claimed byte-exact. Disk admission remains separate: initialization reserves only one native SQLite page and each row write gates its actual owner encoding before SQLite allocation. Logical `DatasetReadMode.CHUNKED` boundaries do not imply whole-file reads: numeric part indexes define deterministic file order, canonical year parsing drives both discovery and window pruning, Parquet statistics prune non-matching row groups, and schema-aware exact Arrow predicates apply before Pandas conversion. Projection labels are exact unique non-empty strings, and unsupported or nonfinite filter scalars fail before Arrow/statistics work. Part discovery rejects hardlinked, symlinked, junction-backed, duplicate, externalized, or noncanonical partition aliases. Dataset-wide null and dictionary metadata stabilize nullable and categorical dtypes across CHUNKED, MATERIALIZED, filtered, and empty results.

`storage/_dataset_key_tracking.py` owns the exact spillable primary-key index used by dataset writes and exact dataset comparison. Final stored Arrow columns feed the typed encoder directly, without an intermediate Pandas materialization. Writer indexing detects chunk-local and cross-part duplicate keys; comparison namespaces encode each row through one bounded append sink, store each persisted primary-key payload with its selected-row payload, and use indexed queries for bounded missing, extra, and changed counts. Persisted null and nonfinite row values remain typed comparison values, map entry order is canonicalized in the same sink, while primary-key null and nonfinite rules still fail closed. The SQLite database is a temporary uniqueness/comparison index, not a business database: its initial page and every encoded row are disk-gated, its bounded page-cache reserve participates in operation headroom, and exact comparison reserves one bounded row payload plus one equally sized transient SQLite BLOB binding copy alongside the reader working set. Comparison uses one session at a time. Ordinary completion and primary-error paths remove the database plus journal, WAL, and SHM sidecars. Scratch names carry an explicit QRP owner tag and PID; recovery deletes only recognized stale ownership, compares directory time with the current process start time to reject PID reuse, normalizes Windows case aliases, and preserves unclaimed sibling directories.

Storage captures one issued Runtime policy after bounded metadata/path preflight and before data-scale decode or materialization; materialized reads reuse that same session for an exact row/string/object sizing pass, deduct the inventory page reserve, and fill exact public Pandas/NumPy/Arrow buffers through a second bounded pass without retaining frames or concatenating them. `payload_target_bytes` bounds one physical payload; `task_memory_budget_bytes` gates diagnosed Storage-created Pandas payload and Arrow bytes across writer copies, persisted readback, materialization transition, PK/category/summary state, and nested consumers. This payload sizing does not prove native memory, RSS, or the full working-set peak; the actual peak remains 5B. Admission-before-action and actual retained-state checks do not introduce a third policy quantity or a guessed concat multiplier. Fingerprinting is fast reject only: ordinary columns avoid a normalization copy, while physical map and top-level physical float32/float64 primary-key normalization plus row hashes are produced and released in calibrated bounded slices；float16 与所有 nested float 保留 bit identity。受支持的比较要求两个 artifact 在整个比较过程中保持静止（quiescent）；顺序执行的 session reopen 和 token checks 只提供尽力而为的已观测漂移检测（best-effort observed-drift detection），不构成联合原子快照。只有 persisted Arrow exact comparison 才能产生 `MATCH`。File and partition differences are bounded layout diagnostics and do not replace content proof.

`storage/atomic.py` owns the single package-internal fail-if-exists directory lifecycle shared by staged datasets and artifact writers. Registered final dataset roots require their exact contract segment; scratch-looking names are never accepted as final layout. Its fixed-length target digest keeps scratch components portable even when the final component is long, target markers disambiguate digest collisions, and glob-safe owner prefixes drive stale recovery without interpreting target-name metacharacters. A recognized dead non-empty stage is removed only when its complete marker matches the exact target; an empty dead stage and its empty, truncated, or unreadable marker can be removed without risking foreign payload. A dead-owner marker whose stage is already absent is also removed without trusting its incomplete payload, while live or unrecognized names are preserved. Windows rename and Linux `renameat2` claim the final filesystem name without replacing a concurrently completed target; a concurrent winner is normalized to the same review-safe already-exists error as an entry-time conflict. Claimed file and directory scratch is removed for ordinary exceptions and catchable control-flow cancellation, while an uncatchable hard kill remains the responsibility of stale-owner recovery. Temporary ownership, retry, cleanup, and recovery are not reimplemented by the dataset writer, and the primitive is not a second dataset layout or public execution mode.

`runtime/resources.py` issues a sealed immutable `RuntimeResourcePolicy` containing only `task_memory_budget_bytes` and `payload_target_bytes`; the task budget is capped at 80% of discovered available memory, non-current settings fail closed, and each operation captures one issued policy before passing the same object downstream. Runtime exposes one Pandas payload diagnostic. Each owner enforces its own retained-state lifecycle from that policy; disk checks remain separate owner calls with an explicit same-volume `reserve_bytes`.

`artifacts/` owns artifact membership validation for generated outputs. Generic
artifact validation checks DataFrame membership values directly and must not use
hash-set conversion when untrusted artifact columns may contain unhashable
values. Membership checks compare the exact stored identifiers and do not strip or
coerce generated artifact membership values.
Chunked prediction output uses the storage-owned exact spillable key index against
the Arrow schema written to its Parquet file, so cross-chunk uniqueness does not
retain a row-proportional Python key set. Its SQLite scratch stays inside the
atomic output stage and is removed before the output bundle is published.

Diagnostics that aggregate dataset schemas and market verification summaries
derive disjoint physical-read and SQLite-cache allocations from the active
`RuntimeResourcePolicy`; they do not fall back to the storage reader's default
target while a key index is simultaneously live.
Output-bundle aggregation consumes Parquet membership and order from the output
manifest, rejects undeclared Parquet, filesystem aliases, and external hardlinks,
and then routes each declared file through the same storage physical iterator.

Research assembly captures one active policy and passes it through date-session
or exact-timestamp chunking; intraday timestamps are not normalized to dates.
The current sizing estimate uses observed Pandas payload only to select an
owner-local time partition. Year-partitioned chunk views borrow the same Storage
inventory through a temporary path selection and release that selection on normal
completion or error. Individual output payload is checked against
`payload_target_bytes`, and known simultaneously retained Pandas payload can fail
against `task_memory_budget_bytes`, but neither check claims to measure Python key
state, allocator overhead, RSS, or the full join/sort/write peak. Scoped traversal,
bounded key state, join-frame release, split/spill, and bounded evaluation and
portfolio aggregation are not guarantees of the current Research implementation.
The deterministic output and business ordering semantics are unchanged.
Local verification validates
prediction windows only for profiles that build prediction specs. Before staged
outputs it performs an explicit same-volume free-space check using owner-selected
reserve bytes independently from Runtime memory policy. Its output index
requires every declared file to exist inside the owning bundle; finalization writes
the human summary before atomically replacing the completed `run.json`. The compare
tool requires that summary, derives matrix/overall status through the Policy owner,
cross-validates run data and validation facts, and typed-exactly binds each run index
entry to its persisted output manifest before comparing file content.

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
