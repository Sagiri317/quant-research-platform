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

`artifacts/` owns artifact membership validation for generated outputs. Generic
artifact validation checks DataFrame membership values directly and must not use
hash-set conversion when untrusted artifact columns may contain unhashable
values. Membership checks compare the exact stored identifiers and do not strip or
coerce generated artifact membership values.

Research assembly uses session planning for date-session datasets and exact
timestamp chunking for intraday datasets. Intraday timestamps are not normalized
to dates or routed through the date-session planner. Local verification validates
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
