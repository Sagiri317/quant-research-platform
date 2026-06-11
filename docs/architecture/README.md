# Architecture

This directory documents the current QRP Clean Rewrite `0.1.0` architecture
boundary. It is not a record of old branch history.

## Reading Order

1. `ai-native-engineering-principles.md`
2. `engineering-standards.md`
3. `quant-research-standards.md`
4. `next-stage-roadmap.md`

## Current Architecture

- Stable contracts live in `src/quant_research_platform/contracts/`.
- Storage layout and privacy checks live in `storage/` and `runtime/`.
- Catalogs are the queryable source of truth for registered platform objects.
- Catalogs expose current factors, labels, and research dataset requirements.
- Internal provider adapters live under `sources/`.
- Market semantics live under `markets/`; the first active market is `a_share`.
- Research assembly, evaluation, prediction, and portfolio simulation live under
  `research/`.
- Workflow orchestration lives under `workflows/` and is a simple local DAG
  executor in this phase.
- API and CLI adapters live under `interfaces/`.
- Private report outputs live under `reports/private_real/`.

## Non-Goals

No website UI, deployment, scheduler, worker, deep learning, feature store, model
registry, external LLM API, broker/exchange connector, paper trading, live
trading, or production backtester is in scope for this phase.
