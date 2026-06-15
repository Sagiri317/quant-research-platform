# Architecture

This directory documents the current QRP Clean Rewrite `0.1.0` architecture boundary. It is not a record of old branch history and is not a prompt to load wholesale for ordinary tasks.

## Reading Order

1. `ai-native-engineering-principles.md`
2. `engineering-standards.md`
3. `agent-governance.md`
4. `quant-research-standards.md`
5. `next-stage-roadmap.md`
6. `agent-governance-redesign-report.md` only when reviewing governance redesign history or future governance changes
7. `agent-governance-adapters-draft.md` only when evaluating optional hooks/MCP/subagent/plugin adapters

## Current Architecture

- Stable contracts live in `src/quant_research_platform/contracts/`.
- Storage layout and privacy checks live in `storage/` and `runtime/`.
- Catalogs are the queryable source of truth for registered platform objects.
- Catalogs expose current factors, labels, and research dataset requirements.
- Internal provider adapters live under `sources/`.
- Market semantics live under `markets/`; the first active market is `a_share`.
- Research assembly, evaluation, prediction, and portfolio simulation live under `research/`.
- Workflow orchestration lives under `workflows/` and is a simple local DAG executor in this phase.
- API and CLI adapters live under `interfaces/`.
- Private report outputs live under `reports/private_real/`.
- Agent governance lives under root `AGENTS.md`, `.agents/`, and durable governance docs in this directory.

## Non-Goals

No website UI, deployment, scheduler, worker, deep learning, feature store, model registry, external LLM API, broker/exchange connector, paper trading, live trading, or production backtester is in scope for this phase.
