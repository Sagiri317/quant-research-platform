# AI-Native Engineering Principles

This project uses AI agents for local implementation work, but QRP Clean Rewrite
`0.1.0` keeps contracts, source topology, and private boundaries explicit.

## Principles

- Stable contracts live in `src/quant_research_platform/contracts`.
- Storage layout and privacy checks live in `src/quant_research_platform/storage` and `runtime`.
- Market-specific semantics stay under `markets/implementations/<market>`.
- Research assembly, evaluation, prediction, and portfolio work stays under `research`.
- Agent proposal modules may create proposals or specs only; they do not write parquet or run private data directly.
- Public API and CLI entry points are spec-first and use `0.1`.
- Private-derived row-level outputs stay under `data/` or `reports/private_real`.

## Guardrails

- Do not add frontend, deployment, scheduler, worker, broker, exchange, or live-trading surfaces in this phase.
- Do not turn old generated artifacts or old command names into source of truth.
- Mechanical contract checks belong in tests or scripts, not only in prose.
