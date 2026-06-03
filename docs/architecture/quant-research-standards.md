# Quant Research Standards

These standards keep QRP research outputs defensible during Clean Rewrite
`0.1.0`.

## Time And Keys

- Processed canonical key is `market_id`, `asset_id`, `timestamp`.
- Use `session_date` for market-local trading sessions.
- Use `available_at` for point-in-time availability.
- `stock_code` and `trade_date` are source or migration fields only.
- `ProcessedDataSpec.grain` declares row meaning and drives duplicate checks,
  joins, partitioning, and evaluation semantics.

## Factors, Labels, Masks

- Factors must not use future information.
- Labels express forward targets through timing, horizon, lookahead, entry rule,
  and exit rule.
- Masks may be as-of, execution, or evaluation masks; future-dependent masks
  must not enter feature sets.
- Current allowed factor sets are `full`, `core`, `smoke`, `experimental`, and
  `deprecated`.

## Evaluation

- Use parity-or-explained-delta for migration validation.
- Classify deltas as expected, numeric tolerance, old bug fixed, or unexplained
  regression.
- Stop on unexplained regression.
- Keep prediction output in reports, not processed data.

## Scope Boundaries

This phase does not claim market alpha, benchmark performance, capacity,
tradability, live results, or investment advice.
