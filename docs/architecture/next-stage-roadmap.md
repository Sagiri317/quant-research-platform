# Next Stage Roadmap

This roadmap is scoped to QRP Clean Rewrite `0.1.0`. It records next-stage
implementation priorities, not historical branch closeout.

## Current Priorities

| Priority | Scope | Status |
|---|---|---|
| C1 | Complete 0.1 source topology and contract skeleton | implemented in branch |
| C2 | Port CSMAR raw-to-interim and A-share processed-data builders | implemented skeleton |
| C3 | Port full price-volume factor and forward-label semantics | implemented skeleton |
| C4 | Port research dataset assembly and validation | implemented skeleton |
| C5 | Port evaluation and prediction smoke flows under report contracts | implemented skeleton |
| C6 | Add 2014 parity-or-explained-delta gates | pending private-data run |
| C7 | Sync docs/code-map/API references and private-boundary checks | in progress |

## Non-Goals For This Stage

- Frontend implementation.
- Deployment.
- Scheduler or worker runtime.
- Deep learning.
- Feature store.
- Model registry.
- External LLM API integration.
- Paper trading or live trading.
- Broker/exchange connector.
- Production backtester.
- Full high-frequency implementation.

## Verification Gates

- Targeted pytest for each migrated behavior slice.
- `ruff check`.
- `git diff --check`.
- rg gate for retired package names, old route/command strings, private markers,
  and version misuse.
- Private-boundary sanitizer tests.
