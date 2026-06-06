# Compute Portfolio Boundary

Focused portfolio compute entry points are split into owner modules:

- `compute/dataframe/construction.py`
- `compute/dataframe/constraints.py`
- `compute/dataframe/turnover.py`
- `compute/dataframe/exposure.py`
- `compute/dataframe/reporting.py`
- `compute/tensor/construction.py`
- `compute/tensor/constraints.py`
- `compute/tensor/turnover.py`
- `compute/tensor/exposure.py`
- `compute/tensor/reporting.py`

These modules must not gain storage, catalog, API, workflow, market,
broker/exchange, paper trading, or live trading responsibilities. New portfolio
logic should be assigned to the focused module that owns the behavior. The old
`compute/dataframe/portfolio.py` and `compute/tensor/portfolio.py` facade modules
are retired clean-rewrite entrypoints.
