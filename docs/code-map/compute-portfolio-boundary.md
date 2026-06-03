# Compute Portfolio Boundary

`compute/dataframe/portfolio.py` and `compute/tensor/portfolio.py` remain compute
backend modules for QRP `0.1`. They must not gain storage, catalog, API,
workflow, market, broker, exchange, paper trading, or live trading
responsibilities.

Accepted 0.1.0 limitation: the files are not split in this pre-merge fix
because the current task is contract and acceptance repair. Future work should
split portfolio compute into:

- construction
- constraints
- turnover
- exposure
- reporting

No new portfolio logic should be added to the monolithic files unless it is
first assigned to one of those future submodules.
