# Clean Rewrite 0.1 Public-Safe Parity Evidence

This evidence records public-safe aggregate validation for QRP Clean Rewrite `0.1.0`. Public identifiers are `data_product_id=a_share_daily_bar`, `market_id=a_share`, and contract version `0.1`.

Machine-readable evidence lives in `docs/reports/clean-rewrite-0.1-parity-evidence.json`.

| Area | Rows | Columns | Period | Status |
| --- | ---: | ---: | --- | --- |
| processed panel parity | 660416 | 48 | 2014-01-01 through 2014-12-31 | verified aggregate |
| factor value parity | 12231140 | 69 | 2014-01-01 through 2025-12-31 | verified aggregate |
| label value parity | 12231140 | 14 | 2014-01-01 through 2025-12-31 | verified aggregate |
| masks / universes parity | 48924560 | 8 | 2014-01-01 through 2025-12-31 | verified aggregate |
| research dataset parity | 12231140 | 75 | 2014-01-01 through 2025-12-31 | verified aggregate |
| evaluation parity | 1446 | 7 | 2014-01-29 through 2014-12-31 | verified aggregate |
| prediction OOS smoke parity | 660416 | 4 | 2014-01-01 through 2014-12-31 | verified aggregate |
| 2014-2025 supporting layers migration | 97849120 | 79 | 2014-01-01 through 2025-12-31 | verified aggregate |
| full historical daily_panel status | 660416 | 48 | 2014-01-01 through 2014-12-31 | accepted explained-delta |

Public-safe status:

- `privacy_level | public_safe_aggregate`
- `contract_version | 0.1`
- `fixture-backed API/CLI execution supported`
- `private real-data execution not exposed through API/CLI in 0.1.0`
- `private aggregate metadata verification run locally`
- no row-level artifacts committed by this evidence update
