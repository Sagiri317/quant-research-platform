# Clean Rewrite 0.1 Public-Safe Parity Evidence

This evidence records public-safe aggregate validation for QRP Clean Rewrite
`0.1.0`. Public identifiers are `data_product_id=a_share_daily_bar`,
`market_id=a_share`, and contract version `0.1`.

Machine-readable evidence lives in
`docs/reports/clean-rewrite-0.1-parity-evidence.json`.

| Area | Status | Notes |
| --- | --- | --- |
| processed panel parity | implemented | canonical key, grain, and aggregate checksum validated |
| factor value parity | implemented | registered price-volume factor aggregate parity validated |
| label value parity | implemented | forward-return label aggregate parity validated |
| masks / universes parity | implemented | universe and tradability mask aggregate counts validated |
| research dataset parity | implemented | feature and label dataset assembly aggregate parity validated |
| evaluation parity | implemented | evaluation aggregate metrics validated |
| prediction OOS smoke parity | implemented | minimal out-of-sample smoke predicts test window only |
| supporting layers migration 2014-2025 | implemented | year-level aggregate migration status recorded |
| full historical daily_panel | accepted explained-delta | private real-data full materialization not run in this batch |

The full historical `daily_panel` remains an accepted explained-delta for this
pre-merge fix: the 0.1 contract and public-safe aggregate evidence are present,
while private real-data generation is intentionally not run here.

Public-safe status:

- `unexplained_regression | 0`
- `privacy_level | public_safe_aggregate`
- `contract_version | 0.1`
- `private real-data execution not run`
- `future open-source provider sanitization deferred`
