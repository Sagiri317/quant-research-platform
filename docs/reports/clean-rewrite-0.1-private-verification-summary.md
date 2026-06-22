# Clean Rewrite 0.1 Private Verification Summary

This summary records aggregate-only sealed dataset inventory for QRP Clean Rewrite `0.1.0`.

Machine-readable summary lives in `docs/reports/clean-rewrite-0.1-private-verification-summary.json`.

Status: `superseded_by_factor_catalog_refactor`; `evidence_pending_regeneration`.
This file records pre-refactor aggregate inventory only and is not current
merge-ready evidence for the catalog-driven `price_volume` artifact.

| Logical Root | Area | Rows | Columns | Period |
| --- | --- | ---: | ---: | --- |
| processed/a_share/annotation/<market_context_data_id(DataFrequency.DAILY)> | supporting_layers_coverage_2014_2025 | 12231140 | 14 | 2014-01-01 through 2025-12-31 |
| processed/a_share/factor/price_volume | factor_value_parity_superseded | 12231140 | 69 | 2014-01-01 through 2025-12-31 |
| processed/a_share/label/forward_returns | label_value_parity | 12231140 | 14 | 2014-01-01 through 2025-12-31 |
| processed/a_share/annotation/as_of | masks_universes_parity | 12231140 | 6 | 2014-01-01 through 2025-12-31 |
| processed/a_share/annotation/evaluation | masks_universes_parity | 12231140 | 7 | 2014-01-01 through 2025-12-31 |
| processed/a_share/annotation/execution | masks_universes_parity | 12231140 | 7 | 2014-01-01 through 2025-12-31 |
| processed/a_share/panel/<market_panel_data_id(DataFrequency.DAILY)> | full_historical_market_panel_data_id_daily_status | 12231140 | 48 | 2014-01-01 through 2025-12-31 |
| research/a_share/price_volume_forward_returns | research_dataset_parity | 12231140 | 75 | 2014-01-01 through 2025-12-31 |
| processed/a_share/annotation/default | masks_universes_parity | 12231140 | 6 | 2014-01-01 through 2025-12-31 |

Public-safe status:

- `privacy_level | public_safe_aggregate`
- `contract_version | 0.1`
- `verification_kind | private_research_data_aggregate_inventory`
- `source_data_visibility | private_local_only`
- `row_level_output_committed | false`
- `covered_roots | 9`
- `uncovered_roots | 0`
- `allowed_unverified_roots | 0`
- `merge_ready | false`
