# Clean Rewrite 0.1 Public-Safe Parity Evidence

This evidence records public-safe aggregate validation for QRP Clean Rewrite `0.1.0`. Public identifiers are `artifact_id=a_share_private_aggregate_evidence`, `market_id=a_share`, and contract version `0.1`.

Machine-readable evidence lives in `docs/reports/clean-rewrite-0.1-parity-evidence.json`.

Status: `superseded_by_factor_catalog_refactor`; `evidence_pending_regeneration`.
This file records pre-refactor aggregate parity only and is not current
merge-ready evidence for the catalog-driven `price_volume` artifact.

| Area | Rows | Columns | Period | Status |
| --- | ---: | ---: | --- | --- |
| processed panel parity | 660416 | 48 | 2014-01-01 through 2014-12-31 | verified aggregate |
| factor value parity | 12231140 | 69 | 2014-01-01 through 2025-12-31 | superseded pending regeneration |
| label value parity | 12231140 | 14 | 2014-01-01 through 2025-12-31 | verified aggregate |
| masks universes parity | 48924560 | 8 | 2014-01-01 through 2025-12-31 | verified aggregate |
| research dataset parity | 12231140 | 75 | 2014-01-01 through 2025-12-31 | verified aggregate |
| evaluation parity | 1446 | 7 | 2014-01-29 through 2014-12-31 | verified aggregate |
| prediction oos smoke parity | 660416 | 4 | 2014-01-01 through 2014-12-31 | verified aggregate |
| supporting layers coverage 2014 2025 | 85617980 | 79 | 2014-01-01 through 2025-12-31 | verified aggregate |
| full historical market_panel_1d status | 12231140 | 48 | 2014-01-01 through 2025-12-31 | verified aggregate |

Public-safe status:

- `privacy_level | public_safe_aggregate`
- `contract_version | 0.1`
- `verification_kind | private_research_data_aggregate`
- `verification_scope | aggregate_public_safe_parity`
- `evidence_generation_mode | local_private_research_public_safe_aggregate`
- `source_data_visibility | private_local_only`
- `row_level_output_committed | false`
- `merge_ready | false`
- private local aggregate run completed without committed row-level artifacts
