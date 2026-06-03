# Clean Rewrite 0.1 Public-Safe Parity Evidence

This report records public-safe aggregate evidence for QRP Clean Rewrite
`0.1.0`. It contains no row samples, stock-code samples, private absolute paths,
credentials, raw source rows, or local machine paths.

Source identifier policy: `csmar` is a public source identifier for catalog
metadata and API routes. It must not appear as row-level private evidence or raw
file evidence in public-safe reports.

## Evidence Scope

| Area | Status | Public-safe evidence type |
| --- | --- | --- |
| processed panel parity | public-safe evidence committed | row_count, column_count, key_count, duplicate_key_count, date_range, column_set_hash, dtype_summary_hash |
| factor value parity | public-safe evidence committed | factor_count, column_set_hash, max_abs_delta, mean_abs_delta, accepted tolerance bucket |
| label value parity | public-safe evidence committed | label_count, horizon_set, max_abs_delta, mean_abs_delta, accepted tolerance bucket |
| masks / universes parity | public-safe evidence committed | mask_count, universe_count, key_count, duplicate_key_count, boolean_rate_hash |
| research dataset parity | public-safe evidence committed | row_count, column_count, key_count, duplicate_key_count, column_set_hash |
| evaluation parity | public-safe evidence committed | metric_set_hash, rankic_delta_tolerance, max_abs_delta, mean_abs_delta |
| prediction smoke parity | public-safe evidence committed | prediction_mode, row_count, key_count, duplicate_key_count, output_root_policy |
| supporting layers migration 2014-2025 | local evidence only | partition_count, manifest_fingerprint, schema_version |
| full historical daily_panel | accepted explained-delta | 2014 validated; full daily panel materialization remains deferred for 0.1.0 |

## Aggregate Result Summary

| Check | Result | Aggregate metadata |
| --- | --- | --- |
| 2014 processed panel | accepted | duplicate_key_count=0; schema_version=0.1; grain=asset_timestamp |
| 2014 factor values | accepted | factor_count=56; max_abs_delta within numeric tolerance; unexplained_regression_count=0 |
| 2014 labels | accepted | horizon_set_hash committed; unexplained_regression_count=0 |
| 2014 masks / universes | accepted | duplicate_key_count=0; accepted_explained_delta=0 |
| 2014 research dataset | accepted | key_count parity accepted; column_set_hash parity accepted |
| 2014 evaluation | accepted | IC and RankIC aggregate deltas within tolerance |
| prediction smoke | accepted | prediction_mode=smoke_in_sample; oos=false; output_policy=reports_private_real_only |

## Historical Migration Status

Supporting factor, label, mask, context, and universe layers for 2014-2025 are
treated as migration-support evidence for the clean rewrite branch. The committed
repository contains only public-safe aggregate metadata; private generated
artifacts remain local evidence only.

The full historical `daily_panel` is recorded as an accepted explained-delta for
QRP 0.1.0: 2014 parity validates the new contracts and writers, while full
daily-panel regeneration is deferred to avoid committing or exposing private
row-level data. This is acceptable for 0.1.0 because the merge target is
architecture acceptance plus spec/catalog/writer safety, not publication of
private generated datasets.

## Stop-Gate Classification

| Classification | Count |
| --- | ---: |
| expected_delta | 1 |
| numeric_tolerance_delta | 1 |
| old_bug_fixed_delta | 0 |
| unexplained_regression | 0 |

If a future parity rerun produces any `unexplained_regression`, the migration
must stop before claiming merge readiness.
