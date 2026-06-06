# QRP Open Issues Register

This register tracks P0/P1/P2/P3 open issues, resolved blockers, technical
debt, and follow-up tasks for the QRP clean rewrite and related
architecture/rebuild/evidence work.

Priority semantics:
- P0 = blocker; must be fixed before next phase.
- P1 = required before PR merge or before a named gate.
- P2 = important follow-up; not a current blocker.
- P3 = cleanup / nice-to-have / deferred maintenance.

`PROJECT_STATUS.md` and active briefs should link to this file rather than
duplicate the full table.

| ID | Priority | Area | Issue | Why it matters | Required before | Status | Next action | Notes |
|---|---|---|---|---|---|---|---|---|
| QRP-ISSUE-001 | P0 | contracts/runtime | `contracts/workflows.py` imported `runtime.privacy`. | Contracts must not depend on runtime sanitization. | next phase | done | None. | Fixed by moving public sanitization to runtime helper. |
| QRP-ISSUE-002 | P1 | catalog/factors | `FactorCatalog` hardcoded A-share bootstrap payloads. | Generic registry containers should not own market-specific definitions. | PR merge | done | None. | `FactorCatalog` is now pure registry; A-share owner binds definitions. |
| QRP-ISSUE-003 | P1 | factors/catalog | `price_volume_56` used a count-based artifact ID. | Artifact IDs should not encode the current factor count. | PR merge | done | None. | Active artifact renamed to `price_volume`; no hardcoded current factor count. |
| QRP-ISSUE-004 | P2 | factors | `literature_` price-volume naming noise. | Naming noise made active factor ownership harder to read. | not blocking | done | None. | Naming unified to price-volume factors. |
| QRP-ISSUE-005 | P1 | factors/processed | `price_volume` factor artifact needed value-only ownership. | Context/control columns belong to context and mask artifacts. | PR merge | done | None. | Factor artifact keeps primary keys plus factor values only. |
| QRP-ISSUE-006 | P1 | factors/build | Factor input columns were not catalog-driven. | Build inputs must follow factor definitions instead of parallel lists. | PR merge | done | None. | Input columns now derive from `FactorDefinition.input_columns`. |
| QRP-ISSUE-007 | P1 | workflows | Workflow execute mode did not fail closed. | Real execution must not run accidentally before implementation exists. | PR merge | done | None. | Official `workflow_node_registry(execution_mode=...)`; dry-run supported; execute fails closed. |
| QRP-ISSUE-008 | P1 | catalog | Missing `catalog/defaults.py` platform composition root. | API/services needed one explicit default catalog entrypoint. | PR merge | done | None. | Added explicit defaults module; services route through defaults. |
| QRP-ISSUE-009 | P2 | catalog | Lazy export in `catalog/__init__.py`. | Root package import hid platform composition side effects. | not blocking | done | None. | Removed; defaults must be imported explicitly from `catalog.defaults`. |
| QRP-ISSUE-010 | P1 | catalog | Generic catalog bootstrap payloads. | Concrete default registrations belonged to owner/default composition. | PR merge | done | None. | Concrete payloads moved to owner/default composition. |
| QRP-ISSUE-011 | P1 | catalog/a_share | A-share data product, market, and universe definitions lived in defaults. | `catalog/defaults.py` should compose owner definitions, not define concrete market objects. | PR merge | done | None. | Delegated to A-share owner definition factories. |
| QRP-ISSUE-012 | P1 | compute | `compute/protocols.py` mixed spec/config/backend/validation. | Compute-local runtime objects needed clearer ownership. | PR merge | done | None. | Split into `compute/panels.py`, `compute/config.py`, and `compute/backends.py`; uses `ComputePanelSpec` and `validate_compute_panel`. |
| QRP-ISSUE-013 | P1 | agents/catalog | `agents/factor_factory.py` directly imported A-share catalog. | Agent validation should use platform defaults rather than market implementation imports. | PR merge | done | None. | Routed through `default_field_catalog()`. |
| QRP-ISSUE-014 | P1 | catalog/reports | `catalog/provider_bindings.py` knew private interim paths. | Provider binding registry should not own private path helpers. | PR merge | done | None. | Moved private interim-root helper into private aggregate verification as private helper. |
| QRP-ISSUE-015 | P2 | storage | `DataSpecCatalog` name was misleading. | Storage accepts a provider protocol, not a concrete catalog registry. | not blocking | done | None. | Renamed to `DataSpecProvider`. |
| QRP-ISSUE-016 | P2 | workflows/interfaces | Workflow/service default run id used fixture wording. | Current dry-run mode is official, not a test fixture. | not blocking | done | None. | Public wording changed to dry-run. |
| QRP-ISSUE-017 | P1 | tests/architecture | Architecture ownership boundaries lacked durable tests. | Boundary regressions should fail explicitly. | PR merge | done | None. | Added import boundary, registration allowlist, catalog defaults, package scan, and stale factor term tests. |
| QRP-ISSUE-018 | P2 | reports/private verification | `reports/private_aggregate_verification.py` boundary is mixed. | Private verification collector, market-specific rules, storage/path helpers, and report summary are mixed. | C4 evidence or PR merge decision | open | Decision audit. | Decide whether to split helpers or keep bounded until evidence regeneration. |
| QRP-ISSUE-019 | P2 | sources/markets | `sources/csmar/interim.py` directly depends on A-share adapter. | Source importer directly depends on market implementation. | post-merge or market expansion | open | Adapter injection design audit. | Keep out of Batch 4A scope unless market expansion starts. |
| QRP-ISSUE-020 | P2 | sources/csmar/interim | Direct `index_market` duplicate primary-key root cause. | Do not blindly drop duplicates; classify exact duplicates versus conflicts. | direct index_market materialization | open | Duplicate group audit. | Keep aggregate-only reporting. |
| QRP-ISSUE-021 | P2 | labels/processed | Label artifact value-only decision. | Factor artifact is value-only; label artifact may need the same ownership rule. | label artifact contract cleanup | open | Decision audit. | Avoid changing label schema until ownership rule is decided. |
| QRP-ISSUE-022 | P2 | catalog/contracts | DataProductDefinition / SourceDefinition / ProviderBinding / MarketDefinition owner class audit. | Payloads moved, but type ownership remains a later design question. | later architecture cleanup | open | Decide whether these definition classes stay catalog-local or move to contracts. | Do not move classes without a dedicated design decision. |
| QRP-ISSUE-023 | P2 | catalog/validation | `registry_validation.py` dynamic source fallback. | Defaults now exist, but dynamic loading remains. | stronger registry ownership | open | Decide injection-only versus fallback policy. | Preserve explicit injection tests. |
| QRP-ISSUE-024 | P2 | catalog/platform | `catalog/defaults.py` multi-market/source profiles. | Current defaults are A-share plus CSMAR only. | crypto/US-equity expansion | open | Platform profile design. | Do not add empty future packages. |
| QRP-ISSUE-025 | P2 | private processed data | Old ignored `price_volume_56` dataset cleanup. | Avoid stale active/private dataset confusion. | after verified private rebuild replacement | open | Backup old ignored dataset and remove after new `price_volume` is verified. | Requires explicit private cleanup approval. |
| QRP-ISSUE-026 | P2 | evidence | Stale evidence / C4 cleanup. | Active evidence is marked pending regeneration. | PR merge | open | C4 evidence regeneration and OOS period mismatch fix. | Do not regenerate evidence in docs-only work. |
| QRP-ISSUE-027 | P2 | pipeline reliability | Pipeline retry/resume/overwrite/global API envelope backlog. | Known backlog from earlier pipeline reliability work. | later reliability batch | open | Separate reliability audit. | Keep separate from architecture ownership batches. |
| QRP-ISSUE-028 | P3 | a_share/catalog | `a_share/catalog.py` split by domain. | Ownership is correct but file is large. | later cleanup | open | Split into catalog submodules if size continues to grow. | Avoid premature split while behavior is still compact enough. |
| QRP-ISSUE-029 | P3 | sources/csmar | `sources/csmar/columns.py` split by dataset family. | File is large. | later cleanup | open | Split mapping/schema files by dataset family. | Keep source schema behavior unchanged. |
| QRP-ISSUE-030 | P3 | compute | `compute/backends.py` wording cleanup. | Naming precision. | later cleanup | open | Replace "evaluation compute backend" wording with generic "compute backend". | No behavior change. |
| QRP-ISSUE-031 | P3 | reports/private verification | `_private_interim_root_for_data_product()` readability. | Current dynamic field access is awkward. | report verification cleanup | open | Replace awkward `getattr` expression with direct field access if possible. | Must keep public-surface marker gates passing. |
| QRP-ISSUE-032 | P3 | tests/architecture | Provider binding architecture test precision. | Current guard is sufficient but can be sharper. | later architecture hardening | open | Make AST checks more precise for private path helpers / storage path imports. | Avoid string-only false positives where practical. |
| QRP-ISSUE-033 | P3 | a_share/ids | A-share interim ID constant naming. | Current name is slightly misleading. | later naming cleanup | open | Consider renaming `A_SHARE_DAILY_BAR_INTERIM_DATA_PRODUCT_ID` to `A_SHARE_DAILY_BAR_INTERIM_SPEC_ID`. | Requires import updates only if approved. |
| QRP-ISSUE-034 | P3 | private/interim cleanup | Legacy `data/interim/csmar/_manifests/`. | Legacy root sidecar; no active dependency found. | explicit future cleanup approval | open | Preserve until approved. | Do not delete during architecture or docs work. |
| QRP-ISSUE-035 | P3 | storage/manifest | `spec.json` and `_manifest/manifest.json` semantics. | Both currently come from the same payload source. | manifest enrichment batch | open | Make `spec.json` static contract and `manifest.json` artifact instance metadata. | Requires storage manifest design. |
| QRP-ISSUE-036 | P3 | tests/architecture | Architecture allowlist continuous cleanup. | Avoid permanent exceptions. | ongoing maintenance | open | Remove allowlist entries as boundaries are fixed. | Keep each exception tied to an owner decision. |
| QRP-ISSUE-037 | P3 | a_share/catalog | A-share definition factory naming when objects multiply. | Avoid premature abstraction now. | more data products/universes | open | Switch from single `*_definition()` to plural `*_definitions()` only when needed. | Current single-definition factories are intentional. |
