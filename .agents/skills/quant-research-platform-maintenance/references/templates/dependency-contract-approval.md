# Dependency Contract Approval

## Task Mode

`dependency-contract-approval`

Use when a task may require a dependency change, public contract change, schema change, `manifest.display` change, API/frontend shared contract change, or runtime boundary change.

## Purpose

Classify the approval needed before implementation. This template does not authorize the change.

## Context Budget

- Start from the user request and exact proposed change.
- Read only the directly affected template, contract, config, or code references needed to classify approval.
- Do not inspect private data, generated artifacts, full docs, or unrelated modules.

## Approval Required Before Implementation

- New dependency, dependency update/removal, package manager change, or environment change.
- Schema, `manifest.display`, public API, frontend shared contract, artifact type, required artifact file, directory structure, deployment/trading scope, or runtime behavior change.
- External API, vendor, broker, exchange connector, model registry, feature store, scheduler, or worker.

## Decision Packet

- Decision question:
- Options A/B/C:
- Recommended option:
- Why:
- Risk if wrong:
- Implementation must wait: yes/no
- Exact files likely to change:
- Exact commands requiring approval:

## Stop Conditions

- Stop before implementation until the user approves the exact dependency or contract boundary.
- Stop if approval would touch private/generated data, secrets, deployment exposure, paper/live trading, force push, amend, rebase, or broad governance rewrite.

## Output Format

- Approval type
- Decision packet
- Files inspected
- Not Changed
- Scope Check
