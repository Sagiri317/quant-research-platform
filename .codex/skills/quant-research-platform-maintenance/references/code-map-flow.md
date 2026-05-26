# Code Map Flow Maintenance

Load this reference when changing or reviewing `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`.

## Scope

- `platform-flow.md` is the concise Chinese source-reading map.
- `platform-flow-interactive.html` is the offline interactive SVG map.
- Keep both files aligned on nodes, edges, API surfaces, entry points, and terminology.
- Do not turn this reference into a generated-run log. Keep commit hashes, temporary screenshots, private paths, and one-time review notes out of it.

## Graph Semantics

- Edges represent data, contract, or payload availability flow. They do not necessarily represent HTTP request direction.
- Keep actual request direction in node detail text when it differs from the visual flow.
- Prefer left-to-right visual flow. Local vertical movement is allowed when it reduces overlap or edge clutter.
- The graph must remain acyclic unless the code actually has a runtime cycle and the doc explicitly explains it.
- Do not collapse frontend and API nodes unless there is a single path, no reuse branch, and no independent responsibility boundary. Composite nodes must explicitly say `Frontend + API`.

## Required Relationships

- A-share main chain: CSMAR raw CSV -> build_interim -> interim CSV -> build_panel -> research panel -> build_factors -> factor table -> A-share adapter -> EvaluationSpec -> run_evaluation -> schema 0.6 artifacts -> `manifest.display` -> API/download/frontend.
- `start_from` runner entries are `existing_factors`, `rebuild_factors`, `rebuild_panel`, and `rebuild_interim`; every executable run continues through evaluation.
- `manifest.display` is the only display contract for evidence payloads, table previews, direct downloads, ZIP downloads, frontend tables, frontend charts, and download presets.
- API-visible evaluation run identity is `(source_alias, run_id)` across run list, evidence, preview, direct download, ZIP cache path, comparison payload, and frontend routing.
- `Artifact ZIP package` / `artifact_package.py` stays visible in the download path and enforces `manifest.display.downloads`; download services orchestrate responses.
- `FastAPI app` creates `ApiContext` and mounts routers. It is the app/container, not a downstream consumer of `run_repository`.
- `Factor Studio` loads templates through `GET /api/factor-templates` / factor-template API. Do not draw a direct frontend-to-registry shortcut.
- `Factor Factory API` and `FactorFactoryRuns UI` stay separate unless their responsibilities are intentionally redesigned.
- `qrp compare` is an offline comparison writer. It is not the API comparison endpoint.
- `qrp evaluate-parquet` is a generic Parquet adapter check. It is not an A-share adapter, A-share private pipeline, or Pipeline Builder acceptance substitute.
- `CSV Upload UI/API` is legacy/dev compatibility and not the A-share private main chain.

## Interactive HTML Rules

- Keep the file self-contained and offline; do not add npm, CDN, or network dependencies.
- Keep source UTF-8 and Chinese-first, preserving necessary English module, class, function, CLI, and API names.
- Use SVG nodes and Bezier edges.
- Do not reintroduce `routeY`, `lane`, or lane-style edge routing. All edges should be editable with the same handle model.
- Every selected edge should expose `source`, `target`, `c1`, and `c2` handles:
  - `source`: drag the source endpoint along the source node border.
  - `target`: drag the target endpoint along the target node border.
  - `c1` and `c2`: drag Bezier control points.
- Reset must restore node positions, endpoint overrides, and curve controls.
- Keep search, group filter, hidden-node controls, collapsible/resizable sidebar, pan/zoom, node dragging, edge dragging, and detail panel behavior working after changes.

## Entry Detail Coverage

Keep detailed parameter views for these entry surfaces when they exist in the code:

- `qrp evaluate-parquet`
- `qrp compare`
- `qrp a-share build-interim`, `build-panel`, `build-factors`, and `evaluate` helper entries
- `python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`
- pipeline snapshot and pipeline run APIs
- evidence, preview, download, and comparison APIs
- factor-template API
- Factor Factory API
- local/legacy CSV upload API, clearly marked as legacy/dev

Mark backend-only or local-debug parameters, such as local file inputs, as not being API/UI acceptance substitutes.

## Review Checklist

Before calling a code-map update complete:

- Compare the Markdown and HTML maps for missing nodes, missing edges, naming drift, and relationship direction errors.
- Check key code surfaces when the graph mentions them: `interfaces/api/app.py`, API routes/services, `artifacts/artifact_package.py`, `artifacts/display_manifest.py`, A-share pipeline runner, A-share evaluation adapter, Factor Factory routes, and React feature entry points.
- Verify there are no `routeY`, `lane`, or `泳道` remnants in the interactive HTML.
- Verify HTML nodes and edges parse, every edge endpoint exists, the graph has no directed cycle, and static node positions do not overlap.
- Verify the default edge paths are Bezier curves.
- Verify edge edit handles exist for `source`, `target`, `c1`, and `c2`.
- Verify Markdown includes the current critical terms: `manifest.display`, `source_alias`, `run_id`, `Artifact ZIP package`, `GET /api/factor-templates`, `Factor Factory`, `evaluate-parquet`, and CSV upload compatibility.
- Run `scripts/verify_code_map_flow.ps1` and `git diff --check`.
- Use browser acceptance when HTML interaction changed. At minimum, open the file in Chrome or Playwright, check for JavaScript errors, select an edge, drag all four edge handles, and verify reset restores the path.
