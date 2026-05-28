# Frontend Feature Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the flat `web/src` React workbench into feature and shared folders without changing API contracts, visual hierarchy, or frontend business behavior.

**Architecture:** Keep the workbench shell as the composition boundary, move API/types/shared UI first, then move one feature at a time: evidence, pipeline, factor studio, factor factory, uploads, and runs. Use relative imports inside each feature and avoid broad barrel imports that hide dependencies or increase bundle risk.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, Recharts, Framer Motion, ESLint, PowerShell on Windows.

---

## Scope

Implement N4 only:

- Move files from flat `web/src` into feature folders.
- Preserve all current UI copy, CSS classes, API request paths, `manifest.display` consumption, and test behavior.
- Keep frontend as a consumer of backend artifacts; do not add frontend-side metric, IC, PnL, position, turnover, eligibility, or tradability calculation.
- Keep `web/src/App.css` and `web/src/index.css` in place for this split unless a tiny import path adjustment is necessary. Visual redesign is out of scope.

Do not:

- Change API contracts or backend code.
- Implement N2 tradability diagnostics UI.
- Implement N3 artifact packaging.
- Delete `web/src/assets/*` in this branch.
- Introduce path aliases unless the project first adds and tests Vite/TS config support.
- Add new dependencies.

## Target Structure

```text
web/src/
  shell/
    App.tsx
    App.test.tsx
  features/
    evidence/
      ArtifactDownloads.tsx
      CombinationSelector.tsx
      CsvBrowser.tsx
      MetricSummary.tsx
      ResultsView.tsx
      RunCharts.tsx
      RunComparison.tsx
      RunHeader.tsx
      chartRange.ts
      charts.tsx
      insights.ts
      reportView.tsx
      runEvidenceState.ts
      useArtifactDownloads.ts
      useRunEvidence.ts
      useTablePreview.ts
      __tests__/
        charts.test.ts
        downloads.test.ts
        runEvidenceState.test.ts
    pipeline/
      DataCheckpoints.tsx
      PipelineBuilder.tsx
    factor-studio/
      FactorStudio.tsx
    factor-factory/
      FactorFactoryRuns.tsx
    uploads/
      CreateRunGuide.tsx
      CreateRunPanel.tsx
      CsvPreviewPanel.tsx
      CsvUploadField.tsx
      runForm.ts
      useCreateRun.ts
    runs/
      Sidebar.tsx
      useDeleteRun.ts
      useRuns.ts
  shared/
    api/
      api.ts
      api.test.ts
    downloads.ts
    factorDraft.ts
    format.ts
    testFixtures.ts
    ui/
      components.tsx
  main.tsx
  App.css
  index.css
  assets/
```

Rationale:

- `features/evidence/` owns evidence state, charts, CSV preview, comparison, report view, and artifact download UI.
- `features/pipeline/` owns Pipeline Builder and Data Checkpoints.
- `features/factor-studio/` owns template editing.
- `features/factor-factory/` owns Factor Factory run browsing.
- `features/uploads/` owns the development CSV upload flow.
- `features/runs/` owns run list/sidebar/delete hooks.
- `shell/` owns top-level workbench composition and view state. It is named `shell` instead of `app` to avoid confusion with future repo-level `apps/web` or framework-specific `src/app` routing.
- `shared/api/` owns backend request types and fetch helpers.
- `shared/factorDraft.ts` owns the factor draft state shared by Factor Studio and Pipeline Builder.
- `shared/ui/` owns small reusable fields, metrics, chart panels, and tables.

## Preflight

**Files:**
- Read: `AGENTS.md`
- Read: `docs/architecture/next-stage-worktree-plan.md`
- Read: `docs/architecture/next-stage-roadmap.md`
- Read: `web/package.json`
- Read: `web/src/App.tsx`
- Read: `web/src/App.test.tsx`

- [ ] **Step 1: Confirm the branch and working tree**

Run:

```powershell
git status --short
git branch --show-current
```

Expected: the N4 implementation branch is `codex/frontend-feature-split`, and status is clean before edits.

- [ ] **Step 2: Confirm the frontend baseline**

Run:

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
```

Expected: all current frontend checks pass before moving files. If they fail, stop and report the failure before refactoring.

## Task 1: Move Shared API And Utilities

**Files:**
- Move: `web/src/api.ts` -> `web/src/shared/api/api.ts`
- Move: `web/src/api.test.ts` -> `web/src/shared/api/api.test.ts`
- Move: `web/src/format.ts` -> `web/src/shared/format.ts`
- Move: `web/src/components.tsx` -> `web/src/shared/ui/components.tsx`
- Move: `web/src/testFixtures.ts` -> `web/src/shared/testFixtures.ts`
- Move: `web/src/downloads.ts` -> `web/src/shared/downloads.ts`
- Modify imports in all moved files and consumers.

- [ ] **Step 1: Move files with Git**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/shared/api","web/src/shared/ui" | Out-Null
git mv "web/src/api.ts" "web/src/shared/api/api.ts"
git mv "web/src/api.test.ts" "web/src/shared/api/api.test.ts"
git mv "web/src/format.ts" "web/src/shared/format.ts"
git mv "web/src/components.tsx" "web/src/shared/ui/components.tsx"
git mv "web/src/testFixtures.ts" "web/src/shared/testFixtures.ts"
git mv "web/src/downloads.ts" "web/src/shared/downloads.ts"
```

- [ ] **Step 2: Update shared imports**

Apply these import rules:

```text
web/src/shared/api/api.test.ts:
  from './api' stays './api'

web/src/shared/ui/components.tsx:
  from './api' -> '../api/api'
  from './format' -> '../format'

web/src/shared/downloads.ts:
  from './api' -> './api/api'

web/src/shared/testFixtures.ts:
  from './api' -> './api/api'
```

- [ ] **Step 3: Update consumers**

Use search:

```powershell
rg -n "from './api'|from './format'|from './components'|from './testFixtures'|from './downloads'" web/src
```

Replace imports by location:

```text
Root files still under web/src:
  './api' -> './shared/api/api'
  './format' -> './shared/format'
  './components' -> './shared/ui/components'
  './testFixtures' -> './shared/testFixtures'
  './downloads' -> './shared/downloads'
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
Push-Location web
npm test -- --run --maxWorkers=1 src/shared/api/api.test.ts src/shell/App.test.tsx
Pop-Location
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add web/src
git commit -m "refactor(web): move shared api utilities"
```

## Task 2: Move Evidence Feature

**Files:**
- Move evidence components and hooks listed in the target structure.
- Move tests into `web/src/features/evidence/__tests__/`.
- Modify imports in moved files and `web/src/shell/App.tsx` after Task 3.

- [ ] **Step 1: Move evidence files**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/features/evidence/__tests__" | Out-Null
git mv "web/src/ArtifactDownloads.tsx" "web/src/features/evidence/ArtifactDownloads.tsx"
git mv "web/src/CombinationSelector.tsx" "web/src/features/evidence/CombinationSelector.tsx"
git mv "web/src/CsvBrowser.tsx" "web/src/features/evidence/CsvBrowser.tsx"
git mv "web/src/MetricSummary.tsx" "web/src/features/evidence/MetricSummary.tsx"
git mv "web/src/ResultsView.tsx" "web/src/features/evidence/ResultsView.tsx"
git mv "web/src/RunCharts.tsx" "web/src/features/evidence/RunCharts.tsx"
git mv "web/src/RunComparison.tsx" "web/src/features/evidence/RunComparison.tsx"
git mv "web/src/RunHeader.tsx" "web/src/features/evidence/RunHeader.tsx"
git mv "web/src/chartRange.ts" "web/src/features/evidence/chartRange.ts"
git mv "web/src/charts.tsx" "web/src/features/evidence/charts.tsx"
git mv "web/src/insights.ts" "web/src/features/evidence/insights.ts"
git mv "web/src/reportView.tsx" "web/src/features/evidence/reportView.tsx"
git mv "web/src/runEvidenceState.ts" "web/src/features/evidence/runEvidenceState.ts"
git mv "web/src/useArtifactDownloads.ts" "web/src/features/evidence/useArtifactDownloads.ts"
git mv "web/src/useRunEvidence.ts" "web/src/features/evidence/useRunEvidence.ts"
git mv "web/src/useTablePreview.ts" "web/src/features/evidence/useTablePreview.ts"
git mv "web/src/charts.test.ts" "web/src/features/evidence/__tests__/charts.test.ts"
git mv "web/src/downloads.test.ts" "web/src/features/evidence/__tests__/downloads.test.ts"
git mv "web/src/runEvidenceState.test.ts" "web/src/features/evidence/__tests__/runEvidenceState.test.ts"
```

- [ ] **Step 2: Update evidence imports**

Use these rules:

```text
Inside web/src/features/evidence:
  './api' -> '../../shared/api/api'
  './components' -> '../../shared/ui/components'
  './downloads' -> '../../shared/downloads'
  './format' -> '../../shared/format'
  './testFixtures' -> '../../shared/testFixtures'

Inside web/src/features/evidence/__tests__:
  '../chartRange' for chartRange.ts
  '../charts' for charts.tsx
  '../runEvidenceState' for runEvidenceState.ts
  '../../../shared/api/api' for api types
  '../../../shared/testFixtures' for fixtures
```

Keep local evidence imports as relative same-folder imports, for example:

```text
./ArtifactDownloads
./CombinationSelector
./CsvBrowser
./MetricSummary
./RunCharts
./RunHeader
```

- [ ] **Step 3: Run evidence tests**

Run:

```powershell
Push-Location web
npm test -- --run --maxWorkers=1 src/features/evidence/__tests__/charts.test.ts src/features/evidence/__tests__/downloads.test.ts src/features/evidence/__tests__/runEvidenceState.test.ts
Pop-Location
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```powershell
git add web/src
git commit -m "refactor(web): split evidence feature"
```

## Task 3: Move Workbench Shell Composition

**Files:**
- Move: `web/src/App.tsx` -> `web/src/shell/App.tsx`
- Move: `web/src/App.test.tsx` -> `web/src/shell/App.test.tsx`
- Modify: `web/src/main.tsx`

- [ ] **Step 1: Move shell files**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/shell" | Out-Null
git mv "web/src/App.tsx" "web/src/shell/App.tsx"
git mv "web/src/App.test.tsx" "web/src/shell/App.test.tsx"
```

- [ ] **Step 2: Update `main.tsx`**

Change:

```ts
import App from './App.tsx'
```

to:

```ts
import App from './shell/App.tsx'
```

- [ ] **Step 3: Update `shell/App.tsx` imports**

Apply these target imports:

```ts
import '../App.css'
import { DataCheckpoints } from '../features/pipeline/DataCheckpoints'
import { FactorFactoryRuns } from '../features/factor-factory/FactorFactoryRuns'
import { FactorStudio } from '../features/factor-studio/FactorStudio'
import { PipelineBuilder } from '../features/pipeline/PipelineBuilder'
import { ResultsView } from '../features/evidence/ResultsView'
import { Sidebar, type ViewMode } from '../features/runs/Sidebar'
import {
  DEFAULT_FACTOR_DRAFT,
  type FactorDraft,
} from '../shared/factorDraft'
import { useArtifactDownloads } from '../features/evidence/useArtifactDownloads'
import { useCreateRun } from '../features/uploads/useCreateRun'
import { useDeleteRun } from '../features/runs/useDeleteRun'
import { useRunEvidence } from '../features/evidence/useRunEvidence'
import { useRuns } from '../features/runs/useRuns'
```

Task 3 will temporarily fail until Tasks 4-7 move the remaining feature files. Do not run full tests between Task 3 and Task 7 unless all import targets exist.

## Task 4: Move Pipeline And Factor Studio Features

**Files:**
- Move: `web/src/DataCheckpoints.tsx` -> `web/src/features/pipeline/DataCheckpoints.tsx`
- Move: `web/src/PipelineBuilder.tsx` -> `web/src/features/pipeline/PipelineBuilder.tsx`
- Move: `web/src/FactorStudio.tsx` -> `web/src/features/factor-studio/FactorStudio.tsx`
- Move: `web/src/factorDraft.ts` -> `web/src/shared/factorDraft.ts`

- [ ] **Step 1: Move pipeline and factor studio files**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/features/pipeline","web/src/features/factor-studio" | Out-Null
git mv "web/src/DataCheckpoints.tsx" "web/src/features/pipeline/DataCheckpoints.tsx"
git mv "web/src/PipelineBuilder.tsx" "web/src/features/pipeline/PipelineBuilder.tsx"
git mv "web/src/FactorStudio.tsx" "web/src/features/factor-studio/FactorStudio.tsx"
git mv "web/src/factorDraft.ts" "web/src/shared/factorDraft.ts"
```

- [ ] **Step 2: Update imports**

Apply:

```text
web/src/features/pipeline/PipelineBuilder.tsx:
  './api' -> '../../shared/api/api'
  './factorDraft' -> '../../shared/factorDraft'

web/src/features/factor-studio/FactorStudio.tsx:
  './api' -> '../../shared/api/api'
  './factorDraft' -> '../../shared/factorDraft'
```

- [ ] **Step 3: Check imports**

Run:

```powershell
rg -n "from './api'|from './factorDraft'|from '../factor-studio/factorDraft'" web/src/features/pipeline web/src/features/factor-studio web/src/shell
```

Expected: no stale root-relative imports.

## Task 5: Move Factor Factory Feature

**Files:**
- Move: `web/src/FactorFactoryRuns.tsx` -> `web/src/features/factor-factory/FactorFactoryRuns.tsx`

- [ ] **Step 1: Move file**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/features/factor-factory" | Out-Null
git mv "web/src/FactorFactoryRuns.tsx" "web/src/features/factor-factory/FactorFactoryRuns.tsx"
```

- [ ] **Step 2: Update imports**

In `web/src/features/factor-factory/FactorFactoryRuns.tsx`, apply:

```text
./api -> ../../shared/api/api
```

- [ ] **Step 3: Keep Factor Factory behavior covered by App test**

Do not extract new logic during this move. Existing `shell/App.test.tsx` cases should still cover:

```text
renders Factor Factory runs without leaking private filesystem paths
keeps Factor Factory detail and downloads visible when table preview fails
chooses a preview table from display tables and renders Factor Factory preset downloads
enables Factor Factory pagination from total rows when has_more is omitted
clears stale Factor Factory detail when switching to a failing run
```

## Task 6: Move Uploads And Run Hooks

**Files:**
- Move uploads files into `web/src/features/uploads/`
- Move run list/delete files into `web/src/features/runs/`
- Move create-run hook into `web/src/features/uploads/`

- [ ] **Step 1: Move upload files**

Run:

```powershell
New-Item -ItemType Directory -Force "web/src/features/uploads","web/src/features/runs" | Out-Null
git mv "web/src/CreateRunGuide.tsx" "web/src/features/uploads/CreateRunGuide.tsx"
git mv "web/src/CreateRunPanel.tsx" "web/src/features/uploads/CreateRunPanel.tsx"
git mv "web/src/CsvPreviewPanel.tsx" "web/src/features/uploads/CsvPreviewPanel.tsx"
git mv "web/src/CsvUploadField.tsx" "web/src/features/uploads/CsvUploadField.tsx"
git mv "web/src/runForm.ts" "web/src/features/uploads/runForm.ts"
git mv "web/src/Sidebar.tsx" "web/src/features/runs/Sidebar.tsx"
git mv "web/src/useDeleteRun.ts" "web/src/features/runs/useDeleteRun.ts"
git mv "web/src/useRuns.ts" "web/src/features/runs/useRuns.ts"
git mv "web/src/useCreateRun.ts" "web/src/features/uploads/useCreateRun.ts"
```

- [ ] **Step 2: Update upload imports**

Apply:

```text
web/src/features/uploads/CreateRunGuide.tsx:
  './api' -> '../../shared/api/api'

web/src/features/uploads/CreateRunPanel.tsx:
  './components' -> '../../shared/ui/components'
  './CreateRunGuide' stays './CreateRunGuide'
  './CsvPreviewPanel' stays './CsvPreviewPanel'
  './CsvUploadField' stays './CsvUploadField'
  './runForm' stays './runForm'

web/src/features/uploads/CsvPreviewPanel.tsx:
  './runForm' stays './runForm'

web/src/features/uploads/CsvUploadField.tsx:
  './components' -> '../../shared/ui/components'

web/src/features/uploads/runForm.ts:
  './api' -> '../../shared/api/api'
```

- [ ] **Step 3: Update run hook imports**

Apply:

```text
web/src/features/runs/Sidebar.tsx:
  './api' -> '../../shared/api/api'

web/src/features/runs/useDeleteRun.ts:
  './api' -> '../../shared/api/api'
  './format' -> '../../shared/format'

web/src/features/runs/useRuns.ts:
  './api' -> '../../shared/api/api'
  './format' -> '../../shared/format'

web/src/features/uploads/useCreateRun.ts:
  './api' -> '../../shared/api/api'
  './format' -> '../../shared/format'
  './runForm' stays './runForm'
```

- [ ] **Step 4: Update `shell/App.tsx` upload import**

Add:

```ts
import { CreateRunPanel } from '../features/uploads/CreateRunPanel'
```

Remove any stale `CreateRunPanel` root import.

## Task 7: Repair Tests And Root Imports

**Files:**
- Modify: `web/src/shell/App.test.tsx`
- Modify moved tests under `web/src/features/evidence/__tests__/`
- Modify any stale imports found by search.

- [ ] **Step 1: Update `shell/App.test.tsx` imports**

Apply:

```ts
import App from './App'
import { FactorFactoryRuns } from '../features/factor-factory/FactorFactoryRuns'
import type { RunSummary } from '../shared/api/api'
import {
  displayPayload,
  evidencePayload,
  jsonResponse,
  tablePayload,
  tableResponse,
} from '../shared/testFixtures'
import { useRunEvidence } from '../features/evidence/useRunEvidence'
```

- [ ] **Step 2: Search stale root imports**

Run:

```powershell
rg -n "from './(api|format|components|downloads|testFixtures|runEvidenceState|FactorFactoryRuns|useRunEvidence|factorDraft|runForm)'|from '../api'|from '../format'|from '../components'" web/src
```

Expected: no imports pointing to old flat files. Imports inside same folder can remain local, such as `./RunCharts` from `ResultsView.tsx`.

- [ ] **Step 3: Run TypeScript build to catch import errors**

Run:

```powershell
Push-Location web
npm run build
Pop-Location
```

Expected: PASS. If it fails, fix import paths only; do not change behavior.

## Task 8: Verify Behavior

**Files:**
- No edits unless verification exposes an import or behavior regression.

- [ ] **Step 1: Run frontend tests**

Run:

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
Pop-Location
```

Expected: PASS.

- [ ] **Step 2: Run lint**

Run:

```powershell
Push-Location web
npm run lint
Pop-Location
```

Expected: PASS.

- [ ] **Step 3: Run build**

Run:

```powershell
Push-Location web
npm run build
Pop-Location
```

Expected: PASS.

- [ ] **Step 4: Run diff check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

## Task 9: Documentation Update

**Files:**
- Modify: `docs/architecture/next-stage-roadmap.md`
- Modify: `docs/code-map/platform-flow.md`
- Modify: `docs/code-map/README.md` if it names old flat frontend paths.
- Modify: `PROJECT_STATUS.md` only after implementation is complete and reviewed.

- [ ] **Step 1: Update roadmap after checks pass**

Under N4 in `docs/architecture/next-stage-roadmap.md`, add:

```markdown
Implementation note:

- `web/src` is split into `shell/`, `features/`, and `shared/` folders.
- Evidence, pipeline, factor studio, Factor Factory, uploads, runs, API, and shared UI keep the same contracts and behavior.
```

- [ ] **Step 2: Update code map references**

Search:

```powershell
rg -n "web/src/(App|ResultsView|PipelineBuilder|FactorFactoryRuns|CreateRunPanel|api|runEvidenceState|components)" docs README.md ARCHITECTURE.md PROJECT_STATUS.md
```

Update only references that describe current code paths. Do not rewrite historical Phase 1 acceptance evidence unless it claims to be current.

- [ ] **Step 3: Avoid premature status claims**

Only update `PROJECT_STATUS.md` after N4 implementation and verification are complete. If this branch only writes the plan, leave project status unchanged.

## Commit Plan

Use small commits:

```text
refactor(web): move shared api utilities
refactor(web): split evidence feature
refactor(web): move app composition
refactor(web): split workflow features
docs: update frontend feature map
```

Avoid combining generated lockfile changes, visual restyling, and feature split in the same commit. `package-lock.json` should not change for this task.

## Self-Review Checklist

- [ ] No API request path changed.
- [ ] No backend code changed.
- [ ] No frontend-only metric or artifact shortcut was added.
- [ ] `manifest.display` remains the source of table/download/chart behavior.
- [ ] No private filesystem path is introduced into UI or tests.
- [ ] `web/src/assets/hero.png`, `react.svg`, and `vite.svg` are not deleted in this branch.
- [ ] All tests still pass after moving files.
