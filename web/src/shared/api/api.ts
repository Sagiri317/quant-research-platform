const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, '')

export const FACTOR_CSV_TEMPLATE_URL = `${API_BASE_URL}/api/templates/factor-csv`

export type RunSummary = {
  source_alias: string
  run_id: string
  public_run_id: string
  project: string
  kind: string
  domain: string
  data_stage: string
  uses_external_data: boolean
  n_rows_evaluated: number
  label_columns?: string[]
  return_source_column?: string | null
  n_combinations?: number
  breakdown_columns?: string[]
  weight_method: string
  portfolio_mode?: string
  mean_final_return?: number
  mean_turnover: number
  total_cost: number
  sharpe_like?: number
  sortino_like?: number
  max_drawdown?: number
  win_rate?: number
  mean_ic?: number
  mean_rankic?: number
  icir?: number
  rankicir?: number
}

export type TablePayload = {
  columns: string[]
  rows: Record<string, unknown>[]
  total_rows?: number
  total_rows_exact?: boolean
  offset?: number
  limit?: number | null
  has_more?: boolean
}

export type RunDetail = {
  run_id: string
  project: string
  kind: string
  domain: string
  data_stage: string
  uses_external_data: boolean
  n_rows_evaluated: number
  label_columns?: string[]
  return_source_column?: string | null
  n_combinations?: number
  breakdown_columns?: string[]
  weight_config?: Record<string, unknown>
  boundary?: string
}

export type RunEvidence = {
  summary: RunDetail
  report?: {
    markdown: string
  } | null
  comparison: TablePayload
  combination_summaries?: TablePayload
  display: RunDisplay
  tables: Record<string, TablePayload>
}

export type CreateRunPayload = {
  run_id: string
  project: string
  kind: string
  domain: string
  data_stage: string
  uses_external_data: boolean
  timestamp_column: string
  asset_column: string
  feature_columns: string[]
  label_columns: string[]
  return_source_column?: string
  segment_columns: string[]
  time_breakdowns: string[]
  kernel: string
  n_groups: number
  weight_method: string
  portfolio_mode: string
  direction: string
  top_quantile: number
  bottom_quantile: number
  cost_bps: number
  risk_column?: string
  portfolio_feature?: string
  tensor_max_memory_mb?: number
  boundary?: string
  csv_text: string
}

export type MarketCapability = {
  id: 'a_share' | 'crypto' | string
  label: string
  status: 'enabled' | 'planned' | 'disabled' | string
  description?: string
}

export type FactorTemplate = {
  id: string
  name: string
  version: string
  market: string
  expression: string
  description?: string
  dependencies?: string[]
  inputs?: string[]
  outputs?: string[]
  capabilities?: string[]
  grid_parameter?: string
  grid_values?: string[]
}

export type PipelineSnapshotRequest = {
  cost_bps: number
  date_end: string
  date_start: string
  factor_expression: string
  factor_template_id?: string
  kernel: string
  label_columns: string[]
  market: string
  return_source_column: string
  portfolio_mode: string
  processed_factor_column?: string
  diagnostics?: string[]
  run_id: string
}

export type PipelineRunRequest = PipelineSnapshotRequest & {
  start_from:
    | 'existing_factors'
    | 'rebuild_factors'
    | 'rebuild_panel'
    | 'rebuild_interim'
}

export type PipelineRunSummary = {
  kind: 'snapshot' | 'run'
  run_id: string
  status: 'queued' | 'running' | 'ready' | 'failed' | string
  message?: string
  run_ref?: string
  warnings?: Record<string, unknown>[]
}

export type PipelineCheckpointSummary = {
  key: string
  label: string
  status: 'pending' | 'ready' | 'warning' | 'blocked' | string
  artifact?: string
  message?: string
  duration_seconds?: number
  row_count?: number
  failure_code?: string
  warnings?: Record<string, unknown>[]
  blocking_issues?: Record<string, unknown>[]
}

export type FactorFactoryRunSummary = {
  artifact_schema_version: string
  artifact_type: string
  candidate_count: number
  created_at?: string
  failed_count: number
  run_id: string
  success_count: number
}

export type FactorFactoryDisplayTable = {
  download_item: string
  group: string
  key: string
  label: string
  path: string
}

export type FactorFactoryDownloadItem = {
  item: string
  kind: string
  label: string
  table?: string
}

export type FactorFactoryDisplay = {
  schema_version: string
  downloads: {
    items: FactorFactoryDownloadItem[]
    presets: RunDisplayDownloadPreset[]
  }
  tables: FactorFactoryDisplayTable[]
}

export type FactorFactoryRunDetail = {
  display: FactorFactoryDisplay
  run: FactorFactoryRunSummary
}

export type RunTableSpec = {
  key: string
  label: string
  path: string
  previewUrl: string
  downloadItem: string
  evidenceKey: string
  format?: string
  previewRows: number
  previewMaxRows: number
  supportsOffset: boolean
  supportsCombinationFilter: boolean
  filterParameter: string | null
}

export type RunDisplayTablePreview = {
  kind: 'table' | string
  table_key: string
  default_limit: number
  max_limit: number
  supports_offset: boolean
  supports_combination_filter: boolean
  filter_parameter?: string | null
}

export type RunDisplayTable = {
  key: string
  label: string
  group: string
  path: string
  download_ref?: RunDisplayRef
  format?: 'csv' | 'parquet' | string
  role?: string
  join_key?: string | null
  preview?: RunDisplayTablePreview
  preview_url?: string
  preview_rows?: number
  supports_combination_filter?: boolean
}

export type RunDisplayChartSeries = {
  color: string
  format?: 'percent' | 'number'
  key: string
  label?: string
}

export type RunDisplayChart = {
  description?: string
  id: string
  include_zero?: boolean
  input_series?: string
  kind: 'line' | 'bar' | 'derived_cumulative_return' | string
  output_series?: string
  section?: 'primary' | 'diagnostic' | string
  series?: RunDisplayChartSeries[]
  source_table_key?: string
  title: string
  transform?: string
  x: string
}

export type RunDisplayRef = {
  kind: string
  key: string
}

export type RunDisplayDownloadItem = {
  item?: string
  key?: string
  kind: string
  label: string
  path?: string
  table?: string
}

export type RunDisplayDownloadPreset = {
  key: string
  label: string
  items: Array<string | RunDisplayRef>
}

export type RunDisplayDownloads = {
  items: RunDisplayDownloadItem[]
  presets: RunDisplayDownloadPreset[]
}

export type RunDisplay = {
  schema_version: string
  charts: RunDisplayChart[]
  downloads: RunDisplayDownloads
  tables: RunDisplayTable[]
}

export async function fetchRuns(): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>('/api/runs')
}

export async function fetchRunEvidence(run: RunSummary): Promise<RunEvidence> {
  return fetchJson<RunEvidence>(
    `${runApiPath(run)}/evidence`,
  )
}

export async function fetchTablePreview(
  path: string,
  limit: number,
  combinationId?: string,
  offset?: number,
): Promise<TablePayload> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (offset !== undefined) {
    params.set('offset', String(offset))
  }
  if (combinationId) {
    params.set('combination_id', combinationId)
  }
  const separator = path.includes('?') ? '&' : '?'
  return fetchJson<TablePayload>(`${path}${separator}${params.toString()}`)
}

export async function createRun(payload: CreateRunPayload): Promise<RunSummary> {
  return fetchJson<RunSummary>('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function listMarkets(): Promise<MarketCapability[]> {
  return fetchJson<MarketCapability[]>('/api/markets')
}

export async function listFactorTemplates(market: string): Promise<FactorTemplate[]> {
  const params = new URLSearchParams({ market })
  return fetchJson<FactorTemplate[]>(`/api/factor-templates?${params.toString()}`)
}

export async function createPipelineSnapshot(
  payload: PipelineSnapshotRequest,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>('/api/pipeline-snapshots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function createPipelineRun(
  payload: PipelineRunRequest,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>('/api/pipeline-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function getPipelineSnapshotStatus(
  runId: string,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>(
    `/api/pipeline-snapshots/${encodeURIComponent(runId)}`,
  )
}

export async function getPipelineRunStatus(
  runId: string,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>(
    `/api/pipeline-runs/${encodeURIComponent(runId)}`,
  )
}

async function getPipelineCheckpoints(
  runId: string,
  kind: 'run' | 'snapshot',
): Promise<PipelineCheckpointSummary[]> {
  const root = kind === 'run' ? 'pipeline-runs' : 'pipeline-snapshots'
  return fetchJson<PipelineCheckpointSummary[]>(
    `/api/${root}/${encodeURIComponent(runId)}/checkpoints`,
  )
}

export async function getPipelineSnapshotCheckpoints(
  runId: string,
): Promise<PipelineCheckpointSummary[]> {
  return getPipelineCheckpoints(runId, 'snapshot')
}

export async function getPipelineRunCheckpoints(
  runId: string,
): Promise<PipelineCheckpointSummary[]> {
  return getPipelineCheckpoints(runId, 'run')
}

export async function listFactorFactoryRuns(): Promise<FactorFactoryRunSummary[]> {
  return fetchJson<FactorFactoryRunSummary[]>('/api/factor-factory-runs')
}

export async function fetchFactorFactoryRun(
  runId: string,
): Promise<FactorFactoryRunDetail> {
  return fetchJson<FactorFactoryRunDetail>(
    `/api/factor-factory-runs/${encodeURIComponent(runId)}`,
  )
}

export async function fetchFactorFactoryTable(
  runId: string,
  tableKey: string,
  limit: number,
  offset = 0,
): Promise<TablePayload> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })
  return fetchJson<TablePayload>(
    `/api/factor-factory-runs/${encodeURIComponent(runId)}/tables/${encodeURIComponent(tableKey)}?${params.toString()}`,
  )
}

export async function deleteRun(run: RunSummary): Promise<void> {
  await fetchJson<{ deleted: string }>(runApiPath(run), {
    method: 'DELETE',
  })
}

export function selectedArtifactsDownloadUrl(
  run: RunSummary,
  items: string[],
): string {
  const params = new URLSearchParams()
  for (const item of items) {
    params.append('items', item)
  }
  return `${API_BASE_URL}${runApiPath(run)}/download?${params.toString()}`
}

export function runTableSpecsFromDisplay(
  run: RunSummary,
  display: RunDisplay | null,
): RunTableSpec[] {
  void run
  if (!display) {
    return []
  }
  return display.tables.map((table) => ({
    key: table.key,
    label: table.label,
    path: table.path,
    previewUrl: tablePreviewUrl(table),
    downloadItem: tableDownloadRef(table),
    evidenceKey: table.key,
    format: table.format,
    previewRows: tablePreviewDefaultLimit(table),
    previewMaxRows: table.preview?.max_limit ?? 1000,
    supportsOffset: table.preview?.supports_offset === true,
    supportsCombinationFilter: tablePreviewSupportsCombinationFilter(table),
    filterParameter: table.preview?.filter_parameter ?? null,
  }))
}

function tablePreviewUrl(table: RunDisplayTable): string {
  if (!table.preview_url) {
    throw new Error(`Display table ${table.key} missing preview_url`)
  }
  return table.preview_url
}

function tablePreviewDefaultLimit(table: RunDisplayTable): number {
  if (!table.preview) {
    throw new Error(`Display table ${table.key} missing preview metadata`)
  }
  if (table.preview.kind !== 'table' || table.preview.table_key !== table.key) {
    throw new Error(`Display table ${table.key} has stale preview metadata`)
  }
  return table.preview.default_limit
}

function tablePreviewSupportsCombinationFilter(table: RunDisplayTable): boolean {
  if (!table.preview) {
    throw new Error(`Display table ${table.key} missing preview metadata`)
  }
  return table.preview.supports_combination_filter
}

export function artifactDownloadUrl(run: RunSummary, downloadItem: string): string {
  const encodedItem = encodeURIComponent(downloadItem)
  return `${API_BASE_URL}${runApiPath(run)}/artifacts/${encodedItem}`
}

export function factorFactoryArtifactDownloadUrl(
  runId: string,
  downloadItem: string,
): string {
  const encodedRunId = encodeURIComponent(runId)
  const encodedItem = downloadItem.split('/').map(encodeURIComponent).join('/')
  return `${API_BASE_URL}/api/factor-factory-runs/${encodedRunId}/artifacts/${encodedItem}`
}

function runApiPath(run: RunSummary): string {
  return `/api/runs/${encodeURIComponent(run.source_alias)}/${encodeURIComponent(run.run_id)}`
}

function tableDownloadRef(table: RunDisplayTable): string {
  if (table.download_ref) {
    return displayRefKey(table.download_ref)
  }
  throw new Error(`Display table ${table.key} missing download_ref`)
}

export function displayRefKey(ref: string | RunDisplayRef): string {
  if (typeof ref === 'string') {
    return ref
  }
  return `${ref.kind}:${ref.key}`
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`
  const response = init === undefined ? await fetch(url) : await fetch(url, init)
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  return response.json() as Promise<T>
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown }
    return formatErrorDetail(payload.detail) ?? `Request failed with status ${response.status}`
  } catch {
    return `Request failed with status ${response.status}`
  }
}

function formatErrorDetail(detail: unknown): string | null {
  if (typeof detail === 'string') {
    return detail
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((issue) => formatFastApiIssue(issue))
      .filter((message): message is string => Boolean(message))
    return messages.length > 0 ? messages.join('; ') : JSON.stringify(detail)
  }
  if (!detail || typeof detail !== 'object') {
    return null
  }
  if ('blocking' in detail && Array.isArray(detail.blocking)) {
    const messages = detail.blocking
      .map((issue) => formatValidationIssue(issue))
      .filter((message): message is string => Boolean(message))
    if (messages.length > 0) {
      return messages.join('; ')
    }
  }
  return JSON.stringify(detail)
}

function formatFastApiIssue(issue: unknown): string | null {
  if (!issue || typeof issue !== 'object') {
    return null
  }
  const message = 'msg' in issue ? String(issue.msg) : ''
  const rawLoc = 'loc' in issue ? issue.loc : undefined
  const loc = Array.isArray(rawLoc) ? rawLoc.map(String).join('.') : ''
  if (loc && message) {
    return `${loc}: ${message}`
  }
  return message || null
}

function formatValidationIssue(issue: unknown): string | null {
  if (!issue || typeof issue !== 'object') {
    return null
  }
  const code = 'code' in issue ? String(issue.code) : ''
  const message = 'message' in issue ? String(issue.message) : ''
  if (code && message) {
    return `${code}: ${message}`
  }
  return message || code || null
}
