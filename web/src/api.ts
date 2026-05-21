const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, '')

export const FACTOR_CSV_TEMPLATE_URL = `${API_BASE_URL}/api/templates/factor-csv`
export const COMPARISON_CSV_URL = `${API_BASE_URL}/api/comparison.csv`

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
  pnl_return_column?: string | null
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
  offset?: number
  limit?: number | null
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
  pnl_return_column?: string | null
  n_combinations?: number
  breakdown_columns?: string[]
  weight_config?: Record<string, unknown>
  boundary?: string
}

export type RunEvidence = {
  summary: RunDetail
  report: {
    markdown: string
  }
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
  pnl_return_column?: string
  segment_columns: string[]
  time_breakdowns: string[]
  engine: string
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
  market: string
  expression: string
  description?: string
  dependencies?: string[]
}

export type PipelineRunRequest = {
  cost_bps: number
  date_end: string
  date_start: string
  factor_expression: string
  factor_template_id?: string
  label_columns: string[]
  market: string
  output_root?: string
  pnl_return_column: string
  portfolio_mode: string
  run_id: string
}

export type PipelineRunSummary = {
  run_id: string
  status: 'queued' | 'running' | 'ready' | 'failed' | string
  message?: string
  snapshot_path?: string
}

export type PipelineCheckpointSummary = {
  key: string
  label: string
  status: 'pending' | 'ready' | 'warning' | 'blocked' | string
  path?: string
  message?: string
}

export type RunTableSpec = {
  key: string
  label: string
  path: string
  downloadItem: string
  evidenceKey: string
  supportsCombinationFilter: boolean
}

export type RunDisplayTable = {
  key: string
  label: string
  group: string
  path: string
  download_item: string
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
  kind: 'line' | 'bar'
  series: RunDisplayChartSeries[]
  table: string
  title: string
  x: string
}

export type RunDisplayDownloadItem = {
  item: string
  kind: string
  label: string
  table?: string
}

export type RunDisplayDownloadPreset = {
  key: 'all' | 'report' | 'tables' | 'repro'
  label: string
  items: string[]
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
): Promise<TablePayload> {
  const params = new URLSearchParams({ limit: String(limit) })
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

export async function createPipelineRun(
  payload: PipelineRunRequest,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>('/api/pipeline-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function getPipelineRunStatus(
  runId: string,
): Promise<PipelineRunSummary> {
  return fetchJson<PipelineRunSummary>(
    `/api/pipeline-runs/${encodeURIComponent(runId)}`,
  )
}

export async function getPipelineCheckpoints(
  runId: string,
): Promise<PipelineCheckpointSummary[]> {
  return fetchJson<PipelineCheckpointSummary[]>(
    `/api/pipeline-runs/${encodeURIComponent(runId)}/checkpoints`,
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
  if (!display) {
    return []
  }
  return display.tables.map((table) => ({
    key: table.key,
    label: table.label,
    path: tableApiPath(run, table),
    downloadItem: table.download_item,
    evidenceKey: table.key,
    supportsCombinationFilter: table.supports_combination_filter !== false,
  }))
}

function tableApiPath(run: RunSummary, table: RunDisplayTable): string {
  const metricName = table.path.split('/').pop()?.replace(/\.csv$/, '') ?? table.key
  const prefix = runApiPath(run)
  if (table.path === 'combinations.csv') {
    return `${prefix}/metrics/combinations`
  }
  if (table.path.startsWith('portfolio/')) {
    return `${prefix}/portfolio/${encodeURIComponent(metricName)}`
  }
  if (table.path.startsWith('diagnostics/')) {
    return `${prefix}/diagnostics/${encodeURIComponent(metricName)}`
  }
  return `${prefix}/metrics/${encodeURIComponent(metricName)}`
}

export function artifactDownloadUrl(run: RunSummary, downloadItem: string): string {
  const encodedItem = downloadItem.split('/').map(encodeURIComponent).join('/')
  return `${API_BASE_URL}${runApiPath(run)}/artifacts/${encodedItem}`
}

export function reportMarkdownUrl(run: RunSummary): string {
  return `${API_BASE_URL}${runApiPath(run)}/report.md`
}

function runApiPath(run: RunSummary): string {
  return `/api/runs/${encodeURIComponent(run.source_alias)}/${encodeURIComponent(run.run_id)}`
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
    const payload = (await response.json()) as { detail?: string }
    return payload.detail ?? `Request failed with status ${response.status}`
  } catch {
    return `Request failed with status ${response.status}`
  }
}
