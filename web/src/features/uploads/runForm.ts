import type { CreateRunPayload } from '../../shared/api/api'

export type CreateFormState = {
  run_id: string
  project: string
  kind: string
  domain: string
  data_stage: string
  uses_external_data: boolean
  timestamp_column: string
  asset_column: string
  feature_columns: string
  label_columns: string
  return_source_column: string
  segment_columns: string
  time_breakdowns: string
  kernel: string
  n_groups: string
  weight_method: string
  portfolio_mode: string
  direction: string
  top_quantile: string
  bottom_quantile: string
  cost_bps: string
  risk_column: string
  portfolio_feature: string
  tensor_max_memory_mb: string
  boundary: string
}

export type CsvPreview = {
  columns: string[]
  rowCount: number
}

export const WEIGHT_METHOD_OPTIONS = [
  { value: 'equal_weight_quantile', label: '等权分位' },
  { value: 'rank_weighted_quantile', label: '按排名加权' },
  { value: 'zscore_weighted_quantile', label: '按 z-score 加权' },
  { value: 'signal_abs_weighted_quantile', label: '按信号绝对值加权' },
  { value: 'inverse_vol_scaled_quantile', label: '按风险缩放' },
]

export const PORTFOLIO_MODE_OPTIONS = [
  { value: 'long_short', label: '多空组合' },
  { value: 'long_only', label: '只做多' },
]

export const DIRECTION_OPTIONS = [
  { value: 'long_high', label: '因子越大越做多' },
  { value: 'long_low', label: '因子越小越做多' },
]

export const KIND_OPTIONS = [{ value: 'factor', label: '因子研究' }]
export const DATA_STAGE_OPTIONS = [
  { value: 'uploaded_panel', label: '用户上传面板' },
]
export const KERNEL_OPTIONS = [
  { value: 'dataframe', label: 'DataFrame 内核' },
  { value: 'tensor', label: 'Tensor 内核' },
]

export const INITIAL_CREATE_FORM: CreateFormState = {
  run_id: `uploaded_${new Date().toISOString().replace(/\D/g, '').slice(0, 14)}`,
  project: 'quant-research-platform',
  kind: 'factor',
  domain: 'user_upload',
  data_stage: 'uploaded_panel',
  uses_external_data: true,
  timestamp_column: 'timestamp',
  asset_column: 'asset',
  feature_columns: 'factor',
  label_columns: 'forward_return',
  return_source_column: 'forward_return_1d',
  segment_columns: '',
  time_breakdowns: '',
  kernel: 'dataframe',
  n_groups: '5',
  weight_method: 'equal_weight_quantile',
  portfolio_mode: 'long_short',
  direction: 'long_high',
  top_quantile: '0.2',
  bottom_quantile: '0.2',
  cost_bps: '0',
  risk_column: '',
  portfolio_feature: '',
  tensor_max_memory_mb: '',
  boundary: '',
}

export function validateCreateForm(
  form: CreateFormState,
  csvPreview: CsvPreview | null,
): string | null {
  const requiredFields: [keyof CreateFormState, string][] = [
    ['run_id', '运行编号'],
    ['project', '项目名'],
    ['kind', '研究类型'],
    ['domain', '市场域'],
    ['data_stage', '数据阶段'],
    ['timestamp_column', '时间列'],
    ['asset_column', '资产列'],
    ['feature_columns', '因子列'],
    ['label_columns', '标签列'],
    ['return_source_column', '收益来源列'],
    ['n_groups', '分组数'],
    ['weight_method', '权重方法'],
    ['portfolio_mode', '组合模式'],
    ['direction', '信号方向'],
    ['kernel', '计算内核'],
    ['top_quantile', '多头分位'],
    ['bottom_quantile', '空头分位'],
  ]
  const missingField = requiredFields.find(([field]) => {
    const value = form[field]
    return typeof value === 'string' && !value.trim()
  })
  if (missingField) {
    return `请填写${missingField[1]}。`
  }
  if (!csvPreview) {
    return '请先选择 factor CSV 文件。'
  }
  if (csvPreview.rowCount < 1) {
    return 'CSV 至少需要一行数据。'
  }
  const requiredColumns = [
    form.timestamp_column,
    form.asset_column,
    ...splitColumns(form.feature_columns),
    ...splitColumns(form.label_columns),
    form.return_source_column,
  ].map((column) => column.trim())
  const missingColumns = requiredColumns.filter(
    (column) => !csvPreview.columns.includes(column),
  )
  if (missingColumns.length > 0) {
    return `CSV 缺少列：${missingColumns.join(', ')}`
  }
  const nGroups = Number.parseInt(form.n_groups, 10)
  if (!Number.isInteger(nGroups) || nGroups < 2) {
    return '分组数必须是不小于 2 的整数。'
  }
  const unsupportedBreakdowns = splitColumns(form.time_breakdowns).filter(
    (value) => !['year', 'quarter', 'month'].includes(value),
  )
  if (unsupportedBreakdowns.length > 0) {
    return `时间分解只支持 year、quarter、month：${unsupportedBreakdowns.join(', ')}`
  }
  if (!['dataframe', 'tensor'].includes(form.kernel)) {
    return '计算内核只支持 dataframe 或 tensor。'
  }
  if (!['long_short', 'long_only'].includes(form.portfolio_mode)) {
    return '组合模式只支持 long_short 或 long_only。'
  }
  for (const [field, label] of [
    ['top_quantile', '多头分位'],
    ['bottom_quantile', '空头分位'],
  ] as const) {
    const value = Number.parseFloat(form[field])
    if (!Number.isFinite(value) || value <= 0 || value > 1) {
      return `${label}必须大于 0 且不超过 1。`
    }
  }
  const costBps = Number.parseFloat(form.cost_bps || '0')
  if (!Number.isFinite(costBps) || costBps < 0) {
    return '成本 bps 必须是非负数字。'
  }
  if (form.tensor_max_memory_mb.trim()) {
    const maxMemory = Number.parseFloat(form.tensor_max_memory_mb)
    if (!Number.isFinite(maxMemory) || maxMemory < 0) {
      return 'Tensor 内存预算必须是非负数字。'
    }
  }
  return null
}

export function parseCsvPreview(csvText: string): CsvPreview {
  const rows = csvText
    .split(/\r?\n/)
    .map((row) => row.trim())
    .filter(Boolean)
  if (rows.length === 0) {
    return { columns: [], rowCount: 0 }
  }
  return {
    columns: rows[0].split(',').map((column) => column.trim()).filter(Boolean),
    rowCount: Math.max(rows.length - 1, 0),
  }
}

export function buildCreatePayload(
  form: CreateFormState,
  csvText: string,
): CreateRunPayload {
  return {
    run_id: form.run_id.trim(),
    project: form.project.trim(),
    kind: form.kind.trim(),
    domain: form.domain.trim(),
    data_stage: form.data_stage.trim(),
    uses_external_data: form.uses_external_data,
    timestamp_column: form.timestamp_column.trim(),
    asset_column: form.asset_column.trim(),
    feature_columns: splitColumns(form.feature_columns),
    label_columns: splitColumns(form.label_columns),
    return_source_column: form.return_source_column.trim(),
    segment_columns: splitColumns(form.segment_columns),
    time_breakdowns: splitColumns(form.time_breakdowns),
    kernel: form.kernel,
    n_groups: Number.parseInt(form.n_groups, 10),
    weight_method: form.weight_method.trim(),
    portfolio_mode: form.portfolio_mode.trim(),
    direction: form.direction.trim(),
    top_quantile: Number.parseFloat(form.top_quantile),
    bottom_quantile: Number.parseFloat(form.bottom_quantile),
    cost_bps: Number.parseFloat(form.cost_bps),
    risk_column: optionalText(form.risk_column),
    portfolio_feature: optionalText(form.portfolio_feature),
    tensor_max_memory_mb: optionalNumber(form.tensor_max_memory_mb),
    boundary: optionalText(form.boundary),
    csv_text: csvText,
  }
}

function splitColumns(value: string): string[] {
  return value
    .split(',')
    .map((column) => column.trim())
    .filter(Boolean)
}

function optionalText(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

function optionalNumber(value: string): number | undefined {
  const trimmed = value.trim()
  return trimmed ? Number.parseFloat(trimmed) : undefined
}

export function inferCreateFormFromColumns(
  form: CreateFormState,
  columns: string[],
): CreateFormState {
  const timestamp = firstExisting(columns, [
    form.timestamp_column,
    'timestamp',
    'date',
    'trade_date',
    'datetime',
  ])
  const asset = firstExisting(columns, [
    form.asset_column,
    'asset',
    'symbol',
    'ticker',
    'stock_code',
    'instrument',
  ])
  const inferredReturnSource = firstExisting(columns, [
    form.return_source_column,
    'forward_return_1d',
    'return_1d',
  ])
  const reserved = new Set([timestamp, asset, inferredReturnSource].filter(Boolean))
  const labelCandidates = columns.filter(
    (column) =>
      !reserved.has(column) &&
      /(^|_)(label|target|forward_return|ret_\d+d|return_\d+d)/i.test(column),
  )
  const featureCandidates = columns.filter(
    (column) =>
      !reserved.has(column) &&
      !labelCandidates.includes(column) &&
      /(factor|feature|signal|score|prediction|alpha)/i.test(column),
  )
  return {
    ...form,
    timestamp_column: timestamp || form.timestamp_column,
    asset_column: asset || form.asset_column,
    feature_columns: featureCandidates.join(',') || form.feature_columns,
    label_columns: labelCandidates.join(',') || form.label_columns,
    return_source_column: inferredReturnSource || form.return_source_column,
  }
}

function firstExisting(columns: string[], candidates: string[]): string {
  return candidates.find((candidate) => columns.includes(candidate)) ?? ''
}
