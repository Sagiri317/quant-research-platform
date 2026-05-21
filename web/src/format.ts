export function messageFrom(caught: unknown): string {
  return caught instanceof Error ? caught.message : '未知 API 错误'
}

export function formatCell(value: unknown): string {
  if (typeof value === 'number') {
    return formatNumber(value)
  }
  if (typeof value === 'boolean') {
    return value ? '是' : '否'
  }
  return value == null ? '' : String(value)
}

export function formatNumber(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) {
    return '--'
  }
  if (Math.abs(value) >= 100) {
    return value.toLocaleString('en-US', { maximumFractionDigits: 0 })
  }
  return value.toLocaleString('en-US', { maximumFractionDigits: 4 })
}

export function formatAxisTick(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return ''
  }
  const normalized = Object.is(value, -0) ? 0 : value
  return formatNumber(normalized)
}

export function formatPercent(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return ''
  }
  const normalized = Object.is(value, -0) ? 0 : value
  return `${(normalized * 100).toLocaleString('en-US', {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  })}%`
}

export function estimateYAxisWidth(
  rows: Record<string, unknown>[],
  keys: string[],
): number {
  const numericValues = yAxisValues(rows, keys)
  if (numericValues.length === 0) {
    return 46
  }
  const domain = estimateYAxisDomain(rows, keys, { includeZero: true })
  const valuesForLabels = [...numericValues, ...domain, 0]
  const longest = numericValues.reduce((maxLength, value) => {
    const label = formatAxisTick(value)
    return Math.max(maxLength, label.length)
  }, 0)
  const longestWithDomain = valuesForLabels.reduce((maxLength, value) => {
    const label = formatAxisTick(value)
    return Math.max(maxLength, label.length)
  }, longest)
  return clamp(18 + longestWithDomain * 7, 46, 72)
}

export function estimateYAxisDomain(
  rows: Record<string, unknown>[],
  keys: string[],
  { includeZero = false }: { includeZero?: boolean } = {},
): [number, number] {
  const numericValues = yAxisValues(rows, keys)
  if (numericValues.length === 0) {
    return [0, 1]
  }
  const rawMin = Math.min(...numericValues)
  const rawMax = Math.max(...numericValues)
  const min = includeZero ? Math.min(rawMin, 0) : rawMin
  const max = includeZero ? Math.max(rawMax, 0) : rawMax
  const range = max - min
  const fallbackPad = Math.max(Math.max(Math.abs(min), Math.abs(max)) * 0.08, 0.01)
  const pad = range > 0 ? range * 0.08 : fallbackPad
  return [min - pad, max + pad]
}

function yAxisValues(
  rows: Record<string, unknown>[],
  keys: string[],
): number[] {
  return rows.flatMap((row) =>
    keys
      .map((key) => row[key])
      .filter(
        (value): value is number =>
          typeof value === 'number' && Number.isFinite(value),
      ),
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
