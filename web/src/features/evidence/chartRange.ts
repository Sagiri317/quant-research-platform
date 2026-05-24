export type ChartRangeRow = Record<string, unknown>

export type ChartRangeSeries = {
  key: string
  label?: string
}

export type ChartRangeReadout = {
  datePart: string
  seriesRanges: Array<{
    label: string
    maxValue: string
    minValue: string
  }>
}

export function rangeReadout(
  rows: ChartRangeRow[],
  series: ChartRangeSeries[],
  formatter: (value: unknown) => string,
  xKey = 'timestamp',
): ChartRangeReadout {
  const firstDate = rows[0]?.[xKey]
  const lastDate = rows[rows.length - 1]?.[xKey]
  const datePart =
    firstDate && lastDate ? `${String(firstDate)} 至 ${String(lastDate)}` : '暂无范围'
  return {
    datePart,
    seriesRanges: series.flatMap((item) => {
      const values = rows
        .map((row) => row[item.key])
        .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
      if (values.length === 0) {
        return []
      }
      return [
        {
          label: item.label ?? item.key,
          minValue: formatter(Math.min(...values)),
          maxValue: formatter(Math.max(...values)),
        },
      ]
    }),
  }
}
