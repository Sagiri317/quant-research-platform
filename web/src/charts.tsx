import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
  type RefObject,
} from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { ChartPanel } from './components'
import {
  estimateYAxisDomain,
  estimateYAxisWidth,
  formatAxisTick,
} from './format'
import { rangeReadout, type ChartRangeReadout } from './chartRange'

const CHART_HEIGHT = 240
const CHART_MARGIN = { top: 14, right: 16, bottom: 18, left: 0 }
const MIN_VISIBLE_POINTS = 8
const X_AXIS_PROPS = {
  height: 34,
  minTickGap: 24,
  tickMargin: 10,
}

type ChartRow = Record<string, unknown>

type ChartSeries = {
  color: string
  key: string
  label?: string
}

type InteractiveChartState = {
  dragStartX: number | null
  hiddenKeys: string[]
  range: [number, number]
  resetKey: string
}

type RunChartProps = {
  chartKind?: string
  description?: string
  includeZero?: boolean
  rows: ChartRow[]
  series: ChartSeries[]
  testId: string
  title: string
  valueFormatter?: (value: unknown) => string
  xKey: string
}

function chartAxis(rows: ChartRow[], series: ChartSeries[], includeZero = false) {
  const keys = series.map((item) => item.key)
  return {
    domain: estimateYAxisDomain(rows, keys, { includeZero }),
    width: estimateYAxisWidth(rows, keys),
  }
}

function SharedYAxis({
  domain,
  tickFormatter = formatAxisTick,
  width,
}: {
  domain: [number, number]
  tickFormatter?: (value: unknown) => string
  width: number
}) {
  return (
    <YAxis
      domain={domain}
      tickFormatter={tickFormatter}
      tickMargin={6}
      width={width}
    />
  )
}

function ChartMeta({
  description,
  hiddenSeries,
  onToggleSeries,
  series,
}: {
  description?: string
  hiddenSeries: Set<string>
  onToggleSeries: (key: string) => void
  series: ChartSeries[]
}) {
  return (
    <div className="chart-meta">
      {description ? <p>{description}</p> : null}
      <div className="chart-legend" aria-label="图例">
        {series.map((item) => (
          <button
            aria-label={`${hiddenSeries.has(item.key) ? '显示' : '隐藏'} ${item.label ?? item.key}`}
            aria-pressed={!hiddenSeries.has(item.key)}
            className={hiddenSeries.has(item.key) ? 'muted' : undefined}
            key={item.key}
            onClick={() => onToggleSeries(item.key)}
            type="button"
          >
            <i style={{ backgroundColor: item.color }} />
            {item.label ?? item.key}
          </button>
        ))}
      </div>
    </div>
  )
}

function ChartEmptyState() {
  return <div className="chart-empty">暂无可绘制数据</div>
}

export function RunLineChartPanel({
  chartKind = 'line',
  description,
  includeZero,
  rows,
  series,
  testId,
  title,
  valueFormatter,
  xKey,
}: RunChartProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const chartState = useInteractiveChart(
    rows,
    series,
    viewportRef,
    chartResetKey(rows, series, xKey, testId),
  )
  const visibleRows = chartState.visibleRows
  const visibleSeries = series.filter((item) => !chartState.hiddenSeries.has(item.key))
  const yAxis = chartAxis(visibleRows, visibleSeries, includeZero)
  const formatter = valueFormatter ?? formatAxisTick
  return (
    <ChartPanel
      chartKind={chartKind}
      chartMarginLeft={CHART_MARGIN.left}
      dataRowCount={visibleRows.length}
      testId={testId}
      title={title}
      yAxisWidth={yAxis.width}
    >
      <ChartMeta
        description={description}
        hiddenSeries={chartState.hiddenSeries}
        onToggleSeries={chartState.toggleSeries}
        series={series}
      />
      <ChartViewportControls
        canReset={chartState.canReset}
        onReset={chartState.resetViewport}
        readout={rangeReadout(visibleRows, visibleSeries, formatter, xKey)}
      />
      {visibleRows.length === 0 || visibleSeries.length === 0 ? (
        <ChartEmptyState />
      ) : (
        <div
          className="chart-viewport"
          onMouseDown={chartState.handleMouseDown}
          onMouseLeave={chartState.handleMouseLeave}
          onMouseMove={chartState.handleMouseMove}
          onMouseUp={chartState.handleMouseUp}
          ref={viewportRef}
        >
          <ResponsiveContainer height={CHART_HEIGHT} width="100%">
            <LineChart data={visibleRows} margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis {...X_AXIS_PROPS} dataKey={xKey} />
              <SharedYAxis
                domain={yAxis.domain}
                tickFormatter={formatter}
                width={yAxis.width}
              />
              <Tooltip formatter={(value) => formatter(value)} />
              {visibleSeries.map((item) => (
                <Line
                  activeDot={false}
                  connectNulls
                  dataKey={item.key}
                  dot={false}
                  isAnimationActive={false}
                  key={item.key}
                  stroke={item.color}
                  strokeWidth={2.2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartPanel>
  )
}

export function RunBarChartPanel({
  chartKind = 'bar',
  description,
  includeZero,
  rows,
  series,
  testId,
  title,
  xKey,
}: RunChartProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const chartState = useInteractiveChart(
    rows,
    series,
    viewportRef,
    chartResetKey(rows, series, xKey, testId),
  )
  const visibleRows = chartState.visibleRows
  const visibleSeries = series.filter((item) => !chartState.hiddenSeries.has(item.key))
  const yAxis = chartAxis(visibleRows, visibleSeries, includeZero)
  return (
    <ChartPanel
      chartKind={chartKind}
      chartMarginLeft={CHART_MARGIN.left}
      dataRowCount={visibleRows.length}
      testId={testId}
      title={title}
      yAxisWidth={yAxis.width}
    >
      <ChartMeta
        description={description}
        hiddenSeries={chartState.hiddenSeries}
        onToggleSeries={chartState.toggleSeries}
        series={series}
      />
      <ChartViewportControls
        canReset={chartState.canReset}
        onReset={chartState.resetViewport}
        readout={rangeReadout(visibleRows, visibleSeries, formatAxisTick, xKey)}
      />
      {visibleRows.length === 0 || visibleSeries.length === 0 ? (
        <ChartEmptyState />
      ) : (
        <div
          className="chart-viewport"
          onMouseDown={chartState.handleMouseDown}
          onMouseLeave={chartState.handleMouseLeave}
          onMouseMove={chartState.handleMouseMove}
          onMouseUp={chartState.handleMouseUp}
          ref={viewportRef}
        >
          <ResponsiveContainer height={CHART_HEIGHT} width="100%">
            <BarChart data={visibleRows} margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis {...X_AXIS_PROPS} dataKey={xKey} />
              <SharedYAxis domain={yAxis.domain} width={yAxis.width} />
              <Tooltip formatter={(value) => formatAxisTick(value)} />
              {visibleSeries.map((item) => (
                <Bar dataKey={item.key} fill={item.color} key={item.key} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </ChartPanel>
  )
}

function ChartViewportControls({
  canReset,
  onReset,
  readout,
}: {
  canReset: boolean
  onReset: () => void
  readout: ChartRangeReadout
}) {
  return (
    <div className="chart-controls">
      <span>
        <b>时间范围</b>
        {readout.datePart}
      </span>
      {readout.seriesRanges.length > 0 ? (
        <div className="chart-series-ranges" aria-label="Series value ranges">
          {readout.seriesRanges.map((range) => (
            <span className="chart-series-range" key={range.label}>
              <b>{range.label}</b>
              <span>
                {range.minValue} ~ {range.maxValue}
              </span>
            </span>
          ))}
        </div>
      ) : null}
      <button disabled={!canReset} onClick={onReset} type="button">
        重置视图
      </button>
    </div>
  )
}

function useInteractiveChart(
  rows: ChartRow[],
  series: ChartSeries[],
  viewportRef: RefObject<HTMLDivElement | null>,
  resetKey: string,
) {
  const maxIndex = Math.max(rows.length - 1, 0)
  const [state, setState] = useState<InteractiveChartState>(() =>
    initialInteractiveChartState(resetKey, maxIndex),
  )
  const currentState =
    state.resetKey === resetKey
      ? state
      : initialInteractiveChartState(resetKey, maxIndex)
  const { dragStartX, hiddenKeys, range } = currentState
  const hiddenSeries = useMemo(() => new Set(hiddenKeys), [hiddenKeys])
  const normalizedRange = normalizeRange(range, maxIndex)
  const visibleRows = rows.slice(normalizedRange[0], normalizedRange[1] + 1)
  const canReset = normalizedRange[0] !== 0 || normalizedRange[1] !== maxIndex
  const wheelStateRef = useRef({
    normalizedRange,
    rowCount: rows.length,
  })

  useEffect(() => {
    wheelStateRef.current = {
      normalizedRange,
      rowCount: rows.length,
    }
  }, [normalizedRange, rows.length])

  const updateState = useCallback(
    (
      updater: (
        current: InteractiveChartState,
        currentMaxIndex: number,
      ) => InteractiveChartState,
    ) => {
      setState((current) => {
        const base =
          current.resetKey === resetKey
            ? current
            : initialInteractiveChartState(resetKey, maxIndex)
        return updater(base, maxIndex)
      })
    },
    [maxIndex, resetKey],
  )

  const updateRange = useCallback(
    (nextRange: [number, number]) => {
      updateState((current, currentMaxIndex) => ({
        ...current,
        range: normalizeRange(nextRange, currentMaxIndex),
      }))
    },
    [updateState],
  )

  function resetViewport() {
    updateState((current, currentMaxIndex) => ({
      ...current,
      range: [0, currentMaxIndex],
    }))
  }

  function toggleSeries(key: string) {
    updateState((current) => ({
      ...current,
      hiddenKeys: current.hiddenKeys.includes(key)
        ? current.hiddenKeys.filter((item) => item !== key)
        : current.hiddenKeys.length >= series.length - 1
          ? current.hiddenKeys
          : [...current.hiddenKeys, key],
    }))
  }

  const handleWheel = useCallback(
    (event: globalThis.WheelEvent) => {
      const { normalizedRange: currentRange, rowCount } = wheelStateRef.current
      if (rowCount <= MIN_VISIBLE_POINTS) {
        return
      }
      event.preventDefault()
      const [start, end] = currentRange
      const span = end - start + 1
      const rect = viewportRef.current?.getBoundingClientRect()
      const ratio =
        rect && rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0.5
      const anchor = start + Math.max(0, Math.min(1, ratio)) * (span - 1)
      const factor = event.deltaY > 0 ? 1.25 : 0.8
      const nextSpan = Math.round(
        Math.min(rowCount, Math.max(MIN_VISIBLE_POINTS, span * factor)),
      )
      const nextStart = Math.round(anchor - (anchor - start) * (nextSpan / span))
      const nextMaxIndex = Math.max(rowCount - 1, 0)
      updateState((current) => ({
        ...current,
        range: normalizeRange([nextStart, nextStart + nextSpan - 1], nextMaxIndex),
      }))
    },
    [updateState, viewportRef],
  )

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }
    viewport.addEventListener('wheel', handleWheel, { passive: false })
    return () => {
      viewport.removeEventListener('wheel', handleWheel)
    }
  }, [handleWheel, viewportRef])

  function handleMouseDown(event: MouseEvent<HTMLDivElement>) {
    updateState((current) => ({ ...current, dragStartX: event.clientX }))
  }

  function handleMouseMove(event: MouseEvent<HTMLDivElement>) {
    if (dragStartX === null) {
      return
    }
    const [start, end] = normalizedRange
    const span = end - start + 1
    const rect = event.currentTarget.getBoundingClientRect()
    const pointsPerPixel = rect.width > 0 ? span / rect.width : 0
    const delta = Math.round((dragStartX - event.clientX) * pointsPerPixel)
    if (delta === 0) {
      return
    }
    updateRange([start + delta, end + delta])
    updateState((current) => ({ ...current, dragStartX: event.clientX }))
  }

  function stopDrag() {
    updateState((current) => ({ ...current, dragStartX: null }))
  }

  return {
    canReset,
    handleMouseDown,
    handleMouseLeave: stopDrag,
    handleMouseMove,
    handleMouseUp: stopDrag,
    handleWheel,
    hiddenSeries,
    resetViewport,
    toggleSeries,
    visibleRows,
  }
}

function initialInteractiveChartState(
  resetKey: string,
  maxIndex: number,
): InteractiveChartState {
  return {
    dragStartX: null,
    hiddenKeys: [],
    range: [0, maxIndex],
    resetKey,
  }
}

function chartResetKey(
  rows: ChartRow[],
  series: ChartSeries[],
  xKey: string,
  testId: string,
): string {
  const first = rows[0]
  const last = rows[rows.length - 1]
  return [
    testId,
    xKey,
    rows.length,
    first?.[xKey],
    last?.[xKey],
    series.map((item) => item.key).join(','),
  ].join('\0')
}

function normalizeRange(range: [number, number], maxIndex: number): [number, number] {
  if (maxIndex <= 0) {
    return [0, 0]
  }
  const span = Math.min(maxIndex + 1, Math.max(MIN_VISIBLE_POINTS, range[1] - range[0] + 1))
  let start = Math.max(0, Math.min(range[0], maxIndex - span + 1))
  let end = start + span - 1
  if (end > maxIndex) {
    end = maxIndex
    start = Math.max(0, end - span + 1)
  }
  return [start, end]
}
