import { useEffect, useMemo, useRef, useState } from 'react'
import {
  type RunSummary,
  type TablePayload,
  fetchTablePreview,
  fetchRunEvidence,
  runTableSpecsFromDisplay,
} from '../../shared/api/api'
import { messageFrom } from '../../shared/format'
import {
  EMPTY_RUN_STATE,
  buildRunStateFromEvidence,
  filterTablesByCombination,
  firstCombinationId,
  firstPreviewTableKey,
  type RunState,
} from './runEvidenceState'
import { useTablePreview } from './useTablePreview'

const TRADABILITY_DIAGNOSTICS_TABLE_KEY = 'tradability_diagnostics'
const LATEST_PREVIEW_SUFFIX = '__latest'

export function useRunEvidence(selectedRun: RunSummary | null, cacheVersion = 0) {
  const [runState, setRunState] = useState<RunState>(EMPTY_RUN_STATE)
  const [activeTable, setActiveTable] = useState('IC')
  const [selectedCombinationId, setSelectedCombinationId] = useState('')
  const [tablePreviewExpanded, setTablePreviewExpanded] = useState(false)
  const [chartPreviewPayloads, setChartPreviewPayloads] = useState<
    Record<string, TablePayload>
  >({})
  const [loadingEvidence, setLoadingEvidence] = useState(false)
  const [evidenceError, setEvidenceError] = useState('')
  const [runStateRunId, setRunStateRunId] = useState('')
  const runCache = useRef(new Map<string, RunState>())
  const runRequests = useRef(new Map<string, Promise<RunState>>())

  useEffect(() => {
    if (!selectedRun) {
      return
    }
    const run = selectedRun
    const cacheKey = `${run.public_run_id}::${cacheVersion}`
    const cached = runCache.current.get(cacheKey)
    if (cached) {
      const tableSpecs = runTableSpecsFromDisplay(run, cached.display)
      setLoadingEvidence(false)
      setRunState(cached)
      setRunStateRunId(cacheKey)
      setSelectedCombinationId(firstCombinationId(cached.tables.combinations))
      setActiveTable(
        firstPreviewTableKey(tableSpecs),
      )
      setTablePreviewExpanded(false)
      setEvidenceError('')
      return
    }
    let cancelled = false
    let pending = runRequests.current.get(cacheKey)
    if (!pending) {
      pending = fetchRunEvidence(run).then((evidence) => {
        const nextRunState = buildRunStateFromEvidence(evidence)
        runCache.current.set(cacheKey, nextRunState)
        return nextRunState
      })
      runRequests.current.set(cacheKey, pending)
      void pending.finally(() => {
        runRequests.current.delete(cacheKey)
      }).catch(() => {})
    }
    const runStatePromise = pending
    async function loadRunEvidence() {
      try {
        setLoadingEvidence(true)
        setRunState(EMPTY_RUN_STATE)
        setRunStateRunId('')
        const nextRunState = await runStatePromise
        if (cancelled) {
          return
        }
        setRunState(nextRunState)
        setRunStateRunId(cacheKey)
        const tableSpecs = runTableSpecsFromDisplay(run, nextRunState.display)
        setSelectedCombinationId(firstCombinationId(nextRunState.tables.combinations))
        setActiveTable(
          firstPreviewTableKey(tableSpecs),
        )
        setTablePreviewExpanded(false)
        setEvidenceError('')
      } catch (caught) {
        if (!cancelled) {
          setEvidenceError(messageFrom(caught))
        }
      } finally {
        if (!cancelled) {
          setLoadingEvidence(false)
        }
      }
    }
    void loadRunEvidence()
    return () => {
      cancelled = true
    }
  }, [cacheVersion, selectedRun])

  const selectedRunKey = selectedRun?.public_run_id ?? ''
  const selectedRunCacheKey = selectedRunKey ? `${selectedRunKey}::${cacheVersion}` : ''
  const runStateMatchesSelectedRun =
    selectedRunCacheKey !== '' && runStateRunId === selectedRunCacheKey
  const exposedRunState = runStateMatchesSelectedRun ? runState : EMPTY_RUN_STATE
  const allTableSpecs = useMemo(
    () =>
      selectedRun
        ? runTableSpecsFromDisplay(selectedRun, exposedRunState.display)
        : [],
    [exposedRunState.display, selectedRun],
  )
  const selectedTableSpecs = allTableSpecs
  const displayCharts = useMemo(
    () => exposedRunState.display?.charts ?? [],
    [exposedRunState.display],
  )
  const selectedCombinationExists = useMemo(
    () =>
      (exposedRunState.tables.combinations?.rows ?? []).some(
        (row) => String(row.combination_id ?? '') === selectedCombinationId,
      ),
    [exposedRunState.tables.combinations, selectedCombinationId],
  )
  const chartTableSpecs = useMemo(() => {
    const tableKeys = new Set(displayCharts.map((chart) => chart.table))
    return selectedTableSpecs.filter((spec) => tableKeys.has(spec.key))
  }, [displayCharts, selectedTableSpecs])
  const selectedCombinationPreviewSpecs = useMemo(() => {
    const previewKeys = new Set(chartTableSpecs.map((spec) => spec.key))
    return selectedTableSpecs.filter((spec) => {
      if (previewKeys.has(spec.key)) {
        return true
      }
      return spec.supportsCombinationFilter
    })
  }, [chartTableSpecs, selectedTableSpecs])
  const chartPreviewPrefix =
    selectedRun && selectedCombinationId
      ? [
          selectedRun.public_run_id,
          cacheVersion,
          selectedCombinationId,
        ].join('::')
      : ''

  useEffect(() => {
    if (
      !selectedRun ||
      !selectedCombinationId ||
      !selectedCombinationExists ||
      selectedCombinationPreviewSpecs.length === 0
    ) {
      return
    }
    const filterableChartSpecs = selectedCombinationPreviewSpecs.filter(
      (spec) => spec.supportsCombinationFilter !== false,
    )
    const missingSpecs = filterableChartSpecs.filter((spec) => {
      const previewKey = `${chartPreviewPrefix}::${spec.key}`
      const latestKey = latestPreviewKey(chartPreviewPrefix, spec.key)
      return (
        chartPreviewPayloads[previewKey] === undefined ||
        (spec.key === TRADABILITY_DIAGNOSTICS_TABLE_KEY &&
          chartPreviewPayloads[latestKey] === undefined)
      )
    })
    if (missingSpecs.length === 0) {
      return
    }
    let cancelled = false
    async function loadSelectedChartTables() {
      try {
        const payloads = await Promise.all(
          missingSpecs.map(async (spec) => {
            const payload = await fetchTablePreview(
              spec.previewUrl,
              combinationPreviewLimit(spec, chartTableSpecs),
              selectedCombinationId,
            )
            const latestPayload =
              spec.key === TRADABILITY_DIAGNOSTICS_TABLE_KEY
                ? await latestDiagnosticPreview(spec, payload, selectedCombinationId)
                : undefined
            return [spec.key, payload, latestPayload] as const
          }),
        )
        if (cancelled) {
          return
        }
        setChartPreviewPayloads((current) => {
          const next = { ...current }
          for (const [key, payload, latestPayload] of payloads) {
            next[`${chartPreviewPrefix}::${key}`] = payload
            if (latestPayload) {
              next[latestPreviewKey(chartPreviewPrefix, key)] = latestPayload
            }
          }
          return next
        })
        setEvidenceError('')
      } catch (caught) {
        if (!cancelled) {
          setEvidenceError(messageFrom(caught))
        }
      }
    }
    void loadSelectedChartTables()
    return () => {
      cancelled = true
    }
  }, [
    chartPreviewPayloads,
    chartPreviewPrefix,
    chartTableSpecs,
    selectedCombinationPreviewSpecs,
    selectedCombinationExists,
    selectedCombinationId,
    selectedRun,
    setEvidenceError,
  ])

  const filteredTables = useMemo(
    () =>
      filterTablesByCombination(
        exposedRunState.tables,
        selectedCombinationId,
        selectedTableSpecs,
      ),
    [exposedRunState.tables, selectedCombinationId, selectedTableSpecs],
  )
  const chartTables = useMemo(() => {
    const next = { ...filteredTables }
    if (!chartPreviewPrefix) {
      return next
    }
    for (const spec of selectedCombinationPreviewSpecs) {
      const payload = chartPreviewPayloads[`${chartPreviewPrefix}::${spec.key}`]
      if (payload !== undefined) {
        const existingRows = next[spec.key]?.rows.length ?? 0
        if (payload.rows.length > 0 || existingRows === 0) {
          next[spec.key] = payload
        }
      }
    }
    return next
  }, [
    chartPreviewPayloads,
    chartPreviewPrefix,
    filteredTables,
    selectedCombinationPreviewSpecs,
  ])
  const activeTableSpec = selectedTableSpecs.find(
    (spec) => spec.key === activeTable,
  )
  const baseActivePayload = chartTables[activeTable]

  const { loadingTablePreview, previewPayload } = useTablePreview({
    activePayload: baseActivePayload,
    activeTableSpec,
    expanded: tablePreviewExpanded,
    cacheVersion,
    selectedCombinationId,
    selectedRun,
    setEvidenceError,
  })
  const activePayload = baseActivePayload ?? previewPayload
  const latestTradabilityDiagnostic = chartPreviewPrefix
    ? chartPreviewPayloads[
        latestPreviewKey(chartPreviewPrefix, TRADABILITY_DIAGNOSTICS_TABLE_KEY)
      ]
    : undefined

  return {
    activePayload,
    activeTable,
    activeTableSpec,
    evidenceError: selectedRun ? evidenceError : '',
    loadingEvidence: selectedRun ? loadingEvidence || !runStateMatchesSelectedRun : false,
    loadingTablePreview,
    latestTradabilityDiagnostic,
    runState: exposedRunState,
    chartTables,
    combinationSummaries: exposedRunState.combinationSummaries,
    display: exposedRunState.display,
    selectedCombinationId,
    selectedTableSpecs,
    setActiveTable,
    setEvidenceError,
    setSelectedCombinationId,
    setTablePreviewExpanded,
    tablePreviewExpanded,
  }
}

async function latestDiagnosticPreview(
  spec: { previewUrl: string; supportsOffset?: boolean },
  previewPayload: TablePayload,
  selectedCombinationId: string,
): Promise<TablePayload> {
  if (previewPayload.rows.length === 0) {
    return previewPayload
  }
  const totalRows = previewPayload.total_rows
  if (
    spec.supportsOffset &&
    typeof totalRows === 'number' &&
    previewPayload.total_rows_exact === true &&
    totalRows > previewPayload.rows.length
  ) {
    return fetchTablePreview(
      spec.previewUrl,
      1,
      selectedCombinationId,
      totalRows - 1,
    )
  }
  if (previewPayload.has_more && previewPayload.total_rows_exact === false) {
    return emptyRowsPayload(previewPayload)
  }
  return singleRowPayload(
    previewPayload,
    previewPayload.rows[previewPayload.rows.length - 1],
  )
}

function emptyRowsPayload(payload: TablePayload): TablePayload {
  return {
    ...payload,
    has_more: false,
    rows: [],
    total_rows: 0,
  }
}

function singleRowPayload(
  payload: TablePayload,
  row: Record<string, unknown>,
): TablePayload {
  return {
    ...payload,
    has_more: false,
    limit: 1,
    rows: [row],
    total_rows: 1,
  }
}

function latestPreviewKey(prefix: string, tableKey: string): string {
  return `${prefix}::${tableKey}${LATEST_PREVIEW_SUFFIX}`
}

function combinationPreviewLimit(
  spec: { key: string; previewRows?: number; previewMaxRows?: number },
  chartTableSpecs: { key: string }[],
): number {
  const previewRows = spec.previewRows ?? 20
  if (chartTableSpecs.some((chartSpec) => chartSpec.key === spec.key)) {
    return Math.min(previewRows, spec.previewMaxRows ?? previewRows)
  }
  return previewRows
}
