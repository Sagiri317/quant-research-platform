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

const CHART_TABLE_ROW_LIMIT = 1000

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
      chartTableSpecs.length === 0
    ) {
      return
    }
    const filterableChartSpecs = chartTableSpecs.filter(
      (spec) => spec.supportsCombinationFilter !== false,
    )
    const missingSpecs = filterableChartSpecs.filter(
      (spec) => chartPreviewPayloads[`${chartPreviewPrefix}::${spec.key}`] === undefined,
    )
    if (missingSpecs.length === 0) {
      return
    }
    let cancelled = false
    async function loadSelectedChartTables() {
      try {
        const payloads = await Promise.all(
          missingSpecs.map(async (spec) => {
            const payload = await fetchTablePreview(
              spec.path,
              Math.max(spec.previewRows ?? 20, CHART_TABLE_ROW_LIMIT),
              selectedCombinationId,
            )
            return [spec.key, payload] as const
          }),
        )
        if (cancelled) {
          return
        }
        setChartPreviewPayloads((current) => {
          const next = { ...current }
          for (const [key, payload] of payloads) {
            next[`${chartPreviewPrefix}::${key}`] = payload
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
    for (const spec of chartTableSpecs) {
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
    chartTableSpecs,
    filteredTables,
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

  return {
    activePayload,
    activeTable,
    activeTableSpec,
    evidenceError: selectedRun ? evidenceError : '',
    loadingEvidence: selectedRun ? loadingEvidence || !runStateMatchesSelectedRun : false,
    loadingTablePreview,
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
