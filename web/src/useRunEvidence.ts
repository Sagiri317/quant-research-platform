import { useEffect, useMemo, useRef, useState } from 'react'
import {
  type RunSummary,
  fetchRunEvidence,
  runTableSpecsFromDisplay,
} from './api'
import { messageFrom } from './format'
import {
  EMPTY_RUN_STATE,
  buildRunStateFromEvidence,
  filterTablesByCombination,
  firstCombinationId,
  firstPreviewTableKey,
  type RunState,
} from './runEvidenceState'
import { useTablePreview } from './useTablePreview'

export function useRunEvidence(selectedRun: RunSummary | null) {
  const [runState, setRunState] = useState<RunState>(EMPTY_RUN_STATE)
  const [activeTable, setActiveTable] = useState('IC')
  const [selectedCombinationId, setSelectedCombinationId] = useState('')
  const [tablePreviewExpanded, setTablePreviewExpanded] = useState(false)
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
    const cacheKey = run.public_run_id
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
  }, [selectedRun])

  const selectedRunKey = selectedRun?.public_run_id ?? ''
  const runStateMatchesSelectedRun = selectedRunKey !== '' && runStateRunId === selectedRunKey
  const exposedRunState = runStateMatchesSelectedRun ? runState : EMPTY_RUN_STATE
  const allTableSpecs = useMemo(
    () =>
      selectedRun
        ? runTableSpecsFromDisplay(selectedRun, exposedRunState.display)
        : [],
    [exposedRunState.display, selectedRun],
  )
  const selectedTableSpecs = allTableSpecs
  const chartTables = useMemo(
    () =>
      filterTablesByCombination(
        exposedRunState.tables,
        selectedCombinationId,
        selectedTableSpecs,
      ),
    [exposedRunState.tables, selectedCombinationId, selectedTableSpecs],
  )
  const activeTableSpec = selectedTableSpecs.find(
    (spec) => spec.key === activeTable,
  )
  const baseActivePayload = chartTables[activeTable]

  const { loadingTablePreview, previewPayload } = useTablePreview({
    activePayload: baseActivePayload,
    activeTableSpec,
    expanded: tablePreviewExpanded,
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
