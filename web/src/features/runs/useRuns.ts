import { useCallback, useEffect, useMemo, useState } from 'react'
import { type RunSummary, fetchRuns } from '../../shared/api/api'
import { messageFrom } from '../../shared/format'

type RefreshRunsOptions = {
  fallbackPublicRunIds?: string[]
  selectPublicRunId?: string
}

function selectedPublicRunId(
  runs: RunSummary[],
  options: RefreshRunsOptions = {},
) {
  const candidates = [
    options.selectPublicRunId,
    ...(options.fallbackPublicRunIds ?? []),
  ].filter((id): id is string => Boolean(id))
  const matched = candidates.find((id) =>
    runs.some((run) => run.public_run_id === id),
  )
  return matched ?? runs[0]?.public_run_id ?? ''
}

export function useRuns() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [runListError, setRunListError] = useState('')

  const refreshRuns = useCallback(async (options: RefreshRunsOptions = {}) => {
    try {
      setLoadingRuns(true)
      const nextRuns = await fetchRuns()
      setRuns(nextRuns)
      setSelectedId(selectedPublicRunId(nextRuns, options))
      setRunListError('')
      return nextRuns
    } catch (caught) {
      setRunListError(messageFrom(caught))
      throw caught
    } finally {
      setLoadingRuns(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    async function loadRuns() {
      try {
        setLoadingRuns(true)
        const nextRuns = await fetchRuns()
        if (cancelled) {
          return
        }
        setRuns(nextRuns)
        setSelectedId(selectedPublicRunId(nextRuns))
        setRunListError('')
      } catch (caught) {
        if (!cancelled) {
          setRunListError(messageFrom(caught))
        }
      } finally {
        if (!cancelled) {
          setLoadingRuns(false)
        }
      }
    }
    void loadRuns()
    return () => {
      cancelled = true
    }
  }, [])

  const selectedRun = useMemo(
    () => runs.find((run) => run.public_run_id === selectedId) ?? null,
    [runs, selectedId],
  )

  function upsertRunAtTop(run: RunSummary) {
    setRuns((previous) => [
      run,
      ...previous.filter((item) => item.public_run_id !== run.public_run_id),
    ])
    setSelectedId(run.public_run_id)
  }

  function removeRun(publicRunId: string) {
    setRuns((previous) => {
      const nextRuns = previous.filter((run) => run.public_run_id !== publicRunId)
      setSelectedId(nextRuns[0]?.public_run_id ?? '')
      return nextRuns
    })
  }

  return {
    loadingRuns,
    refreshRuns,
    removeRun,
    runListError,
    runs,
    selectedId,
    selectedRun,
    setRunListError,
    setSelectedId,
    upsertRunAtTop,
  }
}
