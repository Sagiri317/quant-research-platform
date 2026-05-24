import { useEffect, useMemo, useState } from 'react'
import { type RunSummary, fetchRuns } from '../../shared/api/api'
import { messageFrom } from '../../shared/format'

export function useRuns() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [runListError, setRunListError] = useState('')

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
        setSelectedId(nextRuns[0]?.public_run_id ?? '')
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
