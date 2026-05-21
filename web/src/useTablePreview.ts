import { useEffect, useState } from 'react'
import {
  type RunSummary,
  type RunTableSpec,
  type TablePayload,
  fetchTablePreview,
} from './api'
import { messageFrom } from './format'

const TABLE_PREVIEW_ROW_LIMIT = 20

export function useTablePreview({
  activePayload,
  activeTableSpec,
  expanded,
  selectedCombinationId,
  selectedRun,
  setEvidenceError,
}: {
  activePayload: TablePayload | undefined
  activeTableSpec: RunTableSpec | undefined
  expanded: boolean
  selectedCombinationId: string
  selectedRun: RunSummary | null
  setEvidenceError: (message: string) => void
}) {
  const [loadingTablePreview, setLoadingTablePreview] = useState(false)
  const [previewPayloads, setPreviewPayloads] = useState<Record<string, TablePayload>>({})
  const previewKey =
    selectedRun && activeTableSpec
      ? [
          selectedRun.public_run_id,
          activeTableSpec.key,
          activeTableSpec.supportsCombinationFilter ? selectedCombinationId : 'all',
        ].join('::')
      : ''
  const previewPayload = previewKey ? previewPayloads[previewKey] : undefined

  useEffect(() => {
    if (
      !selectedRun ||
      !expanded ||
      !activeTableSpec ||
      activePayload !== undefined ||
      previewPayload !== undefined
    ) {
      return
    }
    const tableSpec = activeTableSpec
    const combinationFilter = tableSpec.supportsCombinationFilter
      ? selectedCombinationId
      : undefined
    const cacheKey = previewKey
    let cancelled = false
    async function loadTablePreview() {
      try {
        setLoadingTablePreview(true)
        const payload = await fetchTablePreview(
          tableSpec.path,
          TABLE_PREVIEW_ROW_LIMIT,
          combinationFilter,
        )
        if (cancelled) {
          return
        }
        setPreviewPayloads((current) => ({ ...current, [cacheKey]: payload }))
        setEvidenceError('')
      } catch (caught) {
        if (!cancelled) {
          setEvidenceError(messageFrom(caught))
        }
      } finally {
        if (!cancelled) {
          setLoadingTablePreview(false)
        }
      }
    }
    void loadTablePreview()
    return () => {
      cancelled = true
    }
  }, [
    activePayload,
    activeTableSpec,
    expanded,
    previewKey,
    previewPayload,
    selectedCombinationId,
    selectedRun,
    setEvidenceError,
  ])

  return { loadingTablePreview, previewPayload }
}
