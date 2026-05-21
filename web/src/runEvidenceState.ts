import type {
  RunDetail,
  RunDisplay,
  RunEvidence,
  TablePayload,
} from './api'

export type RunState = {
  combinationSummaries: TablePayload | null
  detail: RunDetail | null
  display: RunDisplay | null
  reportMarkdown: string
  tables: Record<string, TablePayload>
  comparison: TablePayload | null
}

export const EMPTY_RUN_STATE: RunState = {
  combinationSummaries: null,
  detail: null,
  display: null,
  reportMarkdown: '',
  tables: {},
  comparison: null,
}

export function buildRunStateFromEvidence(evidence: RunEvidence): RunState {
  return {
    detail: evidence.summary,
    display: evidence.display,
    combinationSummaries: evidence.combination_summaries ?? null,
    reportMarkdown: evidence.report?.markdown ?? '',
    comparison: evidence.comparison ?? null,
    tables: evidence.tables ?? {},
  }
}

export function firstCombinationId(payload: TablePayload | undefined): string {
  const first = payload?.rows[0]?.combination_id
  return first == null ? '' : String(first)
}

export function firstPreviewTableKey(tableSpecs: { key: string }[]): string {
  return tableSpecs.find((spec) => spec.key !== 'combinations')?.key ?? ''
}

export function filterTablesByCombination(
  tables: Record<string, TablePayload>,
  combinationId: string,
  tableSpecs: { key: string; supportsCombinationFilter?: boolean }[] = [],
): Record<string, TablePayload> {
  if (!combinationId) {
    return tables
  }
  const filterableKeys = new Set(
    tableSpecs
      .filter((spec) => spec.supportsCombinationFilter !== false)
      .map((spec) => spec.key),
  )
  return Object.fromEntries(
    Object.entries(tables).map(([name, payload]) => {
      if (
        (tableSpecs.length > 0 && !filterableKeys.has(name)) ||
        !payload.columns.includes('combination_id')
      ) {
        return [name, payload]
      }
      return [
        name,
        {
          ...payload,
          rows: payload.rows.filter(
            (row) => row.combination_id === combinationId,
          ),
        },
      ]
    }),
  )
}
