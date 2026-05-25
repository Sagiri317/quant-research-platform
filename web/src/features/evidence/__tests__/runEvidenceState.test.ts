import { describe, expect, test } from 'vitest'
import type { RunEvidence } from '../../../shared/api/api'
import {
  buildRunStateFromEvidence,
  filterTablesByCombination,
  firstCombinationId,
  firstPreviewTableKey,
} from '../runEvidenceState'
import { displayPayload, evidencePayload, tablePayload } from '../../../shared/testFixtures'

describe('run evidence state helpers', () => {
  test('builds first-view state from the evidence payload', () => {
    const tables = {
      combinations: tablePayload([{ combination_id: 'combo_a' }]),
      cumulative_returns: tablePayload([{ combination_id: 'combo_a', cumulative_net_return: 0.2 }]),
    }
    const evidence = evidencePayload({
      display: displayPayload(tables),
      reportMarkdown: '# Evidence',
      tables,
    }) as RunEvidence

    const state = buildRunStateFromEvidence(evidence)

    expect(state.detail?.run_id).toBe('demo')
    expect(state.reportMarkdown).toBe('# Evidence')
    expect(state.tables.cumulative_returns.rows[0].cumulative_net_return).toBe(0.2)
  })

  test('selects stable defaults for combination and table preview', () => {
    expect(
      firstCombinationId(
        tablePayload([
          { combination_id: 'combo_a' },
          { combination_id: 'combo_b' },
        ]),
      ),
    ).toBe('combo_a')
    expect(
      firstPreviewTableKey([
        { key: 'combinations' },
        { key: 'cumulative_returns' },
        { key: 'ic' },
      ]),
    ).toBe('cumulative_returns')
  })

  test('filters only tables that carry combination_id', () => {
    const tables = {
      cumulative_returns: tablePayload([
        { combination_id: 'combo_a', cumulative_net_return: 0.1 },
        { combination_id: 'combo_b', cumulative_net_return: 0.2 },
      ]),
      comparison: tablePayload([{ public_run_id: 'core/demo' }]),
    }

    const filtered = filterTablesByCombination(tables, 'combo_b')

    expect(filtered.cumulative_returns.rows).toEqual([
      { combination_id: 'combo_b', cumulative_net_return: 0.2 },
    ])
    expect(filtered.comparison.rows).toEqual([{ public_run_id: 'core/demo' }])
  })

  test('respects display tables that opt out of combination filtering', () => {
    const tables = {
      combinations: tablePayload([
        { combination_id: 'combo_a', feature: 'rev_20d' },
        { combination_id: 'combo_b', feature: 'rev_60d' },
      ]),
      cumulative_returns: tablePayload([
        { combination_id: 'combo_a', cumulative_net_return: 0.1 },
        { combination_id: 'combo_b', cumulative_net_return: 0.2 },
      ]),
    }

    const filtered = filterTablesByCombination(tables, 'combo_b', [
      { key: 'combinations', supportsCombinationFilter: false },
      { key: 'cumulative_returns', supportsCombinationFilter: true },
    ])

    expect(filtered.combinations.rows).toHaveLength(2)
    expect(filtered.cumulative_returns.rows).toEqual([
      { combination_id: 'combo_b', cumulative_net_return: 0.2 },
    ])
  })
})
