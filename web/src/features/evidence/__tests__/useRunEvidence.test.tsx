import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, test, vi } from 'vitest'
import type { RunSummary } from '../../../shared/api/api'
import {
  displayPayload,
  evidencePayload,
  jsonResponse,
  tablePayload,
  tableResponse,
} from '../../../shared/testFixtures'
import { useRunEvidence } from '../useRunEvidence'

const originalFetch = globalThis.fetch

afterEach(() => {
  cleanup()
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
})

describe('useRunEvidence diagnostics previews', () => {
  test('fetches selected-combination tradability diagnostics from the preview API', async () => {
    const run: RunSummary = {
      data_stage: 'synthetic_only',
      domain: 'core_synthetic',
      kind: 'factor',
      mean_turnover: 0,
      n_rows_evaluated: 10,
      project: 'quant-research-platform',
      public_run_id: 'core/demo',
      run_id: 'demo',
      source_alias: 'core',
      total_cost: 0,
      uses_external_data: false,
      weight_method: 'equal_weight_quantile',
    }
    const tables = {
      combinations: tablePayload([
        { combination_id: 'combo_a' },
        { combination_id: 'combo_b' },
      ]),
      cumulative_returns: tablePayload([
        { combination_id: 'combo_a', timestamp: '2026-01-01', cumulative_net_return: 0.1 },
      ]),
      ic: tablePayload([
        { combination_id: 'combo_a', timestamp: '2026-01-01', ic: 0.2, rankic: 0.3 },
      ]),
      returns: tablePayload([]),
      tradability_diagnostics: tablePayload([
        {
          combination_id: 'combo_a',
          timestamp: '2026-01-01',
          eligible_rate: 0.9,
        },
      ]),
    }
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            display: displayPayload(tables),
            tables,
          }),
        )
      }
      if (
        url.includes('/api/runs/core/demo/diagnostics/tradability_diagnostics')
      ) {
        if (url.includes('offset=2')) {
          return tableResponse([
            {
              combination_id: 'combo_b',
              timestamp: '2026-01-04',
              eligible_rate: 0.2,
            },
          ])
        }
        return jsonResponse({
          ...tablePayload([
            {
              combination_id: 'combo_b',
              timestamp: '2026-01-02',
              eligible_rate: 0.5,
            },
          ]),
          total_rows: 3,
          offset: 0,
          limit: 20,
          has_more: true,
        })
      }
      return tableResponse([])
    }) as typeof fetch

    function EvidenceProbe() {
      const {
        chartTables,
        latestTradabilityDiagnostic,
        selectedCombinationId,
        setSelectedCombinationId,
      } = useRunEvidence(run)
      const diagnostics =
        chartTables.tradability_diagnostics?.rows[0]?.eligible_rate ?? 'missing'
      const latestDiagnostic =
        latestTradabilityDiagnostic?.rows[0]?.eligible_rate ?? 'missing'
      return (
        <>
          <div data-testid="selected">{selectedCombinationId}</div>
          <div data-testid="diagnostic-rate">{String(diagnostics)}</div>
          <div data-testid="latest-diagnostic-rate">{String(latestDiagnostic)}</div>
          <button onClick={() => setSelectedCombinationId('combo_b')} type="button">
            select combo b
          </button>
        </>
      )
    }

    render(<EvidenceProbe />)

    expect(await screen.findByTestId('selected')).toHaveTextContent('combo_a')
    fireEvent.click(screen.getByRole('button', { name: 'select combo b' }))

    await waitFor(() =>
      expect(screen.getByTestId('diagnostic-rate')).toHaveTextContent('0.5'),
    )
    await waitFor(() =>
      expect(screen.getByTestId('latest-diagnostic-rate')).toHaveTextContent(
        '0.2',
      ),
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/diagnostics/tradability_diagnostics?limit=20&combination_id=combo_b',
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/diagnostics/tradability_diagnostics?limit=1&offset=2&combination_id=combo_b',
    )
  })
})
