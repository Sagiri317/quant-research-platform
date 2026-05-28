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
  test('derives the primary cumulative return chart from the returns preview API', async () => {
    const run: RunSummary = {
      data_stage: 'synthetic_only',
      domain: 'core_synthetic',
      kind: 'factor',
      mean_turnover: 0,
      n_rows_evaluated: 10,
      project: 'quant-research-platform',
      public_run_id: 'core/chart_demo',
      run_id: 'chart_demo',
      source_alias: 'core',
      total_cost: 0,
      uses_external_data: false,
      weight_method: 'equal_weight_quantile',
    }
    const tables = {
      combinations: tablePayload([
        {
          combination_id: 'combo_a',
          feature: 'rev_20d',
          label: 'forward_return',
          kernel: 'dataframe',
          portfolio_mode: 'long_only',
        },
        {
          combination_id: 'combo_b',
          feature: 'rev_60d',
          label: 'forward_return',
          kernel: 'dataframe',
          portfolio_mode: 'long_only',
        },
      ]),
      ic: tablePayload([]),
      returns: tablePayload([]),
    }
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/chart_demo/evidence')) {
        const display = displayPayload(tables, { runId: 'chart_demo' })
        const returns = display.tables.find((table) => table.key === 'returns')
        if (returns?.preview) {
          returns.preview.max_limit = 200
        }
        return jsonResponse(
          evidencePayload({
            display,
            tables,
          }),
        )
      }
      if (
        url.includes('/api/runs/core/chart_demo/tables/returns/preview')
      ) {
        return tableResponse([
          {
            combination_id: 'combo_b',
            timestamp: '2026-01-01',
            net_return: 0.2,
          },
          {
            combination_id: 'combo_b',
            timestamp: '2026-01-02',
            net_return: 0.25,
          },
        ])
      }
      return tableResponse([])
    }) as typeof fetch

    function EvidenceProbe() {
      const {
        chartTables,
        selectedCombinationId,
        setSelectedCombinationId,
      } = useRunEvidence(run)
      return (
        <>
          <div data-testid="selected">{selectedCombinationId}</div>
          <div data-testid="chart-rows">
            {chartTables.primary_cumulative_return?.rows.length ?? 0}
          </div>
          <div data-testid="last-cumulative">
            {String(
              chartTables.primary_cumulative_return?.rows[1]
                ?.cumulative_net_return ?? 'missing',
            )}
          </div>
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
      expect(screen.getByTestId('chart-rows')).toHaveTextContent('2'),
    )
    expect(screen.getByTestId('last-cumulative')).toHaveTextContent('0.5')
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/chart_demo/tables/returns/preview?limit=200&combination_id=combo_b',
    )
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      expect.stringContaining(`/tables/${'cumulative' + '_returns'}/preview`),
    )
  })

  test('paginates chart previews until the selected combination chart is complete', async () => {
    const run: RunSummary = {
      data_stage: 'synthetic_only',
      domain: 'core_synthetic',
      kind: 'factor',
      mean_turnover: 0,
      n_rows_evaluated: 10,
      project: 'quant-research-platform',
      public_run_id: 'core/paged_chart_demo',
      run_id: 'paged_chart_demo',
      source_alias: 'core',
      total_cost: 0,
      uses_external_data: false,
      weight_method: 'equal_weight_quantile',
    }
    const tables = {
      combinations: tablePayload([{ combination_id: 'combo_a' }]),
      ic: tablePayload([]),
      returns: tablePayload([]),
    }
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/paged_chart_demo/evidence')) {
        const display = displayPayload(tables, { runId: 'paged_chart_demo' })
        display.charts = display.charts.filter(
          (chart) => chart.id === 'primary_cumulative_return',
        )
        const returns = display.tables.find((table) => table.key === 'returns')
        if (returns?.preview) {
          returns.preview.max_limit = 1
        }
        return jsonResponse(evidencePayload({ display, tables }))
      }
      if (
        url.includes(
          '/api/runs/core/paged_chart_demo/tables/returns/preview',
        )
      ) {
        if (url.includes('offset=1')) {
          return jsonResponse({
            ...tablePayload([
              {
                combination_id: 'combo_a',
                timestamp: '2026-01-02',
                net_return: 0.1,
              },
            ]),
            has_more: false,
            limit: 1,
            offset: 1,
            total_rows: 2,
            total_rows_exact: false,
          })
        }
        return jsonResponse({
          ...tablePayload([
            {
              combination_id: 'combo_a',
              timestamp: '2026-01-01',
              net_return: 0.1,
            },
          ]),
          has_more: true,
          limit: 1,
          offset: 0,
          total_rows: 2,
          total_rows_exact: false,
        })
      }
      return tableResponse([])
    }) as typeof fetch

    function EvidenceProbe() {
      const { chartTables } = useRunEvidence(run)
      return (
        <div data-testid="chart-rows">
          {chartTables.primary_cumulative_return?.rows.length ?? 0}
        </div>
      )
    }

    render(<EvidenceProbe />)

    await waitFor(() =>
      expect(screen.getByTestId('chart-rows')).toHaveTextContent('2'),
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/paged_chart_demo/tables/returns/preview?limit=1&combination_id=combo_a',
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/paged_chart_demo/tables/returns/preview?limit=1&offset=1&combination_id=combo_a',
    )
  })

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
        url.includes(
          '/api/runs/core/demo/tables/tradability_diagnostics/preview',
        )
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
          total_rows_exact: true,
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
      'http://127.0.0.1:8000/api/runs/core/demo/tables/tradability_diagnostics/preview?limit=20&combination_id=combo_b',
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/tables/tradability_diagnostics/preview?limit=1&offset=2&combination_id=combo_b',
    )
  })

  test('does not infer latest diagnostics from a non-exact preview row count', async () => {
    const run: RunSummary = {
      data_stage: 'synthetic_only',
      domain: 'core_synthetic',
      kind: 'factor',
      mean_turnover: 0,
      n_rows_evaluated: 10,
      project: 'quant-research-platform',
      public_run_id: 'core/non_exact_demo',
      run_id: 'non_exact_demo',
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
      ic: tablePayload([]),
      returns: tablePayload([]),
      tradability_diagnostics: tablePayload([]),
    }
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/non_exact_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            display: displayPayload(tables, { runId: 'non_exact_demo' }),
            tables,
          }),
        )
      }
      if (
        url.includes(
          '/api/runs/core/non_exact_demo/tables/tradability_diagnostics/preview',
        )
      ) {
        if (url.includes('offset=20')) {
          return tableResponse([
            {
              combination_id: 'combo_b',
              timestamp: '2026-01-21',
              eligible_rate: 0.2,
            },
          ])
        }
        return jsonResponse({
          ...tablePayload([
            {
              combination_id: 'combo_b',
              timestamp: '2026-01-20',
              eligible_rate: 0.5,
            },
          ]),
          total_rows: 21,
          total_rows_exact: false,
          offset: 0,
          limit: 20,
          has_more: true,
        })
      }
      return tableResponse([])
    }) as typeof fetch

    function EvidenceProbe() {
      const {
        latestTradabilityDiagnostic,
        selectedCombinationId,
        setSelectedCombinationId,
      } = useRunEvidence(run)
      const latestDiagnostic =
        latestTradabilityDiagnostic?.rows[0]?.eligible_rate ?? 'missing'
      return (
        <>
          <div data-testid="selected">{selectedCombinationId}</div>
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
      expect(screen.getByTestId('latest-diagnostic-rate')).toHaveTextContent(
        'missing',
      ),
    )
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/non_exact_demo/tables/tradability_diagnostics/preview?limit=1&offset=20&combination_id=combo_b',
    )
  })
})
