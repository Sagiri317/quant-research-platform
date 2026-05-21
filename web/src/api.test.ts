import { afterEach, describe, expect, test, vi } from 'vitest'
import {
  fetchRuns,
  fetchTablePreview,
  createPipelineRun,
  getPipelineCheckpoints,
  getPipelineRunStatus,
  listFactorTemplates,
  listMarkets,
  runTableSpecsFromDisplay,
  selectedArtifactsDownloadUrl,
} from './api'

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
})

describe('api client', () => {
  test('reads market, factor template, and pipeline studio endpoints', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/markets')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: 'a_share', label: 'A-share', status: 'enabled' },
            { id: 'crypto', label: 'Crypto', status: 'planned' },
          ],
        } as Response)
      }
      if (url.endsWith('/api/factor-templates?market=a_share')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: 'reversal_20d',
              name: '20日反转',
              market: 'a_share',
              expression: 'neg(return(close, 20))',
              dependencies: ['close'],
            },
          ],
        } as Response)
      }
      if (url.endsWith('/api/pipeline-runs') && init?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ run_id: 'a_share_reversal_smoke', status: 'queued' }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-runs/a_share_reversal_smoke')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ run_id: 'a_share_reversal_smoke', status: 'ready' }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-runs/a_share_reversal_smoke/checkpoints')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { key: 'run_config', label: 'Run config', status: 'ready' },
          ],
        } as Response)
      }
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) } as Response)
    }) as typeof fetch

    await expect(listMarkets()).resolves.toHaveLength(2)
    await expect(listFactorTemplates('a_share')).resolves.toHaveLength(1)
    await expect(
      createPipelineRun({
        cost_bps: 10,
        date_end: '2025-12-31',
        date_start: '2024-01-01',
        factor_expression: 'neg(return(close, 20))',
        label_columns: ['forward_return_20d'],
        market: 'a_share',
        pnl_return_column: 'forward_return_1d',
        portfolio_mode: 'long_only',
        run_id: 'a_share_reversal_smoke',
      }),
    ).resolves.toMatchObject({ status: 'queued' })
    await expect(getPipelineRunStatus('a_share_reversal_smoke')).resolves.toMatchObject({
      status: 'ready',
    })
    await expect(getPipelineCheckpoints('a_share_reversal_smoke')).resolves.toHaveLength(1)
  })

  test('fetchRuns reads the run index from the backend', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        {
          source_alias: 'core',
          run_id: 'demo',
          public_run_id: 'core/demo',
          project: 'quant-research-platform',
          kind: 'factor',
          domain: 'core_synthetic',
          data_stage: 'synthetic_only',
          uses_external_data: false,
          n_rows_evaluated: 600,
          weight_method: 'equal_weight_quantile',
          mean_final_return: 1.2,
          mean_turnover: 1.5,
          total_cost: 0.01,
        },
      ],
    } as Response)

    const runs = await fetchRuns()

    expect(globalThis.fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/runs')
    expect(runs[0].public_run_id).toBe('core/demo')
  })

  test('fetchTablePreview throws a readable error when the backend rejects a request', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'summary.json is missing required fields' }),
    } as Response)

    await expect(fetchTablePreview('/api/comparison', 20)).rejects.toThrow(
      'summary.json is missing required fields',
    )
  })

  test('fetchTablePreview can request a selected combination before limiting', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ columns: [], rows: [] }),
    } as Response)

    await fetchTablePreview('/api/runs/core/demo/metrics/ic', 20, 'combo_b')

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/metrics/ic?limit=20&combination_id=combo_b',
    )
  })

  test('runTableSpecsFromDisplay maps top-level combinations table to preview JSON route', () => {
    const specs = runTableSpecsFromDisplay(
      {
        source_alias: 'core',
        run_id: 'demo',
        public_run_id: 'core/demo',
        project: 'quant-research-platform',
        kind: 'factor',
        domain: 'core_synthetic',
        data_stage: 'synthetic_only',
        uses_external_data: false,
        n_rows_evaluated: 600,
        weight_method: 'equal_weight_quantile',
        mean_turnover: 1,
        total_cost: 0,
      },
      {
        schema_version: '0.5',
        charts: [],
        downloads: { items: [], presets: [] },
        tables: [
          {
            download_item: 'combinations.csv',
            group: 'setup',
            key: 'combinations',
            label: '组合配置',
            path: 'combinations.csv',
            supports_combination_filter: false,
          },
        ],
      },
    )

    expect(specs[0].path).toBe('/api/runs/core/demo/metrics/combinations')
    expect(specs[0].supportsCombinationFilter).toBe(false)
  })

  test('selectedArtifactsDownloadUrl uses query items for native browser downloads', () => {
    const url = selectedArtifactsDownloadUrl(
      {
        source_alias: 'core',
        run_id: 'demo',
        public_run_id: 'core/demo',
        project: 'quant-research-platform',
        kind: 'factor',
        domain: 'core_synthetic',
        data_stage: 'synthetic_only',
        uses_external_data: false,
        n_rows_evaluated: 600,
        weight_method: 'equal_weight_quantile',
        mean_turnover: 1,
        total_cost: 0,
      },
      ['report.md', 'metrics/ic.csv'],
    )

    expect(url).toBe(
      'http://127.0.0.1:8000/api/runs/core/demo/download?items=report.md&items=metrics%2Fic.csv',
    )
  })
})
