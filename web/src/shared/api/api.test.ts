import { afterEach, describe, expect, test, vi } from 'vitest'
import {
  fetchRuns,
  fetchTablePreview,
  fetchFactorFactoryRun,
  fetchFactorFactoryTable,
  factorFactoryArtifactDownloadUrl,
  createPipelineRun,
  createPipelineSnapshot,
  getPipelineRunCheckpoints,
  getPipelineRunStatus,
  getPipelineSnapshotCheckpoints,
  getPipelineSnapshotStatus,
  listFactorFactoryRuns,
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
    let snapshotRequest: Record<string, unknown> | undefined
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
      if (url.endsWith('/api/pipeline-snapshots') && init?.method === 'POST') {
        snapshotRequest = JSON.parse(String(init.body))
        return Promise.resolve({
          ok: true,
          json: async () => ({
            kind: 'snapshot',
            run_id: 'a_share_reversal_smoke',
            run_ref: 'a_share_reversal_smoke_20260523',
            status: 'queued',
          }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-snapshots/a_share_reversal_smoke')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            kind: 'snapshot',
            run_id: 'a_share_reversal_smoke',
            run_ref: 'a_share_reversal_smoke_20260523',
            status: 'ready',
          }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-snapshots/a_share_reversal_smoke/checkpoints')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              key: 'run_config',
              label: 'Run config',
              status: 'ready',
              artifact: 'run_config.yaml',
            },
          ],
        } as Response)
      }
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) } as Response)
    }) as typeof fetch

    await expect(listMarkets()).resolves.toHaveLength(2)
    await expect(listFactorTemplates('a_share')).resolves.toHaveLength(1)
    await expect(
      createPipelineSnapshot({
        cost_bps: 10,
        date_end: '2025-12-31',
        date_start: '2024-01-01',
        factor_expression: 'neg(return(close, 20))',
        kernel: 'dataframe',
        label_columns: ['forward_return_20d'],
        market: 'a_share',
        return_source_column: 'forward_return_1d',
        portfolio_mode: 'long_only',
        run_id: 'a_share_reversal_smoke',
      }),
    ).resolves.toMatchObject({
      kind: 'snapshot',
      run_ref: 'a_share_reversal_smoke_20260523',
      status: 'queued',
    })
    expect(snapshotRequest).toMatchObject({
      kernel: 'dataframe',
      return_source_column: 'forward_return_1d',
    })
    expect(snapshotRequest).not.toHaveProperty('engine')
    expect(snapshotRequest).not.toHaveProperty('input_file')
    expect(snapshotRequest).not.toHaveProperty('pnl_return_column')
    await expect(
      getPipelineSnapshotStatus('a_share_reversal_smoke'),
    ).resolves.toMatchObject({
      kind: 'snapshot',
      status: 'ready',
    })
    await expect(
      getPipelineSnapshotCheckpoints('a_share_reversal_smoke'),
    ).resolves.toMatchObject([
      { artifact: 'run_config.yaml' },
    ])
  })

  test('pipeline run helpers use executable-run endpoints', async () => {
    let runRequest: Record<string, unknown> | undefined
    globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/pipeline-runs') && init?.method === 'POST') {
        runRequest = JSON.parse(String(init.body))
        return Promise.resolve({
          ok: true,
          json: async () => ({
            kind: 'run',
            run_id: 'a_share_runner',
            run_ref: 'a_share_runner',
            status: 'ready',
          }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-runs/a_share_runner')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            kind: 'run',
            run_id: 'a_share_runner',
            status: 'ready',
          }),
        } as Response)
      }
      if (url.endsWith('/api/pipeline-runs/a_share_runner/checkpoints')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              key: 'evaluate',
              label: 'A-share evaluation',
              status: 'ready',
              artifact: 'evaluation/manifest.json',
            },
          ],
        } as Response)
      }
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) } as Response)
    }) as typeof fetch

    await expect(
      createPipelineRun({
        cost_bps: 10,
        date_end: '2025-12-31',
        date_start: '2024-01-01',
        factor_expression: 'rev_20d',
        kernel: 'tensor',
        label_columns: ['forward_return_20d'],
        market: 'a_share',
        return_source_column: 'forward_return_1d',
        portfolio_mode: 'long_only',
        run_id: 'a_share_runner',
        start_from: 'existing_factors',
      }),
    ).resolves.toMatchObject({
      kind: 'run',
      status: 'ready',
    })
    expect(runRequest).toMatchObject({
      kernel: 'tensor',
      return_source_column: 'forward_return_1d',
      start_from: 'existing_factors',
    })
    expect(runRequest).not.toHaveProperty('input_file')
    await expect(getPipelineRunStatus('a_share_runner')).resolves.toMatchObject({
      kind: 'run',
      status: 'ready',
    })
    await expect(getPipelineRunCheckpoints('a_share_runner')).resolves.toMatchObject([
      { artifact: 'evaluation/manifest.json' },
    ])
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/pipeline-runs/a_share_runner/checkpoints',
    )
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

  test('reads Factor Factory runs, detail, table preview, and downloads', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/factor-factory-runs')) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              artifact_schema_version: '0.1',
              artifact_type: 'factor_factory_run',
              candidate_count: 2,
              failed_count: 1,
              run_id: 'factory_demo',
              success_count: 1,
            },
          ],
        } as Response)
      }
      if (url.endsWith('/api/factor-factory-runs/factory_demo')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            display: {
              schema_version: '0.1',
              downloads: {
                items: [
                  {
                    item: 'candidate_summary.csv',
                    kind: 'table',
                    label: 'Candidate summary',
                    table: 'candidate_summary',
                  },
                ],
                presets: [],
              },
              tables: [
                {
                  download_item: 'candidate_summary.csv',
                  group: 'candidate',
                  key: 'candidate_summary',
                  label: 'Candidate summary',
                  path: 'candidate_summary.csv',
                },
              ],
            },
            run: {
              artifact_schema_version: '0.1',
              artifact_type: 'factor_factory_run',
              candidate_count: 2,
              failed_count: 1,
              run_id: 'factory_demo',
              success_count: 1,
            },
          }),
        } as Response)
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_demo/tables/candidate_summary?limit=20&offset=40',
        )
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            columns: ['candidate_id', 'status', 'run_ref'],
            rows: [
              {
                candidate_id: 'reversal_20d@1',
                run_ref: 'pipeline/reversal_20d',
                status: 'success',
              },
            ],
          }),
        } as Response)
      }
      return Promise.resolve({ ok: false, status: 404, json: async () => ({}) } as Response)
    }) as typeof fetch

    await expect(listFactorFactoryRuns()).resolves.toMatchObject([
      { run_id: 'factory_demo', success_count: 1 },
    ])
    await expect(fetchFactorFactoryRun('factory_demo')).resolves.toMatchObject({
      run: { candidate_count: 2 },
      display: { schema_version: '0.1' },
    })
    await expect(
      fetchFactorFactoryTable('factory_demo', 'candidate_summary', 20, 40),
    ).resolves.toMatchObject({
      rows: [{ candidate_id: 'reversal_20d@1' }],
    })
    expect(
      factorFactoryArtifactDownloadUrl('factory_demo', 'candidate_summary.csv'),
    ).toBe(
      'http://127.0.0.1:8000/api/factor-factory-runs/factory_demo/artifacts/candidate_summary.csv',
    )
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

  test('fetchTablePreview can request an offset page', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ columns: [], rows: [] }),
    } as Response)

    await fetchTablePreview('/api/runs/core/demo/diagnostics/tradability_diagnostics', 1, 'combo_b', 129)

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/diagnostics/tradability_diagnostics?limit=1&offset=129&combination_id=combo_b',
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
        schema_version: '0.6',
        charts: [],
        downloads: { items: [], presets: [] },
        tables: [
          {
            download_item: 'dimensions/combinations.csv',
            group: 'setup',
            key: 'combinations',
            label: '组合配置',
            path: 'dimensions/combinations.csv',
            supports_combination_filter: false,
          },
        ],
      },
    )

    expect(specs[0].path).toBe('/api/runs/core/demo/metrics/combinations')
    expect(specs[0].supportsCombinationFilter).toBe(false)
  })

  test('runTableSpecsFromDisplay maps breakdown tables to preview JSON routes', () => {
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
        schema_version: '0.6',
        charts: [],
        downloads: { items: [], presets: [] },
        tables: [
          {
            download_item: 'breakdowns/year/performance_summary.csv',
            group: 'breakdown',
            key: 'performance_by_year',
            label: '按年绩效',
            path: 'breakdowns/year/performance_summary.csv',
            supports_combination_filter: true,
          },
        ],
      },
    )

    expect(specs[0].path).toBe(
      '/api/runs/core/demo/breakdowns/year/performance_summary',
    )
  })

  test('runTableSpecsFromDisplay rejects unsupported table path shapes', () => {
    expect(() =>
      runTableSpecsFromDisplay(
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
          schema_version: '0.6',
          charts: [],
          downloads: { items: [], presets: [] },
          tables: [
            {
              download_item: 'custom/alpha.csv',
              group: 'custom',
              key: 'custom_alpha',
              label: '自定义 Alpha',
              path: 'custom/alpha.csv',
              supports_combination_filter: true,
            },
          ],
        },
      ),
    ).toThrow('Unsupported display table path')
  })

  test('fetch helpers stringify structured FastAPI validation details', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        detail: {
          blocking: [
            {
              code: 'invalid_factor_expression',
              message: 'unsupported expression',
            },
          ],
        },
      }),
    } as Response)

    await expect(getPipelineSnapshotStatus('bad')).rejects.toThrow(
      'invalid_factor_expression: unsupported expression',
    )
  })

  test('fetch helpers stringify FastAPI list validation details', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [
          {
            loc: ['body', 'run_id'],
            msg: 'Field required',
            type: 'missing',
          },
        ],
      }),
    } as Response)

    await expect(getPipelineSnapshotStatus('bad')).rejects.toThrow(
      'body.run_id: Field required',
    )
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
