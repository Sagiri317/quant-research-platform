import '@testing-library/jest-dom/vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import { afterEach, describe, expect, test, vi } from 'vitest'
import App from './App'
import { FactorFactoryRuns } from '../features/factor-factory/FactorFactoryRuns'
import type { RunSummary } from '../shared/api/api'
import {
  displayPayload,
  evidencePayload,
  jsonResponse,
  tablePayload,
  tableResponse,
} from '../shared/testFixtures'
import { useRunEvidence } from '../features/evidence/useRunEvidence'

const originalFetch = globalThis.fetch

afterEach(() => {
  cleanup()
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
})

describe('App', () => {
  test('surfaces Factor Studio, Pipeline Builder, and Data Checkpoints as the platform workflow', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByText('量化研究平台')).toBeInTheDocument()
    expect(screen.getByText('研究证据工作台')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '因子工作台' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByRole('button', { name: '因子工厂' })).toBeInTheDocument()
    expect(screen.getByText('A 股')).toBeInTheDocument()
    expect(screen.getByText('加密资产预留')).toBeInTheDocument()
    expect(screen.getAllByText('neg(return(close, 20))').length).toBeGreaterThan(0)
    expect(screen.queryByText('新建回测')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '流程配置' }))
    expect(screen.getByLabelText('运行编号')).toHaveValue('a_share_reversal_smoke')
    expect(screen.getByText('pipeline/(pending)')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '数据检查点' }))
    expect(screen.getByText('pipeline/a_share_reversal_smoke')).toBeInTheDocument()
    expect(screen.getByText('run_config.yaml')).toBeInTheDocument()
    expect(screen.getByText('评估清单')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '证据工作台' }))
    expect(screen.getAllByText('暂无可用运行').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'CSV 上传（开发）' }))
    expect(screen.getByText('开发用 CSV 上传')).toBeInTheDocument()
  })

  test('renders Factor Factory runs without leaking private filesystem paths', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/factor-factory-runs')) {
        return jsonResponse([
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 2,
            failed_count: 1,
            run_id: 'factory_demo',
            success_count: 1,
          },
        ])
      }
      if (url.endsWith('/api/factor-factory-runs/factory_demo')) {
        return jsonResponse({
          display: {
            schema_version: '0.1',
            downloads: {
              items: [
                {
                  item: 'candidate_summary.csv',
                  kind: 'table',
                  label: 'Candidate summary CSV',
                  table: 'candidate_summary',
                },
              ],
              presets: [
                {
                  items: ['candidate_summary.csv'],
                  key: 'all',
                  label: 'All artifacts',
                },
              ],
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
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_demo/tables/candidate_summary?limit=50&offset=0',
        )
      ) {
        return jsonResponse({
          columns: ['candidate_id', 'status', 'run_ref', 'failure_reason'],
          rows: [
            {
              candidate_id: 'reversal_20d@1',
              failure_reason: '',
              run_ref: 'pipeline/reversal_20d',
              status: 'success',
            },
            {
              candidate_id: 'momentum_60d@1',
              failure_reason: 'invalid expression',
              run_ref: '',
              status: 'failed',
            },
          ],
          total_rows: 2,
        })
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: '因子工厂' }))

    expect(await screen.findByText('Factor Factory')).toBeInTheDocument()
    expect((await screen.findAllByText('factory_demo')).length).toBeGreaterThan(0)
    expect(await screen.findByText('reversal_20d@1')).toBeInTheDocument()
    expect(screen.getByText('momentum_60d@1')).toBeInTheDocument()
    expect(screen.getByText('pipeline/reversal_20d')).toBeInTheDocument()
    expect(screen.getAllByText('Candidate summary CSV')).toHaveLength(2)
    expect(screen.queryByText('candidate_summary.csv')).not.toBeInTheDocument()
    expect(screen.queryByText(/reports\/private_real/)).not.toBeInTheDocument()
    expect(screen.queryByText(/E:\\/)).not.toBeInTheDocument()
  })

  test('keeps Factor Factory detail and downloads visible when table preview fails', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/factor-factory-runs')) {
        return jsonResponse([
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 2,
            failed_count: 0,
            run_id: 'factory_preview_error',
            success_count: 2,
          },
        ])
      }
      if (url.endsWith('/api/factor-factory-runs/factory_preview_error')) {
        return jsonResponse({
          display: {
            schema_version: '0.1',
            downloads: {
              items: [
                {
                  item: 'candidate_summary.csv',
                  kind: 'table',
                  label: 'Candidate summary CSV',
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
            failed_count: 0,
            run_id: 'factory_preview_error',
            success_count: 2,
          },
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_preview_error/tables/candidate_summary?limit=50&offset=0',
        )
      ) {
        return Promise.resolve({
          ok: false,
          status: 422,
          json: async () => ({ detail: 'preview table is unavailable' }),
        } as Response)
      }
      return jsonResponse({ columns: [], rows: [] })
    }) as typeof fetch

    render(<FactorFactoryRuns />)

    expect(await screen.findByRole('heading', { name: 'factory_preview_error' })).toBeInTheDocument()
    expect(
      await screen.findByRole('link', { name: 'Candidate summary CSV' }),
    ).toHaveAttribute(
      'href',
      'http://127.0.0.1:8000/api/factor-factory-runs/factory_preview_error/artifacts/candidate_summary.csv',
    )
    expect(await screen.findByText('preview table is unavailable')).toBeInTheDocument()
    expect(screen.getByText('Table preview is unavailable. Downloads remain available.')).toBeInTheDocument()
  })

  test('chooses a preview table from display tables and renders Factor Factory preset downloads', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/factor-factory-runs')) {
        return jsonResponse([
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 1,
            failed_count: 0,
            run_id: 'factory_display_driven',
            success_count: 1,
          },
        ])
      }
      if (url.endsWith('/api/factor-factory-runs/factory_display_driven')) {
        return jsonResponse({
          display: {
            schema_version: '0.1',
            downloads: {
              items: [
                {
                  item: 'analysis_summary.csv',
                  kind: 'table',
                  label: 'Analysis summary CSV',
                  table: 'analysis_summary',
                },
                {
                  item: 'manifest.json',
                  kind: 'json',
                  label: 'Manifest JSON',
                },
              ],
              presets: [
                {
                  key: 'review_bundle',
                  label: 'Review bundle',
                  items: ['analysis_summary.csv', 'manifest.json'],
                },
              ],
            },
            tables: [
              {
                download_item: 'analysis_summary.csv',
                group: 'candidate',
                key: 'analysis_summary',
                label: 'Analysis summary',
                path: 'analysis_summary.csv',
              },
            ],
          },
          run: {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 1,
            failed_count: 0,
            run_id: 'factory_display_driven',
            success_count: 1,
          },
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_display_driven/tables/analysis_summary?limit=50&offset=0',
        )
      ) {
        return jsonResponse({
          columns: ['candidate_id', 'status', 'score'],
          rows: [{ candidate_id: 'display_candidate', score: 0.4, status: 'success' }],
          total_rows: 1,
        })
      }
      if (url.includes('/tables/candidate_summary')) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ detail: 'candidate_summary should not be requested' }),
        } as Response)
      }
      return jsonResponse({ columns: [], rows: [] })
    }) as typeof fetch

    render(<FactorFactoryRuns />)

    expect(await screen.findByText('display_candidate')).toBeInTheDocument()
    expect(screen.getByText('score')).toBeInTheDocument()
    expect(screen.getByText('0.4')).toBeInTheDocument()
    expect(screen.getByText('Analysis summary')).toBeInTheDocument()
    expect(screen.getByText('Review bundle')).toBeInTheDocument()
    for (const link of screen.getAllByRole('link', {
      name: 'Analysis summary CSV',
    })) {
      expect(link).toHaveAttribute(
        'href',
        'http://127.0.0.1:8000/api/factor-factory-runs/factory_display_driven/artifacts/analysis_summary.csv',
      )
    }
    for (const link of screen.getAllByRole('link', { name: 'Manifest JSON' })) {
      expect(link).toHaveAttribute(
        'href',
        'http://127.0.0.1:8000/api/factor-factory-runs/factory_display_driven/artifacts/manifest.json',
      )
    }
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      expect.stringContaining('/tables/candidate_summary'),
    )
  })

  test('enables Factor Factory pagination from total rows when has_more is omitted', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/factor-factory-runs')) {
        return jsonResponse([
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 75,
            failed_count: 0,
            run_id: 'factory_total_rows',
            success_count: 75,
          },
        ])
      }
      if (url.endsWith('/api/factor-factory-runs/factory_total_rows')) {
        return jsonResponse({
          display: {
            schema_version: '0.1',
            downloads: { items: [], presets: [] },
            tables: [
              {
                group: 'candidate',
                key: 'analysis_summary',
                label: 'Analysis summary',
                path: 'analysis_summary.csv',
              },
            ],
          },
          run: {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 75,
            failed_count: 0,
            run_id: 'factory_total_rows',
            success_count: 75,
          },
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_total_rows/tables/analysis_summary?limit=50&offset=0',
        )
      ) {
        return jsonResponse({
          columns: ['candidate_id', 'score'],
          rows: [{ candidate_id: 'page_1_candidate', score: 0.1 }],
          total_rows: 75,
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_total_rows/tables/analysis_summary?limit=50&offset=50',
        )
      ) {
        return jsonResponse({
          columns: ['candidate_id', 'score'],
          rows: [{ candidate_id: 'page_2_candidate', score: 0.2 }],
          total_rows: 75,
        })
      }
      return jsonResponse({ columns: [], rows: [] })
    }) as typeof fetch

    render(<FactorFactoryRuns />)

    expect(await screen.findByText('page_1_candidate')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '下一页' }))

    expect(await screen.findByText('page_2_candidate')).toBeInTheDocument()
  })

  test('clears stale Factor Factory detail when switching to a failing run', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/factor-factory-runs')) {
        return jsonResponse([
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 1,
            failed_count: 0,
            run_id: 'factory_a',
            success_count: 1,
          },
          {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 1,
            failed_count: 1,
            run_id: 'factory_b',
            success_count: 0,
          },
        ])
      }
      if (url.endsWith('/api/factor-factory-runs/factory_a')) {
        return jsonResponse({
          display: {
            schema_version: '0.1',
            downloads: {
              items: [
                {
                  item: 'candidate_summary.csv',
                  kind: 'table',
                  label: 'Candidate summary A',
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
                label: 'Candidate summary A',
                path: 'candidate_summary.csv',
              },
            ],
          },
          run: {
            artifact_schema_version: '0.1',
            artifact_type: 'factor_factory_run',
            candidate_count: 1,
            failed_count: 0,
            run_id: 'factory_a',
            success_count: 1,
          },
        })
      }
      if (
        url.endsWith(
          '/api/factor-factory-runs/factory_a/tables/candidate_summary?limit=50&offset=0',
        )
      ) {
        return jsonResponse({
          columns: ['candidate_id', 'status', 'run_ref', 'failure_reason'],
          rows: [
            {
              candidate_id: 'candidate_a',
              failure_reason: '',
              run_ref: 'pipeline/a',
              status: 'success',
            },
          ],
        })
      }
      if (url.endsWith('/api/factor-factory-runs/factory_b')) {
        return Promise.resolve({
          ok: false,
          status: 422,
          json: async () => ({ detail: 'factory_b broken' }),
        } as Response)
      }
      return jsonResponse({ columns: [], rows: [] })
    }) as typeof fetch

    render(<FactorFactoryRuns />)

    expect(await screen.findByText('candidate_a')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /factory_b/ }))

    await waitFor(() => expect(screen.getByText('factory_b broken')).toBeInTheDocument())
    expect(screen.queryByText('candidate_a')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Candidate summary A' })).not.toBeInTheDocument()
  })

  test('passes the edited factor expression into the pipeline snapshot request', async () => {
    let pipelineRequest: Record<string, unknown> | undefined
    globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/factor-templates?market=a_share')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/pipeline-snapshots') && init?.method === 'POST') {
        pipelineRequest = JSON.parse(String(init.body))
        return jsonResponse({
          kind: 'snapshot',
          run_id: pipelineRequest?.run_id,
          run_ref: 'a_share_reversal_smoke_20260523',
          status: 'ready',
        })
      }
      if (
        url.endsWith(
          '/api/pipeline-snapshots/a_share_reversal_smoke_20260523/checkpoints',
        )
      ) {
        return jsonResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    const factorExpression = await screen.findByLabelText('因子表达式')
    fireEvent.change(factorExpression, {
      target: { value: 'return(close, 60)' },
    })
    fireEvent.click(screen.getByRole('button', { name: '进入流程配置' }))

    expect(screen.getByLabelText('因子表达式')).toHaveValue('return(close, 60)')

    fireEvent.click(screen.getByRole('button', { name: '生成快照' }))

    await waitFor(() => {
      expect(pipelineRequest).toMatchObject({
        factor_expression: 'return(close, 60)',
        factor_template_id: 'reversal_20d',
        kernel: 'dataframe',
      })
    })
    expect(pipelineRequest).not.toHaveProperty('input_file')
    expect(screen.getByText('pipeline/a_share_reversal_smoke_20260523')).toBeInTheDocument()
  })

  test('can execute an A-share pipeline run with selected kernel and start layer', async () => {
    let pipelineRequest: Record<string, unknown> | undefined
    globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/factor-templates?market=a_share')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/pipeline-runs') && init?.method === 'POST') {
        pipelineRequest = JSON.parse(String(init.body))
        return jsonResponse({
          kind: 'run',
          run_id: pipelineRequest?.run_id,
          run_ref: 'a_share_reversal_smoke',
          status: 'ready',
          message: 'pipeline run ready',
        })
      }
      if (
        url.endsWith('/api/pipeline-runs/a_share_reversal_smoke/checkpoints')
      ) {
        return jsonResponse([
          {
            key: 'evaluate',
            label: 'A-share evaluation',
            status: 'ready',
            artifact: 'evaluation/manifest.json',
          },
        ])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: '进入流程配置' }))
    fireEvent.change(screen.getByLabelText('提交方式'), {
      target: { value: 'run' },
    })
    fireEvent.change(screen.getByLabelText('计算内核'), {
      target: { value: 'tensor' },
    })
    fireEvent.change(screen.getByLabelText('起始步骤'), {
      target: { value: 'existing_factors' },
    })
    fireEvent.click(screen.getByRole('button', { name: '执行流程' }))

    await waitFor(() => {
      expect(pipelineRequest).toMatchObject({
        kernel: 'tensor',
        start_from: 'existing_factors',
        return_source_column: 'forward_return_1d',
      })
    })
    expect(pipelineRequest).not.toHaveProperty('input_file')
    expect(screen.getByText('evaluation/manifest.json')).toBeInTheDocument()
    expect(screen.getByText('pipeline run ready')).toBeInTheDocument()
  })

  test('defaults to an interview workbench with compact diagnostic charts and delivery tabs', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'workbench_demo',
            public_run_id: 'core/workbench_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'a_share',
            data_stage: 'approved_offline_real_data',
            uses_external_data: true,
            n_rows_evaluated: 3984077,
            label_columns: ['forward_return_20d'],
            return_source_column: 'forward_return_1d',
            n_combinations: 4,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.54,
            mean_turnover: 0.37,
            total_cost: 0.08,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/workbench_demo/evidence')) {
        return jsonResponse(evidencePayload({ comparisonRows: [] }))
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByText('研究证据工作台')).toBeInTheDocument()
    expect(
      await screen.findByText('可复现的因子评估工作台'),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '工作台' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByRole('button', { name: '完整报告' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    expect(await screen.findByTestId('cumulative-returns-chart')).toBeInTheDocument()
    expect(screen.getByTestId('ic-rankic-chart')).toBeInTheDocument()
    expect(await screen.findByTestId('net-portfolio-return-chart')).toBeInTheDocument()
    expect(screen.queryByTestId('turnover-chart')).not.toBeInTheDocument()
    expect(screen.queryByTestId('cost-chart')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '换手率' }))
    expect(screen.getByTestId('turnover-chart')).toBeInTheDocument()
    expect(screen.queryByTestId('net-portfolio-return-chart')).not.toBeInTheDocument()

    const delivery = screen.getByLabelText('数据与交付')
    expect(within(delivery).getByRole('tab', { name: '指标详情' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    expect(screen.queryByText('评估报告')).not.toBeInTheDocument()
    fireEvent.click(within(delivery).getByRole('tab', { name: '研究报告' }))
    expect(await screen.findByText('评估报告')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '完整报告' }))
    expect(screen.getByRole('button', { name: '完整报告' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(await screen.findByTestId('net-portfolio-return-chart')).toBeInTheDocument()
    expect(await screen.findByTestId('turnover-chart')).toBeInTheDocument()
    expect(await screen.findByTestId('cost-chart')).toBeInTheDocument()
    expect(screen.getByText('明细表预览')).toBeInTheDocument()
    expect(screen.getByText('结果文件下载')).toBeInTheDocument()
  })

  test('renders run evidence chain and CSV browser from API data', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
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
            label_columns: ['forward_return_20d'],
            return_source_column: 'forward_return_1d',
            weight_method: 'equal_weight_quantile',
            mean_final_return: 1.6099,
            mean_turnover: 1.54,
            total_cost: 0.077,
            sharpe_like: 1.2,
            sortino_like: 1.5,
            max_drawdown: -0.12,
            win_rate: 0.58,
            mean_ic: 0.53,
            mean_rankic: 0.51,
            icir: 2.1,
            rankicir: 1.9,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [
              {
                public_run_id: 'core/demo',
                n_rows_evaluated: 600,
                mean_ic: 0.53,
                mean_rankic: 0.51,
                mean_final_return: 1.6,
                mean_turnover: 1.54,
                total_cost: 0.077,
                sharpe_like: 1.2,
                max_drawdown: -0.12,
                rank_mean_final_return: 1,
              },
            ],
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([
          {
            public_run_id: 'core/demo',
            n_rows_evaluated: 600,
            mean_ic: 0.53,
            mean_rankic: 0.51,
            mean_final_return: 1.6,
            mean_turnover: 1.54,
            total_cost: 0.077,
            sharpe_like: 1.2,
            max_drawdown: -0.12,
            rank_mean_final_return: 1,
          },
        ])
      }
      if (url.includes('/metrics/ic_summary')) {
        return tableResponse([
          {
            feature: 'factor',
            mean_ic: 0.53,
            icir: 2.1,
            mean_rankic: 0.51,
            rankicir: 1.9,
          },
        ])
      }
      if (url.includes('/metrics/ic')) {
        return tableResponse(
          Array.from({ length: 30 }, (_, index) => ({
            timestamp: `2026-01-${String(index + 1).padStart(2, '0')}`,
            ic: 0.4,
            rankic: 0.5,
          })),
        )
      }
      if (url.includes('/portfolio/returns')) {
        return tableResponse([
          {
            timestamp: '2026-01-01',
            net_portfolio_return: 0.02,
            turnover: 1.1,
            cost: 0.001,
          },
        ])
      }
      if (url.includes('/portfolio/cumulative_returns')) {
        return tableResponse([
          {
            timestamp: '2026-01-01',
            net_portfolio_return: 0.02,
            cumulative_net_return: 0.02,
            drawdown: 0,
          },
        ])
      }
      if (url.includes('/metrics/performance_summary')) {
        return tableResponse([
          {
            feature: 'factor',
            mean_final_return: 0.02,
            sharpe_like: 1.2,
            max_drawdown: 0,
          },
        ])
      }
      if (url.includes('/metrics/group_spread_summary')) {
        return tableResponse([{ feature: 'factor', top_bottom_spread_mean: 0.02 }])
      }
      if (url.includes('/portfolio/exposure-summary')) {
        return tableResponse([{ feature: 'factor', mean_gross_exposure: 1 }])
      }
      return tableResponse([{ timestamp: '2026-01-01', value: 1 }])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByText('研究证据工作台')).toBeInTheDocument()
    expect((await screen.findAllByText('core/demo')).length).toBeGreaterThan(0)
    fireEvent.click(await screen.findByRole('button', { name: '完整报告' }))
    expect(
      await screen.findByText('合成本地结果文件 · 不连接外部 API'),
    ).toBeInTheDocument()
    expect(
      screen.getByText((_, element) => element?.textContent === '标签列：forward_return_20d'),
    ).toBeInTheDocument()
    expect(
      screen.getByText((_, element) => element?.textContent === '收益来源: forward_return_1d'),
    ).toBeInTheDocument()
    expect(screen.getByText('收益风险')).toBeInTheDocument()
    expect(screen.getAllByText('IC / RankIC').length).toBeGreaterThan(0)
    expect(screen.getByText('成本换手')).toBeInTheDocument()
    expect(screen.getByText('结果解读')).toBeInTheDocument()
    expect(screen.getByText('研究报告')).toBeInTheDocument()
    expect(await screen.findByText('评估报告')).toBeInTheDocument()
    expect(screen.getByText('报告由量化研究平台根据提供的因子或预测面板生成。')).toBeInTheDocument()
    expect(screen.getByText('指标表')).toBeInTheDocument()
    expect(screen.queryByText(/This report is generated/)).not.toBeInTheDocument()
    expect(screen.getByText('结果文件下载')).toBeInTheDocument()
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/evidence',
    )
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/metrics/ic',
    )
    expect(screen.getByText('在结果文件下载中选择 comparison.csv')).toBeInTheDocument()
    const comparisonTable = screen.getByRole('table', { name: '运行对比表格' })
    expect(within(comparisonTable).getByText('n_rows_evaluated')).toBeInTheDocument()
    expect(
      within(comparisonTable).queryByText('rank_mean_final_return'),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText('运行对比').compareDocumentPosition(
        screen.getByText('结果文件下载'),
      ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(screen.getByRole('checkbox', { name: '研究报告 report.md' })).toBeChecked()
    expect(screen.queryByRole('checkbox', { name: 'Excel 汇总 artifacts.xlsx' })).not.toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: '运行对比 comparison.csv' })).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: '目标权重 CSV' })).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: '可交易诊断 CSV' })).toBeInTheDocument()
    expect(screen.getByText('常用组合')).toBeInTheDocument()
    expect(screen.getByText('选择内容')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '下载所选' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '删除回测' })).not.toBeInTheDocument()
    expect(screen.getAllByText('当前组合最终收益').length).toBeGreaterThan(0)
    const summary = screen.getByLabelText('核心摘要')
    expect(within(summary).getByText('0.02')).toBeInTheDocument()
    expect(
      screen.getByText('当前组合最终收益为 0.02，回撤和收益曲线需要一起看。'),
    ).toBeInTheDocument()
    expect(screen.getAllByText('ICIR').length).toBeGreaterThan(0)
    expect(screen.getByText('核心摘要')).toBeInTheDocument()
    expect(screen.getByText('指标详情')).toBeInTheDocument()
    expect(screen.getByText('收益风险')).toBeInTheDocument()
    expect(screen.getByText('收益风险').closest('details')).toHaveAttribute('open')
    expect(screen.getByText('成本换手')).toBeInTheDocument()
    const cumulativeReturnsChartTitle = await screen.findByText('累计组合收益')
    expect(
      cumulativeReturnsChartTitle.compareDocumentPosition(
        screen.getByText('指标详情'),
      ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      screen.getByText('指标详情').compareDocumentPosition(
        screen.getByText('研究报告'),
      ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(screen.getByRole('combobox', { name: '选择明细表' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '展开预览' })).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'IC / RankIC 表格' })).not.toBeInTheDocument()
    expect(cumulativeReturnsChartTitle).toBeInTheDocument()
    expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute(
      'data-row-count',
      '1',
    )
    expect(Number(screen.getByTestId('cumulative-returns-chart').dataset.yAxisWidth)).toBeGreaterThan(
      46,
    )
    expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute(
      'data-chart-margin-left',
      '0',
    )
    expect(screen.getByTestId('ic-rankic-chart')).toHaveAttribute(
      'data-row-count',
      '30',
    )
    expect(
      Number(screen.getByTestId('ic-rankic-chart').dataset.yAxisWidth),
    ).toBeGreaterThan(46)
    expect(screen.getByTestId('ic-rankic-chart')).toHaveAttribute(
      'data-chart-margin-left',
      '0',
    )
    expect(screen.getByTestId('net-portfolio-return-chart')).toHaveAttribute(
      'data-row-count',
      '1',
    )
    expect(
      Number(screen.getByTestId('net-portfolio-return-chart').dataset.yAxisWidth),
    ).toBeGreaterThan(46)
    expect(screen.getByTestId('net-portfolio-return-chart')).toHaveAttribute(
      'data-chart-margin-left',
      '0',
    )
    expect(screen.getByRole('link', { name: '下载当前 CSV' })).toHaveAttribute(
      'href',
      'http://127.0.0.1:8000/api/runs/core/demo/artifacts/metrics/ic.csv',
    )

    fireEvent.click(screen.getByRole('button', { name: '展开预览' }))
    expect(screen.getByRole('table', { name: 'IC / RankIC 表格' })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: '选择明细表' }), {
      target: { value: 'returns' },
    })
    const table = screen.getByRole('table', { name: '组合收益明细 表格' })
    expect(within(table).getByText('net_portfolio_return')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('combobox', { name: '选择明细表' }), {
      target: { value: 'performance_summary' },
    })
    const performance = await screen.findByRole('table', {
      name: '绩效汇总 表格',
    })
    expect(within(performance).getByText('sharpe_like')).toBeInTheDocument()
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/metrics/performance_summary?limit=20&combination_id=factor__forward_return__dataframe__long_only',
    )
    expect(screen.getByRole('link', { name: '下载当前 CSV' })).toHaveAttribute(
      'href',
      'http://127.0.0.1:8000/api/runs/core/demo/artifacts/metrics/performance_summary.csv',
    )
  })

  test('renders chart legends and descriptions', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
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
          },
        ])
      }
      if (url.endsWith('/api/runs/core/demo/evidence')) {
        return jsonResponse(evidencePayload())
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByTestId('cumulative-returns-chart')).toBeInTheDocument()
    expect(within(screen.getByTestId('cumulative-returns-chart')).getAllByText('cumulative_net_return').length).toBeGreaterThan(1)
    expect(within(screen.getByTestId('cumulative-returns-chart')).getAllByText('drawdown').length).toBeGreaterThan(1)
    expect(within(screen.getByTestId('ic-rankic-chart')).getAllByText('rankic').length).toBeGreaterThan(1)
    expect(
      within(screen.getByTestId('net-portfolio-return-chart')).getAllByText(
        'net_portfolio_return',
      ).length,
    ).toBeGreaterThan(1)
    expect(within(screen.getByTestId('cumulative-returns-chart')).getByText('净值曲线和回撤')).toBeInTheDocument()
  })

  test('renders chart range readouts as structured tokens and splits return diagnostics', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'range_layout_demo',
            public_run_id: 'core/range_layout_demo',
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
        ])
      }
      if (url.endsWith('/api/runs/core/range_layout_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([]),
              cumulative_returns: tablePayload([
                {
                  timestamp: '2026-01-01',
                  cumulative_net_return: -0.1,
                  drawdown: -0.12,
                },
                {
                  timestamp: '2026-01-02',
                  cumulative_net_return: 0.2,
                  drawdown: 0,
                },
              ]),
              returns: tablePayload([
                {
                  timestamp: '2026-01-01',
                  net_portfolio_return: -0.01,
                  turnover: 0.3,
                  cost: 0.0001,
                },
                {
                  timestamp: '2026-01-02',
                  net_portfolio_return: 0.02,
                  turnover: 0.8,
                  cost: 0.0004,
                },
              ]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    const cumulativeReturnsChart = await screen.findByTestId('cumulative-returns-chart')
    expect(within(cumulativeReturnsChart).getByText('2026-01-01 至 2026-01-02')).toBeInTheDocument()
    expect(within(cumulativeReturnsChart).queryByText('数值范围')).not.toBeInTheDocument()
    expect(within(cumulativeReturnsChart).getAllByText('cumulative_net_return').length).toBeGreaterThan(1)
    expect(within(cumulativeReturnsChart).getByText('-10.00% ~ 20.00%')).toBeInTheDocument()
    expect(within(cumulativeReturnsChart).getAllByText('drawdown').length).toBeGreaterThan(1)
    expect(within(cumulativeReturnsChart).getByText('-12.00% ~ 0.00%')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '完整报告' }))
    await waitFor(() => {
      expect(screen.getByTestId('net-portfolio-return-chart')).toBeInTheDocument()
      expect(screen.getByTestId('turnover-chart')).toBeInTheDocument()
      expect(screen.getByTestId('cost-chart')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('return-turnover-cost-chart')).not.toBeInTheDocument()
  })

  test('renders dense bar charts as line charts so time series remain visible', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'dense_chart_demo',
            public_run_id: 'core/dense_chart_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 600,
            weight_method: 'equal_weight_quantile',
          },
        ])
      }
      if (url.endsWith('/api/runs/core/dense_chart_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              cumulative_returns: tablePayload([]),
              ic: tablePayload([]),
              returns: tablePayload(
                Array.from({ length: 180 }, (_, index) => ({
                  timestamp: `2026-01-${String((index % 28) + 1).padStart(2, '0')}`,
                  net_portfolio_return: index / 1000,
                  turnover: 0.4,
                  cost: 0.001,
                })),
              ),
            },
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByTestId('net-portfolio-return-chart')).toHaveAttribute(
      'data-chart-kind',
      'line',
    )
  })

  test('shows chart empty states when evidence tables have no chart rows', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'empty_demo',
            public_run_id: 'core/empty_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 0,
            weight_method: 'equal_weight_quantile',
          },
        ])
      }
      if (url.endsWith('/api/runs/core/empty_demo/evidence')) {
        return jsonResponse({
          ...evidencePayload(),
          tables: {
            ...evidencePayload().tables,
            ic: tablePayload([]),
            returns: tablePayload([]),
            cumulative_returns: tablePayload([]),
          },
        })
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByTestId('cumulative-returns-chart')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '完整报告' }))
    await waitFor(() => {
      expect(screen.getAllByText('暂无可绘制数据').length).toBeGreaterThanOrEqual(5)
    })
  })

  test('renders an API error state', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'backend unavailable' }),
    } as Response)

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('backend unavailable')).toBeInTheDocument()
    })
  })

  test('submits an uploaded CSV run and switches to the result view', async () => {
    globalThis.fetch = vi.fn(
      (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.endsWith('/api/runs') && init?.method === 'POST') {
          return jsonResponse({
            source_alias: 'core',
            run_id: 'uploaded_demo',
            public_run_id: 'core/uploaded_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'user_upload',
            data_stage: 'uploaded_panel',
            uses_external_data: true,
            n_rows_evaluated: 4,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.04,
            mean_turnover: 1,
            total_cost: 0,
            sharpe_like: 0,
            sortino_like: 0,
            max_drawdown: 0,
            win_rate: 1,
            mean_ic: 0.4,
            mean_rankic: 0.5,
            icir: 0,
            rankicir: 0,
          })
        }
        if (url.endsWith('/api/runs')) {
          return jsonResponse([
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
              mean_final_return: 1.6099,
              mean_turnover: 1.54,
              total_cost: 0.077,
            },
          ])
        }
        if (url.endsWith('/api/runs/core/uploaded_demo/evidence')) {
          return jsonResponse(
            evidencePayload({
              summary: {
                run_id: 'uploaded_demo',
                project: 'quant-research-platform',
                kind: 'factor',
                domain: 'user_upload',
                data_stage: 'uploaded_panel',
                uses_external_data: true,
                n_rows_evaluated: 4,
                boundary: 'Uploaded panel.',
              },
              reportMarkdown: '# Uploaded Report\n',
              comparisonRows: [],
            }),
          )
        }
        if (url.endsWith('/api/comparison')) {
          return tableResponse([])
        }
      if (url.includes('/metrics/ic_summary')) {
          return tableResponse([{ feature: 'factor', mean_ic: 0.4, icir: 0 }])
        }
        if (url.includes('/portfolio/cumulative_returns')) {
          return tableResponse([{ timestamp: '2026-01-01', cumulative_net_return: 0.04 }])
        }
        if (url.includes('/portfolio/returns')) {
          return tableResponse([
            { timestamp: '2026-01-01', net_portfolio_return: 0.04 },
          ])
        }
        if (url.includes('/metrics/ic')) {
          return tableResponse([{ timestamp: '2026-01-01', ic: 0.4, rankic: 0.5 }])
        }
        return tableResponse([{ timestamp: '2026-01-01', value: 1 }])
      },
    ) as typeof fetch

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: 'CSV 上传（开发）' }))
    expect(screen.getByLabelText('新建回测说明')).toBeInTheDocument()
    expect(screen.getByText('1 上传面板')).toBeInTheDocument()
    expect(screen.getByText('2 映射列名')).toBeInTheDocument()
    expect(screen.getByText('3 生成结果文件')).toBeInTheDocument()
    expect(screen.getByText('上传因子面板')).toBeInTheDocument()
    expect(screen.queryByText('core/demo')).not.toBeInTheDocument()
    expect(screen.getByText('必填列')).toBeInTheDocument()
    expect(screen.getByText('可选列')).toBeInTheDocument()
    expect(screen.getByText('数据约束')).toBeInTheDocument()
    expect(screen.getAllByText('timestamp').length).toBeGreaterThan(0)
    expect(screen.getAllByText('asset').length).toBeGreaterThan(0)
    expect(screen.getAllByText('forward_return').length).toBeGreaterThan(0)
    expect(screen.getAllByText('forward_return_1d').length).toBeGreaterThan(0)
    expect(screen.getAllByText('必填').length).toBeGreaterThan(0)
    expect(screen.getAllByText('选填').length).toBeGreaterThan(0)
    expect(screen.getByRole('link', { name: '下载模板 CSV' })).toHaveAttribute(
      'href',
      'http://127.0.0.1:8000/api/templates/factor-csv',
    )
    expect(screen.getByLabelText('运行编号 run_id')).toBeInTheDocument()
    expect(screen.getByLabelText('项目名 project')).toBeInTheDocument()
    expect(screen.getByText('基础信息')).toBeInTheDocument()
    expect(screen.getByText('CSV 列映射')).toBeInTheDocument()
    expect(screen.getByText('回测设置')).toBeInTheDocument()
    expect(screen.getByText('高级设置')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '研究类型 kind' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '数据阶段 data_stage' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '权重方法 weight_method' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '组合模式 portfolio_mode' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '信号方向 direction' })).toBeInTheDocument()
    expect(screen.getByLabelText('风险列 risk_column')).toHaveAttribute('type', 'text')
    expect(screen.getByLabelText('组合因子 portfolio_feature')).toHaveAttribute(
      'type',
      'text',
    )
    expect(screen.getByLabelText('上传文件 factor CSV')).toBeInTheDocument()
    expect(
      screen.getAllByRole('button', {
      name: /因子工作台|流程配置|数据检查点|证据工作台|CSV 上传（开发）/,
      }).map(
        (button) => button.textContent,
      ),
    ).toEqual([
      '因子工作台',
      '流程配置',
      '数据检查点',
      '证据工作台',
      'CSV 上传（开发）',
    ])
    expect(screen.getByRole('button', { name: '生成回测' })).toHaveClass(
      'submit-run',
    )

    fireEvent.change(screen.getByLabelText('运行编号 run_id'), {
      target: { value: 'uploaded_demo' },
    })
    const file = new File(
      [
        [
          'timestamp,asset,factor,forward_return,forward_return_1d',
          '2026-01-01,A,0.1,0.02,0.006',
          '2026-01-01,B,0.2,0.01,0.002',
          '2026-01-02,A,0.3,0.03,0.008',
          '2026-01-02,B,0.1,-0.01,-0.002',
        ].join('\n'),
      ],
      'factor.csv',
      { type: 'text/csv' },
    )
    fireEvent.drop(screen.getByLabelText('拖拽上传 CSV'), {
      dataTransfer: { files: [file] },
    })
    expect(await screen.findByText('CSV 预览')).toBeInTheDocument()
    expect(screen.getByText('4 行数据')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '时间列 timestamp_column' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '资产列 asset_column' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: '标签列 label_columns' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '收益来源列 return_source_column' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '风险列 risk_column' })).toBeInTheDocument()
    expect(screen.getAllByText('timestamp').length).toBeGreaterThan(0)
    expect(screen.getAllByText('forward_return').length).toBeGreaterThan(0)
    expect(screen.getAllByText('forward_return_1d').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: '生成回测' }))

    expect(await screen.findByText('已生成回测：core/uploaded_demo')).toBeInTheDocument()
    expect((await screen.findAllByText('core/uploaded_demo')).length).toBeGreaterThan(
      0,
    )
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  test('validates uploaded CSV before posting a new run', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([])
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: 'CSV 上传（开发）' }))
    const file = new File(
      [
        [
          'timestamp,asset,factor',
          '2026-01-01,A,0.1',
          '2026-01-01,B,0.2',
        ].join('\n'),
      ],
      'bad-factor.csv',
      { type: 'text/csv' },
    )
    fireEvent.change(screen.getByLabelText('上传文件 factor CSV'), {
      target: { files: [file] },
    })
    fireEvent.click(screen.getByRole('button', { name: '生成回测' }))

    expect(
      await screen.findByText('CSV 缺少列：forward_return, forward_return_1d'),
    ).toBeInTheDocument()
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  test('posts selected artifact downloads for the active run', async () => {
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
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
            mean_final_return: 1.6099,
            mean_turnover: 1.54,
            total_cost: 0.077,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/demo/evidence')) {
        return jsonResponse(evidencePayload({ comparisonRows: [] }))
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([{ timestamp: '2026-01-01', value: 1 }])
    }) as typeof fetch

    render(<App />)

    await screen.findByText('结果文件下载')
    fireEvent.click(screen.getByRole('tab', { name: '结果文件下载' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'IC / RankIC CSV' }))
    fireEvent.click(screen.getByRole('button', { name: '下载所选' }))

    await waitFor(() => {
      expect(anchorClick).toHaveBeenCalled()
    })
    expect(screen.getByText(/浏览器已接管下载/)).toBeInTheDocument()
    const anchor = anchorClick.mock.contexts[0] as HTMLAnchorElement
    expect(anchor.href).toBe(
      'http://127.0.0.1:8000/api/runs/core/demo/download?items=report.md&items=metrics%2Fic.csv',
    )
    expect(anchorClick).toHaveBeenCalled()
    expect(globalThis.fetch).not.toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/demo/download',
      expect.anything(),
    )
  })

  test('all artifact downloads include only available dynamic breakdown tables', async () => {
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
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
            mean_final_return: 1.6099,
            mean_turnover: 1.54,
            total_cost: 0.077,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([{ timestamp: '2026-01-01', value: 1 }])
    }) as typeof fetch

    render(<App />)

    await screen.findByText('结果文件下载')
    fireEvent.click(screen.getByRole('tab', { name: '结果文件下载' }))
    fireEvent.click(screen.getByRole('button', { name: '全选' }))
    fireEvent.click(screen.getByRole('button', { name: '下载所选' }))

    await waitFor(() => {
      expect(anchorClick).toHaveBeenCalled()
    })
    const anchor = anchorClick.mock.contexts[0] as HTMLAnchorElement
    expect(anchor.href).not.toContain('metrics_by_year%2Fic_summary_by_year.csv')
    expect(anchor.href).not.toContain('metrics_by_quarter%2Fic_summary_by_quarter.csv')
    expect(anchor.href).not.toContain('artifacts.xlsx')
    expect(anchor.href).toContain('items=portfolio%2Freturns.csv')
  })

  test('deletes an uploaded run after confirmation', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/uploaded_demo') && init?.method === 'DELETE') {
        return jsonResponse({ deleted: 'core/uploaded_demo' })
      }
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'uploaded_demo',
            public_run_id: 'core/uploaded_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'user_upload',
            data_stage: 'uploaded_panel',
            uses_external_data: true,
            n_rows_evaluated: 4,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.04,
            mean_turnover: 1,
            total_cost: 0,
          },
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
            mean_final_return: 1.6099,
            mean_turnover: 1.54,
            total_cost: 0.077,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/uploaded_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            summary: {
              run_id: 'uploaded_demo',
              project: 'quant-research-platform',
              kind: 'factor',
              domain: 'user_upload',
              data_stage: 'uploaded_panel',
              uses_external_data: true,
              n_rows_evaluated: 4,
            },
            reportMarkdown: '# Uploaded Report\n',
            comparisonRows: [],
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([{ timestamp: '2026-01-01', value: 1 }])
    }) as typeof fetch

    render(<App />)

    expect((await screen.findAllByText('core/uploaded_demo')).length).toBeGreaterThan(0)
    const deleteButton = await screen.findByRole('button', { name: '删除回测' })
    expect(deleteButton.closest('.reviewer-summary')).toHaveClass('reviewer-summary')
    fireEvent.click(deleteButton)

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/runs/core/uploaded_demo',
        expect.objectContaining({ method: 'DELETE' }),
      )
    })
    expect(confirm).toHaveBeenCalledWith(
      [
        '确认删除回测 core/uploaded_demo？',
        '仅允许删除用户上传面板（data_stage=uploaded_panel）且位于当前运行根目录（run root）下的本地结果目录。',
      ].join('\n'),
    )
    expect(screen.queryByText('core/uploaded_demo')).not.toBeInTheDocument()
    expect(await screen.findByText('已删除回测：core/uploaded_demo')).toBeInTheDocument()
  })

  test('renders tables downloads and charts from evidence display manifest', async () => {
    const tables = {
      performance_summary: tablePayload([
        { timestamp: '2026-01-01', alpha_score: 0.7 },
      ]),
      cumulative_returns: tablePayload([
        {
          timestamp: '2026-01-01',
          cumulative_net_return: 0.12,
          drawdown: -0.02,
        },
      ]),
    }
    const display = displayPayload(tables)
    display.charts = [
      {
        description: 'manifest configured chart',
        id: 'custom_cumulative_returns',
        include_zero: true,
        kind: 'line',
        section: 'primary',
        series: [
          {
            color: '#2563eb',
            format: 'percent',
            key: 'cumulative_net_return',
            label: 'cumulative_net_return',
          },
        ],
        table: 'cumulative_returns',
        title: 'Manifest Returns',
        x: 'timestamp',
      },
    ]
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'manifest_demo',
            public_run_id: 'core/manifest_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 10,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.12,
            mean_turnover: 0,
            total_cost: 0,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/manifest_demo/evidence')) {
        return jsonResponse(evidencePayload({ comparisonRows: [], display, tables }))
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByText('Manifest Returns')).toBeInTheDocument()
    expect(screen.queryByTestId('ic-rankic-chart')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: '结果文件下载' }))
    expect(screen.getByRole('checkbox', { name: '绩效汇总 CSV' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'CSV 明细' }))
    expect(screen.getByRole('combobox', { name: '选择明细表' })).toHaveValue(
      'performance_summary',
    )
    expect(screen.getByRole('link', { name: '下载当前 CSV' })).toHaveAttribute(
      'href',
      'http://127.0.0.1:8000/api/runs/core/manifest_demo/artifacts/metrics/performance_summary.csv',
    )
  })

  test('splits combination selection by available dimensions and shows missing combinations', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'combo_demo',
            public_run_id: 'core/combo_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'a_share',
            data_stage: 'approved_offline_real_data',
            uses_external_data: true,
            label_columns: ['forward_return_20d'],
            return_source_column: 'forward_return_1d',
            n_combinations: 2,
            n_rows_evaluated: 12231140,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 1.05,
            mean_turnover: 0.4,
            total_cost: 0.01,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/combo_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'combo_1',
                  feature: 'rev_20d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
                {
                  combination_id: 'combo_2',
                  feature: 'rev_60d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'market_neutral',
                },
              ]),
              cumulative_returns: tablePayload([
                {
                  combination_id: 'combo_1',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.02,
                  drawdown: 0,
                },
                {
                  combination_id: 'combo_2',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.04,
                  drawdown: 0,
                },
              ]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByLabelText('选择 feature')).toBeInTheDocument()
    expect(screen.getByLabelText('选择 portfolio_mode')).toBeInTheDocument()
    expect(screen.queryByLabelText('展示组合 combination_id')).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('选择 feature'), {
      target: { value: 'rev_20d' },
    })
    fireEvent.change(screen.getByLabelText('选择 portfolio_mode'), {
      target: { value: 'market_neutral' },
    })

    expect(screen.getByText('没有匹配的组合配置')).toBeInTheDocument()
    expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute('data-row-count', '0')
  })

  test('loads selected-combination chart tables instead of filtering truncated evidence', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'combo_chart_demo',
            public_run_id: 'core/combo_chart_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'a_share',
            data_stage: 'approved_offline_real_data',
            uses_external_data: true,
            n_combinations: 2,
            n_rows_evaluated: 100,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.12,
            mean_turnover: 0.4,
            total_cost: 0,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/combo_chart_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'combo_1',
                  feature: 'rev_20d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
                {
                  combination_id: 'combo_2',
                  feature: 'rev_60d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
              ]),
              cumulative_returns: tablePayload([
                {
                  combination_id: 'combo_1',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.02,
                  drawdown: 0,
                },
              ]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (
        url ===
        'http://127.0.0.1:8000/api/runs/core/combo_chart_demo/portfolio/cumulative_returns?limit=1000&combination_id=combo_2'
      ) {
        return tableResponse([
          {
            combination_id: 'combo_2',
            timestamp: '2026-01-01',
            cumulative_net_return: 0.04,
            drawdown: 0,
          },
          {
            combination_id: 'combo_2',
            timestamp: '2026-01-02',
            cumulative_net_return: 0.05,
            drawdown: -0.01,
          },
        ])
      }
      if (
        url ===
        'http://127.0.0.1:8000/api/runs/core/combo_chart_demo/portfolio/cumulative_returns?limit=1000&combination_id=combo_1'
      ) {
        return tableResponse([
          {
            combination_id: 'combo_1',
            timestamp: '2026-01-01',
            cumulative_net_return: 0.02,
            drawdown: 0,
          },
        ])
      }
      if (url.includes('combo_chart_demo') && url.includes('combination_id=')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByTestId('cumulative-returns-chart')).toHaveAttribute(
      'data-row-count',
      '1',
    )
    fireEvent.change(screen.getByLabelText('选择 feature'), {
      target: { value: 'rev_60d' },
    })

    await waitFor(() => {
      expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute(
        'data-row-count',
        '2',
      )
    })
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/runs/core/combo_chart_demo/portfolio/cumulative_returns?limit=1000&combination_id=combo_2',
    )
  })

  test('does not reuse stale chart evidence when selected-combination preview fails', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'combo_chart_error',
            public_run_id: 'core/combo_chart_error',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'a_share',
            data_stage: 'approved_offline_real_data',
            uses_external_data: true,
            n_combinations: 2,
            n_rows_evaluated: 100,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 0.12,
            mean_turnover: 0.4,
            total_cost: 0,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/combo_chart_error/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'combo_1',
                  feature: 'rev_20d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
                {
                  combination_id: 'combo_2',
                  feature: 'rev_60d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
              ]),
              cumulative_returns: tablePayload([
                {
                  combination_id: 'combo_1',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.02,
                  drawdown: 0,
                },
              ]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (
        url ===
        'http://127.0.0.1:8000/api/runs/core/combo_chart_error/portfolio/cumulative_returns?limit=1000&combination_id=combo_2'
      ) {
        return Promise.resolve({
          ok: false,
          status: 500,
          json: async () => ({ detail: 'chart preview unavailable' }),
        } as Response)
      }
      if (url.includes('combo_chart_error') && url.includes('combination_id=')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect(await screen.findByTestId('cumulative-returns-chart')).toHaveAttribute(
      'data-row-count',
      '1',
    )
    fireEvent.change(screen.getByLabelText('选择 feature'), {
      target: { value: 'rev_60d' },
    })

    expect(await screen.findByText('chart preview unavailable')).toBeInTheDocument()
    expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute(
      'data-row-count',
      '0',
    )
  })

  test('caches loaded run evidence when switching runs', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'first',
            public_run_id: 'core/first',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 600,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 1.2,
            mean_turnover: 1,
            total_cost: 0,
          },
          {
            source_alias: 'core',
            run_id: 'second',
            public_run_id: 'core/second',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 700,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 1.3,
            mean_turnover: 1,
            total_cost: 0,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/first/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [
              {
                public_run_id: 'core/first',
                mean_ic: 0.4,
                mean_rankic: 0.5,
                mean_final_return: 0.6,
              },
            ],
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'shared_combo',
                  feature: 'first_factor',
                  label: 'forward_return',
                },
                {
                  combination_id: 'first_other',
                  feature: 'first_other_factor',
                  label: 'forward_return',
                },
              ]),
              cumulative_returns: tablePayload([]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (url.endsWith('/api/runs/core/second/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            summary: {
              run_id: 'second',
              project: 'quant-research-platform',
              kind: 'factor',
              domain: 'core_synthetic',
              data_stage: 'synthetic_only',
              uses_external_data: false,
              n_rows_evaluated: 700,
            },
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'shared_combo',
                  feature: 'second_factor',
                  label: 'forward_return',
                },
                {
                  combination_id: 'second_other',
                  feature: 'second_other_factor',
                  label: 'forward_return',
                },
              ]),
              cumulative_returns: tablePayload([]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    expect((await screen.findAllByText('core/first')).length).toBeGreaterThan(0)
    expect(await screen.findByLabelText('选择 feature')).toHaveValue('first_factor')
    fireEvent.click(screen.getByRole('tab', { name: '运行对比' }))
    expect(screen.getByRole('table', { name: '运行对比表格' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /core\/second/ }))
    expect((await screen.findAllByText('core/second')).length).toBeGreaterThan(0)
    expect(await screen.findByLabelText('选择 feature')).toHaveValue('second_factor')
    expect(
      within(screen.getByLabelText('数据与交付')).getByRole('tab', {
        name: '指标详情',
      }),
    ).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('table', { name: '运行对比表格' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /core\/first/ }))
    expect(
      await screen.findByRole('heading', { name: 'core/first' }),
    ).toBeInTheDocument()
    expect(await screen.findByLabelText('选择 feature')).toHaveValue('first_factor')

    const firstEvidenceCalls = vi.mocked(globalThis.fetch).mock.calls.filter(
      ([input]) => String(input).endsWith('/api/runs/core/first/evidence'),
    )
    expect(firstEvidenceCalls).toHaveLength(1)
  })

  test('reloads evidence for the same public run id after cache version changes', async () => {
    const run: RunSummary = {
      source_alias: 'core',
      run_id: 'uploaded_demo',
      public_run_id: 'core/uploaded_demo',
      project: 'quant-research-platform',
      kind: 'factor',
      domain: 'user_upload',
      data_stage: 'uploaded_panel',
      uses_external_data: true,
      n_rows_evaluated: 4,
      weight_method: 'equal_weight_quantile',
      mean_turnover: 1,
      total_cost: 0,
    }
    let evidenceCalls = 0
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs/core/uploaded_demo/evidence')) {
        evidenceCalls += 1
        return jsonResponse(
          evidencePayload({
            reportMarkdown: `# Uploaded Report ${evidenceCalls}`,
            summary: {
              run_id: 'uploaded_demo',
              project: 'quant-research-platform',
              kind: 'factor',
              domain: 'user_upload',
              data_stage: 'uploaded_panel',
              uses_external_data: true,
              n_rows_evaluated: 4,
            },
          }),
        )
      }
      return tableResponse([])
    }) as typeof fetch

    function EvidenceProbe({
      cacheVersion,
    }: {
      cacheVersion: number
    }) {
      const { runState } = useRunEvidence(run, cacheVersion)
      return <div data-testid="report-markdown">{runState.reportMarkdown}</div>
    }

    const { rerender } = render(<EvidenceProbe cacheVersion={0} />)

    expect(await screen.findByText('# Uploaded Report 1')).toBeInTheDocument()
    rerender(<EvidenceProbe cacheVersion={1} />)

    expect(await screen.findByText('# Uploaded Report 2')).toBeInTheDocument()
    const evidenceFetches = vi.mocked(globalThis.fetch).mock.calls.filter(
      ([input]) => String(input).endsWith('/api/runs/core/uploaded_demo/evidence'),
    )
    expect(evidenceFetches).toHaveLength(2)
  })

  test('formats chart values as percentages and toggles series from the legend', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'chart_demo',
            public_run_id: 'core/chart_demo',
            project: 'quant-research-platform',
            kind: 'factor',
            domain: 'core_synthetic',
            data_stage: 'synthetic_only',
            uses_external_data: false,
            n_rows_evaluated: 600,
            weight_method: 'equal_weight_quantile',
            mean_final_return: 1.2,
            mean_turnover: 1,
            total_cost: 0,
          },
        ])
      }
      if (url.endsWith('/api/runs/core/chart_demo/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([]),
              cumulative_returns: tablePayload([
                {
                  timestamp: '2026-01-01',
                  cumulative_net_return: 1.1219,
                  drawdown: -0.3156,
                },
                {
                  timestamp: '2026-01-02',
                  cumulative_net_return: 0.8844,
                  drawdown: -0.4856,
                },
              ]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    const chart = await screen.findByTestId('cumulative-returns-chart')
    expect(within(chart).getByText(/112\.19%/)).toBeInTheDocument()
    const drawdownToggle = within(chart).getByRole('button', { name: '隐藏 drawdown' })
    fireEvent.click(drawdownToggle)
    expect(within(chart).getByRole('button', { name: '显示 drawdown' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
  })

  test('updates summary metrics when the selected combination changes', async () => {
    globalThis.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/runs')) {
        return jsonResponse([
          {
            source_alias: 'core',
            run_id: 'combo_metrics',
            public_run_id: 'core/combo_metrics',
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
        ])
      }
      if (url.endsWith('/api/runs/core/combo_metrics/evidence')) {
        return jsonResponse(
          evidencePayload({
            comparisonRows: [],
            tables: {
              combinations: tablePayload([
                {
                  combination_id: 'combo_a',
                  feature: 'rev_20d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
                {
                  combination_id: 'combo_b',
                  feature: 'rev_60d',
                  label: 'forward_return_20d',
                  kernel: 'dataframe',
                  portfolio_mode: 'long_only',
                },
              ]),
              cumulative_returns: tablePayload([
                {
                  combination_id: 'combo_a',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.11,
                  drawdown: -0.01,
                },
                {
                  combination_id: 'combo_b',
                  timestamp: '2026-01-01',
                  cumulative_net_return: 0.33,
                  drawdown: -0.03,
                },
              ]),
              returns: tablePayload([]),
              ic: tablePayload([]),
            },
            combinationSummaryRows: [
              {
                combination_id: 'combo_a',
                final_cumulative_net_return: 0.11,
                sharpe_like: 1.1,
                sortino_like: 1.2,
                max_drawdown: -0.1,
                win_rate: 0.55,
                mean_ic: 0.01,
                icir: 0.2,
                mean_rankic: 0.03,
                rankicir: 0.4,
                mean_turnover: 0.8,
                total_cost: 0.05,
              },
              {
                combination_id: 'combo_b',
                final_cumulative_net_return: 0.33,
                sharpe_like: 2.2,
                sortino_like: 2.4,
                max_drawdown: -0.2,
                win_rate: 0.66,
                mean_ic: 0.07,
                icir: 0.8,
                mean_rankic: 0.09,
                rankicir: 1.1,
                mean_turnover: 0.9,
                total_cost: 0.06,
              },
            ],
          }),
        )
      }
      if (url.endsWith('/api/comparison')) {
        return tableResponse([])
      }
      return tableResponse([])
    }) as typeof fetch

    render(<App />)

    await screen.findByRole('heading', { name: 'core/combo_metrics' })
    expect(screen.getAllByText('0.11').length).toBeGreaterThan(0)
    expect(screen.getAllByText('1.1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0.03').length).toBeGreaterThan(0)

    fireEvent.change(screen.getByRole('combobox', { name: '选择 feature' }), {
      target: { value: 'rev_60d' },
    })

    expect(screen.getAllByText('0.33').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2.2').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0.09').length).toBeGreaterThan(0)
    expect(screen.getByTestId('cumulative-returns-chart')).toHaveAttribute('data-row-count', '1')

    fireEvent.click(screen.getByRole('tab', { name: 'CSV 明细' }))
    fireEvent.change(screen.getByRole('combobox', { name: '选择明细表' }), {
      target: { value: 'cumulative_returns' },
    })
    fireEvent.click(screen.getByRole('button', { name: '展开预览' }))
    const cumulativeReturnsTable = screen.getByRole('table', { name: '累计收益曲线 表格' })
    expect(within(cumulativeReturnsTable).queryByText('combo_a')).not.toBeInTheDocument()
    expect(within(cumulativeReturnsTable).getByText('combo_b')).toBeInTheDocument()
  })
})
