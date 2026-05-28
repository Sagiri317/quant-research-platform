import { expect, test } from '@playwright/test'

const webBaseUrl = process.env.QRP_E2E_WEB_BASE_URL ?? 'http://127.0.0.1:5173'
const apiBaseUrl = process.env.QRP_E2E_API_BASE_URL ?? 'http://127.0.0.1:8000'
const executablePath = process.env.QRP_E2E_CHROMIUM_EXECUTABLE_PATH

if (executablePath) {
  test.use({ launchOptions: { executablePath } })
}

test('schema 0.7 workbench renders diagnostics and display-backed downloads', async ({
  page,
  request,
}) => {
  const runsResponse = await request.get(`${apiBaseUrl}/api/runs`)
  expect(runsResponse.ok()).toBeTruthy()
  const runs = (await runsResponse.json()) as Array<{
    public_run_id: string
    run_id: string
    source_alias: string
  }>
  expect(runs.length).toBeGreaterThan(0)
  const run = runs[0]

  const evidenceResponse = await request.get(
    `${apiBaseUrl}/api/runs/${run.source_alias}/${run.run_id}/evidence`,
  )
  expect(evidenceResponse.ok()).toBeTruthy()
  const evidence = (await evidenceResponse.json()) as {
    display: {
      tables: Array<{ key: string }>
      downloads: {
        items: Array<{ kind: string; key: string }>
        presets: Array<{ key: string; label: string }>
      }
    }
    summary: { run_id: string }
  }
  expect(evidence.summary.run_id).toBe(run.run_id)
  expect(evidence.display.tables.some((table) => table.key === 'diagnostic_summary')).toBe(
    true,
  )
  const exposesWeightsDownload = evidence.display.downloads.items.some(
    (item) => item.kind === 'table' && item.key === 'weights',
  )
  const firstDownloadPreset = evidence.display.downloads.presets[0]
  expect(firstDownloadPreset).toBeTruthy()
  if (exposesWeightsDownload) {
    expect(evidence.display.tables.some((table) => table.key === 'weights')).toBe(false)
  }

  const downloadResponse = await request.get(
    `${apiBaseUrl}/api/runs/${run.source_alias}/${run.run_id}/download?items=table%3Adiagnostic_summary&items=metadata_file%3Arun_context`,
  )
  expect(downloadResponse.ok()).toBeTruthy()

  const factoryResponse = await request.get(`${apiBaseUrl}/api/factor-factory-runs`)
  expect(factoryResponse.ok()).toBeTruthy()
  const factoryRuns = (await factoryResponse.json()) as Array<{ run_id: string }>
  expect(factoryRuns.length).toBeGreaterThan(0)

  await page.goto(webBaseUrl)
  await page.getByRole('button', { exact: true, name: '证据工作台' }).click()
  await expect(page.locator('.run-list')).toContainText(run.public_run_id)
  await page.getByRole('button', { name: run.public_run_id }).click()

  await page.getByRole('tab', { exact: true, name: 'CSV 明细' }).click()
  await page.locator('.csv-table-select select').selectOption('diagnostic_summary')
  await page.getByRole('button', { exact: true, name: '展开预览' }).click()
  await expect(page.locator('table')).toContainText('diagnostic_status')

  await page.getByRole('tab', { exact: true, name: '结果文件下载' }).click()
  await page
    .getByRole('button', { exact: true, name: firstDownloadPreset.label })
    .click()
  const download = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { exact: true, name: '下载所选' }).click(),
  ]).then(([artifactDownload]) => artifactDownload)
  expect(download.suggestedFilename()).toContain('selected_artifacts.zip')

  await page.getByRole('button', { exact: true, name: '因子工厂' }).click()
  await expect(page.locator('.factory-run-list')).toContainText(factoryRuns[0].run_id)
})
