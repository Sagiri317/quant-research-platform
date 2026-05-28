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
  await page.getByTestId('view-mode-results').click()
  await expect(page.locator('.run-list')).toContainText(run.public_run_id)
  await page.getByTestId(`run-selector-${run.public_run_id}`).click()

  await page.getByTestId('delivery-tab-csv').click()
  await page.getByTestId('csv-table-select').selectOption('diagnostic_summary')
  await page.getByTestId('csv-preview-toggle').click()
  await expect(page.locator('table')).toContainText('diagnostic_status')

  await page.getByTestId('delivery-tab-downloads').click()
  await page.getByTestId(`download-preset-${firstDownloadPreset.key}`).click()
  const download = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('download-selected-artifacts').click(),
  ]).then(([artifactDownload]) => artifactDownload)
  expect(download.suggestedFilename()).toContain('selected_artifacts.zip')

  await page.getByTestId('view-mode-factory').click()
  await expect(page.locator('.factory-run-list')).toContainText(factoryRuns[0].run_id)
})
