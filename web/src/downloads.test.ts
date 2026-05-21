import { describe, expect, test } from 'vitest'
import { buildDownloadOptions, defaultDownloadItems, presetDownloadItems } from './downloads'
import type { RunDisplayDownloads } from './api'

describe('artifact download contract helpers', () => {
  test('do not invent download items when manifest display downloads are absent', () => {
    expect(buildDownloadOptions()).toEqual([])
    expect(defaultDownloadItems()).toEqual([])
    expect(presetDownloadItems(undefined, 'all')).toEqual([])
  })

  test('uses manifest presets for default download selection', () => {
    const displayDownloads: RunDisplayDownloads = {
      items: [
        { item: 'report.md', kind: 'file', label: 'Report' },
        { item: 'comparison.csv', kind: 'file', label: 'Comparison' },
        { item: 'metrics/ic.csv', kind: 'table', label: 'IC' },
      ],
      presets: [
        { key: 'all', label: 'All', items: ['report.md', 'comparison.csv', 'metrics/ic.csv'] },
        { key: 'report', label: 'Report pack', items: ['report.md', 'comparison.csv'] },
      ],
    }

    expect(defaultDownloadItems(displayDownloads)).toEqual(['report.md', 'comparison.csv'])
    expect(presetDownloadItems(displayDownloads, 'all')).toEqual([
      'report.md',
      'comparison.csv',
      'metrics/ic.csv',
    ])
  })
})
