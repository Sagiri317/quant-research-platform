import { describe, expect, test } from 'vitest'
import { buildDownloadOptions, defaultDownloadItems, presetDownloadItems } from '../../../shared/downloads'
import type { RunDisplayDownloads } from '../../../shared/api/api'

describe('artifact download contract helpers', () => {
  test('do not invent download items when manifest display downloads are absent', () => {
    expect(buildDownloadOptions()).toEqual([])
    expect(defaultDownloadItems()).toEqual([])
    expect(presetDownloadItems(undefined, 'all')).toEqual([])
  })

  test('uses manifest presets for default download selection', () => {
    const displayDownloads: RunDisplayDownloads = {
      items: [
        { kind: 'metadata_file', key: 'report', label: 'Report' },
        { kind: 'dynamic', key: 'comparison', label: 'Comparison' },
        { kind: 'table', key: 'ic', label: 'IC' },
      ],
      presets: [
        {
          key: 'all',
          label: 'All',
          items: [
            { kind: 'metadata_file', key: 'report' },
            { kind: 'table', key: 'ic' },
          ],
        },
        {
          key: 'report',
          label: 'Report pack',
          items: [
            { kind: 'metadata_file', key: 'report' },
            { kind: 'dynamic', key: 'comparison' },
          ],
        },
      ],
    }

    expect(defaultDownloadItems(displayDownloads)).toEqual([
      'metadata_file:report',
      'dynamic:comparison',
    ])
    expect(presetDownloadItems(displayDownloads, 'all')).toEqual([
      'metadata_file:report',
      'table:ic',
    ])
  })
})
