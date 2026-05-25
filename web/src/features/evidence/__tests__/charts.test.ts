import '@testing-library/jest-dom/vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, test } from 'vitest'
import { rangeReadout } from '../chartRange'
import { RunLineChartPanel } from '../charts'

describe('chart range readouts', () => {
  test('reports value ranges per series instead of merging unrelated curves', () => {
    const readout = rangeReadout(
      [
        { timestamp: '2026-01-01', cumulative_net_return: 0.1, drawdown: -0.03 },
        { timestamp: '2026-01-02', cumulative_net_return: 0.5, drawdown: -0.2 },
      ],
      [
        { key: 'cumulative_net_return', label: 'cumulative_net_return' },
        { key: 'drawdown', label: 'drawdown' },
      ],
      (value) => `${value}`,
    )

    expect(readout.datePart).toBe('2026-01-01 至 2026-01-02')
    expect(readout.seriesRanges).toEqual([
      { label: 'cumulative_net_return', minValue: '0.1', maxValue: '0.5' },
      { label: 'drawdown', minValue: '-0.2', maxValue: '-0.03' },
    ])
  })

  test('uses the configured x key for date ranges', () => {
    const readout = rangeReadout(
      [
        { period: '2026-Q1', value: 0.1 },
        { period: '2026-Q2', value: 0.2 },
      ],
      [{ key: 'value' }],
      (value) => `${value}`,
      'period',
    )

    expect(readout.datePart).toBe('2026-Q1 至 2026-Q2')
  })
})

describe('interactive chart panels', () => {
  test('resets hidden series when chart data changes', async () => {
    const series = [
      { color: '#2563eb', key: 'value_a', label: 'value_a' },
      { color: '#dc2626', key: 'value_b', label: 'value_b' },
    ]
    const { rerender } = render(
      createElement(RunLineChartPanel, {
        rows: [{ timestamp: '2026-01-01', value_a: 1, value_b: 2 }],
        series,
        testId: 'reset-chart',
        title: 'Reset Chart',
        xKey: 'timestamp',
      }),
    )

    fireEvent.click(screen.getByRole('button', { name: '隐藏 value_b' }))
    expect(screen.getByRole('button', { name: '显示 value_b' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )

    rerender(
      createElement(RunLineChartPanel, {
        rows: [{ timestamp: '2026-01-02', value_a: 3, value_b: 4 }],
        series,
        testId: 'reset-chart',
        title: 'Reset Chart',
        xKey: 'timestamp',
      }),
    )

    await waitFor(() => {
      expect(
        within(screen.getByTestId('reset-chart')).getByRole('button', {
          name: '隐藏 value_b',
        }),
      ).toHaveAttribute('aria-pressed', 'true')
    })
  })
})
