import '@testing-library/jest-dom/vitest'
import { render, screen, within } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { tablePayload } from '../../../shared/testFixtures'
import { TradabilityDiagnostics } from '../TradabilityDiagnostics'

describe('TradabilityDiagnostics', () => {
  test('renders the selected combination diagnostics from API artifact rows', () => {
    render(
      <TradabilityDiagnostics
        latestDiagnostic={tablePayload([
          {
            combination_id: 'combo_b',
            timestamp: '2026-01-02',
            universe_count: 50,
            eligible_count: 25,
            can_enter_count: 20,
            can_exit_count: 15,
            entry_blocked_count: 5,
            exit_blocked_count: 10,
            eligible_rate: 0.5,
            can_enter_rate: 0.4,
            can_exit_rate: 0.3,
            entry_blocked_rate: 0.1,
            exit_blocked_rate: 0.2,
            fallback_reason: 'can_enter missing; treated as tradable',
          },
        ])}
      />,
    )

    const panel = screen.getByLabelText('可交易性诊断')
    expect(within(panel).getByText('2026-01-02')).toBeInTheDocument()
    expect(within(panel).getByText('50')).toBeInTheDocument()
    expect(within(panel).getByText('50.00%')).toBeInTheDocument()
    expect(within(panel).getByText('40.00%')).toBeInTheDocument()
    expect(within(panel).getByText('30.00%')).toBeInTheDocument()
    expect(
      within(panel).getByText('can_enter missing; treated as tradable'),
    ).toBeInTheDocument()
    expect(within(panel).queryByText('2026-01-01')).not.toBeInTheDocument()
  })

  test('stays absent when manifest display did not expose diagnostics rows', () => {
    const { container } = render(
      <TradabilityDiagnostics
        latestDiagnostic={undefined}
      />,
    )

    expect(container).toBeEmptyDOMElement()
  })
})
